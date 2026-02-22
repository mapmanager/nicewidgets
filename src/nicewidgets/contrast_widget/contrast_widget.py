"""Contrast widget for image display adjustment.

Self-contained NiceGUI widget with color LUT, log checkbox, lazy histogram,
and min/max sliders. Uses Plotly dicts only for ui.plotly (never go.Figure).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import plotly.graph_objects as go
from nicegui import ui

from nicewidgets.contrast_widget.colorscales import COLORSCALE_OPTIONS
from nicewidgets.contrast_widget.histogram import histogram_plot_plotly
from nicewidgets.contrast_widget.theme import ThemeMode
from nicewidgets.utils.lazy_section import LazySection, LazySectionConfig


def _resolve_theme(theme: Union[str, ThemeMode]) -> ThemeMode:
    """Convert str to ThemeMode. Default to LIGHT."""
    if isinstance(theme, ThemeMode):
        return theme
    s = str(theme).lower()
    if s in ("dark", "plotly_dark"):
        return ThemeMode.DARK
    return ThemeMode.LIGHT


def _safe_call(func: Callable, *args, **kwargs) -> None:
    """Safely call a function, catching 'client deleted' RuntimeErrors only."""
    try:
        func(*args, **kwargs)
    except RuntimeError as e:
        if "deleted" not in str(e).lower():
            raise


@dataclass(frozen=True)
class ContrastParams:
    """Immutable contrast display parameters."""

    colorscale: str
    zmin: int
    zmax: int


OnContrastChange = Callable[[ContrastParams], None]


class ContrastWidget:
    """Reusable contrast adjustment widget.

    Displays color LUT select, Log checkbox, lazy-loaded histogram,
    and min/max sliders. Emits ContrastParams via on_change callback.
    """

    def __init__(
        self,
        *,
        on_change: OnContrastChange,
        theme: Union[str, ThemeMode] = "light",
    ) -> None:
        self._on_change = on_change
        self._theme = _resolve_theme(theme)
        self._current_image: Optional[np.ndarray] = None
        self._display_params: Optional[ContrastParams] = None
        self._updating_programmatically = False

        self._colorscale_select: Optional[ui.select] = None
        self._log_checkbox: Optional[ui.checkbox] = None
        self._histogram_plot: Optional[ui.plotly] = None
        self._min_slider: Optional[ui.slider] = None
        self._max_slider: Optional[ui.slider] = None
        self._min_value_label: Optional[ui.label] = None
        self._max_value_label: Optional[ui.label] = None

    def render(self) -> None:
        """Create the contrast controls UI inside the current container."""
        self._colorscale_select = None
        self._log_checkbox = None
        self._histogram_plot = None
        self._min_slider = None
        self._max_slider = None
        self._min_value_label = None
        self._max_value_label = None
        self._updating_programmatically = False

        # Row 1: Color LUT and Log checkbox
        with ui.row().classes("w-full gap-4 items-center"):
            colorscale_options = [opt["value"] for opt in COLORSCALE_OPTIONS]
            self._colorscale_select = ui.select(
                colorscale_options,
                value="Gray",
                label="Color LUT",
            ).classes("flex-1")
            self._colorscale_select.on("update:model-value", self._on_colorscale_change)

            self._log_checkbox = ui.checkbox("Log", value=True)
            self._log_checkbox.on("update:model-value", self._on_log_toggle)

        # Row 2: Lazy histogram (Plotly dict only)
        empty_histogram_dict = go.Figure().to_dict()

        def _render_histogram(container: ui.element) -> None:
            with container:
                self._histogram_plot = ui.plotly(empty_histogram_dict).classes("w-full h-48")
            self._update_histogram()

        LazySection(
            "Histogram",
            render_fn=_render_histogram,
            config=LazySectionConfig(
                render_once=False,
                clear_on_close=True,
                show_spinner=True,
                on_clear=lambda: setattr(self, "_histogram_plot", None),
            ),
        )

        # Row 3: Min slider
        with ui.row().classes("w-full items-center gap-2"):
            ui.label("Min:").classes("w-12")
            self._min_slider = ui.slider(
                min=0, max=255, value=0, step=1
            ).classes("flex-1")
            self._min_slider.on(
                "update:model-value",
                self._on_slider_change,
                throttle=0.2,
            )
            self._min_value_label = ui.label("0").classes("w-16")

        # Row 4: Max slider
        with ui.row().classes("w-full items-center gap-2"):
            ui.label("Max:").classes("w-12")
            self._max_slider = ui.slider(
                min=0, max=255, value=255, step=1
            ).classes("flex-1")
            self._max_slider.on(
                "update:model-value",
                self._on_slider_change,
                throttle=0.2,
            )
            self._max_value_label = ui.label("255").classes("w-16")

    def set_image(self, image: Optional[np.ndarray]) -> None:
        """Update histogram and slider range for new image."""
        _safe_call(self._set_image_impl, image)

    def _set_image_impl(self, image: Optional[np.ndarray]) -> None:
        self._current_image = image
        colorscale = self._display_params.colorscale if self._display_params else "Gray"
        if image is None:
            self._display_params = ContrastParams(colorscale=colorscale, zmin=0, zmax=255)
            self._updating_programmatically = True
            try:
                if self._min_slider is not None:
                    self._min_slider.props("max=255")
                    self._min_slider.value = 0
                if self._max_slider is not None:
                    self._max_slider.props("max=255")
                    self._max_slider.value = 255
                if self._min_value_label is not None:
                    self._min_value_label.text = "0"
                if self._max_value_label is not None:
                    self._max_value_label.text = "255"
            finally:
                self._updating_programmatically = False
        else:
            image_max = int(np.max(image))
            self._display_params = ContrastParams(colorscale=colorscale, zmin=0, zmax=image_max)
            self._updating_programmatically = True
            try:
                if self._min_slider is not None:
                    self._min_slider.props(f"max={image_max}")
                    self._min_slider.value = 0
                if self._max_slider is not None:
                    self._max_slider.props(f"max={image_max}")
                    self._max_slider.value = image_max
                if self._min_value_label is not None:
                    self._min_value_label.text = "0"
                if self._max_value_label is not None:
                    self._max_value_label.text = str(image_max)
            finally:
                self._updating_programmatically = False
        self._update_histogram()

    def set_params(self, params: ContrastParams) -> None:
        """Update sliders and colorscale from ContrastParams."""
        _safe_call(self._set_params_impl, params)

    def _set_params_impl(self, params: ContrastParams) -> None:
        self._display_params = params
        self._updating_programmatically = True
        try:
            if self._colorscale_select is not None and params.colorscale:
                self._colorscale_select.value = params.colorscale
            if params.zmin is not None and self._min_slider is not None:
                self._min_slider.value = params.zmin
                if self._min_value_label is not None:
                    self._min_value_label.text = str(params.zmin)
            if params.zmax is not None and self._max_slider is not None:
                self._max_slider.value = params.zmax
                if self._max_value_label is not None:
                    self._max_value_label.text = str(params.zmax)
        finally:
            self._updating_programmatically = False
        self._update_histogram()

    def set_theme(self, theme: Union[str, ThemeMode]) -> None:
        """Update theme for histogram."""
        _safe_call(self._set_theme_impl, theme)

    def _set_theme_impl(self, theme: Union[str, ThemeMode]) -> None:
        self._theme = _resolve_theme(theme)
        self._update_histogram()

    def _update_histogram(self) -> None:
        if self._histogram_plot is None:
            return
        zmin = (
            self._display_params.zmin
            if self._display_params is not None
            else 0
        )
        zmax = (
            self._display_params.zmax
            if self._display_params is not None
            else 255
        )
        log_scale = self._log_checkbox.value if self._log_checkbox is not None else True
        fig_dict = histogram_plot_plotly(
            image=self._current_image,
            zmin=zmin,
            zmax=zmax,
            log_scale=log_scale,
            theme=self._theme,
        )
        try:
            self._histogram_plot.update_figure(fig_dict)
        except RuntimeError as e:
            if "deleted" not in str(e).lower():
                raise

    def _on_colorscale_change(self) -> None:
        if self._updating_programmatically or self._colorscale_select is None:
            return
        colorscale = self._colorscale_select.value
        zmin = self._display_params.zmin if self._display_params else 0
        zmax = self._display_params.zmax if self._display_params else 255
        self._emit(ContrastParams(colorscale=colorscale, zmin=zmin, zmax=zmax))

    def _on_slider_change(self) -> None:
        if self._updating_programmatically:
            return
        if self._min_slider is None or self._max_slider is None:
            return
        new_zmin = int(self._min_slider.value)
        new_zmax = int(self._max_slider.value)
        if new_zmin > new_zmax:
            self._updating_programmatically = True
            try:
                if self._min_slider.value == new_zmin:
                    new_zmax = new_zmin
                    self._max_slider.value = new_zmax
                    if self._max_value_label:
                        self._max_value_label.text = str(new_zmax)
                else:
                    new_zmin = new_zmax
                    self._min_slider.value = new_zmin
                    if self._min_value_label:
                        self._min_value_label.text = str(new_zmin)
            finally:
                self._updating_programmatically = False
        if self._min_value_label:
            self._min_value_label.text = str(new_zmin)
        if self._max_value_label:
            self._max_value_label.text = str(new_zmax)
        colorscale = (
            self._display_params.colorscale
            if self._display_params
            else "Gray"
        )
        self._emit(ContrastParams(colorscale=colorscale, zmin=new_zmin, zmax=new_zmax))
        self._update_histogram()

    def _on_log_toggle(self) -> None:
        self._update_histogram()

    def _emit(self, params: ContrastParams) -> None:
        self._on_change(params)
