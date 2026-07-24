"""One-row contrast widget: color LUT select, Auto button, min/max range.

The widget is intentionally host-application-agnostic. It owns its UI state and emits
a single :class:`ContrastChangedIntent` whenever the user changes any control.
Auto contrast is computed locally inside the widget using an injected
``auto_contrast_callback`` so no controller round-trip is needed; the
originating slice ndarray is supplied via :meth:`set_image_ext`.

External callers use ``*_ext`` setters to push state without emitting intents
(mirrors the :class:`nicewidgets.image_toolbar_widget.ImageToolbarWidget` pattern).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
from nicegui import ui
from nicegui.events import ValueChangeEventArguments

from nicewidgets.compact_select_styles import (
    COMPACT_SELECT_CLASS,
    COMPACT_SELECT_PROPS,
    ensure_compact_select_styles,
)
from nicewidgets.contrast_widget.colorscales import (
    COLORSCALE_OPTIONS,
    colorscale_option_value_to_label,
)
from nicewidgets.contrast_widget.intent import ContrastChangedIntent
from nicewidgets.utils.logging import get_logger

logger = get_logger(__name__)

OnContrastIntent = Callable[[ContrastChangedIntent], None]
AutoContrastCallback = Callable[[np.ndarray], tuple[int, int]]

DEFAULT_LUT = 'Gray'
DEFAULT_RANGE_MIN = 0
DEFAULT_RANGE_MAX = 255


class ContrastWidget:
    """NiceGUI contrast controls: color LUT select, Auto button, and min/max range.

    The widget builds its five child controls inside one internal
    ``ui.row`` with ``flex-nowrap`` so the contrast group always travels
    together: when the host row narrows, the whole group wraps as one unit
    instead of letting the range and its companion min/max labels break
    onto separate lines. The widget itself does not inherit from
    ``ui.row`` — the caller still chooses where this group sits in its own
    layout (same composition pattern used by :class:`EChartWidget` and
    :class:`PlotlyRasterViewer`).

    The range slider is bounded by ``min-w-32 max-w-48`` and grows with
    ``flex-1`` so it uses available space without dominating its row.

    Args:
        on_intent: Callback invoked with a single :class:`ContrastChangedIntent`
            whenever the user changes any control. ``None`` disables emission
            entirely.
        auto_contrast_callback: Callable invoked when the user clicks ``Auto``;
            it receives the current 2D ndarray (from
            :meth:`set_image_ext`) and returns ``(value_min, value_max)``. The
            widget updates its range to these values and emits the intent.
            ``None`` disables the Auto button computation (button is still
            visible but does nothing).
        widget_name: Identifier used for log messages.
    """

    def __init__(
        self,
        *,
        on_intent: OnContrastIntent | None = None,
        auto_contrast_callback: AutoContrastCallback | None = None,
        widget_name: str = 'contrast_widget',
    ) -> None:
        self._widget_name = widget_name
        self._on_intent = on_intent
        self._auto_contrast_callback = auto_contrast_callback
        self._suppress_intent = False
        self._enabled = True

        self._color_lut: str = DEFAULT_LUT
        self._value_min: int = DEFAULT_RANGE_MIN
        self._value_max: int = DEFAULT_RANGE_MAX
        self._img_min: int = DEFAULT_RANGE_MIN
        self._img_max: int = DEFAULT_RANGE_MAX
        self._image: np.ndarray | None = None

        ensure_compact_select_styles()

        with ui.row().classes('items-center gap-1 flex-nowrap'):
            with ui.row().classes('items-center gap-1'):
                ui.label('Color LUT')
                self._lut_select = ui.select(
                    options=colorscale_option_value_to_label(),
                    value=self._color_lut,
                    on_change=self._on_lut_change,
                ).props(COMPACT_SELECT_PROPS).classes(f'w-28 {COMPACT_SELECT_CLASS}')
            self._auto_btn = ui.button('Auto', on_click=self._on_auto_click)
            self._auto_btn.tooltip('Auto contrast (percentile clip)')
            self._min_label = ui.label(str(self._value_min)).classes('w-8 text-right')
            self._range = ui.range(
                min=self._img_min,
                max=self._img_max,
                value={'min': self._value_min, 'max': self._value_max},
                step=1,
                on_change=self._on_range_change,
            ).classes('flex-1 min-w-32 max-w-48').props('debounce=200')
            self._max_label = ui.label(str(self._value_max)).classes('w-8 text-left')

            self._min_label.bind_text_from(
                self._range, 'value', backward=lambda v: str(int(v['min']))
            )
            self._max_label.bind_text_from(
                self._range, 'value', backward=lambda v: str(int(v['max']))
            )

    @contextmanager
    def _intent_suppressed(self) -> Iterator[None]:
        """Suppress intent emission while programmatic state updates are applied."""
        prev = self._suppress_intent
        self._suppress_intent = True
        try:
            yield
        finally:
            self._suppress_intent = prev

    def _emit(self, intent: ContrastChangedIntent) -> None:
        """Emit one intent unless suppressed, disabled, or no callback was registered."""
        if self._suppress_intent or not self._enabled or self._on_intent is None:
            return
        self._on_intent(intent)

    # -- Public state APIs (no emit) --------------------------------------------------

    def set_lut_ext(self, lut: str) -> None:
        """Set the color LUT selection without emitting an intent.

        Args:
            lut: LUT identifier; should match one of ``COLORSCALE_OPTIONS`` values
                but unknown values are accepted as pass-through.
        """
        self._color_lut = str(lut)
        with self._intent_suppressed():
            self._lut_select.value = self._color_lut

    def set_range_ext(self, *, value_min: int, value_max: int) -> None:
        """Set the current min/max range without emitting an intent.

        Args:
            value_min: Minimum value of the displayed range.
            value_max: Maximum value of the displayed range. Swapped with
                ``value_min`` when ``value_min > value_max``.
        """
        lo = int(value_min)
        hi = int(value_max)
        if lo > hi:
            lo, hi = hi, lo
        self._value_min = lo
        self._value_max = hi
        with self._intent_suppressed():
            self._range.value = {'min': self._value_min, 'max': self._value_max}

    def set_image_ext(self, image: np.ndarray | None) -> None:
        """Set the current 2D image used for range bounds and Auto computation.

        The widget keeps a reference (no copy) so callers must not mutate the
        array in place. When ``image`` is ``None`` the range bounds revert to
        :data:`DEFAULT_RANGE_MIN` / :data:`DEFAULT_RANGE_MAX`.

        Args:
            image: 2D ndarray ``(Y, X)`` displayed by the host viewer, or
                ``None`` to clear.
        """
        if image is None or image.size == 0:
            self._image = None
            self._img_min = DEFAULT_RANGE_MIN
            self._img_max = DEFAULT_RANGE_MAX
        else:
            self._image = image
            self._img_min = int(image.min())
            self._img_max = int(image.max())
        with self._intent_suppressed():
            self._range.min = self._img_min
            self._range.max = self._img_max

    def set_enabled_ext(self, enabled: bool) -> None:
        """Enable or disable user interaction without emitting an intent.

        Args:
            enabled: When ``False``, all child controls are disabled and intents
                are suppressed even for direct handler calls (defensive). When
                ``True``, normal operation resumes.
        """
        self._enabled = bool(enabled)
        with self._intent_suppressed():
            self._lut_select.set_enabled(self._enabled)
            self._auto_btn.set_enabled(self._enabled)
            self._range.set_enabled(self._enabled)

    # -- Read-only getters (mostly for tests) ----------------------------------------

    def get_color_lut(self) -> str:
        """Return the current LUT identifier."""
        return self._color_lut

    def get_range(self) -> tuple[int, int]:
        """Return the current ``(value_min, value_max)``."""
        return self._value_min, self._value_max

    def get_image_bounds(self) -> tuple[int, int]:
        """Return the current ``(img_min, img_max)`` range bounds."""
        return self._img_min, self._img_max

    def get_image(self) -> np.ndarray | None:
        """Return the current 2D ndarray reference (may be ``None``)."""
        return self._image

    # -- User handlers ----------------------------------------------------------------

    def _on_lut_change(self, _e: ValueChangeEventArguments | None = None) -> None:
        """Handle user-driven LUT changes and emit a full-state intent."""
        if self._suppress_intent or not self._enabled:
            return
        raw = self._lut_select.value
        if raw is None:
            return
        self._color_lut = str(raw)
        self._emit(
            ContrastChangedIntent(
                color_lut=self._color_lut,
                value_min=self._value_min,
                value_max=self._value_max,
                from_auto=False,
            )
        )

    def _on_range_change(self, e: ValueChangeEventArguments) -> None:
        """Handle user-driven range changes and emit a full-state intent."""
        if self._suppress_intent or not self._enabled:
            return
        value = e.value
        lo = int(value['min'])
        hi = int(value['max'])
        if lo > hi:
            lo, hi = hi, lo
        self._value_min = lo
        self._value_max = hi
        self._emit(
            ContrastChangedIntent(
                color_lut=self._color_lut,
                value_min=self._value_min,
                value_max=self._value_max,
                from_auto=False,
            )
        )

    def _on_auto_click(self) -> None:
        """Handle Auto button clicks.

        Computes ``(value_min, value_max)`` locally via the injected callback,
        updates the range programmatically (no echo intent from the range
        on_change handler), and emits exactly one full-state intent.
        """
        if not self._enabled:
            return
        if self._image is None or self._auto_contrast_callback is None:
            logger.info(
                '%s: Auto click ignored (image=%s callback=%s)',
                self._widget_name,
                None if self._image is None else self._image.shape,
                self._auto_contrast_callback is not None,
            )
            return
        try:
            lo, hi = self._auto_contrast_callback(self._image)
        except Exception:
            logger.exception('%s: auto_contrast_callback raised', self._widget_name)
            return
        value_min = int(lo)
        value_max = int(hi)
        if value_min > value_max:
            value_min, value_max = value_max, value_min
        self._value_min = value_min
        self._value_max = value_max
        with self._intent_suppressed():
            self._range.value = {'min': self._value_min, 'max': self._value_max}
        self._emit(
            ContrastChangedIntent(
                color_lut=self._color_lut,
                value_min=self._value_min,
                value_max=self._value_max,
                from_auto=True,
            )
        )


__all__ = [
    'AutoContrastCallback',
    'ContrastWidget',
    'OnContrastIntent',
    'COLORSCALE_OPTIONS',
]
