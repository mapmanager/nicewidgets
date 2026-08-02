"""Demo controller wiring MultiChannelRasterView, toolbar, and ContrastWidget.

This module owns demo state (current dataset, active channel, in-memory ROIs)
and translates widget intents into public coordinator APIs. It follows the
nicewidgets host-application pattern:

- User gestures arrive as frozen intent dataclasses via ``on_intent``.
- Programmatic state pushes use the ``*_ext`` widget methods, which never
  re-emit intents (no feedback loops).

The controller is deliberately framework-thin so it reads as a template for
any host application. It only imports nicewidgets and NiceGUI.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from nicegui import background_tasks, ui

from nicewidgets.contrast_widget.colorscales import get_colorscale
from nicewidgets.contrast_widget.contrast_widget import DEFAULT_LUT, ContrastWidget
from nicewidgets.contrast_widget.intent import ContrastChangedIntent
from nicewidgets.image_toolbar_widget.image_toolbar_widget import ImageToolbarWidget
from nicewidgets.image_toolbar_widget.intent import (
    ImageToolbarIntent,
    ImageToolbarRoiAddRequestIntent,
    ImageToolbarRoiApplyFullHeightIntent,
    ImageToolbarRoiApplyFullWidthIntent,
    ImageToolbarRoiDeleteRequestIntent,
    ImageToolbarRoiEditCancelIntent,
    ImageToolbarRoiEditStartIntent,
    ImageToolbarRoiEditSubmitIntent,
    ImageToolbarSelectChannelIntent,
    ImageToolbarSelectRoiIntent,
)
from nicewidgets.raster_viewer.frontend.roi_overlay import RectRoiOverlay
from nicewidgets.raster_viewer.multichannel import (
    ChannelDisplayStyle,
    ChannelPlane,
    MultiChannelRasterView,
    MultiChannelRasterViewConfig,
    MosaicOrientation,
    RasterLayoutMode,
)
from nicewidgets.utils.logging import get_logger

logger = get_logger(__name__)


def percentile_auto_contrast(
    plane: np.ndarray,
    *,
    percentile_low: float = 1.0,
    percentile_high: float = 99.5,
) -> tuple[int, int]:
    """Return an integer intensity window via percentile clipping.

    Passed to :class:`ContrastWidget` as ``auto_contrast_callback``.

    Args:
        plane: 2D image array supplied by the contrast widget.
        percentile_low: Lower clip percentile.
        percentile_high: Upper clip percentile.

    Returns:
        ``(value_min, value_max)`` integer pair for the range slider.
    """
    lo = int(np.percentile(plane, percentile_low))
    hi = int(np.percentile(plane, percentile_high))
    if lo >= hi:
        hi = lo + 1
    return lo, hi


class RasterDemoController:
    """Own demo state and wire toolbar/contrast intents to MultiChannelRasterView.

    The catalog must provide ``names``, ``channels(name)``, ``grid(name)``,
    and ``get_plane(name, channel)`` (see ``sample_data.SampleDataCatalog``).

    ROIs are demo-local, in-memory :class:`RectRoiOverlay` objects keyed by
    integer id; a real host application would own ROIs in its domain model.

    Args:
        catalog: Sample data source.
    """

    def __init__(self, catalog, *, dark_mode: bool = False) -> None:
        self._catalog = catalog
        self._view = MultiChannelRasterView(
            config=MultiChannelRasterViewConfig(
                layout_mode='single',
                mosaic_orientation='horizontal',
                link_viewport=True,
            ),
        )
        self._toolbar: ImageToolbarWidget | None = None
        self._contrast: ContrastWidget | None = None

        self._dataset_name: str = catalog.names[0]
        self._channel: int = 0
        self._plane: np.ndarray | None = None
        self._channel_styles: dict[int, ChannelDisplayStyle] = {}
        self._rois: dict[int, RectRoiOverlay] = {}
        self._selected_roi_id: int | None = None
        self._next_roi_id: int = 1
        self._initial_load_done = False
        self._dark_mode = bool(dark_mode)
        self._view.set_dark_mode(self._dark_mode)

        self._layout_select: ui.select | None = None
        self._orientation_select: ui.select | None = None
        self._link_checkbox: ui.checkbox | None = None

    @property
    def viewer(self) -> MultiChannelRasterView:
        """Return the multi-channel coordinator (demo extra controls / x-range)."""
        return self._view

    @property
    def dataset_name(self) -> str:
        """Return the currently loaded dataset name."""
        return self._dataset_name

    def set_dark_mode(self, enabled: bool) -> None:
        """Toggle the Plotly raster viewer layout theme.

        Args:
            enabled: Whether dark mode is enabled.
        """
        self._dark_mode = bool(enabled)
        self._view.set_dark_mode(self._dark_mode)

    def build(self) -> None:
        """Build toolbar, multichannel controls, contrast, and the coordinator."""
        with ui.row().classes('w-full items-center flex-wrap gap-1'):
            self._toolbar = ImageToolbarWidget(on_intent=self._on_toolbar_intent)
            with ui.element('div').classes('ml-auto'):
                self._contrast = ContrastWidget(
                    on_intent=self._on_contrast_intent,
                    auto_contrast_callback=percentile_auto_contrast,
                )
        self._contrast.set_enabled_ext(False)
        self._toolbar.set_enabled_ext(False)

        with ui.row().classes('w-full items-end gap-3 flex-wrap'):
            self._layout_select = ui.select(
                options={
                    'single': 'Single channel',
                    'mosaic': 'All channels (mosaic)',
                },
                value='single',
                label='Layout',
                on_change=self._on_layout_mode_change,
            ).classes('w-56')
            self._orientation_select = ui.select(
                options={
                    'horizontal': 'Side by side (1×N)',
                    'vertical': 'Stacked (N×1)',
                },
                value='horizontal',
                label='Mosaic orientation',
                on_change=self._on_orientation_change,
            ).classes('w-56')
            self._link_checkbox = ui.checkbox(
                'Link pan/zoom',
                value=True,
                on_change=lambda e: self._view.set_link_viewport(bool(e.value)),
            )
            self._orientation_select.set_enabled(False)

        root = self._view.build()
        root.classes('w-full')
        # First data load waits until a pane Plotly element exists.
        for plot in self._view._pane_plots.values():
            plot.on('plotly_afterplot', self._on_afterplot)
            break

    async def _on_afterplot(self, _event: object) -> None:
        if self._initial_load_done:
            return
        self._initial_load_done = True
        await self.load_dataset(self._dataset_name)

    async def _on_layout_mode_change(self, e) -> None:
        """Rebuild panes in the UI event slot (do not wrap in background_tasks)."""
        mode = str(e.value)
        layout_mode: RasterLayoutMode = 'mosaic' if mode == 'mosaic' else 'single'
        try:
            await self._view.set_layout_mode(layout_mode)
        except NotImplementedError as exc:
            ui.notify(str(exc), type='warning')
            return
        if self._orientation_select is not None:
            self._orientation_select.set_enabled(layout_mode == 'mosaic')

    async def _on_orientation_change(self, e) -> None:
        """Rebuild mosaic orientation in the UI event slot."""
        orientation = str(e.value)
        value: MosaicOrientation = (
            'vertical' if orientation == 'vertical' else 'horizontal'
        )
        await self._view.set_mosaic_orientation(value)

    # -- Dataset / channel loading ----------------------------------------------------

    def _build_planes(self, name: str) -> list[ChannelPlane]:
        planes: list[ChannelPlane] = []
        for channel in self._catalog.channels(name):
            data = self._catalog.get_plane(name, channel)
            style = self._channel_styles.get(channel, ChannelDisplayStyle())
            planes.append(
                ChannelPlane(
                    channel_id=channel,
                    data=data,
                    style=style,
                    label=str(channel),
                )
            )
        return planes

    async def load_dataset(self, name: str) -> None:
        """Load all channels of ``name``, clearing ROIs and resetting the viewport."""
        self._dataset_name = name
        self._channel = 0
        self._rois.clear()
        self._selected_roi_id = None
        self._channel_styles.clear()
        self._view.set_rois([])

        planes = self._build_planes(name)
        grid = self._catalog.grid(name)
        self._plane = planes[0].data if planes else None
        logger.info(
            'load_dataset: %r channels=%s shape=%s dx=%s dy=%s',
            name,
            [p.channel_id for p in planes],
            None if self._plane is None else self._plane.shape,
            grid.dx,
            grid.dy,
        )
        await self._view.set_channels(planes, grid=grid)
        await self._view.set_active_channel(self._channel)

        assert self._contrast is not None and self._toolbar is not None
        if self._plane is not None:
            self._contrast.set_image_ext(self._plane)
            img_min, img_max = self._contrast.get_image_bounds()
            self._contrast.set_lut_ext(DEFAULT_LUT)
            self._contrast.set_range_ext(value_min=img_min, value_max=img_max)
            self._contrast.set_enabled_ext(True)
            self._channel_styles[self._channel] = ChannelDisplayStyle(
                zmin=float(img_min),
                zmax=float(img_max),
                colorscale=get_colorscale(DEFAULT_LUT),
            )

        channel_options = [str(c) for c in self._catalog.channels(name)]
        self._toolbar.set_file_ext(
            name,
            self._channel,
            self._selected_roi_id,
            channel_options=channel_options,
            roi_options=[],
        )
        self._toolbar.set_enabled_ext(True)

    async def _change_channel(self, channel: int) -> None:
        """Select active channel (single pane swap / contrast target in mosaic)."""
        self._channel = channel
        plane = self._catalog.get_plane(self._dataset_name, channel)
        self._plane = plane
        # Coordinator rebuilds/reloads the single pane from stored planes.
        await self._view.set_active_channel(channel)
        assert self._contrast is not None
        self._contrast.set_image_ext(plane)
        style = self._channel_styles.get(channel)
        if style is not None and style.zmin is not None and style.zmax is not None:
            self._contrast.set_range_ext(
                value_min=int(style.zmin),
                value_max=int(style.zmax),
            )

    # -- Contrast wiring ---------------------------------------------------------------

    def _on_contrast_intent(self, intent: ContrastChangedIntent) -> None:
        """Apply LUT and intensity window to the active channel pane."""
        colorscale = get_colorscale(intent.color_lut)
        style = ChannelDisplayStyle(
            visible=True,
            zmin=float(intent.value_min),
            zmax=float(intent.value_max),
            colorscale=colorscale,
        )
        self._channel_styles[self._channel] = style
        background_tasks.create(
            self._view.set_channel_style(self._channel, style)
        )

    # -- Toolbar wiring ----------------------------------------------------------------

    def _on_toolbar_intent(self, intent: ImageToolbarIntent) -> None:
        """Route toolbar intents to demo state changes and viewer calls."""
        match intent:
            case ImageToolbarSelectChannelIntent(channel=channel):
                if channel is not None:
                    background_tasks.create(self._change_channel(channel))
            case ImageToolbarSelectRoiIntent(roi_id=roi_id):
                self._selected_roi_id = roi_id
                self._view.select_roi(roi_id)
            case ImageToolbarRoiAddRequestIntent():
                self._add_roi()
            case ImageToolbarRoiDeleteRequestIntent(roi_id=roi_id):
                self._delete_roi(roi_id)
            case ImageToolbarRoiEditStartIntent():
                ui.notify('Edit mode: use Full width / Full height, then OK or Cancel.')
            case ImageToolbarRoiApplyFullWidthIntent(roi_id=roi_id):
                self._apply_full_extent(roi_id, axis='x')
            case ImageToolbarRoiApplyFullHeightIntent(roi_id=roi_id):
                self._apply_full_extent(roi_id, axis='y')
            case ImageToolbarRoiEditSubmitIntent() | ImageToolbarRoiEditCancelIntent():
                pass

    # -- ROI helpers -------------------------------------------------------------------

    def _physical_extent(self) -> tuple[float, float]:
        """Return ``(x_max, y_max)`` of the current plane in plot physical coordinates."""
        assert self._plane is not None
        rows, cols = self._plane.shape
        grid = self._catalog.grid(self._dataset_name)
        return float(cols) * float(grid.dx), float(rows) * float(grid.dy)

    def _sync_roi_options(self) -> None:
        assert self._toolbar is not None
        self._toolbar.set_roi_options_and_selection_ext(sorted(self._rois), self._selected_roi_id)

    def _add_roi(self) -> None:
        """Create a centered ROI (staggered so repeated adds stay visible)."""
        x_max, y_max = self._physical_extent()
        roi_id = self._next_roi_id
        self._next_roi_id += 1
        offset = 0.05 * (roi_id % 5)
        x0 = (0.25 + offset) * x_max
        y0 = (0.25 + offset) * y_max
        roi = RectRoiOverlay(
            roi_id=roi_id,
            x0=x0,
            x1=x0 + 0.25 * x_max,
            y0=y0,
            y1=y0 + 0.25 * y_max,
            label=f'ROI {roi_id}',
        )
        self._rois[roi_id] = roi
        self._selected_roi_id = roi_id
        self._view.add_roi(roi)
        self._view.select_roi(roi_id)
        self._sync_roi_options()

    def _delete_roi(self, roi_id: int) -> None:
        if roi_id not in self._rois:
            return
        del self._rois[roi_id]
        self._view.delete_roi(roi_id)
        remaining = sorted(self._rois)
        self._selected_roi_id = remaining[-1] if remaining else None
        self._view.select_roi(self._selected_roi_id)
        self._sync_roi_options()

    def _apply_full_extent(self, roi_id: int, *, axis: str) -> None:
        """Stretch one ROI to the full x or y physical extent."""
        roi = self._rois.get(roi_id)
        if roi is None:
            return
        x_max, y_max = self._physical_extent()
        if axis == 'x':
            updated = replace(roi, x0=0.0, x1=x_max)
        else:
            updated = replace(roi, y0=0.0, y1=y_max)
        self._rois[roi_id] = updated
        self._view.add_roi(updated)
