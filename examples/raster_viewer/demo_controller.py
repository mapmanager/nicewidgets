"""Demo controller wiring PlotlyRasterViewer, ImageToolbarWidget, and ContrastWidget.

This module owns demo state (current dataset, channel, in-memory ROIs) and
translates widget intents into public viewer API calls. It follows the
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
from nicewidgets.raster_viewer.backend.image_model import BackendImage
from nicewidgets.raster_viewer.backend.pyramid import ImagePyramid
from nicewidgets.raster_viewer.frontend.plotly_viewer import PlotlyRasterViewer
from nicewidgets.raster_viewer.frontend.roi_overlay import RectRoiOverlay
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

    TODO: promote to a shared nicewidgets utility (for example
    ``nicewidgets.contrast_widget.auto_contrast``) so demos and host
    applications stop re-implementing percentile clipping.

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
    """Own demo state and wire toolbar/contrast intents to the raster viewer.

    The catalog must provide ``names``, ``channels(name)``, ``grid(name)``,
    and ``get_plane(name, channel)`` (see ``sample_data.SampleDataCatalog``).

    ROIs are demo-local, in-memory :class:`RectRoiOverlay` objects keyed by
    integer id; a real host application would own ROIs in its domain model.

    Args:
        catalog: Sample data source.
    """

    def __init__(self, catalog, *, dark_mode: bool = False) -> None:
        self._catalog = catalog
        self._viewer = PlotlyRasterViewer()
        self._toolbar: ImageToolbarWidget | None = None
        self._contrast: ContrastWidget | None = None

        self._dataset_name: str = catalog.names[0]
        self._channel: int = 0
        self._plane: np.ndarray | None = None
        self._rois: dict[int, RectRoiOverlay] = {}
        self._selected_roi_id: int | None = None
        self._next_roi_id: int = 1
        self._initial_load_done = False
        self._dark_mode = bool(dark_mode)
        self._viewer.set_dark_mode(self._dark_mode)

    @property
    def viewer(self) -> PlotlyRasterViewer:
        """Return the wrapped raster viewer (for demo-only extra controls)."""
        return self._viewer

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
        self._viewer.set_dark_mode(self._dark_mode)

    def build(self) -> None:
        """Build the toolbar row and the viewer in the current NiceGUI slot.

        Neither widget owns a layout container; this controller places both on
        one shared row (contrast pushed right) with the viewer below.
        """
        with ui.row().classes('w-full items-center flex-wrap gap-1'):
            self._toolbar = ImageToolbarWidget(on_intent=self._on_toolbar_intent)
            with ui.element('div').classes('ml-auto'):
                self._contrast = ContrastWidget(
                    on_intent=self._on_contrast_intent,
                    auto_contrast_callback=percentile_auto_contrast,
                )
        # Widgets stay disabled until the first plane is loaded.
        self._contrast.set_enabled_ext(False)
        self._toolbar.set_enabled_ext(False)

        plot = self._viewer.build()
        plot.classes('w-full h-[65vh]')
        # The first data load must wait for the Plotly element to exist in the
        # browser; afterplot fires once the initial empty figure is drawn.
        plot.on('plotly_afterplot', self._on_afterplot)

    async def _on_afterplot(self, _event: object) -> None:
        if self._initial_load_done:
            return
        self._initial_load_done = True
        await self.load_dataset(self._dataset_name)

    # -- Dataset / channel loading ----------------------------------------------------

    async def load_dataset(self, name: str) -> None:
        """Load ``name`` at channel 0, clearing ROIs and resetting the viewport."""
        self._dataset_name = name
        self._channel = 0
        self._rois.clear()
        self._selected_roi_id = None
        self._viewer.set_rois([])

        plane = self._catalog.get_plane(name, self._channel)
        grid = self._catalog.grid(name)
        self._plane = plane
        logger.info(f'load_dataset: {name!r} shape={plane.shape} dx={grid.dx} dy={grid.dy}')
        await self._viewer.set_data(plane, grid=grid)

        # set_data resets the viewer to its default colorscale and full
        # intensity window, so reseed the contrast widget to match.
        assert self._contrast is not None and self._toolbar is not None
        self._contrast.set_image_ext(plane)
        img_min, img_max = self._contrast.get_image_bounds()
        self._contrast.set_lut_ext(DEFAULT_LUT)
        self._contrast.set_range_ext(value_min=img_min, value_max=img_max)
        self._contrast.set_enabled_ext(True)

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
        """Swap the displayed plane, preserving viewport, contrast, and ROIs."""
        self._channel = channel
        plane = self._catalog.get_plane(self._dataset_name, channel)
        grid = self._catalog.grid(self._dataset_name)
        self._plane = plane
        source = BackendImage(plane, grid=grid)
        await self._viewer.swap_slice_plane(plane, grid=grid, pyramid=ImagePyramid(source))
        assert self._contrast is not None
        # Keep the user's LUT/window; only refresh Auto-contrast source bounds.
        self._contrast.set_image_ext(plane)

    # -- Contrast wiring ---------------------------------------------------------------

    def _on_contrast_intent(self, intent: ContrastChangedIntent) -> None:
        """Apply LUT and intensity window in one browser round trip."""
        colorscale = get_colorscale(intent.color_lut)
        background_tasks.create(
            self._viewer.set_heatmap_style(
                colorscale=colorscale,
                zmin=float(intent.value_min),
                zmax=float(intent.value_max),
                preserve_viewport=True,
            )
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
                self._viewer.select_roi(roi_id)
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
                # This demo applies edits immediately. A real host application
                # would stage pending bounds and commit on OK / restore on
                # Cancel (see the widget-api docs).
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
        self._viewer.add_roi(roi)
        self._viewer.select_roi(roi_id)
        self._sync_roi_options()

    def _delete_roi(self, roi_id: int) -> None:
        if roi_id not in self._rois:
            return
        del self._rois[roi_id]
        self._viewer.delete_roi(roi_id)
        remaining = sorted(self._rois)
        self._selected_roi_id = remaining[-1] if remaining else None
        self._viewer.select_roi(self._selected_roi_id)
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
        # add_roi replaces an existing overlay with the same roi_id.
        self._viewer.add_roi(updated)
