"""NiceGUI coordinator for single / mosaic / composite multi-channel rasters.

Owns one :class:`PlotlyRasterViewer` per visible pane (or one RGB composite
pane), shares ROIs across panes, and optionally links pan/zoom.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from nicegui import background_tasks, ui

from nicewidgets.raster_viewer.backend.image_model import BackendImage, RasterGridSpec
from nicewidgets.raster_viewer.backend.pyramid import ImagePyramid
from nicewidgets.raster_viewer.frontend.plotly_coord_transform import PlotlyCoordTransform
from nicewidgets.raster_viewer.frontend.plotly_display_options import (
    PlotlyRasterViewerDisplayOptions,
)
from nicewidgets.raster_viewer.frontend.plotly_protocol import PlotlyViewportPayload
from nicewidgets.raster_viewer.frontend.plotly_viewer import (
    DisplayAxisRanges,
    OnPlotlyViewportChanged,
    PlotlyRasterViewer,
)
from nicewidgets.raster_viewer.frontend.roi_overlay import RectRoiOverlay
from nicewidgets.raster_viewer.multichannel.compose import (
    CompositeChannelLimitError,
    DEFAULT_COMPOSITE_MAX_PIXELS,
    build_image_rgb_response,
    select_composite_planes,
    validate_same_shape,
)
from nicewidgets.raster_viewer.multichannel.models import (
    ChannelDisplayStyle,
    ChannelPlane,
    MosaicOrientation,
    MultiChannelRasterViewConfig,
    RasterLayoutMode,
)
from nicewidgets.utils.logging import get_logger

if TYPE_CHECKING:
    from nicegui.element import Element

logger = get_logger(__name__)

# Sentinel viewer key for the composite RGB pane.
COMPOSITE_VIEWER_KEY = -2


class MultiChannelRasterView:
    """Coordinate one or more :class:`PlotlyRasterViewer` panes for N channels.

    Typical host loop::

        view = MultiChannelRasterView(config=MultiChannelRasterViewConfig(link_viewport=True))
        root = view.build()
        await view.set_channels(planes, grid=grid)
        view.set_layout_mode('mosaic')
        view.set_mosaic_orientation('horizontal')
        view.set_rois([...])
    """

    def __init__(
        self,
        *,
        config: MultiChannelRasterViewConfig | None = None,
        display_options: PlotlyRasterViewerDisplayOptions | None = None,
        on_viewport_changed: OnPlotlyViewportChanged | None = None,
    ) -> None:
        self._config = config or MultiChannelRasterViewConfig()
        self._display_options = display_options or PlotlyRasterViewerDisplayOptions()
        self._on_viewport_changed = on_viewport_changed

        self._planes: list[ChannelPlane] = []
        self._grid: RasterGridSpec | None = None
        self._pyramids: dict[int, ImagePyramid] = {}
        self._active_channel_id: int | None = None

        self._root: Element | None = None
        self._panes_host: Element | None = None
        self._viewers: dict[int, PlotlyRasterViewer] = {}
        self._pane_plots: dict[int, Element] = {}

        self._rois: list[RectRoiOverlay] = []
        self._selected_roi_id: int | None = None
        self._syncing_viewport = False
        self._dark_mode = self._display_options.theme == 'dark'
        self._afterplot_future: asyncio.Future[bool] | None = None

    # -- public properties --------------------------------------------------------

    @property
    def config(self) -> MultiChannelRasterViewConfig:
        """Return the current layout / link configuration."""
        return self._config

    @property
    def active_channel_id(self) -> int | None:
        """Return the channel targeted by contrast / single-pane display."""
        return self._active_channel_id

    @property
    def viewers(self) -> Mapping[int, PlotlyRasterViewer]:
        """Return the live pane viewers keyed by ``channel_id``."""
        return self._viewers

    @property
    def has_data(self) -> bool:
        """Return ``True`` when channels have been set."""
        return bool(self._planes) and self._grid is not None

    def get_viewport(self) -> DisplayAxisRanges | None:
        """Return the viewport from the first live pane, if any."""
        for viewer in self._viewers.values():
            viewport = viewer.get_viewport()
            if viewport is not None:
                return viewport
        return None

    # -- build / layout -----------------------------------------------------------

    def build(self) -> Element:
        """Create the NiceGUI host container and initial pane layout."""
        self._root = ui.column().classes('w-full gap-1')
        with self._root:
            self._panes_host = ui.element('div').classes('w-full')
        self._rebuild_panes()
        return self._root

    async def set_layout_mode(self, mode: RasterLayoutMode) -> None:
        """Set ``single`` / ``mosaic`` / ``composite`` and rebuild panes.

        Call from a NiceGUI event handler (or ``with panes_host``) so pane UI
        is created in a valid slot. Reloads channel data after rebuild.

        Raises:
            CompositeChannelLimitError: If ``composite`` would use more than
                three visible channels.
        """
        if mode == 'composite' and self._planes:
            # Validate before mutating config / tearing down panes.
            select_composite_planes(self._planes)
        if mode == self._config.layout_mode:
            return
        self._config = replace(self._config, layout_mode=mode)
        self._rebuild_panes()
        if self.has_data:
            await self._reload_after_rebuild(reset=False)

    async def set_mosaic_orientation(self, orientation: MosaicOrientation) -> None:
        """Set mosaic grid to ``horizontal`` (1×N) or ``vertical`` (N×1)."""
        if orientation == self._config.mosaic_orientation:
            return
        self._config = replace(self._config, mosaic_orientation=orientation)
        if self._config.layout_mode != 'mosaic':
            return
        self._rebuild_panes()
        if self.has_data:
            await self._reload_after_rebuild(reset=False)

    def set_link_viewport(self, enabled: bool) -> None:
        """Enable or disable pan/zoom linking across mosaic panes."""
        self._config = replace(self._config, link_viewport=bool(enabled))

    # -- data ---------------------------------------------------------------------

    async def set_channels(
        self,
        planes: Sequence[ChannelPlane],
        *,
        grid: RasterGridSpec,
        pyramids: Mapping[int, ImagePyramid] | None = None,
    ) -> None:
        """Load N same-shaped channels (full reset of pan viewport / contrast).

        Args:
            planes: Channel planes (same ``(rows, cols)`` shape required).
            grid: Shared physical grid for all channels.
            pyramids: Optional prebuilt pyramids keyed by ``channel_id``. Missing
                entries are built on demand (monitor cost for large N).
        """
        plane_list = list(planes)
        validate_same_shape(plane_list)
        self._planes = plane_list
        self._grid = grid
        self._pyramids = dict(pyramids or {})
        if self._active_channel_id is None or self._active_channel_id not in {
            p.channel_id for p in plane_list
        }:
            self._active_channel_id = plane_list[0].channel_id if plane_list else None
        self._rebuild_panes()
        await self._wait_for_panes_ready()
        await self._load_all_panes(reset=True)

    async def swap_channels(
        self,
        planes: Sequence[ChannelPlane],
        *,
        grid: RasterGridSpec,
        pyramids: Mapping[int, ImagePyramid] | None = None,
    ) -> None:
        """Replace channel arrays while preserving viewport and ROIs."""
        plane_list = list(planes)
        validate_same_shape(plane_list)
        self._planes = plane_list
        self._grid = grid
        self._pyramids = dict(pyramids or {})
        if self._active_channel_id is None or self._active_channel_id not in {
            p.channel_id for p in plane_list
        }:
            self._active_channel_id = plane_list[0].channel_id if plane_list else None
        wanted = self._wanted_viewer_keys()
        if wanted != set(self._viewers):
            self._rebuild_panes()
            await self._wait_for_panes_ready()
        await self._load_all_panes(reset=False)

    async def set_active_channel(self, channel_id: int) -> None:
        """Select the channel for single-pane display and contrast targeting."""
        if channel_id not in {p.channel_id for p in self._planes}:
            raise ValueError(f'unknown channel_id={channel_id}')
        if channel_id == self._active_channel_id and self._config.layout_mode != 'single':
            return
        previous = self._active_channel_id
        self._active_channel_id = int(channel_id)
        if self._config.layout_mode == 'single' and previous != channel_id:
            self._rebuild_panes()
            if self.has_data:
                await self._reload_after_rebuild(reset=False)

    async def set_channel_style(self, channel_id: int, style: ChannelDisplayStyle) -> None:
        """Update one channel's style and refresh its pane when present."""
        updated: list[ChannelPlane] = []
        found = False
        for plane in self._planes:
            if plane.channel_id == channel_id:
                updated.append(replace(plane, style=style))
                found = True
            else:
                updated.append(plane)
        if not found:
            raise ValueError(f'unknown channel_id={channel_id}')
        visibility_changed = any(
            (a.style.visible != b.style.visible)
            for a, b in zip(self._planes, updated, strict=True)
            if a.channel_id == channel_id
        )
        self._planes = updated
        if self._config.layout_mode == 'composite':
            await self._load_composite_pane(reset=False)
            return
        if visibility_changed and self._config.layout_mode == 'mosaic':
            self._rebuild_panes()
            await self._reload_after_rebuild(reset=False)
            return
        viewer = self._viewers.get(channel_id)
        if viewer is None or not viewer.has_data:
            return
        if style.zmin is not None and style.zmax is not None:
            await viewer.set_heatmap_style(
                colorscale=style.colorscale,
                zmin=float(style.zmin),
                zmax=float(style.zmax),
                preserve_viewport=True,
            )
        else:
            await viewer.set_heatmap_colorscale(style.colorscale)

    # -- ROIs (shared across panes) -----------------------------------------------

    def set_rois(self, rois: Sequence[RectRoiOverlay]) -> None:
        """Replace ROIs on every live pane."""
        self._rois = list(rois)
        for viewer in self._viewers.values():
            viewer.set_rois(self._rois)

    def select_roi(self, roi_id: int | None) -> None:
        """Select one ROI on every live pane."""
        self._selected_roi_id = roi_id
        for viewer in self._viewers.values():
            viewer.select_roi(roi_id)

    def add_roi(self, roi: RectRoiOverlay) -> None:
        """Add or replace one ROI on every live pane."""
        self._rois = [r for r in self._rois if r.roi_id != roi.roi_id] + [roi]
        for viewer in self._viewers.values():
            viewer.add_roi(roi)

    def delete_roi(self, roi_id: int) -> None:
        """Delete one ROI from every live pane."""
        self._rois = [r for r in self._rois if r.roi_id != roi_id]
        for viewer in self._viewers.values():
            viewer.delete_roi(roi_id)

    # -- viewport helpers (demo / host) -------------------------------------------

    async def set_x_axis_range(self, *, x_min: float, x_max: float) -> None:
        """Set x range on all panes (y preserved per pane)."""
        for viewer in self._viewers.values():
            await viewer.set_x_axis_range(x_min=x_min, x_max=x_max)

    async def set_viewport(self, viewport: DisplayAxisRanges) -> None:
        """Apply a full viewport to all panes and refresh raster content."""
        (x_lo, x_hi), (y_lo, y_hi) = viewport
        self._syncing_viewport = True
        try:
            for viewer in self._viewers.values():
                await viewer.set_axis_ranges(
                    x_min=x_lo,
                    x_max=x_hi,
                    y_min=y_lo,
                    y_max=y_hi,
                    refresh_raster=True,
                )
        finally:
            self._syncing_viewport = False

    def set_dark_mode(self, enabled: bool) -> None:
        """Toggle theme on all panes."""
        self._dark_mode = bool(enabled)
        for viewer in self._viewers.values():
            viewer.set_dark_mode(self._dark_mode)

    def plane_for(self, channel_id: int) -> ChannelPlane | None:
        """Return the stored plane for ``channel_id``, if any."""
        for plane in self._planes:
            if plane.channel_id == channel_id:
                return plane
        return None

    # -- internals ----------------------------------------------------------------

    def _visible_planes_for_layout(self) -> list[ChannelPlane]:
        if self._config.layout_mode == 'composite':
            return []
        if self._config.layout_mode == 'single':
            if self._active_channel_id is None:
                return []
            for plane in self._planes:
                if plane.channel_id == self._active_channel_id:
                    return [plane]
            return []
        return [p for p in self._planes if p.style.visible]

    def _wanted_viewer_keys(self) -> set[int]:
        if self._config.layout_mode == 'composite':
            return {COMPOSITE_VIEWER_KEY}
        return {p.channel_id for p in self._visible_planes_for_layout()}

    def _rebuild_panes(self) -> None:
        """Recreate pane elements inside ``_panes_host`` (explicit slot).

        Safe from background tasks when ``_panes_host`` is already mounted:
        ``with self._panes_host`` restores the NiceGUI slot stack.
        """
        if self._panes_host is None:
            return
        self._viewers.clear()
        self._pane_plots.clear()
        with self._panes_host:
            self._panes_host.clear()
            if self._config.layout_mode == 'composite':
                if self._planes:
                    self._mount_composite_pane()
                else:
                    self._mount_placeholder_pane()
                return

            planes = self._visible_planes_for_layout()
            # Placeholder pane so the host can arm afterplot before the first
            # ``set_channels`` (mirrors single-viewer demo timing).
            if not planes:
                self._mount_placeholder_pane()
                return

            orientation = self._config.mosaic_orientation
            use_row = (
                orientation == 'horizontal' or self._config.layout_mode == 'single'
            )
            layout_ctx = (
                ui.row().classes('w-full gap-2 items-stretch')
                if use_row
                else ui.column().classes('w-full gap-2 items-stretch')
            )
            with layout_ctx:
                for plane in planes:
                    self._mount_pane(plane, fill_row=use_row)

    def _mount_placeholder_pane(self) -> None:
        viewer = PlotlyRasterViewer(
            display_options=replace(
                self._display_options,
                theme='dark' if self._dark_mode else 'light',
            ),
        )
        plot = viewer.build()
        plot.classes('w-full h-[65vh]')
        plot.on('plotly_afterplot', self._resolve_afterplot_future)
        self._viewers[-1] = viewer
        self._pane_plots[-1] = plot

    def _mount_composite_pane(self) -> None:
        with ui.column().classes('w-full min-w-0 gap-0'):
            ui.label('Composite RGB').classes('text-caption text-grey-7')
            viewer = PlotlyRasterViewer(
                display_options=replace(
                    self._display_options,
                    theme='dark' if self._dark_mode else 'light',
                ),
                on_viewport_changed=self._on_composite_viewport_changed,
                on_raster_refresh=self._composite_raster_refresh,
            )
            plot = viewer.build()
            plot.classes('w-full h-[65vh]')
            plot.on('plotly_afterplot', self._resolve_afterplot_future)
        self._viewers[COMPOSITE_VIEWER_KEY] = viewer
        self._pane_plots[COMPOSITE_VIEWER_KEY] = plot

    def _mount_pane(self, plane: ChannelPlane, *, fill_row: bool) -> None:
        channel_id = plane.channel_id
        label = plane.label if plane.label is not None else str(channel_id)
        visible_count = len(self._visible_planes_for_layout())
        height_class = (
            'h-[65vh]'
            if self._config.layout_mode == 'single' or visible_count == 1
            else 'h-[40vh]'
        )
        # Side-by-side (row): flex-1 shares width. Stacked (column): w-full so
        # the plot is not shrink-wrapped to a narrow intrinsic Plotly width.
        pane_classes = (
            'flex-1 min-w-0 w-full gap-0'
            if fill_row
            else 'w-full min-w-0 gap-0'
        )
        with ui.column().classes(pane_classes):
            ui.label(f'Channel {label}').classes('text-caption text-grey-7')
            viewer = PlotlyRasterViewer(
                display_options=replace(
                    self._display_options,
                    theme='dark' if self._dark_mode else 'light',
                ),
                on_viewport_changed=lambda vp, cid=channel_id: self._on_pane_viewport_changed(
                    cid, vp
                ),
            )
            plot = viewer.build()
            plot.classes(f'w-full {height_class}')
            plot.on('plotly_afterplot', self._resolve_afterplot_future)
        self._viewers[channel_id] = viewer
        self._pane_plots[channel_id] = plot

    def _resolve_afterplot_future(self, _event: object = None) -> None:
        fut = self._afterplot_future
        if fut is not None and not fut.done():
            fut.set_result(True)

    async def _wait_for_panes_ready(self) -> None:
        """Wait until at least one pane has fired ``plotly_afterplot`` (or timeout).

        Uses ``asyncio`` only — never ``ui.timer`` — so this is safe inside
        ``background_tasks`` where the NiceGUI slot stack is empty.
        """
        if not self._pane_plots:
            return
        loop = asyncio.get_running_loop()
        self._afterplot_future = loop.create_future()
        fut = self._afterplot_future
        try:
            await asyncio.wait_for(asyncio.shield(fut), timeout=0.5)
        except asyncio.TimeoutError:
            if not fut.done():
                fut.set_result(False)
        finally:
            if self._afterplot_future is fut:
                self._afterplot_future = None

    async def _reload_after_rebuild(self, *, reset: bool) -> None:
        await self._wait_for_panes_ready()
        await self._load_all_panes(reset=reset)

    def _pyramid_for(self, plane: ChannelPlane) -> ImagePyramid:
        cached = self._pyramids.get(plane.channel_id)
        if cached is not None:
            return cached
        assert self._grid is not None
        source = BackendImage(plane.data, grid=self._grid)
        pyramid = ImagePyramid(source)
        self._pyramids[plane.channel_id] = pyramid
        logger.debug(
            'built pyramid for channel_id=%s shape=%s levels=%s',
            plane.channel_id,
            plane.data.shape,
            pyramid.num_levels,
        )
        return pyramid

    async def _load_all_panes(self, *, reset: bool) -> None:
        if self._grid is None:
            return
        if self._config.layout_mode == 'composite':
            await self._load_composite_pane(reset=reset)
            return
        viewport = None if reset else self.get_viewport()
        for plane in self._visible_planes_for_layout():
            viewer = self._viewers.get(plane.channel_id)
            if viewer is None:
                continue
            pyramid = self._pyramid_for(plane)
            if reset or not viewer.has_data:
                await viewer.set_data_from_pyramid(
                    plane.data,
                    grid=self._grid,
                    pyramid=pyramid,
                )
            else:
                await viewer.swap_slice_plane(
                    plane.data,
                    grid=self._grid,
                    pyramid=pyramid,
                    display_axis_ranges=viewport,
                )
            if plane.style.zmin is not None and plane.style.zmax is not None:
                await viewer.set_heatmap_style(
                    colorscale=plane.style.colorscale,
                    zmin=float(plane.style.zmin),
                    zmax=float(plane.style.zmax),
                    preserve_viewport=True,
                )
            viewer.set_rois(self._rois)
            viewer.select_roi(self._selected_roi_id)

    async def _load_composite_pane(self, *, reset: bool) -> None:
        if self._grid is None:
            return
        viewer = self._viewers.get(COMPOSITE_VIEWER_KEY)
        if viewer is None:
            return
        viewport = None if reset else self.get_viewport()
        if viewport is None:
            shape = validate_same_shape(self._planes)
            transform = PlotlyCoordTransform(
                nrows=shape[0],
                ncols=shape[1],
                grid=self._grid,
            )
            full = transform.full_row_col_bounds()
            viewport = (
                transform.row_col_to_plot_x_range(full),
                transform.row_col_to_plot_y_range(full),
            )
        await self._apply_composite_viewport(viewport, reset_uirevision=reset)
        viewer.set_rois(self._rois)
        viewer.select_roi(self._selected_roi_id)

    async def _apply_composite_viewport(
        self,
        viewport: DisplayAxisRanges,
        *,
        reset_uirevision: bool = False,
    ) -> None:
        if self._grid is None or not self._planes:
            return
        viewer = self._viewers.get(COMPOSITE_VIEWER_KEY)
        if viewer is None:
            return
        shape = validate_same_shape(self._planes)
        transform = PlotlyCoordTransform(
            nrows=shape[0],
            ncols=shape[1],
            grid=self._grid,
        )
        (x_lo, x_hi), (y_lo, y_hi) = viewport
        bounds = transform.plot_xy_ranges_to_row_col(x_lo, x_hi, y_lo, y_hi)
        response = build_image_rgb_response(
            self._planes,
            grid=self._grid,
            bounds=bounds,
            max_pixels=DEFAULT_COMPOSITE_MAX_PIXELS,
        )
        await viewer.apply_rgb_response(
            response,
            grid=self._grid,
            shape=shape,
            display_axis_ranges=None if reset_uirevision else viewport,
            reset_uirevision=reset_uirevision,
        )

    async def _composite_raster_refresh(
        self,
        payload: PlotlyViewportPayload,
        display_axis_ranges: DisplayAxisRanges,
    ) -> bool:
        try:
            # Double-click sets full_reset so image+axes land in one figure
            # rebuild (same pattern as single-channel full_image_png). Settle
            # zooms keep reset_uirevision=False (restyle; browser owns axes).
            await self._apply_composite_viewport(
                display_axis_ranges,
                reset_uirevision=bool(payload.full_reset),
            )
        except CompositeChannelLimitError:
            logger.exception('composite refresh failed: channel limit')
            return True
        except Exception:
            logger.exception('composite refresh failed')
            return True
        return True

    def _on_composite_viewport_changed(self, viewport: DisplayAxisRanges) -> None:
        if self._on_viewport_changed is not None:
            self._on_viewport_changed(viewport)

    def _on_pane_viewport_changed(
        self,
        source_channel_id: int,
        viewport: DisplayAxisRanges,
    ) -> None:
        if self._on_viewport_changed is not None:
            self._on_viewport_changed(viewport)
        if not self._config.link_viewport or self._syncing_viewport:
            return
        if len(self._viewers) < 2:
            return
        background_tasks.create(
            self._propagate_viewport(source_channel_id, viewport)
        )

    async def _propagate_viewport(
        self,
        source_channel_id: int,
        viewport: DisplayAxisRanges,
    ) -> None:
        (x_lo, x_hi), (y_lo, y_hi) = viewport
        self._syncing_viewport = True
        try:
            for channel_id, viewer in self._viewers.items():
                if channel_id == source_channel_id:
                    continue
                await viewer.set_axis_ranges(
                    x_min=x_lo,
                    x_max=x_hi,
                    y_min=y_lo,
                    y_max=y_hi,
                    refresh_raster=True,
                )
        finally:
            self._syncing_viewport = False
