"""Instance-scoped NiceGUI component wrapping the JavaScript raster viewer."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy.typing as npt
from nicegui import events, ui

from .channel_api import ChannelApi
from .config import RasterViewerConfig, ViewerLayout, ViewerTheme
from .events import (
    CHANNEL_SELECTED_EVENT,
    DISPLAY_CHANGE_EVENT,
    ERROR_EVENT,
    PERFORMANCE_EVENT,
    PLANE_CHANGE_EVENT,
    READY_EVENT,
    ROI_ADD_REQUESTED_EVENT,
    ROI_CREATE_REQUESTED_EVENT,
    ROI_DELETE_REQUESTED_EVENT,
    ROI_EDIT_CANCEL_REQUESTED_EVENT,
    ROI_EDIT_COMMITTED_EVENT,
    ROI_EDIT_REQUESTED_EVENT,
    ROI_SELECTED_EVENT,
    ROI_STATE_CHANGE_EVENT,
    TOOLBAR_ACTION_EVENT,
    VIEW_CHANGE_EVENT,
    RasterChannelSelectedEvent,
    RasterDisplayChangeEvent,
    RasterErrorEvent,
    RasterEvent,
    RasterEventHandler,
    RasterPerformanceEvent,
    RasterPlaneChangeEvent,
    RasterReadyEvent,
    RasterRoiAddRequestedEvent,
    RasterRoiCreateRequestedEvent,
    RasterRoiDeleteRequestedEvent,
    RasterRoiEditCancelRequestedEvent,
    RasterRoiEditCommittedEvent,
    RasterRoiEditRequestedEvent,
    RasterRoiSelectedEvent,
    RasterRoiStateChangeEvent,
    RasterToolbarActionEvent,
    RasterViewChangeEvent,
)
from .models import RasterChannelDisplay
from .numpy_source import NumPyRasterSource
from .roi import Roi
from .roi_api import RoiApi
from .source import RasterDataSource
from .source_registry import REGISTRY, ROUTE_PREFIX, ensure_routes_registered
from .xy_plot_api import XYPlotApi

JAVASCRIPT_TIMEOUT_SECONDS = 5.0
WEB_ASSETS = Path(__file__).resolve().parent / "web"
RasterEventT = TypeVar("RasterEventT", bound=RasterEvent)


class RasterViewerWidget(ui.element, component="web/raster_viewer_component.js"):
    """Display one instance-scoped JavaScript raster viewer in NiceGUI.

    The widget owns either a Python ``RasterDataSource`` registration or an
    external descriptor URL. Public methods are the supported Python API;
    callers do not need to access JavaScript or component internals. Loading a
    new source replaces the dataset atomically in the browser, aborts old plane
    requests, clears its decoded-plane cache, and releases the previous Python
    registration.

    ROI mutations initiated by Python are silent. Only genuine browser user
    interactions emit ROI callbacks. Instant add/delete and edit-start chrome
    either mutate locally (``RoiHostMode.LOCAL``) or emit request events
    (``RoiHostMode.DELEGATED``) until the host validates and calls silent
    ``rois`` APIs. Creation/editing geometry remains transactional until Python
    validates the proposal and calls ``rois.complete_commit``.
    """

    def __init__(
        self,
        source: RasterDataSource | None = None,
        *,
        descriptor_url: str | None = None,
        config: RasterViewerConfig | None = None,
        on_channel_selected: Callable[[RasterChannelSelectedEvent], Any] | None = None,
        on_display_changed: Callable[[RasterDisplayChangeEvent], Any] | None = None,
    ) -> None:
        """Create one viewer from a Python source or external descriptor URL.

        Args:
            source: Python object implementing ``RasterDataSource``.
            descriptor_url: Existing browser-readable descriptor endpoint.
            config: Initial presentation configuration.
            on_channel_selected: Optional user-originated channel callback.
            on_display_changed: Optional user-originated display-state callback.

        Raises:
            ValueError: If both or neither source forms are supplied.
        """
        if source is not None and descriptor_url is not None:
            raise ValueError("provide at most one of source or descriptor_url")
        super().__init__()
        self.rois = RoiApi(self._run)
        self.xy_plots = XYPlotApi(self._run)
        self.channels = ChannelApi(self._run)
        self._source_token: str | None = None
        self._config = config or RasterViewerConfig()
        self.add_resource(WEB_ASSETS)
        if source is not None:
            ensure_routes_registered()
            self._source_token = REGISTRY.register(source)
            descriptor_url = self._url_for_token(self._source_token)
        self.client.on_delete(self._release_source)
        self._props.update(
            {
                "descriptor-url": descriptor_url,
                "initial-theme": self._config.theme.value,
                "initial-layout": self._config.layout.value,
                "initial-axes-visible": self._config.axes_visible,
                "initial-rois-visible": (
                    self._config.roi_chrome_enabled and self._config.rois_visible
                ),
                "initial-channel-toolbars-visible": self._config.channel_toolbars_visible,
                "initial-roi-toolbar-visible": (
                    self._config.roi_chrome_enabled and self._config.roi_toolbar_visible
                ),
                "roi-chrome-enabled": self._config.roi_chrome_enabled,
                "roi-host-mode": self._config.roi_host_mode.value,
                "invert-slice-wheel": self._config.invert_slice_wheel,
                "wheel-zoom-factor": self._config.wheel_zoom_factor,
            }
        )
        self.classes("w-full h-full min-h-0")
        if on_channel_selected is not None:
            self.on_channel_selected(on_channel_selected)
        if on_display_changed is not None:
            self.on_display_change(on_display_changed)

    @classmethod
    def from_descriptor_url(
        cls,
        descriptor_url: str,
        *,
        config: RasterViewerConfig | None = None,
        on_channel_selected: Callable[[RasterChannelSelectedEvent], Any] | None = None,
        on_display_changed: Callable[[RasterDisplayChangeEvent], Any] | None = None,
    ) -> RasterViewerWidget:
        """Create a widget backed by an existing descriptor service.

        Args:
            descriptor_url: Browser-readable descriptor endpoint.
            config: Initial presentation configuration.
            on_channel_selected: Optional user-originated channel callback.
            on_display_changed: Optional user-originated display-state callback.

        Returns:
            Newly created widget.
        """
        return cls(
            descriptor_url=descriptor_url,
            config=config,
            on_channel_selected=on_channel_selected,
            on_display_changed=on_display_changed,
        )

    @classmethod
    def from_channels(
        cls,
        channels: Sequence[npt.NDArray[Any]] | Mapping[str, npt.NDArray[Any]],
        *,
        dims: Sequence[str],
        physical_units: Sequence[float],
        physical_units_labels: Sequence[str],
        rois: Sequence[Roi | Mapping[str, object]] = (),
        source_id: str | None = None,
        label: str = "NumPy raster",
        default_luts: Sequence[str] | None = None,
        channel_displays: Sequence[RasterChannelDisplay] | None = None,
        config: RasterViewerConfig | None = None,
        on_channel_selected: Callable[[RasterChannelSelectedEvent], Any] | None = None,
        on_display_changed: Callable[[RasterDisplayChangeEvent], Any] | None = None,
    ) -> RasterViewerWidget:
        """Create a widget directly from separate NumPy channel arrays.

        Every channel must have the same shape and dtype. Per-channel ``dims``
        end in ``("Y", "X")`` and may contain leading ``T`` and ``Z`` axes; a
        channel axis is not included because each array already represents one channel. Source
        arrays remain Python-owned and planes are fetched lazily as raw binary.

        Args:
            channels: Ordered arrays or stable-channel-ID mapping.
            dims: Dimension names describing every channel array axis.
            physical_units: Positive sample spacing corresponding to ``dims``.
            physical_units_labels: Display unit labels corresponding to ``dims``.
            rois: Initial typed ROIs or external descriptor envelopes.
            source_id: Optional stable dataset identity.
            label: Human-readable dataset label.
            default_luts: Optional LUT name for every logical channel.
            channel_displays: Optional complete initial display state per channel.
            config: Initial presentation configuration.
            on_channel_selected: Optional user-originated channel callback.
            on_display_changed: Optional user-originated display-state callback.

        Returns:
            Mounted viewer backed by a registered ``NumPyRasterSource``.
        """
        source = NumPyRasterSource.from_channels(
            channels,
            dims=dims,
            physical_units=physical_units,
            physical_units_labels=physical_units_labels,
            rois=rois,
            source_id=source_id,
            label=label,
            default_luts=default_luts,
            channel_displays=channel_displays,
        )
        return cls(
            source=source,
            config=config,
            on_channel_selected=on_channel_selected,
            on_display_changed=on_display_changed,
        )

    @classmethod
    def from_array(
        cls,
        data: npt.NDArray[Any],
        *,
        dims: Sequence[str],
        physical_units: Sequence[float],
        physical_units_labels: Sequence[str],
        rois: Sequence[Roi | Mapping[str, object]] = (),
        source_id: str | None = None,
        label: str = "NumPy raster",
        channel_ids: Sequence[str] | None = None,
        default_luts: Sequence[str] | None = None,
        channel_displays: Sequence[RasterChannelDisplay] | None = None,
        config: RasterViewerConfig | None = None,
        on_channel_selected: Callable[[RasterChannelSelectedEvent], Any] | None = None,
        on_display_changed: Callable[[RasterDisplayChangeEvent], Any] | None = None,
    ) -> RasterViewerWidget:
        """Create a widget from one explicitly dimensioned NumPy array.

        Supported layouts end in Y/X and may contain C, T, and Z. A named ``C``
        axis is split into logical channel views and excluded from the browser
        header. The display applies transpose followed by bottom-origin flip-Y;
        callers keep data and ROI geometry in original NumPy coordinates.

        Args:
            data: Contiguous or strided uint16/float32 source array.
            dims: Unique dimension names describing every input axis.
            physical_units: Positive sample spacing corresponding to ``dims``.
            physical_units_labels: Display unit labels corresponding to ``dims``.
            rois: Initial typed ROIs or external descriptor envelopes.
            source_id: Optional stable dataset identity.
            label: Human-readable dataset label.
            channel_ids: Optional stable IDs matching the named C axis.
            default_luts: Optional LUT names matching logical channels.
            channel_displays: Optional complete initial display state per channel.
            config: Initial presentation configuration.
            on_channel_selected: Optional user-originated channel callback.
            on_display_changed: Optional user-originated display-state callback.

        Returns:
            Mounted viewer backed by a registered ``NumPyRasterSource``.
        """
        source = NumPyRasterSource.from_array(
            data,
            dims=dims,
            physical_units=physical_units,
            physical_units_labels=physical_units_labels,
            rois=rois,
            source_id=source_id,
            label=label,
            channel_ids=channel_ids,
            default_luts=default_luts,
            channel_displays=channel_displays,
        )
        return cls(
            source=source,
            config=config,
            on_channel_selected=on_channel_selected,
            on_display_changed=on_display_changed,
        )

    @staticmethod
    def _url_for_token(token: str) -> str:
        """Return the internal descriptor URL for an opaque source token."""
        return f"{ROUTE_PREFIX}/{token}/descriptor"

    async def load_source(self, source: RasterDataSource) -> str:
        """Replace the current Python source after loading it in the browser.

        Args:
            source: New protocol-compatible raster source.

        Returns:
            Loaded source identifier reported by JavaScript.
        """
        ensure_routes_registered()
        new_token = REGISTRY.register(source)
        try:
            result = await self.run_method(
                "loadDescriptorUrl",
                self._url_for_token(new_token),
                timeout=JAVASCRIPT_TIMEOUT_SECONDS,
            )
        except Exception:
            REGISTRY.unregister(new_token)
            raise
        previous_token = self._source_token
        self._source_token = new_token
        if previous_token is not None:
            REGISTRY.unregister(previous_token)
        return str(result)

    async def load_descriptor_url(self, descriptor_url: str) -> str:
        """Replace the dataset from an external descriptor endpoint.

        Args:
            descriptor_url: Browser-readable URL returning the exact supported
                versioned descriptor schema.

        Returns:
            Loaded dataset identifier reported by JavaScript.
        """
        result = await self.run_method(
            "loadDescriptorUrl", descriptor_url, timeout=JAVASCRIPT_TIMEOUT_SECONDS
        )
        if self._source_token is not None:
            REGISTRY.unregister(self._source_token)
            self._source_token = None
        return str(result)

    async def set_theme(self, theme: ViewerTheme | str) -> str:
        """Apply a viewer chrome theme.

        Args:
            theme: Supported enum or ``light``/``dark`` value.

        Returns:
            Applied normalized theme value.
        """
        value = theme.value if isinstance(theme, ViewerTheme) else theme
        return str(await self._run("setTheme", value))

    async def set_layout(self, layout: ViewerLayout | str) -> str:
        """Apply a channel-pane layout.

        Args:
            layout: Supported enum or layout value.

        Returns:
            Applied normalized layout value.
        """
        value = layout.value if isinstance(layout, ViewerLayout) else layout
        return str(await self._run("setLayout", value))

    async def set_axes_visible(self, visible: bool) -> bool:
        """Set axis visibility.

        Args:
            visible: Whether fixed axis gutters and labels are drawn.

        Returns:
            Applied visibility.
        """
        return bool(await self._run("setAxesVisible", visible))

    async def set_rois_visible(self, visible: bool) -> bool:
        """Set committed ROI-overlay visibility.

        Args:
            visible: Whether committed ROI overlays are drawn.

        Returns:
            Applied visibility; active ROI drafts cannot be hidden.
        """
        return bool(await self._run("setRoisVisible", visible))

    async def set_channel_toolbars_visible(self, visible: bool) -> bool:
        """Set complete pane-header toolbar visibility.

        Args:
            visible: Whether headers containing channel controls and Copy are shown.

        Returns:
            Applied visibility for all current and future panes.
        """
        return bool(await self._run("setChannelToolbarsVisible", visible))

    async def set_roi_toolbar_visible(self, visible: bool) -> bool:
        """Set top-toolbar ROI strip visibility (dropdown + CRUD controls).

        Args:
            visible: Whether the ROI chrome strip is shown.

        Returns:
            Applied visibility.
        """
        return bool(await self._run("setRoiToolbarVisible", visible))

    async def set_x_range(self, minimum: float, maximum: float) -> dict[str, float]:
        """Set the physical display-X range without changing the Y transform.

        Args:
            minimum: Requested lower bound in the header's display-X units.
            maximum: Requested upper bound in the header's display-X units.

        Returns:
            Applied, image-clamped minimum and maximum.
        """
        result = await self._run("setXRange", minimum, maximum)
        return result if isinstance(result, dict) else {}

    async def set_y_range(self, minimum: float, maximum: float) -> dict[str, float]:
        """Set the physical display-Y range without changing the X transform.

        Args:
            minimum: Requested lower bound in the header's display-Y units.
            maximum: Requested upper bound in the header's display-Y units.

        Returns:
            Applied, image-clamped minimum and maximum.
        """
        result = await self._run("setYRange", minimum, maximum)
        return result if isinstance(result, dict) else {}

    async def set_physical_range(
        self,
        x_minimum: float,
        x_maximum: float,
        y_minimum: float,
        y_maximum: float,
    ) -> dict[str, object]:
        """Set physical display X and Y ranges in one viewport update.

        Prefer this over sequential :meth:`set_x_range` / :meth:`set_y_range`
        when restoring a reconnect viewport so the browser paints once.

        Args:
            x_minimum: Requested lower bound in display-X units.
            x_maximum: Requested upper bound in display-X units.
            y_minimum: Requested lower bound in display-Y units.
            y_maximum: Requested upper bound in display-Y units.

        Returns:
            Applied clamped ``{"x": {...}, "y": {...}}`` mapping when available.
        """
        result = await self._run(
            "setPhysicalRange",
            x_minimum,
            x_maximum,
            y_minimum,
            y_maximum,
        )
        return result if isinstance(result, dict) else {}

    async def set_z_index(self, z_index: int) -> dict[str, int | None]:
        """Select a zero-based Z plane.

        When the active dataset has no Z axis this is a no-op and returns the
        current plane selection.

        Args:
            z_index: Requested index, clamped to the active Z extent.

        Returns:
            Complete applied T/Z and sliding-Z selection.
        """
        result = await self._run("setZIndex", z_index)
        return result if isinstance(result, dict) else {}

    async def set_t_index(self, t_index: int) -> dict[str, int | None]:
        """Select a zero-based T plane.

        When the active dataset has no T axis this is a no-op and returns the
        current plane selection.

        Args:
            t_index: Requested index, clamped to the active T extent.

        Returns:
            Complete applied T/Z and sliding-Z selection.
        """
        result = await self._run("setTIndex", t_index)
        return result if isinstance(result, dict) else {}

    async def set_physical_calibration(
        self,
        physical_units: Sequence[float],
        physical_units_labels: Sequence[str],
    ) -> dict[str, object]:
        """Update runtime calibration without reloading pixel planes.

        Args:
            physical_units: Positive sample spacing aligned with active ``dims``.
            physical_units_labels: Display labels aligned with active ``dims``.

        Returns:
            Applied physical units and labels reported by the browser.
        """
        result = await self._run(
            "setPhysicalCalibration", list(physical_units), list(physical_units_labels)
        )
        return result if isinstance(result, dict) else {}

    async def reset_view(self) -> bool:
        """Restore the full X/Y image extent and emit final viewport events.

        Returns:
            True after every current pane is reset.
        """
        return bool(await self._run("resetView"))

    async def reset_x_range(self) -> dict[str, float]:
        """Restore full X extent while preserving the current Y transform.

        Returns:
            Applied physical X minimum and maximum.
        """
        result = await self._run("resetXRange")
        return result if isinstance(result, dict) else {}

    async def clear_source(self) -> bool:
        """Clear browser dataset state and release the registered Python source.

        Returns:
            True after the viewer returns to its empty state.
        """
        result = bool(await self._run("clear"))
        self._release_source()
        return result

    async def set_sliding_z(
        self, enabled: bool, plus_minus_slices: int = 1
    ) -> dict[str, int | None]:
        """Configure a centered sliding-Z maximum projection.

        Args:
            enabled: Whether the backend projection is active.
            plus_minus_slices: Non-negative Z radius around the selected plane.

        Returns:
            Complete applied T/Z and sliding-Z selection.
        """
        result = await self._run("setSlidingZ", enabled, plus_minus_slices)
        return result if isinstance(result, dict) else {}

    async def _run(self, method: str, *arguments: object) -> Any:
        """Invoke one declared component method with the shared timeout."""
        return await self.run_method(
            method, *arguments, timeout=JAVASCRIPT_TIMEOUT_SECONDS
        )

    def on_viewer_event(
        self, event_name: str, handler: RasterEventHandler, **event_options: Any
    ) -> RasterViewerWidget:
        """Register an advanced callback for a raw raster custom event.

        Prefer the named typed ``on_*`` helpers for stable application code.

        Args:
            event_name: JavaScript custom-event name.
            handler: Callback receiving NiceGUI's generic event wrapper.
            **event_options: Additional options forwarded to ``ui.element.on``.

        Returns:
            This widget for fluent registration.
        """
        self.on(
            event_name,
            handler,
            js_handler="event => emit(event.detail)",
            **event_options,
        )
        return self

    def on_ready(self, handler: Callable[[RasterReadyEvent], Any]) -> RasterViewerWidget:
        """Register a viewer-ready callback."""
        return self._on_typed(READY_EVENT, RasterReadyEvent, handler)

    def on_error(self, handler: Callable[[RasterErrorEvent], Any]) -> RasterViewerWidget:
        """Register a viewer-error callback."""
        return self._on_typed(ERROR_EVENT, RasterErrorEvent, handler)

    def on_view_change(
        self, handler: Callable[[RasterViewChangeEvent], Any], **options: Any
    ) -> RasterViewerWidget:
        """Register a viewport-change callback."""
        return self._on_typed(VIEW_CHANGE_EVENT, RasterViewChangeEvent, handler, **options)

    def on_display_change(
        self, handler: Callable[[RasterDisplayChangeEvent], Any], **options: Any
    ) -> RasterViewerWidget:
        """Register a channel-display-change callback."""
        return self._on_typed(
            DISPLAY_CHANGE_EVENT, RasterDisplayChangeEvent, handler, **options
        )

    def on_channel_selected(
        self, handler: Callable[[RasterChannelSelectedEvent], Any]
    ) -> RasterViewerWidget:
        """Register a user-originated active-channel callback."""
        return self._on_typed(CHANNEL_SELECTED_EVENT, RasterChannelSelectedEvent, handler)

    def on_toolbar_action(
        self, handler: Callable[[RasterToolbarActionEvent], Any]
    ) -> RasterViewerWidget:
        """Register a viewer-toolbar callback."""
        return self._on_typed(TOOLBAR_ACTION_EVENT, RasterToolbarActionEvent, handler)

    def on_plane_change(
        self, handler: Callable[[RasterPlaneChangeEvent], Any]
    ) -> RasterViewerWidget:
        """Register a plane-selection callback."""
        return self._on_typed(PLANE_CHANGE_EVENT, RasterPlaneChangeEvent, handler)

    def on_performance(
        self, handler: Callable[[RasterPerformanceEvent], Any]
    ) -> RasterViewerWidget:
        """Register a browser performance-metric callback."""
        return self._on_typed(PERFORMANCE_EVENT, RasterPerformanceEvent, handler)

    def _on_typed(
        self,
        event_name: str,
        event_type: type[RasterEventT],
        handler: Callable[[RasterEventT], Any],
        **event_options: Any,
    ) -> RasterViewerWidget:
        """Convert a NiceGUI event before calling a typed public handler."""

        def typed_handler(event: events.GenericEventArguments) -> Any:
            """Translate and forward one NiceGUI custom event."""
            converted = event_type.from_nicegui(event)
            return handler(converted)

        return self.on_viewer_event(event_name, typed_handler, **event_options)

    def on_roi_selected(
        self, handler: Callable[[RasterRoiSelectedEvent], Any]
    ) -> RasterViewerWidget:
        """Register a user-originated ROI-selection callback."""
        return self._on_typed(ROI_SELECTED_EVENT, RasterRoiSelectedEvent, handler)

    def on_roi_add_requested(
        self, handler: Callable[[RasterRoiAddRequestedEvent], Any]
    ) -> RasterViewerWidget:
        """Register a user request to add an ROI (host chooses identity/geometry)."""
        return self._on_typed(
            ROI_ADD_REQUESTED_EVENT, RasterRoiAddRequestedEvent, handler
        )

    def on_roi_delete_requested(
        self, handler: Callable[[RasterRoiDeleteRequestedEvent], Any]
    ) -> RasterViewerWidget:
        """Register a user request to delete one ROI."""
        return self._on_typed(
            ROI_DELETE_REQUESTED_EVENT, RasterRoiDeleteRequestedEvent, handler
        )

    def on_roi_edit_requested(
        self, handler: Callable[[RasterRoiEditRequestedEvent], Any]
    ) -> RasterViewerWidget:
        """Register a user request to enter ROI edit mode."""
        return self._on_typed(
            ROI_EDIT_REQUESTED_EVENT, RasterRoiEditRequestedEvent, handler
        )

    def on_roi_edit_cancel_requested(
        self, handler: Callable[[RasterRoiEditCancelRequestedEvent], Any]
    ) -> RasterViewerWidget:
        """Register a user request to cancel an active ROI draft."""
        return self._on_typed(
            ROI_EDIT_CANCEL_REQUESTED_EVENT,
            RasterRoiEditCancelRequestedEvent,
            handler,
        )

    def on_roi_create_requested(
        self, handler: Callable[[RasterRoiCreateRequestedEvent], Any]
    ) -> RasterViewerWidget:
        """Register a user ROI-creation proposal callback."""
        return self._on_typed(
            ROI_CREATE_REQUESTED_EVENT, RasterRoiCreateRequestedEvent, handler
        )

    def on_roi_edit_committed(
        self, handler: Callable[[RasterRoiEditCommittedEvent], Any]
    ) -> RasterViewerWidget:
        """Register a user ROI-edit proposal callback."""
        return self._on_typed(
            ROI_EDIT_COMMITTED_EVENT, RasterRoiEditCommittedEvent, handler
        )

    def on_roi_state_change(
        self, handler: Callable[[RasterRoiStateChangeEvent], Any]
    ) -> RasterViewerWidget:
        """Register a ROI interaction-state callback."""
        return self._on_typed(ROI_STATE_CHANGE_EVENT, RasterRoiStateChangeEvent, handler)

    def delete(self) -> None:
        """Delete the component and unregister its Python source."""
        self._release_source()
        super().delete()

    def _handle_delete(self) -> None:
        """Release source state when NiceGUI removes this element indirectly."""
        self._release_source()
        super()._handle_delete()

    def _release_source(self) -> None:
        """Idempotently unregister the current Python source."""
        if self._source_token is not None:
            REGISTRY.unregister(self._source_token)
            self._source_token = None
