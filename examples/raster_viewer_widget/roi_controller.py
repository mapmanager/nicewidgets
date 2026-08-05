"""Coordinate the NiceWidgets image toolbar with demo ROI state and viewer APIs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

from nicegui import background_tasks, ui
from nicegui.elements.select import Select
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

from examples.raster_viewer_widget.roi_store import DemoRoiStore
from nicewidgets.raster_viewer_widget.events import (
    RasterRoiEditCommittedEvent,
    RasterRoiSelectedEvent,
    RasterRoiStateChangeEvent,
)
from nicewidgets.raster_viewer_widget.roi import RectRoi, RoiInteractionState
from nicewidgets.raster_viewer_widget.roi_api import RoiApi

LOGGER = logging.getLogger(__name__)


class RoiStoreProvider(Protocol):
    """Describe dataset state required by the demo ROI controller."""

    def get_roi_store(self, dataset_id: str) -> DemoRoiStore:
        """Return one dataset's authoritative ROI store."""
        ...

    def channel_indices(self, dataset_id: str) -> tuple[int, ...]:
        """Return zero-based logical channel indices for one dataset."""
        ...


class RoiViewerBridge(Protocol):
    """Describe viewer operations and callbacks required by the controller."""

    def on_roi_selected(self, handler: Callable[[RasterRoiSelectedEvent], Any]) -> object:
        """Register a typed ROI-selection callback."""
        ...

    def on_roi_state_change(
        self, handler: Callable[[RasterRoiStateChangeEvent], Any]
    ) -> object:
        """Register a typed ROI-state callback."""
        ...

    def on_roi_edit_committed(
        self, handler: Callable[[RasterRoiEditCommittedEvent], Any]
    ) -> object:
        """Register a typed ROI-edit callback."""
        ...

    rois: RoiApi


class DemoRoiController:
    """Translate toolbar intents into rectangle ROI store and viewer operations.

    ``ImageToolbarWidget`` owns controls and emits typed user intent. The
    controller owns demo-specific coordination: committed storage, dataset
    locking, rectangle-only toolbar options, and the viewer's transactional
    edit lifecycle. Line ROIs remain supported by the viewer but are not
    managed by this rectangle toolbar.
    """

    def __init__(
        self,
        datasets: RoiStoreProvider,
        dataset_selector: Select,
        viewer: RoiViewerBridge,
        initial_dataset_id: str,
    ) -> None:
        """Build the reusable toolbar and initialize application coordination.

        Args:
            datasets: Provider of per-dataset channels and committed ROI stores.
            dataset_selector: Dataset selector locked during an ROI transaction.
            viewer: Raster viewer wrapper exposing typed ROI APIs and events.
            initial_dataset_id: Initially loaded dataset identity.
        """
        self._datasets = datasets
        self._dataset_selector = dataset_selector
        self._viewer = viewer
        self._dataset_id = initial_dataset_id
        self._state = RoiInteractionState.IDLE
        self._selected_roi_id: int | None = None
        self._toolbar = ImageToolbarWidget(
            on_intent=self._on_toolbar_intent,
            widget_name="raster_demo_roi_toolbar",
        )
        self._register_viewer_events()
        self.set_dataset(initial_dataset_id)

    @property
    def state(self) -> RoiInteractionState:
        """Return the current viewer ROI interaction state."""
        return self._state

    @property
    def _store(self) -> DemoRoiStore:
        """Return the active dataset's authoritative committed ROI store."""
        return self._datasets.get_roi_store(self._dataset_id)

    def _register_viewer_events(self) -> None:
        """Register selection, state, and committed-edit callbacks."""
        self._viewer.on_roi_selected(self._handle_viewer_selection)
        self._viewer.on_roi_state_change(self._handle_viewer_state)
        self._viewer.on_roi_edit_committed(self._handle_edit_commit)

    def set_dataset(self, dataset_id: str) -> None:
        """Switch controller and toolbar state to another dataset.

        Args:
            dataset_id: Newly selected dataset identifier.
        """
        self._dataset_id = dataset_id
        rectangle_ids = self._rectangle_ids()
        self._selected_roi_id = rectangle_ids[0] if rectangle_ids else None
        self._state = RoiInteractionState.IDLE
        self._dataset_selector.set_enabled(True)
        self._reset_toolbar_mode()

    def _channel_options(self) -> list[str]:
        """Return channel values required by the current toolbar contract."""
        return [str(index) for index in self._datasets.channel_indices(self._dataset_id)]

    def _rectangle_ids(self) -> list[int]:
        """Return committed rectangle IDs in stable store order."""
        return [roi.roi_id for roi in self._store if isinstance(roi, RectRoi)]

    def _reset_toolbar_mode(self) -> None:
        """Synchronize all toolbar state and force its mode back to idle."""
        channel_options = self._channel_options()
        self._toolbar.set_file_ext(
            self._dataset_id,
            int(channel_options[0]) if channel_options else None,
            self._selected_roi_id,
            channel_options=channel_options,
            roi_options=self._rectangle_ids(),
        )

    def _sync_rectangle_selection(self) -> None:
        """Atomically synchronize rectangle options and selection."""
        self._toolbar.set_roi_options_and_selection_ext(
            self._rectangle_ids(), self._selected_roi_id
        )

    def _set_state(self, state: RoiInteractionState) -> None:
        """Apply viewer transaction state to application-owned controls.

        Args:
            state: Named viewer ROI interaction state.
        """
        self._state = state
        self._dataset_selector.set_enabled(state is RoiInteractionState.IDLE)
        if state is RoiInteractionState.IDLE:
            self._reset_toolbar_mode()

    def _on_toolbar_intent(self, intent: ImageToolbarIntent) -> None:
        """Route one synchronous toolbar intent to demo coordination.

        Args:
            intent: Frozen intent emitted by ``ImageToolbarWidget``.
        """
        match intent:
            case ImageToolbarSelectChannelIntent(channel=channel):
                LOGGER.info(
                    "Ignoring ImageToolbar channel selection: dataset=%s channel=%s",
                    self._dataset_id,
                    channel,
                )
            case ImageToolbarSelectRoiIntent(roi_id=roi_id):
                background_tasks.create(self._select_rectangle(roi_id))
            case ImageToolbarRoiAddRequestIntent():
                background_tasks.create(self._add_rectangle())
            case ImageToolbarRoiDeleteRequestIntent(roi_id=roi_id):
                background_tasks.create(self._delete_rectangle(roi_id))
            case ImageToolbarRoiEditStartIntent(roi_id=roi_id):
                background_tasks.create(self._begin_edit(roi_id))
            case ImageToolbarRoiEditSubmitIntent():
                background_tasks.create(self._commit_edit())
            case ImageToolbarRoiEditCancelIntent():
                background_tasks.create(self._cancel_edit())
            case ImageToolbarRoiApplyFullWidthIntent(roi_id=roi_id):
                self._unsupported_full_extent("width", roi_id)
            case ImageToolbarRoiApplyFullHeightIntent(roi_id=roi_id):
                self._unsupported_full_extent("height", roi_id)

    async def _select_rectangle(self, roi_id: int | None) -> None:
        """Select one toolbar-managed rectangle in the viewer.

        Args:
            roi_id: Rectangle ID emitted by the toolbar.
        """
        if self._state is not RoiInteractionState.IDLE:
            return
        if roi_id is not None and roi_id not in self._rectangle_ids():
            LOGGER.warning("Ignoring non-rectangle toolbar selection: roi=%s", roi_id)
            return
        self._selected_roi_id = roi_id
        await self._viewer.rois.select(roi_id)
        LOGGER.info("Rectangle selected from ImageToolbar: roi=%s", roi_id)

    async def _add_rectangle(self) -> None:
        """Create and immediately commit one centered rectangle ROI."""
        if self._state is not RoiInteractionState.IDLE:
            return
        roi = self._store.create_rect(self._store.suggested_rect_bounds())
        self._selected_roi_id = roi.roi_id
        await self._viewer.rois.add(roi)
        await self._viewer.rois.select(roi.roi_id)
        self._sync_rectangle_selection()
        LOGGER.info("Rectangle added from ImageToolbar: roi=%s", roi.to_json())

    async def _delete_rectangle(self, roi_id: int) -> None:
        """Delete one rectangle and select a neighboring rectangle.

        Args:
            roi_id: Toolbar-selected rectangle identity.
        """
        if self._state is not RoiInteractionState.IDLE:
            return
        rectangle_ids = self._rectangle_ids()
        if roi_id not in rectangle_ids:
            LOGGER.warning("Ignoring delete for non-rectangle ROI: roi=%s", roi_id)
            return
        deleted_index = rectangle_ids.index(roi_id)
        self._store.delete(roi_id)
        remaining = self._rectangle_ids()
        self._selected_roi_id = (
            remaining[min(deleted_index, len(remaining) - 1)] if remaining else None
        )
        await self._viewer.rois.remove(roi_id)
        await self._viewer.rois.select(self._selected_roi_id)
        self._sync_rectangle_selection()
        LOGGER.info("Rectangle deleted from ImageToolbar: roi=%d", roi_id)

    async def _begin_edit(self, roi_id: int) -> None:
        """Begin transactional editing for a selected rectangle.

        Args:
            roi_id: Existing rectangle identity.
        """
        if self._state is not RoiInteractionState.IDLE or roi_id not in self._rectangle_ids():
            return
        self._selected_roi_id = roi_id
        self._set_state(RoiInteractionState.EDITING)
        started = await self._viewer.rois.begin_edit(roi_id)
        if not started:
            self._set_state(RoiInteractionState.IDLE)
            ui.notify("Could not start ROI editing", type="warning")
            return
        LOGGER.info("Rectangle edit started from ImageToolbar: roi=%d", roi_id)

    async def _commit_edit(self) -> None:
        """Ask the viewer to emit the active rectangle draft proposal."""
        if self._state is not RoiInteractionState.EDITING:
            return
        await self._viewer.rois.commit_edit()

    async def _cancel_edit(self) -> None:
        """Discard the browser-local draft and return controls to idle."""
        if self._state is not RoiInteractionState.EDITING:
            return
        await self._viewer.rois.cancel_edit()
        self._set_state(RoiInteractionState.IDLE)
        LOGGER.info("Rectangle edit cancelled from ImageToolbar")

    def _unsupported_full_extent(self, axis: str, roi_id: int) -> None:
        """Report a toolbar action intentionally unsupported by this demo.

        Args:
            axis: Human-readable extent axis, ``width`` or ``height``.
            roi_id: Rectangle being edited.
        """
        LOGGER.info(
            "Ignoring unsupported ImageToolbar full-%s action: roi=%d", axis, roi_id
        )
        ui.notify(f"Full {axis} is not implemented in this demo", type="info")

    def _handle_viewer_selection(self, event: RasterRoiSelectedEvent) -> None:
        """Reflect a viewer rectangle selection in the toolbar.

        Line selections intentionally remain viewer-only because this toolbar
        manages rectangles.

        Args:
            event: User-originated viewer selection event.
        """
        if event.dataset_id != self._dataset_id or self._state is not RoiInteractionState.IDLE:
            return
        if event.roi_id not in self._rectangle_ids():
            LOGGER.info("Ignoring viewer line ROI selection in rectangle toolbar: %s", event.roi_id)
            return
        self._selected_roi_id = event.roi_id
        self._sync_rectangle_selection()
        LOGGER.info("Rectangle selected from viewer: roi=%s", event.roi_id)

    def _handle_viewer_state(self, event: RasterRoiStateChangeEvent) -> None:
        """Synchronize application controls with viewer edit state.

        Args:
            event: Viewer interaction-state event.
        """
        if event.dataset_id != self._dataset_id:
            return
        self._set_state(event.state)
        LOGGER.info("Viewer ROI state received: %s", event.payload)

    async def _handle_edit_commit(self, event: RasterRoiEditCommittedEvent) -> None:
        """Validate and commit an edited rectangle proposal.

        Args:
            event: Viewer event containing proposed committed geometry.
        """
        if (
            event.dataset_id != self._dataset_id
            or self._state is not RoiInteractionState.EDITING
            or not isinstance(event.roi, RectRoi)
        ):
            return
        roi = self._store.update(event.roi)
        self._selected_roi_id = roi.roi_id
        await self._viewer.rois.complete_commit(roi)
        self._set_state(RoiInteractionState.IDLE)
        LOGGER.info("Rectangle edit committed from viewer: roi=%s", roi.to_json())
