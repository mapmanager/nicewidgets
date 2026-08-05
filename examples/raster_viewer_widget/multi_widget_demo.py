"""Dedicated two-instance acceptance page for the reusable raster viewer."""

from __future__ import annotations

import logging
from typing import Any

from nicegui import events, ui
from nicegui.elements.select import Select

from examples.raster_viewer_widget.raster_demo import DemoDatasetCollection
from nicewidgets.raster_viewer_widget import RasterViewerConfig, RasterViewerWidget, ViewerTheme
from nicewidgets.raster_viewer_widget.events import (
    RasterErrorEvent,
    RasterPlaneChangeEvent,
    RasterReadyEvent,
    RasterRoiSelectedEvent,
    RasterToolbarActionEvent,
)

LOGGER = logging.getLogger(__name__)


class MultiWidgetPane:
    """Own one viewer instance, its controls, and callback wiring."""

    def __init__(
        self,
        datasets: DemoDatasetCollection,
        initial_dataset_id: str,
        title: str,
    ) -> None:
        """Build one independent acceptance-test pane.

        Args:
            datasets: Shared provider of immutable image datasets and ROI snapshots.
            initial_dataset_id: Dataset loaded into this pane initially.
            title: Human-readable pane identity used in UI and logs.
        """
        self._datasets = datasets
        self._title = title
        self._viewer: RasterViewerWidget | None = None
        choices = {item["id"]: item["label"] for item in datasets.summaries()}
        with ui.column().classes("w-full h-full min-h-0 gap-2 p-2"):
            with ui.row().classes("w-full items-center gap-2"):
                ui.label(title).classes("font-medium")
                self._status = ui.label("Loading…").classes("opacity-70")
                ui.space()
                ui.button("Recreate", on_click=self._recreate).props(
                    'outline aria-label="Destroy and recreate this viewer"'
                )
            self._dataset_selector: Select = ui.select(
                choices,
                value=initial_dataset_id,
                label="Dataset",
                on_change=self._load_selected_dataset,
            ).classes("w-full")
            with ui.row().classes("items-center gap-3"):
                self._axes = ui.checkbox(
                    "Axes", value=True, on_change=self._set_axes
                )
                self._rois = ui.checkbox(
                    "ROIs", value=True, on_change=self._set_rois
                )
                self._dark = ui.checkbox(
                    "Dark", value=True, on_change=self._set_theme
                )
                ui.button("Select ROI 1", on_click=self._select_first_roi).props("outline")
                ui.button("Clear ROI", on_click=self._clear_roi_selection).props("outline")
            self._viewer_container = ui.column().classes("w-full grow min-h-0")
        self._create_viewer()

    @property
    def viewer(self) -> RasterViewerWidget:
        """Return the current live viewer instance.

        Returns:
            Viewer currently mounted in this pane.

        Raises:
            RuntimeError: If the pane is between destruction and recreation.
        """
        if self._viewer is None:
            raise RuntimeError(f"{self._title} viewer is not mounted")
        return self._viewer

    def _create_viewer(self) -> None:
        """Create and wire one new component instance inside the pane container."""
        dataset_id = str(self._dataset_selector.value)
        config = RasterViewerConfig(
            theme=ViewerTheme.DARK if bool(self._dark.value) else ViewerTheme.LIGHT,
            axes_visible=bool(self._axes.value),
            rois_visible=bool(self._rois.value),
        )
        with self._viewer_container:
            viewer = RasterViewerWidget(
                source=self._datasets.source(dataset_id),
                config=config,
            )
        self._viewer = viewer
        viewer.on_ready(self._handle_ready)
        viewer.on_error(self._handle_error)
        viewer.on_toolbar_action(self._handle_toolbar_action)
        viewer.on_plane_change(self._handle_plane_change)
        viewer.on_roi_selected(self._handle_roi_selected)
        LOGGER.info("%s created widget for dataset %s", self._title, dataset_id)

    async def _recreate(self) -> None:
        """Destroy and recreate only this pane's viewer instance."""
        self._status.set_text("Recreating…")
        self._viewer = None
        self._viewer_container.clear()
        self._create_viewer()

    async def _load_selected_dataset(
        self,
        event: events.ValueChangeEventArguments[Any],
    ) -> None:
        """Load a new Python source into this pane only.

        Args:
            event: Dataset selector change carrying the new dataset ID.
        """
        dataset_id = str(event.value)
        self._status.set_text("Loading…")
        await self.viewer.load_source(self._datasets.source(dataset_id))
        LOGGER.info("%s loaded dataset %s", self._title, dataset_id)

    async def _set_axes(
        self,
        event: events.ValueChangeEventArguments[bool | None],
    ) -> None:
        """Set this pane's axis visibility.

        Args:
            event: Checkbox event containing the requested visibility.
        """
        await self.viewer.set_axes_visible(bool(event.value))

    async def _set_rois(
        self,
        event: events.ValueChangeEventArguments[bool | None],
    ) -> None:
        """Set this pane's ROI-overlay visibility.

        Args:
            event: Checkbox event containing the requested visibility.
        """
        await self.viewer.set_rois_visible(bool(event.value))

    async def _set_theme(
        self,
        event: events.ValueChangeEventArguments[bool | None],
    ) -> None:
        """Set this pane's viewer theme independently.

        Args:
            event: Checkbox event that is true for dark mode.
        """
        theme = ViewerTheme.DARK if bool(event.value) else ViewerTheme.LIGHT
        await self.viewer.set_theme(theme)

    async def _select_first_roi(self) -> None:
        """Select ROI ID 1 through this viewer's public Python API."""
        applied = await self.viewer.rois.select(1)
        self._status.set_text("ROI 1 selected" if applied else "ROI 1 unavailable")

    async def _clear_roi_selection(self) -> None:
        """Clear this viewer's ROI selection through its public Python API."""
        await self.viewer.rois.clear_selection()
        self._status.set_text("ROI selection cleared")

    def _handle_ready(self, event: RasterReadyEvent) -> None:
        """Report successful initialization for this pane.

        Args:
            event: Typed ready event carrying the dataset ID.
        """
        self._status.set_text(f"Ready: {event.dataset_id}")
        LOGGER.info("%s ready: %s", self._title, event.payload)

    def _handle_error(self, event: RasterErrorEvent) -> None:
        """Report a viewer failure without affecting the other pane.

        Args:
            event: Typed viewer error.
        """
        self._status.set_text(f"Error: {event.message}")
        LOGGER.error("%s viewer error: %s", self._title, event.message)

    def _handle_toolbar_action(self, event: RasterToolbarActionEvent) -> None:
        """Mirror JavaScript visibility actions into this pane's controls.

        Args:
            event: Typed toolbar action originating in this viewer.
        """
        if event.action == "axes":
            self._axes.set_value(bool(event.payload.get("visible")))
        elif event.action == "rois":
            self._rois.set_value(bool(event.payload.get("visible")))
        LOGGER.info("%s toolbar action: %s", self._title, event.payload)

    def _handle_plane_change(self, event: RasterPlaneChangeEvent) -> None:
        """Log this pane's committed Z-plane changes.

        Args:
            event: Typed plane-change event.
        """
        LOGGER.info("%s plane change: %s", self._title, event.payload)

    def _handle_roi_selected(self, event: RasterRoiSelectedEvent) -> None:
        """Report pointer selection from this viewer only.

        Args:
            event: Typed ROI-selection event.
        """
        self._status.set_text(f"Selected ROI: {event.payload.get('roi_id')}")
        LOGGER.info("%s ROI selection: %s", self._title, event.payload)


class MultiWidgetDemo:
    """Build a focused page containing two independent raster widgets."""

    def __init__(self, datasets: DemoDatasetCollection) -> None:
        """Initialize the showcase with shared dataset definitions.

        Args:
            datasets: Dataset provider used to construct independent sources.
        """
        self._datasets = datasets

    def build_page(self) -> None:
        """Build the two-pane widget isolation and lifecycle showcase."""
        summaries = self._datasets.summaries()
        first_id = summaries[0]["id"]
        second_id = summaries[2]["id"] if len(summaries) > 2 else summaries[-1]["id"]
        with ui.header().classes("items-center gap-3"):
            ui.label("RasterViewerWidget Multi-Instance Test").classes(
                "text-h6 font-medium"
            )
            ui.space()
            ui.link("Main demo", "/").classes("text-white")
        ui.label(
            "Change either viewer independently, resize the divider, and recreate one instance."
        ).classes("opacity-70")
        splitter = (
            ui.splitter(value=50, limits=(25, 75))
            .classes("w-full")
            .style("height: 82vh; min-height: 520px")
        )
        with splitter.before:
            MultiWidgetPane(self._datasets, first_id, "Viewer A")
        with splitter.after:
            MultiWidgetPane(self._datasets, second_id, "Viewer B")
