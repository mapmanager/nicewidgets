"""NiceGUI application object and HTTP routes for raster-viewer development."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nicegui import background_tasks, events, ui

from examples.raster_viewer_widget.datasets import RasterChannel, RasterDataset, SyntheticDatasetFactory
from examples.raster_viewer_widget.header import DemoHeader
from examples.raster_viewer_widget.roi_controller import DemoRoiController
from examples.raster_viewer_widget.roi_store import DemoRoiStore
from nicewidgets.raster_viewer_widget import (
    ImageBounds,
    LineEndpoints,
    NumPyRasterSource,
    RasterViewerConfig,
    RasterViewerWidget,
    RectRoiBounds,
    RoiHostMode,
    ViewerTheme,
    XYPlot,
    XYPlotMode,
    XYPlotStyle,
)
from nicewidgets.raster_viewer_widget.events import (
    RasterDisplayChangeEvent,
    RasterErrorEvent,
    RasterPerformanceEvent,
    RasterPlaneChangeEvent,
    RasterReadyEvent,
    RasterToolbarActionEvent,
    RasterViewChangeEvent,
)
from nicewidgets.raster_viewer_widget.models import RASTER_DESCRIPTOR_SCHEMA_VERSION

LOGGER = logging.getLogger(__name__)


class DemoDatasetCollection:
    """Own the small immutable collection of in-memory demo datasets."""

    def __init__(self, datasets: tuple[RasterDataset, ...]) -> None:
        """Initialize and index datasets.

        Args:
            datasets: Unique synthetic raster datasets.

        Raises:
            ValueError: If the collection is empty or dataset IDs repeat.
        """
        if not datasets:
            raise ValueError("at least one demo dataset is required")
        indexed = {dataset.dataset_id: dataset for dataset in datasets}
        if len(indexed) != len(datasets):
            raise ValueError("demo dataset IDs must be unique")
        self._datasets = indexed
        self._roi_stores = {
            dataset.dataset_id: self._create_roi_store(dataset) for dataset in datasets
        }

    @staticmethod
    def _create_roi_store(dataset: RasterDataset) -> DemoRoiStore:
        """Create deterministic sample rectangle and line ROIs for one dataset.

        Args:
            dataset: Dataset whose shape defines valid ROI bounds.

        Returns:
            Store containing two image-bounded mixed-shape sample ROIs.
        """
        height, width = dataset.plane_shape
        store = DemoRoiStore(ImageBounds(width=width, height=height))
        store.create_rect(
            RectRoiBounds(
                row_start=height // 8,
                row_stop=max(height // 8 + 1, height // 2),
                col_start=width // 8,
                col_stop=max(width // 8 + 1, width // 2),
            )
        )
        store.create_line(
            LineEndpoints(
                row0=height // 2,
                col0=width // 4,
                row1=min(height - 1, 3 * height // 4),
                col1=min(width - 1, 3 * width // 4),
            )
        )
        return store

    def summaries(self) -> list[dict[str, str]]:
        """Return dataset choices for the NiceGUI selector.

        Returns:
            Ordered dataset ID and label mappings.
        """
        return [
            {"id": dataset.dataset_id, "label": dataset.label}
            for dataset in self._datasets.values()
        ]

    def get_dataset(self, dataset_id: str) -> RasterDataset:
        """Return a dataset by ID.

        Args:
            dataset_id: Requested dataset identifier.

        Returns:
            Matching raster dataset.

        Raises:
            KeyError: If no dataset has the requested ID.
        """
        return self._datasets[dataset_id]

    def get_channel(self, dataset_id: str, channel_id: str) -> RasterChannel:
        """Return one channel from a dataset.

        Args:
            dataset_id: Owning dataset identifier.
            channel_id: Requested channel identifier.

        Returns:
            Matching raster channel.

        Raises:
            KeyError: If the dataset or channel does not exist.
        """
        dataset = self.get_dataset(dataset_id)
        for channel in dataset.channels:
            if channel.channel_id == channel_id:
                return channel
        raise KeyError(channel_id)

    def channel_indices(self, dataset_id: str) -> tuple[int, ...]:
        """Return zero-based logical channel indices for one dataset.

        Args:
            dataset_id: Requested dataset identifier.

        Returns:
            Stable zero-based logical channel indices for the dataset.
        """
        dataset = self.get_dataset(dataset_id)
        return tuple(range(len(dataset.channels)))

    def get_roi_store(self, dataset_id: str) -> DemoRoiStore:
        """Return one dataset's committed ROI store.

        Args:
            dataset_id: Owning dataset identifier.

        Returns:
            Matching mutable ROI store.

        Raises:
            KeyError: If the dataset is unknown.
        """
        return self._roi_stores[dataset_id]

    def summary_text(self, dataset_id: str) -> str:
        """Return a compact human-readable dataset summary.

        Args:
            dataset_id: Requested dataset identifier.

        Returns:
            Channel count, shared shape, and unique channel dtype names.

        Raises:
            KeyError: If the dataset does not exist.
        """
        dataset = self.get_dataset(dataset_id)
        dtypes = sorted({channel.served_dtype for channel in dataset.channels})
        channel_word = "channel" if len(dataset.channels) == 1 else "channels"
        shape_text = " × ".join(str(size) for size in dataset.shape)
        return f"{len(dataset.channels)} {channel_word} | {shape_text} | {'/'.join(dtypes)}"

    def display_x_range(self, dataset_id: str) -> tuple[float, float]:
        """Return the full transposed display-X range in physical units.

        Args:
            dataset_id: Requested dataset identifier.

        Returns:
            Inclusive lower and upper display-X edge values.

        Raises:
            KeyError: If the dataset is unknown.
        """
        dataset = self.get_dataset(dataset_id)
        height, _width = dataset.plane_shape
        return 0.0, height * dataset.header.physical_units[-2]

    def sample_xy_plot(self, dataset_id: str, revision: int = 0) -> XYPlot:
        """Build a deterministic physical-coordinate plot for API exercises.

        The first and last points intentionally extend beyond the image. This
        demonstrates that plot coordinates are not clamped to the raster and
        helps callers diagnose coordinate-system mismatches visually.

        Args:
            dataset_id: Dataset whose physical image extent scales the plot.
            revision: Non-negative update number used to alter style and data.

        Returns:
            Plot containing markers, line gaps, stable IDs, and outside points.

        Raises:
            ValueError: If ``revision`` is negative.
        """
        if revision < 0:
            raise ValueError("revision must be non-negative")
        dataset = self.get_dataset(dataset_id)
        height, width = dataset.plane_shape
        y_step, x_step = dataset.header.physical_units[-2:]
        x_maximum = height * y_step
        y_maximum = width * x_step
        offset = min(0.08 + revision * 0.01, 0.2)
        return XYPlot(
            plot_id="demo_xy",
            name="Demo X/Y data",
            x=(-0.08 * x_maximum, 0.2 * x_maximum, 0.45 * x_maximum,
               float("nan"), 0.72 * x_maximum, 1.08 * x_maximum),
            y=(0.15 * y_maximum, (0.3 + offset) * y_maximum, 0.65 * y_maximum,
               0.5 * y_maximum, 0.82 * y_maximum, 0.55 * y_maximum),
            point_ids=tuple(f"sample-{index}" for index in range(6)),
            mode=XYPlotMode.LINES_MARKERS,
            style=XYPlotStyle(
                color="#22d3ee" if revision % 2 == 0 else "#facc15",
                marker_size=7,
                line_width=2,
            ),
        )

    def source(self, dataset_id: str) -> NumPyRasterSource:
        """Build a reusable widget source over one demo dataset.

        Args:
            dataset_id: Requested dataset identifier.

        Returns:
            NumPy-backed source containing current committed ROIs.
        """
        dataset = self.get_dataset(dataset_id)
        return NumPyRasterSource.from_channels(
            {channel.channel_id: channel.values for channel in dataset.channels},
            dims=dataset.header.dims,
            physical_units=dataset.header.physical_units,
            physical_units_labels=dataset.header.physical_units_labels,
            rois=self.get_roi_store(dataset_id).to_json(),
            source_id=dataset.dataset_id,
            label=dataset.label,
            default_luts=[channel.default_lut for channel in dataset.channels],
        )

    def descriptor(self, dataset_id: str) -> dict[str, Any]:
        """Build the browser-facing dataset descriptor.

        Args:
            dataset_id: Requested dataset identifier.

        Returns:
            JSON-compatible dataset and channel metadata.

        Raises:
            KeyError: If the dataset does not exist.
        """
        dataset = self.get_dataset(dataset_id)
        height, width = dataset.plane_shape
        y_step, x_step = dataset.header.physical_units[-2:]
        y_label, x_label = dataset.header.physical_units_labels[-2:]
        return {
            "schema_version": RASTER_DESCRIPTOR_SCHEMA_VERSION,
            "id": dataset.dataset_id,
            "label": dataset.label,
            "header": dataset.header.to_json(),
            "width": width,
            "height": height,
            "layout": "row-major",
            "endianness": "little",
            "display_orientation": {"transpose": True, "flip_y": True},
            "axes": {
                "x": {"label": x_label, "step": x_step, "unit": ""},
                "y": {"label": y_label, "step": y_step, "unit": ""},
            },
            "rois": self.get_roi_store(dataset_id).to_json(),
            "channels": [
                {
                    "id": channel.channel_id,
                    "index": index,
                    "label": channel.label,
                    "dtype": channel.served_dtype,
                    "encoding": channel.encoding,
                    "byte_length": height * width * channel.values.dtype.itemsize,
                    "display": {
                        "lut": channel.default_lut,
                        "value_min": None,
                        "value_max": None,
                        "visible": True,
                    },
                    "data_url": (
                        f"/api/raster/datasets/{dataset.dataset_id}/channels/"
                        f"{channel.channel_id}/plane"
                    ),
                }
                for index, channel in enumerate(dataset.channels)
            ],
        }


class RasterViewerDemo:
    """Configure and render the self-contained NiceGUI viewer demo."""

    def __init__(self, example_root: Path) -> None:
        """Initialize demo data, static assets, and HTTP routes.

        Args:
            example_root: Directory containing this example and its optional data.
        """
        self._example_root = example_root
        data_directory = example_root / "data"
        self._datasets = DemoDatasetCollection(
            SyntheticDatasetFactory(data_directory=data_directory).create_all()
        )
        self._viewer: RasterViewerWidget | None = None
        LOGGER.info("Initialized raster demo with %d datasets", len(self._datasets.summaries()))

    @property
    def datasets(self) -> DemoDatasetCollection:
        """Return the dataset collection shared by demo pages.

        Returns:
            In-memory dataset and ROI provider.
        """
        return self._datasets

    def set_dark_mode(self, enabled: bool) -> None:
        """Apply the combined demo's theme to the mounted raster viewer.

        Args:
            enabled: True for dark viewer chrome; false for light chrome.
        """
        if self._viewer is not None:
            theme = ViewerTheme.DARK if enabled else ViewerTheme.LIGHT
            background_tasks.create(self._viewer.set_theme(theme))

    def build_page(
        self,
        *,
        embedded: bool = False,
        dark_mode: bool = True,
    ) -> RasterViewerDemo:
        """Build the NiceGUI page and mount the JavaScript raster viewer.

        Args:
            embedded: Whether the shared main-demo shell owns page chrome.
            dark_mode: Initial viewer theme when embedded.

        Returns:
            This themeable demo handle.
        """
        header = None if embedded else DemoHeader()
        if not embedded:
            with ui.row().classes("w-full justify-end"):
                ui.link("Multi-widget test", "/multi-widget")
        choices = {item["id"]: item["label"] for item in self._datasets.summaries()}
        first_id = next(iter(choices))
        with ui.row().classes("items-center gap-4"):
            selector = ui.select(
                choices,
                value=first_id,
                label="Synthetic dataset",
            ).classes("w-80")
            summary = ui.label(self._datasets.summary_text(first_id)).classes("opacity-70")
        initial_x_min, initial_x_max = self._datasets.display_x_range(first_id)
        with ui.row().classes("items-end gap-3"):
            api_axes = ui.checkbox("Axes", value=True)
            api_rois = ui.checkbox("ROIs", value=True)
            x_minimum = ui.number("X Min", value=initial_x_min).classes("w-32")
            x_maximum = ui.number("X Max", value=initial_x_max).classes("w-32")
            set_x_axis = ui.button("Set X Axis")
        with ui.row().classes("items-center gap-2"):
            ui.label("X/Y plot API")
            add_xy_plot = ui.button("Add")
            update_xy_plot = ui.button("Update")
            show_xy_plot = ui.button("Show")
            hide_xy_plot = ui.button("Hide")
            remove_xy_plot = ui.button("Remove")
            xy_plot_status = ui.label("No plot").classes("opacity-70")
        status = ui.label("Viewer loading…")
        with ui.column().classes("w-full gap-1"):
            splitter_height = "height: 100%; min-height: 520px" if embedded else (
                "height: 72vh; min-height: 520px"
            )
            splitter = (
                ui.splitter(horizontal=True, value=78, limits=(35, 92))
                .classes("w-full flex-1 min-h-0")
                .style(splitter_height)
            )
            with splitter.before:
                viewer = RasterViewerWidget(
                    source=self._datasets.source(first_id),
                    config=RasterViewerConfig(
                        theme=(
                            ViewerTheme.DARK
                            if dark_mode
                            else ViewerTheme.LIGHT
                        )
                        if header is None
                        else ViewerTheme(header.theme.value),
                        roi_host_mode=RoiHostMode.DELEGATED,
                    ),
                )
            with splitter.after, ui.column().classes(
                "w-full h-full items-center justify-center"
            ):
                ui.label("Resize placeholder").classes("opacity-70")
        self._viewer = viewer
        if header is not None:
            header.bind_theme_change(viewer.set_theme)
        roi_controller = DemoRoiController(self._datasets, selector, viewer, first_id)
        xy_plot_revision = 0

        async def load_selected_dataset(event: events.ValueChangeEventArguments[str]) -> None:
            """Tell the mounted JavaScript viewer to load a selected dataset.

            Args:
                event: NiceGUI value-change event containing the dataset ID.
            """
            dataset_id = str(event.value)
            roi_controller.set_dataset(dataset_id)
            summary.set_text(self._datasets.summary_text(dataset_id))
            full_minimum, full_maximum = self._datasets.display_x_range(dataset_id)
            x_minimum.set_value(full_minimum)
            x_maximum.set_value(full_maximum)
            await viewer.load_source(self._datasets.source(dataset_id))
            xy_plot_status.set_text("Plots cleared by dataset change")

        async def add_plot_from_python() -> None:
            """Add the demo X/Y plot through the namespaced widget API."""
            plot_id = await viewer.xy_plots.add(
                self._datasets.sample_xy_plot(str(selector.value))
            )
            xy_plot_status.set_text(f"Added {plot_id}")
            LOGGER.info("Python added JS X/Y plot: %s", plot_id)

        async def update_plot_from_python() -> None:
            """Replace the demo plot data and style through the public API."""
            nonlocal xy_plot_revision
            xy_plot_revision += 1
            updated = await viewer.xy_plots.update(
                self._datasets.sample_xy_plot(str(selector.value), xy_plot_revision)
            )
            xy_plot_status.set_text(f"Updated demo_xy: {updated}")
            LOGGER.info("Python updated JS X/Y plot revision %d", xy_plot_revision)

        async def set_plot_visibility(visible: bool) -> None:
            """Exercise namespaced plot visibility without replacing data.

            Args:
                visible: True to show the demo plot; false to hide it.
            """
            applied = (
                await viewer.xy_plots.show("demo_xy")
                if visible
                else await viewer.xy_plots.hide("demo_xy")
            )
            xy_plot_status.set_text(f"{'Shown' if visible else 'Hidden'}: {applied}")
            LOGGER.info("Python set JS X/Y plot visibility=%s applied=%s", visible, applied)

        async def remove_plot_from_python() -> None:
            """Remove the demo X/Y plot through the namespaced widget API."""
            removed = await viewer.xy_plots.remove("demo_xy")
            xy_plot_status.set_text(f"Removed: {removed}")
            LOGGER.info("Python removed JS X/Y plot: %s", removed)

        def handle_ready(event: RasterReadyEvent) -> None:
            """Update status and X controls after JavaScript initialization.

            Args:
                event: Viewer-ready event containing the full physical X range.
            """
            status.set_text("Viewer ready")
            x_axis = event.x_axis
            x_minimum.set_value(x_axis.minimum)
            x_maximum.set_value(x_axis.maximum)

        def handle_error(event: RasterErrorEvent) -> None:
            """Display a JavaScript viewer error in the NiceGUI page.

            Args:
                event: NiceGUI event carrying the viewer error payload.
            """
            status.set_text(event.message)

        def handle_view_change(event: RasterViewChangeEvent) -> None:
            """Log a viewport interaction received from JavaScript.

            Args:
                event: NiceGUI event whose arguments contain the JavaScript
                    viewport-change payload.
            """
            LOGGER.info("JS view change received: %s", event.payload)

        def handle_display_change(event: RasterDisplayChangeEvent) -> None:
            """Log channel display settings received from JavaScript.

            Args:
                event: NiceGUI event whose arguments contain the JavaScript
                    display-change payload.
            """
            LOGGER.info("JS display change received: %s", event.payload)

        def handle_toolbar_action(event: RasterToolbarActionEvent) -> None:
            """Log a discrete viewer-toolbar action received from JavaScript.

            Args:
                event: NiceGUI event whose arguments contain the JavaScript
                    toolbar-action payload.
            """
            payload = event.payload
            if payload.get("action") == "axes":
                api_axes.set_value(bool(payload.get("visible")))
            elif payload.get("action") == "rois":
                api_rois.set_value(bool(payload.get("visible")))
            LOGGER.info("JS toolbar action received: %s", payload)

        def handle_plane_change(event: RasterPlaneChangeEvent) -> None:
            """Log a committed Z/sliding-Z plane selection from JavaScript.

            Args:
                event: NiceGUI event containing Z center and projection radius.
            """
            LOGGER.info("JS plane change received: %s", event.payload)

        def handle_performance(event: RasterPerformanceEvent) -> None:
            """Log browser plane-loading and rendering timing metrics.

            Args:
                event: NiceGUI event containing a measured JavaScript phase.
            """
            LOGGER.info("JS performance metric: %s", event.payload)

        async def set_axes_from_python(
            event: events.ValueChangeEventArguments[bool | None],
        ) -> None:
            """Exercise the public JavaScript axes-visibility API.

            Args:
                event: NiceGUI checkbox event containing the desired visibility.
            """
            visible = bool(event.value)
            applied = await viewer.set_axes_visible(visible)
            LOGGER.info("Python set JS axes visibility: requested=%s applied=%s", visible, applied)

        async def set_rois_from_python(
            event: events.ValueChangeEventArguments[bool | None],
        ) -> None:
            """Exercise the public JavaScript ROI-visibility API.

            Args:
                event: NiceGUI checkbox event containing the desired visibility.
            """
            visible = bool(event.value)
            applied = await viewer.set_rois_visible(visible)
            if applied != visible:
                api_rois.set_value(applied)
            LOGGER.info("Python set JS ROI visibility: requested=%s applied=%s", visible, applied)

        async def set_x_range_from_python() -> None:
            """Apply the NiceGUI physical X range through the public viewer API."""
            if x_minimum.value is None or x_maximum.value is None:
                ui.notify("X Min and X Max are required", type="warning")
                return
            minimum = float(x_minimum.value)
            maximum = float(x_maximum.value)
            if minimum >= maximum:
                ui.notify("X Min must be smaller than X Max", type="warning")
                return
            full_minimum, full_maximum = self._datasets.display_x_range(str(selector.value))
            minimum = max(full_minimum, min(minimum, full_maximum))
            maximum = max(full_minimum, min(maximum, full_maximum))
            if minimum >= maximum:
                ui.notify("X range is outside the displayed dataset", type="warning")
                return
            applied = await viewer.set_x_range(minimum, maximum)
            if isinstance(applied, dict):
                x_minimum.set_value(float(applied["minimum"]))
                x_maximum.set_value(float(applied["maximum"]))
            LOGGER.info(
                "Python set JS physical X range: requested=(%s, %s) applied=%s",
                minimum,
                maximum,
                applied,
            )

        selector.on_value_change(load_selected_dataset)
        api_axes.on_value_change(set_axes_from_python)
        api_rois.on_value_change(set_rois_from_python)
        set_x_axis.on_click(set_x_range_from_python)
        add_xy_plot.on_click(add_plot_from_python)
        update_xy_plot.on_click(update_plot_from_python)
        show_xy_plot.on_click(lambda: set_plot_visibility(True))
        hide_xy_plot.on_click(lambda: set_plot_visibility(False))
        remove_xy_plot.on_click(remove_plot_from_python)
        viewer.on_ready(handle_ready)
        viewer.on_error(handle_error)
        viewer.on_view_change(
            handle_view_change,
            throttle=0.1,
            leading_events=True,
            trailing_events=True,
        )
        viewer.on_display_change(
            handle_display_change,
            throttle=0.1,
            leading_events=True,
            trailing_events=True,
        )
        viewer.on_toolbar_action(handle_toolbar_action)
        viewer.on_plane_change(handle_plane_change)
        viewer.on_performance(handle_performance)
        return self
