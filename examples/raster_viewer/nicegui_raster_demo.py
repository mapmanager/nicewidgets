"""Small NiceGUI demo for the raster viewer.

This demo keeps the GUI thin. It generates a few dummy 2D numpy arrays with
per-dataset physical spacing (``dx``/``dy``) and axis units, passes them into
``PlotlyRasterViewer.set_data()``, and leaves multiresolution policy inside the
backend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from nicegui import ui

from nicewidgets.raster_viewer.backend.image_model import RasterGridSpec
from nicewidgets.raster_viewer.frontend.plotly_protocol import DEFAULT_HEATMAP_COLORSCALE
from nicewidgets.raster_viewer.frontend.plotly_viewer import PlotlyRasterViewer

from nicewidgets.utils.logging import get_logger, setup_logging

setup_logging(level='DEBUG')

logger = get_logger(__name__)

# Plotly built-in sequential colorscale names (``Gray`` / ``Grays`` are invalid).
COLORSCALE_OPTIONS = ('Greys', 'Viridis', 'Reds', 'Greens', 'Hot')


@dataclass(frozen=True)
class DemoDataset:
    """One selectable demo raster with physical grid metadata."""

    name: str
    factory: Callable[[], np.ndarray]
    grid: RasterGridSpec


class DummyArrayCatalog:
    """Provide a few lazily generated demo arrays with grids."""

    def __init__(self) -> None:
        self._datasets: list[DemoDataset] = [
            DemoDataset(
                name='Diagonal bands',
                factory=self._make_diagonal_bands,
                grid=RasterGridSpec(dx=1.0, dy=1.0, x_unit='Pixels', y_unit='Pixels'),
            ),
            DemoDataset(
                name='Gaussian blobs',
                factory=self._make_gaussian_blobs,
                grid=RasterGridSpec(dx=0.01, dy=0.02, x_unit='time (s)', y_unit='um'),
            ),
            DemoDataset(
                name='Noisy stripes',
                factory=self._make_noisy_stripes,
                grid=RasterGridSpec(dx=0.5, dy=1.0, x_unit='index', y_unit='Pixels'),
            ),
        ]
        self._by_name: dict[str, DemoDataset] = {d.name: d for d in self._datasets}

    @property
    def names(self) -> list[str]:
        """Return demo dataset names."""
        return [d.name for d in self._datasets]

    def get(self, name: str) -> tuple[np.ndarray, RasterGridSpec]:
        """Return the selected array and its grid."""
        d = self._by_name[name]
        return d.factory().astype(np.float32), d.grid

    @staticmethod
    def _make_diagonal_bands() -> np.ndarray:
        rows, cols = 5000, 1024
        y = np.arange(rows, dtype=np.float32)[:, None]
        x = np.arange(cols, dtype=np.float32)[None, :]
        return 0.5 + 0.5 * np.sin((x * 0.035) + (y * 0.11))

    @staticmethod
    def _make_gaussian_blobs() -> np.ndarray:
        rows, cols = 4096, 768
        y = np.linspace(-1.0, 1.0, rows, dtype=np.float32)[:, None]
        x = np.linspace(-3.0, 3.0, cols, dtype=np.float32)[None, :]
        blob1 = np.exp(-(((x + 1.2) ** 2) / 0.12 + ((y + 0.25) ** 2) / 0.08))
        blob2 = 0.8 * np.exp(-(((x - 0.8) ** 2) / 0.45 + ((y - 0.1) ** 2) / 0.03))
        blob3 = 0.6 * np.exp(-(((x - 1.8) ** 2) / 0.20 + ((y + 0.45) ** 2) / 0.10))
        return blob1 + blob2 + blob3

    @staticmethod
    def _make_noisy_stripes() -> np.ndarray:
        rows, cols = 12000, 512
        rng = np.random.default_rng(7)
        y = np.arange(rows, dtype=np.float32)[:, None]
        x = np.arange(cols, dtype=np.float32)[None, :]
        base = 0.45 + 0.25 * np.sin(x * 0.012)
        stripes = 0.20 * np.sin((x * 0.06) + (y * 0.20))
        noise = rng.normal(loc=0.0, scale=0.04, size=(rows, cols)).astype(np.float32)
        return base + stripes + noise


@ui.page('/')
def home() -> None:
    """Build the demo page."""
    catalog = DummyArrayCatalog()
    viewer = PlotlyRasterViewer()

    selected_name = {'value': catalog.names[0]}
    initial_load_done = {'value': False}

    with ui.column().classes('w-full gap-4'):
        ui.label('Raster Viewer Demo')
        viewer_plot = viewer.build()
        viewer_plot.classes('w-full h-[70vh]')

        async def load_selected() -> None:
            data, grid = catalog.get(selected_name['value'])
            logger.info(f'data.shape: {data.shape}, dx={grid.dx}, dy={grid.dy}')
            await viewer.set_data(data, grid=grid)

        async def on_afterplot(_) -> None:
            if initial_load_done['value']:
                return
            initial_load_done['value'] = True
            await load_selected()

        async def on_dataset_change(event) -> None:
            selected_name['value'] = str(event.value)
            await load_selected()

        viewer_plot.on('plotly_afterplot', on_afterplot)

        with ui.row().classes('items-center gap-2'):
            ui.select(
                options=catalog.names,
                value=selected_name['value'],
                label='Dummy dataset',
                on_change=on_dataset_change,
            )
            ui.button('Reload current dataset', on_click=load_selected)

        async def on_colorscale_change(event) -> None:
            try:
                await viewer.set_heatmap_colorscale(str(event.value))
            except RuntimeError as exc:
                ui.notify(str(exc), type='warning')

        with ui.row().classes('items-end gap-4 flex-wrap'):
            ui.select(
                options=list(COLORSCALE_OPTIONS),
                value=DEFAULT_HEATMAP_COLORSCALE,
                label='Heatmap colorscale',
                on_change=on_colorscale_change,
            )

        with ui.row().classes('items-end gap-4 flex-wrap'):
            zmin_input = ui.number(label='Heatmap zmin', value=0.0, format='%.6g')
            zmax_input = ui.number(label='Heatmap zmax', value=1.0, format='%.6g')

            async def apply_heatmap_contrast() -> None:
                try:
                    await viewer.set_heatmap_contrast(
                        zmin=float(zmin_input.value),
                        zmax=float(zmax_input.value),
                    )
                except RuntimeError as exc:
                    ui.notify(str(exc), type='warning')

            ui.button('Apply heatmap contrast', on_click=apply_heatmap_contrast)

        with ui.row().classes('items-end gap-4 flex-wrap'):
            x_axis_min = ui.number(label='Plot x min (physical)', value=0.0, format='%.6g')
            x_axis_max = ui.number(label='Plot x max (physical)', value=1.0, format='%.6g')

            async def apply_x_axis_range() -> None:
                try:
                    await viewer.set_x_axis_range(
                        x_min=float(x_axis_min.value),
                        x_max=float(x_axis_max.value),
                    )
                except RuntimeError as exc:
                    ui.notify(str(exc), type='warning')

            ui.button('Set X axis range', on_click=apply_x_axis_range)


if __name__ in {'__main__', '__mp_main__'}:
    ui.run(reload=False, native=True)
