"""NiceGUI raster viewer demo: PlotlyRasterViewer + ImageToolbarWidget + ContrastWidget.

Run from the repository root:

    uv run python examples/raster_viewer/nicegui_raster_demo.py

Layout and wiring live in ``demo_controller.py``; synthetic datasets live in
``sample_data.py``. This entry module only builds the page and demo-only
controls (dataset select, x-axis range).
"""

from __future__ import annotations

from nicegui import background_tasks, ui

from nicewidgets.utils.logging import setup_logging

try:
    from examples.raster_viewer.demo_controller import RasterDemoController
    from examples.raster_viewer.sample_data import SampleDataCatalog
except ImportError:
    # Running as a plain script puts this directory on sys.path.
    from demo_controller import RasterDemoController
    from sample_data import SampleDataCatalog

setup_logging(level='DEBUG')


@ui.page('/')
def home() -> None:
    """Build the demo page."""
    catalog = SampleDataCatalog()
    controller = RasterDemoController(catalog)

    with ui.column().classes('w-full gap-4'):
        ui.label('Raster Viewer Demo').classes('text-h5')
        controller.build()

        with ui.row().classes('items-center gap-2'):
            ui.select(
                options=catalog.names,
                value=controller.dataset_name,
                label='Dataset',
                on_change=lambda e: background_tasks.create(controller.load_dataset(str(e.value))),
            )
            ui.button(
                'Reload current dataset',
                on_click=lambda: background_tasks.create(controller.load_dataset(controller.dataset_name)),
            )

        with ui.row().classes('items-end gap-4 flex-wrap'):
            x_axis_min = ui.number(label='Plot x min (physical)', value=0.0, format='%.6g')
            x_axis_max = ui.number(label='Plot x max (physical)', value=1.0, format='%.6g')

            async def apply_x_axis_range() -> None:
                try:
                    await controller.viewer.set_x_axis_range(
                        x_min=float(x_axis_min.value),
                        x_max=float(x_axis_max.value),
                    )
                except RuntimeError as exc:
                    ui.notify(str(exc), type='warning')

            ui.button('Set X axis range', on_click=apply_x_axis_range)


if __name__ in {'__main__', '__mp_main__'}:
    ui.run(reload=False, native=True)
