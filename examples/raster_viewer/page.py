"""Reusable page builder for the raster viewer demo."""

from __future__ import annotations

from nicegui import background_tasks, ui

try:
    from examples.raster_viewer.demo_controller import RasterDemoController
    from examples.raster_viewer.sample_data import SampleDataCatalog
except ImportError:
    # Running the standalone entry as a plain script puts this directory on sys.path.
    from demo_controller import RasterDemoController  # type: ignore[no-redef]
    from sample_data import SampleDataCatalog  # type: ignore[no-redef]


def build_raster_demo_page(
    *,
    embedded: bool = False,
    dark_mode: bool = False,
) -> RasterDemoController:
    """Build the raster demo in the current NiceGUI slot.

    Args:
        embedded: Whether the page is hosted inside another full-height layout.
        dark_mode: Initial Plotly layout theme.

    Returns:
        Controller owning the demo widgets and state.
    """
    catalog = SampleDataCatalog()
    controller = RasterDemoController(catalog, dark_mode=dark_mode)
    height_class = 'h-full' if embedded else 'min-h-screen'

    with ui.column().classes(f'w-full {height_class} gap-4 p-4'):
        ui.label('Raster Viewer Demo').classes('text-h5')
        controller.build()

        with ui.row().classes('items-center gap-2'):
            ui.select(
                options=catalog.names,
                value=controller.dataset_name,
                label='Dataset',
                on_change=lambda e: background_tasks.create(
                    controller.load_dataset(str(e.value))
                ),
            )
            ui.button(
                'Reload current dataset',
                on_click=lambda: background_tasks.create(
                    controller.load_dataset(controller.dataset_name)
                ),
            )

        with ui.row().classes('items-end gap-4 flex-wrap'):
            x_axis_min = ui.number(
                label='Plot x min (physical)', value=0.0, format='%.6g'
            )
            x_axis_max = ui.number(
                label='Plot x max (physical)', value=1.0, format='%.6g'
            )

            async def apply_x_axis_range() -> None:
                try:
                    await controller.viewer.set_x_axis_range(
                        x_min=float(x_axis_min.value),
                        x_max=float(x_axis_max.value),
                    )
                except RuntimeError as exc:
                    ui.notify(str(exc), type='warning')

            ui.button('Set X axis range', on_click=apply_x_axis_range)

    return controller
