"""Reusable page builder for the NicePool demo."""

from __future__ import annotations

from nicegui import ui

try:
    from examples.nicepool.demo_controller import NicePoolDemoController
    from examples.nicepool.sample_data import SampleDataCatalog
except ImportError:
    # Running the standalone entry as a plain script puts this directory on sys.path.
    from demo_controller import NicePoolDemoController  # type: ignore[no-redef]
    from sample_data import SampleDataCatalog  # type: ignore[no-redef]


def build_nicepool_demo_page(
    *,
    embedded: bool = False,
    dark_mode: bool = False,
) -> NicePoolDemoController:
    """Build the NicePool demo in the current NiceGUI slot.

    Args:
        embedded: Whether the page is hosted inside another full-height layout.
        dark_mode: Initial Plotly layout theme.

    Returns:
        Controller owning the demo widget and state.
    """
    catalog = SampleDataCatalog()
    controller = NicePoolDemoController(catalog, dark_mode=dark_mode)
    height_class = 'h-full' if embedded else 'h-screen'

    with ui.column().classes(f'w-full {height_class} min-h-0 gap-4 p-4'):
        ui.label('NicePool Demo').classes('text-h5')
        ui.label(
            'Synthetic velocity-pool DataFrame (one row per file/channel/ROI). '
            'Use the pre-filter dropdowns, plot-type controls, and named presets; '
            'click a point or table row to link the selection.'
        ).classes('text-caption text-grey-7')

        with ui.row().classes('items-center gap-4 flex-wrap'):
            ui.select(
                options=catalog.names,
                value=controller.dataset_name,
                label='Dataset',
                on_change=lambda e: controller.load_dataset(str(e.value)),
            ).classes('min-w-64')
            ui.button(
                'Select accepted rows',
                on_click=controller.select_accepted_rows,
            )
            if not embedded:
                ui.switch(
                    'Dark plots',
                    value=dark_mode,
                    on_change=lambda e: controller.set_dark_mode(bool(e.value)),
                )

        selection_label = ui.label('Selected: (none)').classes('text-caption')
        controller.bind_selection_label(selection_label)

        with ui.column().classes('w-full flex-1 min-h-0'):
            controller.build()

    return controller
