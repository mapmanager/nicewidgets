"""Reusable page builder for the PlotlyPlotWidget demo."""

from __future__ import annotations

from nicegui import ui

try:
    from examples.plotly_plot.demo_controller import PlotlyPlotDemoController
    from examples.plotly_plot.sample_data import SampleDataCatalog
except ImportError:
    # Running the standalone entry as a plain script puts this directory on sys.path.
    from demo_controller import PlotlyPlotDemoController  # type: ignore[no-redef]
    from sample_data import SampleDataCatalog  # type: ignore[no-redef]


def build_plotly_plot_demo_page(
    *,
    embedded: bool = False,
    dark_mode: bool = False,
) -> PlotlyPlotDemoController:
    """Build the PlotlyPlotWidget demo in the current NiceGUI slot.

    Args:
        embedded: Whether the page is hosted inside another full-height layout.
        dark_mode: Initial Plotly layout theme.

    Returns:
        Controller owning the demo widget and state.
    """
    catalog = SampleDataCatalog()
    controller = PlotlyPlotDemoController(catalog, dark_mode=dark_mode)
    # Natural page height: the plot uses a fixed h-96. Avoid h-full here so the
    # embedded flex/overflow shell cannot collapse the plot on first navigation.
    height_class = 'min-h-0' if embedded else 'min-h-screen'

    with ui.column().classes(f'w-full {height_class} gap-4 p-4'):
        ui.label('PlotlyPlotWidget Demo').classes('text-h5')
        ui.label(
            'Continuous scattergl line plus sparse peak markers. Zoom/pan the '
            'x-axis, drag the horizontal threshold, or use the buttons below. '
            'Right-click the plot for display options and Copy To Clipboard.'
        ).classes('text-caption text-grey-7')

        x_range_label = ui.label('x-range: auto').classes('text-caption')
        measurement_label = ui.label('measurement: none').classes('text-caption')
        controller.bind_labels(
            x_range_label=x_range_label,
            measurement_label=measurement_label,
        )
        controller.build()

        with ui.row().classes('items-center gap-2 flex-wrap'):
            ui.button('Zoom 8-14 s', on_click=controller.zoom_window)
            ui.button('Reset x-axis', on_click=controller.reset_x_axis)
            ui.button('Move peaks', on_click=controller.move_peaks)
            if not embedded:
                ui.switch(
                    'Dark plots',
                    value=dark_mode,
                    on_change=lambda e: controller.set_dark_mode(bool(e.value)),
                )

    return controller
