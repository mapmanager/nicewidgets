"""Demo controller wiring PlotlyPlotWidget to synthetic signal data.

Owns demo state and translates demo-only controls into the public widget API.
Follows the nicewidgets host-application pattern: callbacks in, ``*_ext``-style
programmatic updates out (here via public setters such as
``set_x_axis_limits`` and ``set_dark_mode``).
"""

from __future__ import annotations

from nicegui import ui

from nicewidgets.plotly_plot.models import MeasurementChangeEvent
from nicewidgets.plotly_plot.widget import PlotlyPlotWidget
from nicewidgets.utils.logging import get_logger

try:
    from examples.plotly_plot.sample_data import DemoSignal, SampleDataCatalog
except ImportError:
    from sample_data import DemoSignal, SampleDataCatalog  # type: ignore[no-redef]

logger = get_logger(__name__)


class PlotlyPlotDemoController:
    """Own demo state and wire PlotlyPlotWidget callbacks to demo labels.

    Args:
        catalog: Sample data source.
        dark_mode: Initial Plotly layout theme.
    """

    def __init__(
        self,
        catalog: SampleDataCatalog,
        *,
        dark_mode: bool = False,
    ) -> None:
        self._catalog = catalog
        self._dataset_name: str = catalog.names[0]
        self._dark_mode = bool(dark_mode)
        self._plot: PlotlyPlotWidget | None = None
        self._x_range_label: ui.label | None = None
        self._measurement_label: ui.label | None = None
        self._peak_offset = 0.0

    @property
    def dataset_name(self) -> str:
        """Return the currently loaded dataset name."""
        return self._dataset_name

    @property
    def plot(self) -> PlotlyPlotWidget:
        """Return the built plot widget.

        Raises:
            RuntimeError: If accessed before :meth:`build`.
        """
        if self._plot is None:
            raise RuntimeError('PlotlyPlotWidget has not been built yet; call build() first.')
        return self._plot

    def bind_labels(
        self,
        *,
        x_range_label: ui.label,
        measurement_label: ui.label,
    ) -> None:
        """Bind demo labels used to echo range and measurement callbacks."""
        self._x_range_label = x_range_label
        self._measurement_label = measurement_label

    def build(self) -> PlotlyPlotWidget:
        """Build the plot widget and load the current dataset.

        Callers must size ``plot.container`` (for example
        ``plot.container.classes('w-full h-96')``); the widget does not set a
        default height.

        Returns:
            The built :class:`PlotlyPlotWidget`.
        """
        self._plot = PlotlyPlotWidget(
            x_label='Time (s)',
            y_label='Normalized intensity',
            on_x_range_changed=self._on_x_range_changed,
            on_measurement_changed=self._on_measurement_changed,
        )
        # Widget defaults to h-full; that collapses to ~0px when the parent
        # height is content-sized (common on first SPA navigation into an
        # embedded demo). Use an explicit height and drop h-full.
        self._plot.container.classes(remove='h-full')
        self._plot.container.classes(add='w-full h-96')
        self._plot.set_dark_mode(self._dark_mode)
        self._load_signal(self._catalog.get_signal(self._dataset_name))
        # First SPA navigation can leave Plotly at a zero client size; nudge
        # a resize after the element is in the DOM.
        ui.timer(0.05, self._nudge_plotly_resize, once=True)
        return self._plot

    def load_dataset(self, name: str) -> None:
        """Replace series from another named dataset.

        Args:
            name: Dataset name from the catalog.
        """
        self._dataset_name = name
        self._peak_offset = 0.0
        self._load_signal(self._catalog.get_signal(name))
        self._set_x_range_text('auto')
        self._set_measurement_text('none')

    def zoom_window(self) -> None:
        """Programmatically zoom to a fixed time window."""
        self.plot.set_x_axis_limits(8.0, 14.0)

    def reset_x_axis(self) -> None:
        """Reset x-axis limits to the visible line-trace extent."""
        self.plot.reset_x_axis_limits()

    def move_peaks(self) -> None:
        """Nudge scatter peak markers to exercise ``update_scatter``."""
        signal = self._catalog.get_signal(self._dataset_name)
        self._peak_offset += 0.25
        self.plot.update_scatter(
            name='peaks',
            x=[value + self._peak_offset for value in signal.peak_x],
            y=list(signal.peak_y),
        )

    def set_dark_mode(self, enabled: bool) -> None:
        """Toggle the Plotly layout theme.

        Args:
            enabled: Whether dark mode is enabled.
        """
        self._dark_mode = bool(enabled)
        if self._plot is not None:
            self._plot.set_dark_mode(self._dark_mode)

    def _nudge_plotly_resize(self) -> None:
        """Ask Plotly to remeasure after the container has a real client size."""
        # Belt-and-suspenders with the widget's figure sync: SPA first paint can
        # leave Plotly sized correctly but still needing a resize pass.
        ui.run_javascript(
            """
(() => {
  const el = document.querySelector('.nw-plotly-plot .js-plotly-plot');
  if (el && window.Plotly && window.Plotly.Plots) {
    window.Plotly.Plots.resize(el);
  }
})();
"""
        )

    def _load_signal(self, signal: DemoSignal) -> None:
        plot = self.plot
        plot.clear_traces()
        plot.clear_scatters()
        plot.add_trace(name='normalized intensity', x=list(signal.x), y=list(signal.y))
        plot.plot_scatter(name='peaks', x=list(signal.peak_x), y=list(signal.peak_y))
        # Replace the demo threshold line so reload stays deterministic.
        try:
            plot.remove_measurement_line('manual threshold')
        except KeyError:
            pass
        plot.add_measurement_line(
            name='manual threshold',
            orientation='horizontal',
            value=signal.threshold,
        )
        plot.reset_x_axis_limits()

    def _on_x_range_changed(self, x_min: float | None, x_max: float | None) -> None:
        if x_min is None or x_max is None:
            self._set_x_range_text('auto')
            return
        self._set_x_range_text(f'{x_min:.3f} to {x_max:.3f} s')

    def _on_measurement_changed(self, event: MeasurementChangeEvent) -> None:
        if event.kind == 'line':
            self._set_measurement_text(f'{event.name}: {event.position:.4f}')
            return
        self._set_measurement_text(
            f'{event.name}: {event.position1:.3f} to {event.position2:.3f}; '
            f'delta={event.delta:.3f}'
        )

    def _set_x_range_text(self, text: str) -> None:
        if self._x_range_label is not None:
            self._x_range_label.text = f'x-range: {text}'

    def _set_measurement_text(self, text: str) -> None:
        if self._measurement_label is not None:
            self._measurement_label.text = f'measurement: {text}'
