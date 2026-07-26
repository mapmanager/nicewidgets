"""Synthetic sample data for the PlotlyPlotWidget demo.

Pure data module: no NiceGUI imports. Returns plain Python float sequences
suitable for :meth:`PlotlyPlotWidget.add_trace` and
:meth:`PlotlyPlotWidget.plot_scatter`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DemoSignal:
    """One synthetic fluorescence-like time series plus sparse peak markers.

    Args:
        name: Display name for the continuous line trace.
        x: Time values in seconds.
        y: Intensity values.
        peak_x: X coordinates of sparse peak markers.
        peak_y: Y coordinates of sparse peak markers.
        threshold: Suggested horizontal measurement-line value.
    """

    name: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    peak_x: tuple[float, ...]
    peak_y: tuple[float, ...]
    threshold: float


class SampleDataCatalog:
    """Provide deterministic synthetic plot datasets."""

    def __init__(self, *, n_points: int = 8_000) -> None:
        self._datasets: dict[str, DemoSignal] = {
            'Normalized intensity': self._make_intensity_signal(n_points),
        }

    @property
    def names(self) -> list[str]:
        """Return selectable dataset names."""
        return list(self._datasets)

    def get_signal(self, name: str) -> DemoSignal:
        """Return one named synthetic signal.

        Args:
            name: Dataset name from :attr:`names`.

        Returns:
            Frozen signal payload for the plot demo.
        """
        return self._datasets[name]

    @staticmethod
    def _make_intensity_signal(n_points: int) -> DemoSignal:
        x = tuple(i * 0.005 for i in range(n_points))
        centers = (4.0, 11.0, 17.5, 29.0, 35.0)
        y_list: list[float] = []
        for t in x:
            baseline = 1.0 + 0.04 * math.sin(t * 0.7)
            carrier = 0.08 * math.sin(t * 8.0)
            peak = 0.0
            for center in centers:
                peak += 0.45 * math.exp(-((t - center) ** 2) / 0.015)
            y_list.append(baseline + carrier + peak)
        y = tuple(y_list)
        peak_x, peak_y = _peak_points(x, y, centers)
        return DemoSignal(
            name='Normalized intensity',
            x=x,
            y=y,
            peak_x=peak_x,
            peak_y=peak_y,
            threshold=1.25,
        )


def _peak_points(
    x: Sequence[float],
    y: Sequence[float],
    centers: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return sparse peak marker coordinates nearest each center time."""
    xs: list[float] = []
    ys: list[float] = []
    for center in centers:
        index = min(range(len(x)), key=lambda i: abs(x[i] - center))
        xs.append(float(x[index]))
        ys.append(float(y[index]))
    return tuple(xs), tuple(ys)
