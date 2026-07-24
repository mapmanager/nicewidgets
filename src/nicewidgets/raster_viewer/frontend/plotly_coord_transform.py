"""Map numpy row/column space to Plotly plot coordinates (physical units)."""

from __future__ import annotations

from dataclasses import dataclass

from nicewidgets.raster_viewer.backend.image_model import RowColBounds, RasterGridSpec


@dataclass(frozen=True)
class PlotlyCoordTransform:
    """Plot-x aligns with row index (dim 0); plot-y with column index (dim 1).

    Physical plot-x = row * dx, plot-y = col * dy. Plotly y-axis uses a
    bottom-left origin, so increasing column index runs upward on screen
    (``range`` is ``[y_lo, y_hi]``).
    """

    nrows: int
    ncols: int
    grid: RasterGridSpec

    def row_col_to_plot_x_range(self, bounds: RowColBounds) -> tuple[float, float]:
        """Return ``(x_min, x_max)`` for ``layout.xaxis.range`` (ordered low, high)."""
        b = bounds.clipped_to_shape((self.nrows, self.ncols))
        r0, r1 = sorted((b.row_min, b.row_max))
        return (r0 * self.grid.dx, r1 * self.grid.dx)

    def row_col_to_plot_y_range(self, bounds: RowColBounds) -> tuple[float, float]:
        """Return ``(y_min, y_max)`` for ``layout.yaxis.range`` (ordered low, high)."""
        b = bounds.clipped_to_shape((self.nrows, self.ncols))
        c0, c1 = sorted((b.col_min, b.col_max))
        y_lo = c0 * self.grid.dy
        y_hi = c1 * self.grid.dy
        return (y_lo, y_hi)

    def plot_xy_ranges_to_row_col(
        self,
        x_lo: float,
        x_hi: float,
        y_a: float,
        y_b: float,
    ) -> RowColBounds:
        """Convert plot axis extents to :class:`RowColBounds` (clipped)."""
        xa, xb = sorted((x_lo, x_hi))
        row_min = xa / self.grid.dx
        row_max = xb / self.grid.dx
        ca = min(y_a, y_b) / self.grid.dy
        cb = max(y_a, y_b) / self.grid.dy
        col_min = min(ca, cb)
        col_max = max(ca, cb)
        return RowColBounds(
            row_min=row_min,
            row_max=row_max,
            col_min=col_min,
            col_max=col_max,
        ).clipped_to_shape((self.nrows, self.ncols))

    def full_row_col_bounds(self) -> RowColBounds:
        """Return bounds covering the full source array."""
        return RowColBounds(
            row_min=0.0,
            row_max=float(self.nrows),
            col_min=0.0,
            col_max=float(self.ncols),
        )


def merge_partial_relayout(
    relayout: dict[str, object],
    transform: PlotlyCoordTransform,
    fallback: RowColBounds,
) -> dict[str, object]:
    """Fill missing axis range keys from ``fallback`` (x-only / y-only / wheel)."""
    out = dict(relayout)
    has_x = any(k.startswith('xaxis.range') for k in relayout)
    has_y = any(k.startswith('yaxis.range') for k in relayout)
    if not has_x:
        x_lo, x_hi = transform.row_col_to_plot_x_range(fallback)
        out['xaxis.range'] = [x_lo, x_hi]
    if not has_y:
        y_lo, y_hi = transform.row_col_to_plot_y_range(fallback)
        out['yaxis.range'] = [y_lo, y_hi]
    return out
