"""Core backend data models for raster viewing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

RenderMode = Literal['image_png', 'heatmap_z', 'image_rgb']


@dataclass(frozen=True)
class ViewportSize:
    """Browser plot area size in screen pixels.

    Attributes:
        width_px: Width of the visible plotting area in pixels.
        height_px: Height of the visible plotting area in pixels.
    """

    width_px: int
    height_px: int


@dataclass(frozen=True)
class RasterDisplayStyle:
    """Colorscale and intensity window shared by PNG overview and heatmap.

    ``zmin`` / ``zmax`` of ``None`` mean use the data min / max of each clip.
    ``colorscale`` accepts either a Plotly built-in name or a list of
    ``[stop, color]`` pairs; both forms are valid for the heatmap trace and
    for the PNG encoder via ``plotly.colors.sample_colorscale``.
    """

    colorscale: str | list[list[float | str]] = 'Greys'
    zmin: float | None = None
    zmax: float | None = None


@dataclass(frozen=True)
class RasterGridSpec:
    """Physical spacing and axis labels for a 2D raster.

    Plotly uses row index as the horizontal (x) axis and column index as the
    vertical (y) axis at the viewer layer. The frontend displays y with a
    bottom-left origin, so increasing column index moves upward on screen.
    ``dx`` / ``dy`` scale index to physical units.

    Attributes:
        dx: Physical extent per row index step; must be strictly positive.
        dy: Physical extent per column index step; must be strictly positive.
        x_unit: Label for the row axis (plot x); may be empty.
        y_unit: Label for the column axis (plot y); may be empty.
    """

    dx: float
    dy: float
    x_unit: str
    y_unit: str

    def __post_init__(self) -> None:
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError(f'dx and dy must be positive, got dx={self.dx}, dy={self.dy}')


@dataclass(frozen=True)
class RowColBounds:
    """Visible bounds in numpy row/column index space (full-resolution).

    Rows are dimension ``0``, columns dimension ``1``. Values may be fractional
    (sub-index). The viewer maps these to Plotly plot coordinates.
    """

    row_min: float
    row_max: float
    col_min: float
    col_max: float

    def row_span(self) -> float:
        """Return the absolute row extent."""
        return abs(self.row_max - self.row_min)

    def col_span(self) -> float:
        """Return the absolute column extent."""
        return abs(self.col_max - self.col_min)

    def clipped_to_shape(self, shape: tuple[int, int]) -> RowColBounds:
        """Return bounds clipped to a 2D array shape ``(nrows, ncols)``."""
        nrows, ncols = shape
        r0, r1 = sorted((self.row_min, self.row_max))
        c0, c1 = sorted((self.col_min, self.col_max))
        return RowColBounds(
            row_min=max(0.0, min(float(nrows), r0)),
            row_max=max(0.0, min(float(nrows), r1)),
            col_min=max(0.0, min(float(ncols), c0)),
            col_max=max(0.0, min(float(ncols), c1)),
        )


@dataclass(frozen=True)
class ViewRequest:
    """Request for a rendered view of the image.

    Attributes:
        bounds: Visible bounds in full-resolution row/column coordinates.
        viewport: Plot viewport size in screen pixels.
        prefer_mode: Optional render-mode override. When ``None``, the service
            chooses the mode.
    """

    bounds: RowColBounds
    viewport: ViewportSize
    prefer_mode: RenderMode | None = None


@dataclass(frozen=True)
class PyramidLevelInfo:
    """Metadata for one pyramid level.

    Attributes:
        level: Zero-based pyramid level index.
        downsample: Integer downsample factor relative to the full-resolution
            image.
        shape: Level shape as ``(rows, columns)``.
    """

    level: int
    downsample: int
    shape: tuple[int, int]


@dataclass(frozen=True)
class RenderResponse:
    """Backend response for a viewer update.

    Attributes:
        mode: Render payload type.
        level: Pyramid level used to satisfy the request.
        bounds: Bounds that informed the render request (row/column space).
        shape: Full-resolution source shape ``(nrows, ncols)``.
        grid: Physical spacing and axis labels.
        x0: Plot-x origin (physical units, row direction).
        y0: Plot-y origin (physical units, column direction).
        dx: Plot-x step per raster column of the returned image (physical).
        dy: Plot-y step per raster row of the returned image (physical).
        png_data_uri: PNG data URI for ``image_png`` mode.
        z: Numeric 2D array for ``heatmap_z`` mode (shape ``(ncols, nrows)`` for Plotly).
        rgb: Uint8 RGB array for ``image_rgb`` mode, Plotly image orientation
            ``(ncols, nrows, 3)`` (plot-y, plot-x, channel).
        zmin: Lower intensity bound for numeric rendering.
        zmax: Upper intensity bound for numeric rendering.
    """

    mode: RenderMode
    level: int
    bounds: RowColBounds
    shape: tuple[int, int]
    grid: RasterGridSpec
    x0: float
    y0: float
    dx: float
    dy: float
    png_data_uri: str | None = None
    z: np.ndarray | None = None
    rgb: np.ndarray | None = None
    zmin: float | None = None
    zmax: float | None = None

    def __str__(self) -> str:
        uri = (
            None
            if self.png_data_uri is None
            else f"<data URI, {len(self.png_data_uri)} chars>"
        )
        z = (
            None
            if self.z is None
            else f"<ndarray shape={self.z.shape} dtype={self.z.dtype}>"
        )
        rgb = (
            None
            if self.rgb is None
            else f"<ndarray shape={self.rgb.shape} dtype={self.rgb.dtype}>"
        )
        lines = [
            "RenderResponse(",
            f"  mode={self.mode!r},",
            f"  level={self.level},",
            f"  bounds={self.bounds!r},",
            f"  shape={self.shape},",
            f"  grid={self.grid!r},",
            f"  x0={self.x0},",
            f"  y0={self.y0},",
            f"  dx={self.dx},",
            f"  dy={self.dy},",
            f"  png_data_uri={uri},",
            f"  z={z},",
            f"  rgb={rgb},",
            f"  zmin={self.zmin},",
            f"  zmax={self.zmax},",
            ")",
        ]
        return "\n".join(lines)

class BackendImage:
    """Hold the full-resolution source image and its physical grid."""

    def __init__(self, data: np.ndarray, grid: RasterGridSpec) -> None:
        """Initialize the image model.

        Args:
            data: Full-resolution 2D numpy array.
            grid: Physical spacing and axis labels (``dx``/``dy`` must be positive).

        Raises:
            ValueError: If ``data`` is not 2D.
        """
        if data.ndim != 2:
            raise ValueError(f'Expected a 2D array, got shape={data.shape}')
        self._data = data
        self._grid = grid

    @property
    def data(self) -> np.ndarray:
        """Return the full-resolution source array."""
        return self._data

    @property
    def grid(self) -> RasterGridSpec:
        """Return the raster grid specification."""
        return self._grid

    @property
    def shape(self) -> tuple[int, int]:
        """Return image shape as ``(rows, columns)``."""
        return self._data.shape

    @property
    def height(self) -> int:
        """Return the number of rows in the image."""
        return self._data.shape[0]

    @property
    def width(self) -> int:
        """Return the number of columns in the image."""
        return self._data.shape[1]

    def clip(self, bounds: RowColBounds) -> np.ndarray:
        """Return a full-resolution clip in row/column coordinates.

        Args:
            bounds: Bounds in full-resolution row/column space.

        Returns:
            A 2D view or copy of the requested region ``data[row, col]``.
        """
        clipped = bounds.clipped_to_shape(self.shape)
        r0 = max(0, int(np.floor(min(clipped.row_min, clipped.row_max))))
        r1 = min(self.height, int(np.ceil(max(clipped.row_min, clipped.row_max))))
        c0 = max(0, int(np.floor(min(clipped.col_min, clipped.col_max))))
        c1 = min(self.width, int(np.ceil(max(clipped.col_min, clipped.col_max))))
        return self._data[r0:r1, c0:c1]


def physical_steps_for_level(grid: RasterGridSpec, downsample: int) -> tuple[float, float]:
    """Return ``(plot_dx, plot_dy)`` for one pyramid level."""
    ds = float(downsample)
    return ds * grid.dx, ds * grid.dy
