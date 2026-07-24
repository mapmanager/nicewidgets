"""Runtime rendering policy for raster viewer requests."""

from __future__ import annotations

import base64
import io

import numpy as np
import plotly.colors as plotly_colors
from PIL import Image

from nicewidgets.raster_viewer.backend.image_model import (
    BackendImage,
    RasterDisplayStyle,
    RasterGridSpec,
    RenderMode,
    RenderResponse,
    RowColBounds,
    ViewRequest,
)
from nicewidgets.raster_viewer.backend.pyramid import ImagePyramid

from nicewidgets.utils.logging import get_logger

logger = get_logger(__name__)

# Plotly.py resolves these named colorscales in the OPPOSITE direction from
# Plotly.js. The heatmap trace is rendered by Plotly.js, but the PNG LUT is
# built here with Plotly.py's ``sample_colorscale``. For these names we sample
# at reversed positions so the encoded PNG matches the heatmap (e.g. for
# ``Greys``: low intensity -> black, high intensity -> white). Explicit
# ``[stop, color]`` lists are read identically by both renderers and are
# unaffected.
_PLOTLY_PY_REVERSED_VS_JS: frozenset[str] = frozenset({'Greys', 'Greens', 'Blues'})


class RasterViewService:
    """Serve view-specific raster payloads from an image pyramid.

    The service answers one question: given visible row/column bounds and a
    viewport, what is the best payload to send to the frontend right now?
    """

    def __init__(
        self,
        source: BackendImage,
        pyramid: ImagePyramid,
        heatmap_max_values: int = 500_000,
    ) -> None:
        """Initialize the service.

        Args:
            source: Full-resolution backend image.
            pyramid: Precomputed pyramid for ``source``.
            heatmap_max_values: Maximum number of values allowed before the
                service prefers PNG output over raw heatmap values.
        """
        self._source = source
        self._pyramid = pyramid
        self._heatmap_max_values = heatmap_max_values

    @property
    def source(self) -> BackendImage:
        """Return the source backend image."""
        return self._source

    @property
    def pyramid(self) -> ImagePyramid:
        """Return the pyramid."""
        return self._pyramid

    @property
    def grid(self) -> RasterGridSpec:
        """Return the source grid specification."""
        return self._source.grid

    def render(
        self,
        request: ViewRequest,
        *,
        display_style: RasterDisplayStyle | None = None,
    ) -> RenderResponse:
        """Render a response for the requested bounds and viewport.

        Args:
            request: View request describing visible bounds and viewport size.

        Returns:
            A render response containing either a PNG data URI or numeric heatmap
            data. Trace ``z`` (heatmap) is oriented ``(ncols, nrows)`` for Plotly
            with plot-x along rows and plot-y along columns.
        """
        bounds = request.bounds.clipped_to_shape(self._source.shape)
        request = ViewRequest(bounds=bounds, viewport=request.viewport, prefer_mode=request.prefer_mode)

        level = self.choose_level(request)
        clip = self._pyramid.clip_from_level(level=level, bounds=bounds)
        mode = self.choose_mode(request=request, clip=clip)
        ds = float(self._pyramid.get_downsample(level))
        grid = self._source.grid

        row0 = float(np.floor(min(bounds.row_min, bounds.row_max) / ds) * ds)
        col0 = float(np.floor(min(bounds.col_min, bounds.col_max) / ds) * ds)
        plot_dx = ds * grid.dx
        plot_dy = ds * grid.dy
        plot_x0 = row0 * grid.dx
        plot_y0 = col0 * grid.dy

        style = display_style if display_style is not None else RasterDisplayStyle()
        shape = self._source.shape
        if mode == 'image_png':
            png_arr = np.asarray(clip).T

            # abb leave this comment here
            # logger.debug(
            #     'render image_png: level=%s shape=%s bounds=%s',
            #     level,
            #     png_arr.shape,
            #     bounds,
            # )

            return RenderResponse(
                mode='image_png',
                level=level,
                bounds=bounds,
                shape=shape,
                grid=grid,
                x0=plot_x0,
                y0=plot_y0,
                dx=plot_dx,
                dy=plot_dy,
                png_data_uri=self.array_to_png_data_uri(png_arr, style=style),
            )

        z_plot = np.asarray(clip).astype(np.float32).T
        z = z_plot
        zmin_data = float(np.nanmin(z)) if z.size else 0.0
        zmax_data = float(np.nanmax(z)) if z.size else 0.0
        zmin_eff = float(style.zmin) if style.zmin is not None else zmin_data
        zmax_eff = float(style.zmax) if style.zmax is not None else zmax_data

        # abb leave this comment here
        # logger.debug(
        #     'render heatmap_z: level=%s shape=%s bounds=%s',
        #     level,
        #     z.shape,
        #     bounds,
        # )

        return RenderResponse(
            mode='heatmap_z',
            level=level,
            bounds=bounds,
            shape=shape,
            grid=grid,
            x0=plot_x0,
            y0=plot_y0,
            dx=plot_dx,
            dy=plot_dy,
            z=z,
            zmin=zmin_eff,
            zmax=zmax_eff,
        )

    def full_image_png(
        self,
        level: int | None = None,
        *,
        display_style: RasterDisplayStyle | None = None,
        max_pixels: int | None = None,
    ) -> RenderResponse:
        """Return a full-image PNG response.

        Args:
            level: Optional pyramid level. When provided, it takes precedence
                over ``max_pixels``.
            display_style: Optional colorscale / intensity window. Defaults to
                the service default style.
            max_pixels: Optional pixel budget used only when ``level`` is
                ``None``. The finest pyramid level whose array size does not
                exceed this budget is selected, so small images render at full
                resolution instead of a coarse overview. When both ``level``
                and ``max_pixels`` are ``None``, a conservative coarse overview
                level is used.

        Returns:
            PNG render response for the full image extent.
        """
        style = display_style if display_style is not None else RasterDisplayStyle()
        if level is None:
            level = self._overview_level(max_pixels)
        arr = self._pyramid.get_level(level)
        ds = float(self._pyramid.get_downsample(level))
        grid = self._source.grid
        bounds = RowColBounds(
            row_min=0.0,
            row_max=float(self._source.height),
            col_min=0.0,
            col_max=float(self._source.width),
        )
        png_arr = np.asarray(arr).T
        plot_dx = ds * grid.dx
        plot_dy = ds * grid.dy
        shape = self._source.shape
        logger.debug(
            'full_image_png: level=%s shape=%s max_pixels=%s',
            level,
            png_arr.shape,
            max_pixels,
        )
        return RenderResponse(
            mode='image_png',
            level=level,
            bounds=bounds,
            shape=shape,
            grid=grid,
            x0=0.0,
            y0=0.0,
            dx=plot_dx,
            dy=plot_dy,
            png_data_uri=self.array_to_png_data_uri(png_arr, style=style),
        )

    def _overview_level(self, max_pixels: int | None) -> int:
        """Select a full-extent overview level.

        Args:
            max_pixels: Optional pixel budget. When ``None``, a conservative
                coarse level is returned (``min(num_levels - 1, 3)``). Otherwise
                the finest level whose array size does not exceed ``max_pixels``
                is returned; when no level fits, the coarsest level is used.

        Returns:
            Pyramid level index.
        """
        if max_pixels is None:
            return min(self._pyramid.num_levels - 1, 3)
        # ``level_info`` is ordered finest (level 0) to coarsest; the first
        # level that fits the budget is the finest acceptable overview.
        for info in self._pyramid.level_info():
            if info.shape[0] * info.shape[1] <= max_pixels:
                return info.level
        return self._pyramid.num_levels - 1

    def choose_level(self, request: ViewRequest) -> int:
        """Choose a pyramid level based on visible-data density.

        The selected level is the coarsest level whose downsample factor does
        not exceed the current source-pixels-per-screen-pixel requirement.
        When that level cannot tile the visible bounds on both axes (common for
        skinny kymographs when time density drives a very large downsample),
        walk to finer levels until the clip fully covers the request.

        Args:
            request: View request.

        Returns:
            Pyramid level index.
        """
        visible_rows = max(1.0, request.bounds.row_span())
        visible_cols = max(1.0, request.bounds.col_span())
        sx = visible_rows / max(1, request.viewport.width_px)
        sy = visible_cols / max(1, request.viewport.height_px)
        target = max(sx, sy)

        best = 0
        for info in self._pyramid.level_info():
            if info.downsample <= target:
                best = info.level
            else:
                break

        bounds = request.bounds.clipped_to_shape(self._source.shape)
        level = best
        while level > 0 and not self._level_covers_bounds(level, bounds):
            level -= 1
        return level

    def _level_covers_bounds(self, level: int, bounds: RowColBounds) -> bool:
        """Return whether pyramid level ``level`` can tile ``bounds`` without truncation.

        Uses the same index math as :meth:`ImagePyramid.clip_from_level` to
        verify that the level array is large enough along both axes.

        Args:
            level: Pyramid level index.
            bounds: Visible bounds in full-resolution row/column coordinates.

        Returns:
            ``True`` when ``clip_from_level`` would not truncate at the array edge.
        """
        ds = float(self._pyramid.get_downsample(level))
        arr = self._pyramid.get_level(level)
        row_hi = max(bounds.row_min, bounds.row_max)
        col_hi = max(bounds.col_min, bounds.col_max)
        needed_r1 = int(np.ceil(row_hi / ds))
        needed_c1 = int(np.ceil(col_hi / ds))
        return needed_r1 <= arr.shape[0] and needed_c1 <= arr.shape[1]

    def choose_mode(self, request: ViewRequest, clip: np.ndarray) -> RenderMode:
        """Choose the render mode for a request.

        Args:
            request: View request.
            clip: Level clip selected for the request (row, col) order.

        Returns:
            ``'heatmap_z'`` or ``'image_png'``.
        """
        if request.prefer_mode is not None:
            return request.prefer_mode
        if int(clip.size) <= self._heatmap_max_values:
            return 'heatmap_z'
        return 'image_png'

    @staticmethod
    def array_to_png_data_uri(arr: np.ndarray, *, style: RasterDisplayStyle) -> str:
        """Encode a 2D float array as an RGB PNG using Plotly's named ``colorscale``.

        Args:
            arr: Input 2D array (already oriented for Plotly image trace).
            style: Plotly colorscale name and optional ``zmin`` / ``zmax`` window.

        Returns:
            Data URI with PNG payload (RGB).
        """
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0:
            image = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8), mode='RGB')
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            encoded = base64.b64encode(buf.getvalue()).decode('ascii')
            return f'data:image/png;base64,{encoded}'
        else:
            lo = float(style.zmin) if style.zmin is not None else float(np.nanmin(a))
            hi = float(style.zmax) if style.zmax is not None else float(np.nanmax(a))
            finite = np.isfinite(a)
            if not finite.any() or hi <= lo or not np.isfinite(lo) or not np.isfinite(hi):
                rgb = np.zeros((*a.shape, 3), dtype=np.uint8)
            else:
                stops = np.linspace(0.0, 1.0, 256, dtype=np.float64)
                query = stops
                if isinstance(style.colorscale, str) and style.colorscale in _PLOTLY_PY_REVERSED_VS_JS:
                    query = 1.0 - stops
                tuples = plotly_colors.sample_colorscale(
                    style.colorscale,
                    query.tolist(),
                    colortype='tuple',
                )
                lut = np.array(tuples, dtype=np.float64)
                lut = np.clip(lut * 255.0, 0.0, 255.0).astype(np.uint8)

                t = np.zeros_like(a, dtype=np.float64)
                np.divide(a - lo, hi - lo, out=t, where=finite)
                t = np.clip(t, 0.0, 1.0)
                t = np.where(finite, t, 0.0)
                idx = np.minimum((t * 255.0).astype(np.int32), 255)
                rgb = lut[idx]

        image = Image.fromarray(rgb, mode='RGB')
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        encoded = base64.b64encode(buf.getvalue()).decode('ascii')
        return f'data:image/png;base64,{encoded}'

    @staticmethod
    def to_png_data_uri(arr: np.ndarray) -> str:
        """Convert a 2D numeric array into a PNG data URI (legacy grayscale path).

        Prefer :meth:`array_to_png_data_uri` with a :class:`RasterDisplayStyle` so
        overview PNG matches heatmap styling.
        """
        scaled = RasterViewService.normalize_to_uint8(arr)
        image = Image.fromarray(scaled, mode='L')
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        encoded = base64.b64encode(buf.getvalue()).decode('ascii')
        return f'data:image/png;base64,{encoded}'

    @staticmethod
    def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
        """Normalize a 2D array to uint8 grayscale.

        Args:
            arr: Input 2D array.

        Returns:
            Uint8 array of the same shape.
        """
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0:
            return np.zeros(a.shape, dtype=np.uint8)

        finite = np.isfinite(a)
        if not finite.any():
            return np.zeros(a.shape, dtype=np.uint8)

        lo = float(np.nanmin(a))
        hi = float(np.nanmax(a))
        if hi <= lo:
            return np.zeros(a.shape, dtype=np.uint8)

        scaled = (a - lo) / (hi - lo)
        return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)
