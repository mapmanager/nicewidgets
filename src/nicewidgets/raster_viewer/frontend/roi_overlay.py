"""Plotly rectangular ROI overlay helpers for raster viewer widgets."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


ROI_SHAPE_NAME_PREFIX = 'roi:'


@dataclass(frozen=True, slots=True)
class RectRoiOverlay:
    """Rectangular ROI overlay in Plotly physical coordinates.

    Physical coordinates are pixel indices scaled by the dataset grid spacing
    (``row * dx`` for x, ``column * dy`` for y), matching
    :class:`~nicewidgets.raster_viewer.frontend.plotly_coord_transform.PlotlyCoordTransform`.
    They are not raw pixel indices unless ``dx == dy == 1``.

    Args:
        roi_id: Stable backend ROI identifier.
        x0: First x coordinate in Plotly coordinate space.
        x1: Second x coordinate in Plotly coordinate space.
        y0: First y coordinate in Plotly coordinate space.
        y1: Second y coordinate in Plotly coordinate space.
        label: Optional display label.
        editable: Whether Plotly should allow direct shape editing.
    """

    roi_id: int
    x0: float
    x1: float
    y0: float
    y1: float
    label: str = ''
    editable: bool = False


@dataclass(frozen=True, slots=True)
class RectRoiStyleConfig:
    """Plotly styling for normal and selected rectangular ROI overlays.

    Args:
        line_width: Normal rectangle line width in pixels.
        line_color: Normal rectangle line color.
        fill_color: Normal rectangle fill color.
        selected_line_width: Selected rectangle line width in pixels.
        selected_line_color: Selected rectangle line color.
        selected_fill_color: Selected rectangle fill color.
        editing_line_width: Editing rectangle line width in pixels.
        editing_line_color: Editing rectangle line color.
        editing_fill_color: Editing rectangle fill color. Keep this slightly
            non-transparent so the editable shape body is a reliable pointer
            target for Plotly shape dragging.
    """

    line_width: int = 2
    line_color: str = 'rgba(255, 193, 7, 0.95)'
    # Fully transparent fills render outline-only ROI rectangles.
    fill_color: str = 'rgba(0, 0, 0, 0)'
    selected_line_width: int = 3
    selected_line_color: str = 'rgba(244, 67, 54, 1.0)'
    selected_fill_color: str = 'rgba(0, 0, 0, 0)'
    editing_line_width: int = 3
    editing_line_color: str = 'rgba(0, 188, 212, 1.0)'
    editing_fill_color: str = 'rgba(0, 188, 212, 0.05)'


class PlotlyRoiOverlayLayer:
    """Manage rectangular ROI overlays in a Plotly ``layout.shapes`` list.

    This class owns ROI overlay state and converts that state to Plotly shape
    dictionaries. It does not call NiceGUI or Plotly directly; callers mutate
    their figure dict and push returned shapes with ``Plotly.relayout``.

    Args:
        style: ROI style configuration.
    """

    def __init__(self, style: RectRoiStyleConfig | None = None) -> None:
        self._style = style or RectRoiStyleConfig()
        self._rois: dict[int, RectRoiOverlay] = {}
        self._selected_roi_id: int | None = None
        self._editing_roi_id: int | None = None

    @property
    def selected_roi_id(self) -> int | None:
        """Return the selected ROI id.

        Returns:
            Selected ROI id, or None when no ROI is selected.
        """
        return self._selected_roi_id

    @property
    def editing_roi_id(self) -> int | None:
        """Return the editing ROI id.

        Returns:
            Editing ROI id, or None when editing is inactive.
        """
        return self._editing_roi_id

    @property
    def rois(self) -> tuple[RectRoiOverlay, ...]:
        """Return managed ROIs in insertion order.

        Returns:
            Tuple of ROI overlays.
        """
        return tuple(self._rois.values())

    @staticmethod
    def shape_name(roi_id: int) -> str:
        """Return the Plotly shape name for a ROI id.

        Args:
            roi_id: ROI identifier.

        Returns:
            Stable Plotly shape name.
        """
        return f'{ROI_SHAPE_NAME_PREFIX}{int(roi_id)}'

    @staticmethod
    def roi_id_from_shape(shape: dict[str, object]) -> int | None:
        """Return ROI id encoded in a Plotly shape dictionary.

        Args:
            shape: Plotly layout shape dictionary.

        Returns:
            ROI id when ``shape`` is a managed ROI shape, otherwise None.
        """
        name = shape.get('name')
        if not isinstance(name, str) or not name.startswith(ROI_SHAPE_NAME_PREFIX):
            return None
        raw = name[len(ROI_SHAPE_NAME_PREFIX):]
        try:
            return int(raw)
        except ValueError:
            return None

    @classmethod
    def is_roi_shape(cls, shape: object) -> bool:
        """Return whether an object is a managed ROI shape.

        Args:
            shape: Object from a Plotly ``layout.shapes`` list.

        Returns:
            True if ``shape`` is a dictionary named ``roi:<id>``.
        """
        return isinstance(shape, dict) and cls.roi_id_from_shape(shape) is not None

    def set_rois(self, rois: Sequence[RectRoiOverlay]) -> None:
        """Replace all managed ROI overlays.

        Args:
            rois: New ROI overlay list.

        Returns:
            None.
        """
        self._rois = {int(roi.roi_id): roi for roi in rois}
        if self._selected_roi_id not in self._rois:
            self._selected_roi_id = None
        if self._editing_roi_id not in self._rois:
            self._editing_roi_id = None

    def select_roi(self, roi_id: int | None) -> None:
        """Set the selected ROI id.

        Args:
            roi_id: ROI id to select, or None to clear selection.

        Returns:
            None.
        """
        if roi_id is None:
            self._selected_roi_id = None
            return
        roi_id = int(roi_id)
        self._selected_roi_id = roi_id if roi_id in self._rois else None

    def set_roi_editing(self, roi_id: int | None) -> None:
        """Set the ROI that should be editable.

        Args:
            roi_id: ROI id to edit, or None to disable editing.

        Returns:
            None.
        """
        if roi_id is None:
            self._editing_roi_id = None
            return
        roi_id = int(roi_id)
        self._editing_roi_id = roi_id if roi_id in self._rois else None

    def add_roi(self, roi: RectRoiOverlay) -> None:
        """Add or replace one ROI overlay.

        Args:
            roi: ROI overlay to insert.

        Returns:
            None.
        """
        self._rois[int(roi.roi_id)] = roi

    def delete_roi(self, roi_id: int) -> None:
        """Delete one ROI overlay.

        Args:
            roi_id: ROI identifier to remove.

        Returns:
            None.
        """
        roi_id = int(roi_id)
        self._rois.pop(roi_id, None)
        if self._selected_roi_id == roi_id:
            self._selected_roi_id = None
        if self._editing_roi_id == roi_id:
            self._editing_roi_id = None

    def update_roi_bounds(
        self,
        roi_id: int,
        *,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
    ) -> RectRoiOverlay | None:
        """Replace one ROI overlay's bounds.

        Args:
            roi_id: ROI identifier.
            x0: First x coordinate in Plotly coordinate space.
            x1: Second x coordinate in Plotly coordinate space.
            y0: First y coordinate in Plotly coordinate space.
            y1: Second y coordinate in Plotly coordinate space.

        Returns:
            Updated ROI overlay, or None when ``roi_id`` is unknown.
        """
        roi_id = int(roi_id)
        roi = self._rois.get(roi_id)
        if roi is None:
            return None
        updated = RectRoiOverlay(
            roi_id=roi.roi_id,
            x0=float(x0),
            x1=float(x1),
            y0=float(y0),
            y1=float(y1),
            label=roi.label,
            editable=roi.editable,
        )
        self._rois[roi_id] = updated
        return updated

    def merge_shapes(self, existing_shapes: Iterable[object]) -> list[dict[str, object]]:
        """Return non-ROI shapes plus current ROI shapes.

        Args:
            existing_shapes: Current Plotly ``layout.shapes`` entries.

        Returns:
            New shape list containing all non-ROI shapes and generated ROI
            shapes.
        """
        merged: list[dict[str, object]] = [
            dict(shape)
            for shape in existing_shapes
            if isinstance(shape, dict) and not self.is_roi_shape(shape)
        ]
        merged.extend(self.to_shapes())
        return merged

    def to_shapes(self) -> list[dict[str, object]]:
        """Return current ROI overlays as Plotly shape dictionaries.

        Returns:
            List of Plotly rectangle shape dictionaries.
        """
        ordered = list(self._rois.values())
        ordered.sort(
            key=lambda roi: (
                roi.roi_id == self._selected_roi_id,
                roi.roi_id == self._editing_roi_id,
            )
        )
        return [
            self._roi_to_shape(
                roi,
                selected=(roi.roi_id == self._selected_roi_id),
                editable=(roi.roi_id == self._editing_roi_id),
            )
            for roi in ordered
        ]

    def _roi_to_shape(
        self,
        roi: RectRoiOverlay,
        *,
        selected: bool,
        editable: bool,
    ) -> dict[str, object]:
        """Convert one ROI overlay to a Plotly rectangle shape.

        Args:
            roi: ROI overlay.
            selected: Whether to apply selected styling.
            editable: Whether Plotly should allow direct shape editing.

        Returns:
            Plotly rectangle shape dictionary.
        """
        if editable:
            line_width = self._style.editing_line_width
            line_color = self._style.editing_line_color
            fill_color = self._style.editing_fill_color
        elif selected:
            line_width = self._style.selected_line_width
            line_color = self._style.selected_line_color
            fill_color = self._style.selected_fill_color
        else:
            line_width = self._style.line_width
            line_color = self._style.line_color
            fill_color = self._style.fill_color
        shape: dict[str, object] = {
            'type': 'rect',
            'name': self.shape_name(roi.roi_id),
            'xref': 'x',
            'yref': 'y',
            'x0': float(min(roi.x0, roi.x1)),
            'x1': float(max(roi.x0, roi.x1)),
            'y0': float(min(roi.y0, roi.y1)),
            'y1': float(max(roi.y0, roi.y1)),
            'line': {
                'width': int(line_width),
                'color': line_color,
            },
            'fillcolor': fill_color,
            'layer': 'above',
            'editable': bool(editable),
            'selectable': False,
        }
        if roi.label:
            shape['label'] = {'text': str(roi.label), 'textposition': 'top left'}
        return shape
