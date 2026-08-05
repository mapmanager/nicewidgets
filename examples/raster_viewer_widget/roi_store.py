"""Application-owned mixed-shape ROI storage for the NiceGUI demo."""

from __future__ import annotations

from collections.abc import Iterator

from nicewidgets.raster_viewer_widget.roi import (
    ImageBounds,
    LineEndpoints,
    LineRoi,
    RectRoi,
    RectRoiBounds,
    Roi,
)


class DemoRoiStore:
    """Own and validate one dataset's committed rectangle and line ROIs."""

    def __init__(self, image_bounds: ImageBounds) -> None:
        """Initialize an empty store bounded to one 2D source image."""
        self.image_bounds = image_bounds
        self._rois: dict[int, Roi] = {}
        self._next_id = 1
        self._next_name = 0

    def __iter__(self) -> Iterator[Roi]:
        """Iterate over committed ROIs in creation order."""
        return iter(self._rois.values())

    def get(self, roi_id: int) -> Roi:
        """Return one committed ROI by stable ID."""
        return self._rois[roi_id]

    def create_rect(self, bounds: RectRoiBounds, *, note: str = "") -> RectRoi:
        """Create one image-bounded rectangle with the next identity."""
        roi = RectRoi(
            self._next_id, str(self._next_name), bounds.clamped_to(self.image_bounds), note
        )
        self._install_new(roi)
        return roi

    def create_line(self, endpoints: LineEndpoints, *, note: str = "") -> LineRoi:
        """Create one image-bounded line segment with the next identity."""
        roi = LineRoi(
            self._next_id, str(self._next_name), endpoints.clamped_to(self.image_bounds), note
        )
        self._install_new(roi)
        return roi

    def _install_new(self, roi: Roi) -> None:
        """Install a newly allocated ROI and advance monotonic counters."""
        self._rois[roi.roi_id] = roi
        self._next_id += 1
        self._next_name += 1

    def update(self, roi: Roi) -> Roi:
        """Replace geometry while preserving authoritative identity metadata."""
        current = self.get(roi.roi_id)
        if isinstance(roi, RectRoi):
            updated: Roi = RectRoi(
                current.roi_id,
                current.name,
                roi.bounds.clamped_to(self.image_bounds),
                current.note,
            )
        else:
            updated = LineRoi(
                current.roi_id,
                current.name,
                roi.endpoints.clamped_to(self.image_bounds),
                current.note,
            )
        self._rois[roi.roi_id] = updated
        return updated

    def delete(self, roi_id: int) -> None:
        """Delete one committed ROI by stable ID."""
        del self._rois[roi_id]

    def suggested_rect_bounds(self) -> RectRoiBounds:
        """Return a centered rectangle covering one quarter of each dimension."""
        width = max(1, self.image_bounds.width // 4)
        height = max(1, self.image_bounds.height // 4)
        col_start = (self.image_bounds.width - width) // 2
        row_start = (self.image_bounds.height - height) // 2
        return RectRoiBounds(
            row_start, row_start + height, col_start, col_start + width
        ).clamped_to(self.image_bounds)

    def suggested_line_endpoints(self) -> LineEndpoints:
        """Return a centered horizontal line spanning one quarter of image width."""
        row = (self.image_bounds.height - 1) // 2
        span = max(1, self.image_bounds.width // 4)
        col0 = max(0, (self.image_bounds.width - span) // 2)
        col1 = min(self.image_bounds.width - 1, col0 + span)
        return LineEndpoints(row, col0, row, col1)

    @property
    def suggested_name(self) -> str:
        """Return the numeric name reserved for the next committed ROI."""
        return str(self._next_name)

    def to_json(self) -> list[dict[str, object]]:
        """Serialize every committed ROI in creation order."""
        return [roi.to_json() for roi in self._rois.values()]

    def selected_neighbor(self, deleted_index: int) -> int | None:
        """Return the next sensible selection after deletion."""
        ids = list(self._rois)
        return ids[min(deleted_index, len(ids) - 1)] if ids else None

    def index_of(self, roi_id: int) -> int:
        """Return one ROI's creation-order index."""
        return list(self._rois).index(roi_id)
