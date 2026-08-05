"""Typed, source-coordinate ROI contracts for the reusable raster viewer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class RoiType(StrEnum):
    """Supported ROI geometry discriminators."""

    RECTROI = "rectroi"
    LINESEGMENTROI = "linesegmentroi"


class RoiInteractionState(StrEnum):
    """Transactional browser ROI interaction states."""

    IDLE = "idle"
    CREATING = "creating"
    EDITING = "editing"


@dataclass(frozen=True, slots=True)
class ImageBounds:
    """Valid source-image extent in columns and rows."""

    width: int
    height: int

    def __post_init__(self) -> None:
        """Require positive source dimensions."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image width and height must be positive")


@dataclass(frozen=True, slots=True)
class RectRoiBounds:
    """Integer half-open rectangle edges in source NumPy coordinates."""

    row_start: int
    row_stop: int
    col_start: int
    col_stop: int

    def clamped_to(self, image: ImageBounds) -> RectRoiBounds:
        """Return normalized, non-empty bounds constrained to ``image``."""
        row_start, row_stop = sorted((self.row_start, self.row_stop))
        col_start, col_stop = sorted((self.col_start, self.col_stop))
        row_start = max(0, min(row_start, image.height - 1))
        row_stop = max(row_start + 1, min(row_stop, image.height))
        col_start = max(0, min(col_start, image.width - 1))
        col_stop = max(col_start + 1, min(col_stop, image.width))
        return RectRoiBounds(row_start, row_stop, col_start, col_stop)

    def to_json(self) -> dict[str, int]:
        """Return the canonical snake_case rectangle data block."""
        return {
            "row_start": self.row_start,
            "row_stop": self.row_stop,
            "col_start": self.col_start,
            "col_stop": self.col_stop,
        }

    @classmethod
    def from_mapping(cls, value: object) -> RectRoiBounds:
        """Parse the strict public rectangle data block."""
        if not isinstance(value, Mapping):
            raise ValueError("rectangle ROI data must be a mapping")
        return cls(*(round(float(value[key])) for key in (
            "row_start", "row_stop", "col_start", "col_stop"
        )))


@dataclass(frozen=True, slots=True)
class LineEndpoints:
    """Integer line endpoints identifying source-array pixel centers."""

    row0: int
    col0: int
    row1: int
    col1: int

    def clamped_to(self, image: ImageBounds) -> LineEndpoints:
        """Clamp both endpoints independently to valid pixel indices."""
        return LineEndpoints(
            row0=max(0, min(self.row0, image.height - 1)),
            col0=max(0, min(self.col0, image.width - 1)),
            row1=max(0, min(self.row1, image.height - 1)),
            col1=max(0, min(self.col1, image.width - 1)),
        )

    def to_json(self) -> dict[str, int]:
        """Return the canonical snake_case line data block."""
        return {"row0": self.row0, "col0": self.col0, "row1": self.row1, "col1": self.col1}

    @classmethod
    def from_mapping(cls, value: object) -> LineEndpoints:
        """Parse the strict public line data block."""
        if not isinstance(value, Mapping):
            raise ValueError("line ROI data must be a mapping")
        return cls(*(round(float(value[key])) for key in ("row0", "col0", "row1", "col1")))


@dataclass(frozen=True, slots=True)
class RectRoi:
    """Committed rectangular ROI."""

    roi_id: int
    name: str
    bounds: RectRoiBounds
    note: str = ""

    def __post_init__(self) -> None:
        """Validate stable identity and display name."""
        _validate_identity(self.roi_id, self.name)

    @property
    def roi_type(self) -> RoiType:
        """Return the stable rectangle discriminator."""
        return RoiType.RECTROI

    def to_json(self) -> dict[str, object]:
        """Return the versioned committed ROI envelope."""
        return _envelope(self, self.bounds.to_json())


@dataclass(frozen=True, slots=True)
class LineRoi:
    """Committed two-endpoint line-segment ROI."""

    roi_id: int
    name: str
    endpoints: LineEndpoints
    note: str = ""

    def __post_init__(self) -> None:
        """Validate stable identity and display name."""
        _validate_identity(self.roi_id, self.name)

    @property
    def roi_type(self) -> RoiType:
        """Return the stable line-segment discriminator."""
        return RoiType.LINESEGMENTROI

    def to_json(self) -> dict[str, object]:
        """Return the versioned committed ROI envelope."""
        return _envelope(self, self.endpoints.to_json())


type Roi = RectRoi | LineRoi


def _envelope(roi: Roi, data: dict[str, int]) -> dict[str, object]:
    """Serialize fields shared by every committed ROI shape."""
    return {
        "roi_id": roi.roi_id,
        "roi_type": roi.roi_type.value,
        "version": "1.0",
        "name": roi.name,
        "note": roi.note,
        "data": data,
    }


def _validate_identity(roi_id: int, name: str) -> None:
    """Validate fields shared by every committed ROI value."""
    if roi_id <= 0:
        raise ValueError("roi_id must be positive")
    if not name:
        raise ValueError("ROI name must not be empty")


def roi_from_mapping(value: object) -> Roi:
    """Parse one strict version-1 committed ROI envelope."""
    if not isinstance(value, Mapping):
        raise ValueError("ROI envelope must be a mapping")
    if str(value.get("version")) != "1.0":
        raise ValueError("ROI version must be '1.0'")
    roi_id = int(value["roi_id"])
    name = str(value["name"])
    note = str(value.get("note", ""))
    roi_type = RoiType(str(value["roi_type"]))
    data = value.get("data")
    if roi_type is RoiType.RECTROI:
        return RectRoi(roi_id, name, RectRoiBounds.from_mapping(data), note)
    return LineRoi(roi_id, name, LineEndpoints.from_mapping(data), note)


@dataclass(frozen=True, slots=True)
class RectRoiCreate:
    """Uncommitted rectangle creation specification."""

    name: str
    bounds: RectRoiBounds
    note: str = ""

    def to_json(self) -> dict[str, object]:
        """Return the browser creation specification."""
        return {"roi_type": RoiType.RECTROI.value, "name": self.name,
                "note": self.note, "data": self.bounds.to_json()}


@dataclass(frozen=True, slots=True)
class LineRoiCreate:
    """Uncommitted line creation specification."""

    name: str
    endpoints: LineEndpoints
    note: str = ""

    def to_json(self) -> dict[str, object]:
        """Return the browser creation specification."""
        return {"roi_type": RoiType.LINESEGMENTROI.value, "name": self.name,
                "note": self.note, "data": self.endpoints.to_json()}


type RoiCreate = RectRoiCreate | LineRoiCreate
