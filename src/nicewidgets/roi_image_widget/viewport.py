# nicewidgets/src/nicewidgets/roi_image_widget/viewport.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Tuple


@dataclass
class Viewport:
    """Viewport describing the currently visible region of an image.

    Coordinates are in full-image pixel space.
    """

    img_width: int
    img_height: int
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0

    def __post_init__(self) -> None:
        # If x_max and y_max are zero, show full image.
        if self.x_max == 0.0 and self.y_max == 0.0:
            self.reset()
        else:
            self._clamp_to_image()

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Viewport":
        return cls(**data)

    # ------------------ core operations ------------------

    def reset(self) -> None:
        """Show full image."""
        self.x_min = 0.0
        self.x_max = float(self.img_width)
        self.y_min = 0.0
        self.y_max = float(self.img_height)

    def zoom_around(
        self,
        x_center: float,
        y_center: float,
        factor_x: float,
        factor_y: float,
        min_width: float = 10.0,
        min_height: float = 5.0,
    ) -> None:
        """Zoom around (x_center, y_center).

        factor < 1 -> zoom in
        factor > 1 -> zoom out
        """
        # X axis zoom
        left_offset = self.x_min - x_center
        right_offset = self.x_max - x_center
        new_x_min = x_center + left_offset * factor_x
        new_x_max = x_center + right_offset * factor_x

        # Y axis zoom
        top_offset = self.y_min - y_center
        bottom_offset = self.y_max - y_center
        new_y_min = y_center + top_offset * factor_y
        new_y_max = y_center + bottom_offset * factor_y

        self.x_min, self.x_max = new_x_min, new_x_max
        self.y_min, self.y_max = new_y_min, new_y_max
        self._clamp_to_image()

        self._ensure_min_size(min_width, min_height)
        self._clamp_to_image()

    # def pan(self, dx: float, dy: float) -> None:
    #     """Pan viewport by (dx, dy) in full-image coords."""
    #     self.x_min -= dx
    #     self.x_max -= dx
    #     self.y_min -= dy
    #     self.y_max -= dy
    #     self._clamp_to_image()

    def pan(self, dx: float, dy: float) -> None:
        """Pan the viewport by the given deltas in full-image coordinates.

        Args:
            dx: Delta in X (positive moves the visible window right).
            dy: Delta in Y (positive moves the visible window down).

        Notes:
            Panning is clamped so the viewport never leaves the image bounds,
            **while preserving the current width and height**.
        """
        # Current size
        # w = self.width
        # h = self.height

        # Apply raw shift
        self.x_min -= dx
        self.x_max -= dx
        self.y_min -= dy
        self.y_max -= dy

        # Keep width/height exactly the same, but clamp inside image.
        # If we go past the left edge, shift right.
        if self.x_min < 0.0:
            self.x_max -= self.x_min  # subtract a negative -> add
            self.x_min = 0.0
        # If we go past the right edge, shift left.
        if self.x_max > float(self.img_width):
            overshoot = self.x_max - float(self.img_width)
            self.x_min -= overshoot
            self.x_max = float(self.img_width)

        # Same logic for Y
        if self.y_min < 0.0:
            self.y_max -= self.y_min
            self.y_min = 0.0
        if self.y_max > float(self.img_height):
            overshoot = self.y_max - float(self.img_height)
            self.y_min -= overshoot
            self.y_max = float(self.img_height)

        # As a safety, if the viewport somehow got larger than the image,
        # fall back to full view.
        if self.width > float(self.img_width) or self.height > float(self.img_height):
            self.reset()

    def get_int_slice(self) -> Tuple[int, int, int, int]:
        """Return (y_min, y_max, x_min, x_max) as ints, clamped."""
        y_min = max(0, min(self.img_height, int(round(self.y_min))))
        y_max = max(0, min(self.img_height, int(round(self.y_max))))
        x_min = max(0, min(self.img_width, int(round(self.x_min))))
        x_max = max(0, min(self.img_width, int(round(self.x_max))))
        return y_min, y_max, x_min, x_max

    # ------------------ internal helpers ------------------

    # def _clamp_to_image(self) -> None:
    #     self.x_min = max(0.0, min(float(self.img_width), self.x_min))
    #     self.x_max = max(0.0, min(float(self.img_width), self.x_max))
    #     self.y_min = max(0.0, min(float(self.img_height), self.y_min))
    #     self.y_max = max(0.0, min(float(self.img_height), self.y_max))

    #     if self.x_max <= self.x_min:
    #         self.x_min = 0.0
    #         self.x_max = float(self.img_width)
    #     if self.y_max <= self.y_min:
    #         self.y_min = 0.0
    #         self.y_max = float(self.img_height)

    def _clamp_to_image(self) -> None:
        """Clamp viewport edges to remain within the image bounds.

        This is primarily used after zoom operations. It keeps the viewport
        inside [0, img_width] x [0, img_height]. If the viewport becomes
        degenerate, it falls back to full-image view.
        """
        self.x_min = max(0.0, min(float(self.img_width), self.x_min))
        self.x_max = max(0.0, min(float(self.img_width), self.x_max))
        self.y_min = max(0.0, min(float(self.img_height), self.y_min))
        self.y_max = max(0.0, min(float(self.img_height), self.y_max))

        # If somehow inverted or collapsed, reset to full image.
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            self.reset()

    def _ensure_min_size(self, min_width: float, min_height: float) -> None:
        # X
        if self.width < min_width:
            cx = 0.5 * (self.x_min + self.x_max)
            half_w = min_width / 2.0
            self.x_min = cx - half_w
            self.x_max = cx + half_w
        # Y
        if self.height < min_height:
            cy = 0.5 * (self.y_min + self.y_max)
            half_h = min_height / 2.0
            self.y_min = cy - half_h
            self.y_max = cy + half_h


def full_to_view(
    x: float,
    y: float,
    viewport: Viewport,
    disp_w: int,
    disp_h: int,
) -> tuple[float, float]:
    """Full-image coords -> display coords."""
    Wv = viewport.width
    Hv = viewport.height
    if Wv <= 0 or Hv <= 0:
        return 0.0, 0.0

    vx = (x - viewport.x_min) / Wv * disp_w
    vy = (y - viewport.y_min) / Hv * disp_h
    return vx, vy


def view_to_full(
    vx: float,
    vy: float,
    viewport: Viewport,
    disp_w: int,
    disp_h: int,
) -> tuple[float, float]:
    """Display coords -> full-image coords."""
    Wv = viewport.width
    Hv = viewport.height
    if Wv <= 0 or Hv <= 0:
        return viewport.x_min, viewport.y_min

    x = viewport.x_min + (vx / disp_w) * Wv
    y = viewport.y_min + (vy / disp_h) * Hv
    return x, y
