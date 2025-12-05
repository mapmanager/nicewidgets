# stub for source code for nicegui roiimagewidget
# nicewidgets/src/nicewidgets/roi_image_widget/roi_image_widget.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict

from typing import Callable

import numpy as np
from nicegui import ui, events
from PIL import Image
from matplotlib import cm

from nicewidgets.utils.logging import get_logger
from .viewport import Viewport, view_to_full, full_to_view

logger = get_logger(__name__)


class RoiDict(TypedDict):
    id: str
    left: int
    top: int
    right: int
    bottom: int


@dataclass
class _Roi:
    """Internal ROI representation in full-image coordinates (floats)."""

    id: str
    left: float
    top: float
    right: float
    bottom: float

    def to_dict(self) -> RoiDict:
        return {
            "id": self.id,
            "left": int(round(self.left)),
            "top": int(round(self.top)),
            "right": int(round(self.right)),
            "bottom": int(round(self.bottom)),
        }


@dataclass
class RoiImageConfig:
    # Wheel / zoom behavior
    wheel_default_axis: str = "both"        # "both", "x", or "y"
    wheel_shift_axis: str | None = "x"      # axis when Shift held
    wheel_ctrl_axis: str | None = "y"       # axis when Ctrl held
    edge_tolerance_px: float = 5.0          # edge hit-test tolerance (display px)
    zoom_in_factor: float = 0.8             # factor for zooming in
    zoom_out_factor: float = 1.25           # factor for zooming out

    # ROI appearance
    roi_color: str = "red"
    roi_selected_color: str = "lime"
    roi_line_width: float = 2.0
    roi_fill_color: str = "red"
    roi_fill_opacity: float = 0.15

    # ROI geometry constraints (in full-image pixels)
    min_roi_width: float = 3.0
    min_roi_height: float = 3.0

    # Panning behavior
    enable_panning: bool = True
    pan_modifier: str = "shift"             # currently only "shift" supported

    # Display resolution (logical pixel grid)
    display_width_px: int | None = None
    display_height_px: int | None = None
    # image border
    image_border_width: int = 0             # in pixels


def array_to_pil(
    arr: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "gray",
) -> Image.Image:
    """Map a 2D NumPy array to an 8-bit RGB PIL image with a colormap."""
    arr = np.asarray(arr, dtype=float)

    if vmin is None:
        vmin = float(np.nanmin(arr))
    if vmax is None:
        vmax = float(np.nanmax(arr))

    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = (arr - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)

    cmap_fn = cm.get_cmap(cmap)
    rgba = cmap_fn(norm)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


class RoiImageWidget:
    """Reusable NiceGUI widget for images with rectangular ROIs and zoom.

    - Input: 2D numpy array (grayscale).
    - Optional: initial ROIs as list of RoiDict.
    - Public API uses dicts; internal uses `_Roi`.
    
    Events (via callback registration):
        on_roi_created(handler): Handler called as handler(roi_dict)
        on_roi_updated(handler): Handler called as handler(roi_dict)
        on_roi_deleted(handler): Handler called as handler(roi_id)
        on_roi_selected(handler): Handler called as handler(roi_id or None)
    """

    def __init__(
        self,
        image: np.ndarray,
        *,
        rois: List[RoiDict] | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = "gray",
        parent=None,
        config: RoiImageConfig | None = None,
    ) -> None:
        if image.ndim != 2:
            raise ValueError("RoiImageWidget expects a 2D numpy array")

        self.image = np.asarray(image)
        self._dtype = self.image.dtype

        self.img_height, self.img_width = self.image.shape

        # Configuration: either provided or default
        if config is None:
            self.config = RoiImageConfig()
        else:
            self.config = config

        # Logical display size: default to image size, but allow override via config
        if self.config.display_width_px is not None:
            self.DISPLAY_W = int(self.config.display_width_px)
        else:
            self.DISPLAY_W = self.img_width

        if self.config.display_height_px is not None:
            self.DISPLAY_H = int(self.config.display_height_px)
        else:
            self.DISPLAY_H = self.img_height

        # Contrast / colormap
        self._vmin = float(vmin) if vmin is not None else float(np.nanmin(self.image))
        self._vmax = float(vmax) if vmax is not None else float(np.nanmax(self.image))
        self._cmap = cmap

        # Viewport: single source of truth for visible region
        self.viewport = Viewport(
            img_width=self.img_width,
            img_height=self.img_height,
        )

        # ROI storage
        self._rois: Dict[str, _Roi] = {}
        self._next_id: int = 1
        self._selected_id: Optional[str] = None

        # Last mouse full-image position
        self._last_mouse_x_full: Optional[float] = None
        self._last_mouse_y_full: Optional[float] = None
        
        # Callback registries (like CustomAgGrid pattern - no psygnal)
        self._roi_created_handlers: List[Callable[[dict], None]] = []
        self._roi_updated_handlers: List[Callable[[dict], None]] = []
        self._roi_deleted_handlers: List[Callable[[str], None]] = []
        self._roi_selected_handlers: List[Callable[[Optional[str]], None]] = []

        # Interaction state for ROI drawing/moving
        self._mode: str = "idle"  # "idle", "drawing", "moving", "resizing_*", "panning"
        self._start_x_full: Optional[float] = None
        self._start_y_full: Optional[float] = None
        self._drag_roi_id: Optional[str] = None
        self._drag_orig: Optional[tuple[float, float, float, float]] = None  # (l,t,r,b)

        # Panning state
        self._last_pan_x_full: float | None = None
        self._last_pan_y_full: float | None = None
        self._pan_viewport_orig: dict | None = None  # snapshot at pan start, fix jitter
        self._start_vx: Optional[float] = None       # display-space at pan start
        self._start_vy: Optional[float] = None

        # Container / interactive image
        container = parent if parent is not None else ui.element("div").classes("w-full")

        with container:
            pil_img = self._render_view_pil()
            self.interactive = (
                ui.interactive_image(
                    pil_img,
                    cross=True,
                    events=["mousedown", "mousemove", "mouseup"],
                )
                .classes("w-full")
                .style(
                    f"aspect-ratio: {self.img_width} / {self.img_height}; "
                    "object-fit: contain; border: 1px solid #666;"
                )
            )
            self.interactive.on_mouse(self._on_mouse)
            self.interactive.on("wheel", self._on_wheel)

            # Global key handler for Delete / Backspace / Enter
            ui.on("keydown", self._on_key)

        # Initialize ROIs (if provided) and perform a full initial draw
        if rois:
            self.set_rois(rois)

        self._update_image()
        
        logger.info(
            f"RoiImageWidget initialized: image={self.img_width}x{self.img_height}, "
            f"display={self.DISPLAY_W}x{self.DISPLAY_H}, cmap={self._cmap}, "
            f"vmin={self._vmin:.2f}, vmax={self._vmax:.2f}"
        )

    # ------------- properties -------------

    @property
    def dtype(self):
        """Return dtype of underlying image array."""
        return self._dtype

    # ------------- public ROI API -------------

    def get_rois(self) -> List[RoiDict]:
        """Return current ROIs as a list of RoiDict."""
        return [roi.to_dict() for roi in self._rois.values()]

    def set_rois(self, rois: List[RoiDict]) -> None:
        """Overwrite ROIs with the given list (ids preserved)."""
        self._rois.clear()
        for r in rois:
            roi_id = str(r["id"])
            left = float(r["left"])
            top = float(r["top"])
            right = float(r["right"])
            bottom = float(r["bottom"])
            self._rois[roi_id] = _Roi(
                id=roi_id,
                left=self._clamp_x(left),
                top=self._clamp_y(top),
                right=self._clamp_x(right),
                bottom=self._clamp_y(bottom),
            )

        # Ensure we don't reuse IDs like "roi-1" that already exist.
        # Find the largest numeric suffix of ids that look like "roi-N"
        # and set _next_id to > that.
        max_idx = self._next_id
        for roi_id in self._rois.keys():
            if roi_id.startswith("roi-"):
                try:
                    n = int(roi_id.split("-", 1)[1])
                except ValueError:
                    continue
                if n >= max_idx:
                    max_idx = n + 1
        self._next_id = max_idx

        self._redraw_overlays()
        logger.debug(f"set_rois: loaded {len(self._rois)} ROIs")

    def delete_selected_roi(self) -> None:
        """Delete the currently selected ROI, if any."""
        if self._selected_id is None:
            return
        rid = self._selected_id
        if rid in self._rois:
            del self._rois[rid]
            # Notify handlers
            for handler in list(self._roi_deleted_handlers):
                try:
                    handler(rid)
                except Exception:
                    logger.exception("Error in roi_deleted handler")
            logger.info(f"Deleted ROI: {rid}")
        self._selected_id = None
        # Notify handlers
        for handler in list(self._roi_selected_handlers):
            try:
                handler(None)
            except Exception:
                logger.exception("Error in roi_selected handler")
        self._redraw_overlays()

    def select_roi(self, roi_id: Optional[str]) -> None:
        """Select an ROI by id (or None to clear selection)."""
        if roi_id is not None and roi_id not in self._rois:
            return
        self._selected_id = roi_id
        # Notify handlers
        for handler in list(self._roi_selected_handlers):
            try:
                handler(roi_id)
            except Exception:
                logger.exception("Error in roi_selected handler")
        self._redraw_overlays()
    
    # ------------- public event registration API -------------
    
    def on_roi_created(self, handler: Callable[[dict], None]) -> None:
        """Register callback for ROI creation events.
        
        Handler is called with: roi_dict (RoiDict)
        """
        self._roi_created_handlers.append(handler)
    
    def on_roi_updated(self, handler: Callable[[dict], None]) -> None:
        """Register callback for ROI update events.
        
        Handler is called with: roi_dict (RoiDict)
        """
        self._roi_updated_handlers.append(handler)
    
    def on_roi_deleted(self, handler: Callable[[str], None]) -> None:
        """Register callback for ROI deletion events.
        
        Handler is called with: roi_id (str)
        """
        self._roi_deleted_handlers.append(handler)
    
    def on_roi_selected(self, handler: Callable[[Optional[str]], None]) -> None:
        """Register callback for ROI selection events.
        
        Handler is called with: roi_id (str or None)
        """
        self._roi_selected_handlers.append(handler)

    # ------------- public viewport API -------------

    def reset_view(self) -> None:
        """Show full image and redraw."""
        self.viewport.reset()
        self._update_image()
        logger.debug("reset_view: viewport reset to full image")

    def get_viewport(self) -> dict:
        """Return current viewport as a dict."""
        return self.viewport.to_dict()

    def set_viewport(self, vp_dict: dict) -> None:
        """Set viewport from a dict and redraw."""
        self.viewport = Viewport.from_dict(vp_dict)
        self._update_image()

    # ------------- public image appearance API -------------

    def set_contrast(self, vmin: float | None, vmax: float | None) -> None:
        """Update vmin/vmax and redraw."""
        if vmin is None or vmax is None:
            self._vmin = float(np.nanmin(self.image))
            self._vmax = float(np.nanmax(self.image))
        else:
            self._vmin = float(vmin)
            self._vmax = float(vmax)
        self._update_image()

    def set_cmap(self, cmap: str) -> None:
        """Update colormap and redraw."""
        self._cmap = cmap
        self._update_image()

    def auto_contrast(self, low_percentile=2.0, high_percentile=98.0) -> None:
        """Set vmin/vmax from image percentiles and redraw."""
        flat = self.image.ravel()
        self._vmin = float(np.nanpercentile(flat, low_percentile))
        self._vmax = float(np.nanpercentile(flat, high_percentile))
        self._update_image()

    # ------------- internals: rendering -------------

    def _view_array(self) -> np.ndarray:
        """Return the current viewport region as a NumPy array.

        Uses the logical float viewport for bounds, but only computes
        integer slice indices for the actual image data. The viewport
        itself is NOT mutated here.
        """
        y_min, y_max, x_min, x_max = self.viewport.get_int_slice()
        return self.image[y_min:y_max, x_min:x_max]

    def _render_view_pil(self) -> Image.Image:
        """Render current viewport region as a PIL RGB image.

        The viewport slice is always rescaled to a fixed DISPLAY_W x DISPLAY_H,
        so the on-screen widget size stays constant while zooming.
        """
        sub = self._view_array()
        img = array_to_pil(sub, vmin=self._vmin, vmax=self._vmax, cmap=self._cmap)

        # Always render into the fixed display size
        if (self.DISPLAY_W, self.DISPLAY_H) != (sub.shape[1], sub.shape[0]):
            img = img.resize((self.DISPLAY_W, self.DISPLAY_H), Image.BILINEAR)

        return img

    def _update_image(self) -> None:
        """Redraw image + overlays."""
        pil_img = self._render_view_pil()
        self.interactive.set_source(pil_img)

        # Keep DOM aspect ratio in sync with the display size
        self.interactive.style(
            f"aspect-ratio: {self.DISPLAY_W} / {self.DISPLAY_H}; "
            f"object-fit: contain; border: {self.config.image_border_width}px solid #666;"
        )

        self._redraw_overlays()

    def _redraw_overlays(self) -> None:
        """Draw ROI rectangles as SVG overlay in display coordinates.

        Uses the same logical viewport that controls the image slice so ROIs
        move correctly with zoom and pan.
        """
        vp = self.viewport
        svg_parts: list[str] = []

        for roi in self._rois.values():
            # Skip ROIs outside viewport
            if (
                roi.right < vp.x_min
                or roi.left > vp.x_max
                or roi.bottom < vp.y_min
                or roi.top > vp.y_max
            ):
                continue

            # Intersect ROI with viewport
            left_full = max(roi.left, vp.x_min)
            right_full = min(roi.right, vp.x_max)
            top_full = max(roi.top, vp.y_min)
            bottom_full = min(roi.bottom, vp.y_max)

            # Convert to display coords
            left_vx, top_vy = full_to_view(
                left_full, top_full, vp, self.DISPLAY_W, self.DISPLAY_H
            )
            right_vx, bottom_vy = full_to_view(
                right_full, bottom_full, vp, self.DISPLAY_W, self.DISPLAY_H
            )

            w = right_vx - left_vx
            h = bottom_vy - top_vy

            is_selected = roi.id == self._selected_id
            stroke = (
                self.config.roi_selected_color
                if is_selected
                else self.config.roi_color
            )

            svg_parts.append(
                f'<rect x="{left_vx}" y="{top_vy}" '
                f'width="{w}" height="{h}" '
                f'stroke="{stroke}" stroke-width="{self.config.roi_line_width}" '
                f'fill="{self.config.roi_fill_color}" '
                f'fill-opacity="{self.config.roi_fill_opacity}" />'
            )

        self.interactive.content = "".join(svg_parts)
        self.interactive.update()

    # ------------- internals: events -------------

    def _on_mouse(self, e: events.MouseEventArguments) -> None:
        """Handle NiceGUI mouse events for ROI interactions + mouse-move signal."""
        # Clamp to display size
        vx = max(0.0, min(float(self.DISPLAY_W - 1), e.image_x))
        vy = max(0.0, min(float(self.DISPLAY_H - 1), e.image_y))

        # Use the logical viewport for converting display coords back to full
        x_full, y_full = view_to_full(
            vx,
            vy,
            self.viewport,
            self.DISPLAY_W,
            self.DISPLAY_H,
        )

        # Track mouse position in full-image space
        self._last_mouse_x_full = x_full
        self._last_mouse_y_full = y_full

        # ---------- MOUSEDOWN (start drawing / moving / resizing / panning) ----------
        if e.type == "mousedown" and e.button == 0:
            # Panning if enabled and modifier pressed (anchored to pan start)
            if (
                self.config.enable_panning
                and self.config.pan_modifier == "shift"
                and e.shift
            ):
                # If we're already showing the full image, there is nothing to pan.
                full_w = float(self.img_width)
                full_h = float(self.img_height)
                if (
                    abs(self.viewport.width - full_w) < 1e-6
                    and abs(self.viewport.height - full_h) < 1e-6
                ):
                    # Ignore Shift+drag in full-view mode.
                    return

                self._mode = "panning"
                self._start_x_full = x_full
                self._start_y_full = y_full
                self._start_vx = vx  # keeping image under mouse cursor
                self._start_vy = vy
                self._last_pan_x_full = x_full  # optional/debug
                self._last_pan_y_full = y_full
                # Snapshot the viewport at the start of the pan gesture
                self._pan_viewport_orig = self.viewport.to_dict()
                return

            # ROI hit-testing
            roi_id, mode = self._hit_test(x_full, y_full)

            if roi_id is not None and mode is not None:
                # Existing ROI hit
                self._selected_id = roi_id
                # Notify handlers
                for handler in list(self._roi_selected_handlers):
                    try:
                        handler(roi_id)
                    except Exception:
                        logger.exception("Error in roi_selected handler")
                roi = self._rois[roi_id]

                self._start_x_full = x_full
                self._start_y_full = y_full
                self._drag_roi_id = roi_id
                self._drag_orig = (roi.left, roi.top, roi.right, roi.bottom)

                if mode == "body":
                    self._mode = "moving"
                else:
                    # mode is one of "resizing_left", etc.
                    self._mode = mode

                self._redraw_overlays()
                return

            # Not over any ROI: start drawing a new one
            roi = self._create_new_roi(x_full, y_full)
            self._mode = "drawing"
            self._start_x_full = x_full
            self._start_y_full = y_full
            self._selected_id = roi.id
            # Notify selection handlers
            for handler in list(self._roi_selected_handlers):
                try:
                    handler(roi.id)
                except Exception:
                    logger.exception("Error in roi_selected handler")

            # Notify creation handlers with initial (zero-area) ROI
            for handler in list(self._roi_created_handlers):
                try:
                    handler(roi.to_dict())
                except Exception:
                    logger.exception("Error in roi_created handler")
            logger.info(f"Started creating ROI: {roi.id} at ({x_full:.1f}, {y_full:.1f})")

            self._redraw_overlays()
            return

        # ---------- MOUSEMOVE with left button held ----------
        if e.type == "mousemove" and (e.buttons & 1):
            # Panning (anchored to pan start, derived from display-space deltas)
            if self._mode == "panning":
                if (
                    self._start_vx is not None
                    and self._start_vy is not None
                    and self._pan_viewport_orig is not None
                ):
                    orig = self._pan_viewport_orig
                    full_w = float(orig["x_max"] - orig["x_min"])
                    full_h = float(orig["y_max"] - orig["y_min"])

                    # How far did the mouse move in display pixels?
                    dx_v = vx - self._start_vx
                    dy_v = vy - self._start_vy

                    # Convert display delta to full-image delta using the ORIGINAL viewport.
                    # This ensures that the point under the cursor at pan start
                    # stays under the cursor as we drag.
                    dx_full = dx_v * (full_w / float(self.DISPLAY_W))
                    dy_full = dy_v * (full_h / float(self.DISPLAY_H))

                    # Reset viewport to the original snapshot, then pan once
                    vp = self.viewport
                    vp.x_min = float(orig["x_min"])
                    vp.x_max = float(orig["x_max"])
                    vp.y_min = float(orig["y_min"])
                    vp.y_max = float(orig["y_max"])

                    vp.pan(dx_full, dy_full)

                    # Optional: keep these for debugging / introspection
                    self._last_pan_x_full = x_full
                    self._last_pan_y_full = y_full

                    self._update_image()
                return

            # Drawing new ROI (rubber-banding)
            if self._mode == "drawing" and self._selected_id is not None:
                roi = self._rois.get(self._selected_id)
                if (
                    roi is None
                    or self._start_x_full is None
                    or self._start_y_full is None
                ):
                    return

                roi.left = self._clamp_x(min(self._start_x_full, x_full))
                roi.right = self._clamp_x(max(self._start_x_full, x_full))
                roi.top = self._clamp_y(min(self._start_y_full, y_full))
                roi.bottom = self._clamp_y(max(self._start_y_full, y_full))

                self._enforce_min_size(roi)

                # Notify handlers
                for handler in list(self._roi_updated_handlers):
                    try:
                        handler(roi.to_dict())
                    except Exception:
                        logger.exception("Error in roi_updated handler")
                self._redraw_overlays()
                return

            # Moving an existing ROI
            if self._mode == "moving" and self._drag_roi_id is not None:
                roi = self._rois.get(self._drag_roi_id)
                if (
                    roi is None
                    or self._start_x_full is None
                    or self._start_y_full is None
                    or self._drag_orig is None
                ):
                    return

                dx = x_full - self._start_x_full
                dy = y_full - self._start_y_full
                l0, t0, r0, b0 = self._drag_orig

                roi.left = self._clamp_x(l0 + dx)
                roi.right = self._clamp_x(r0 + dx)
                roi.top = self._clamp_y(t0 + dy)
                roi.bottom = self._clamp_y(b0 + dy)

                self._enforce_min_size(roi)

                # Notify handlers
                for handler in list(self._roi_updated_handlers):
                    try:
                        handler(roi.to_dict())
                    except Exception:
                        logger.exception("Error in roi_updated handler")
                self._redraw_overlays()
                return

            # Resizing an existing ROI
            if self._mode.startswith("resizing") and self._drag_roi_id is not None:
                roi = self._rois.get(self._drag_roi_id)
                if roi is None or self._drag_orig is None:
                    return

                l0, t0, r0, b0 = self._drag_orig

                if self._mode == "resizing_left":
                    roi.left = self._clamp_x(min(x_full, r0))
                elif self._mode == "resizing_right":
                    roi.right = self._clamp_x(max(x_full, l0))
                elif self._mode == "resizing_top":
                    roi.top = self._clamp_y(min(y_full, b0))
                elif self._mode == "resizing_bottom":
                    roi.bottom = self._clamp_y(max(y_full, t0))

                self._enforce_min_size(roi)

                # Notify handlers
                for handler in list(self._roi_updated_handlers):
                    try:
                        handler(roi.to_dict())
                    except Exception:
                        logger.exception("Error in roi_updated handler")
                self._redraw_overlays()
                return

            return  # other modes: nothing to do

        # ---------- MOUSEUP (finish drawing/moving/resizing) ----------
        if e.type == "mouseup" and e.button == 0:
            if self._mode == "panning":
                self._mode = "idle"
                self._start_x_full = None
                self._start_y_full = None
                self._start_vx = None
                self._start_vy = None
                self._last_pan_x_full = None
                self._last_pan_y_full = None
                self._pan_viewport_orig = None
                return

            if self._mode == "drawing" and self._selected_id is not None:
                roi = self._rois.get(self._selected_id)
                if roi is not None:
                    # Remove zero-area ROIs
                    if int(round(roi.left)) == int(round(roi.right)) or int(
                        round(roi.top)
                    ) == int(round(roi.bottom)):
                        rid = roi.id
                        del self._rois[rid]
                        # Notify deleted handlers
                        for handler in list(self._roi_deleted_handlers):
                            try:
                                handler(rid)
                            except Exception:
                                logger.exception("Error in roi_deleted handler")
                        self._selected_id = None
                        # Notify selected handlers
                        for handler in list(self._roi_selected_handlers):
                            try:
                                handler(None)
                            except Exception:
                                logger.exception("Error in roi_selected handler")
                        self._redraw_overlays()
                        logger.debug(f"Deleted zero-area ROI: {rid}")
                    else:
                        logger.info(
                            f"Completed ROI: {roi.id} - "
                            f"L={roi.left:.1f}, T={roi.top:.1f}, "
                            f"R={roi.right:.1f}, B={roi.bottom:.1f}"
                        )

            # Reset interaction state
            self._mode = "idle"
            self._drag_roi_id = None
            self._drag_orig = None
            self._start_x_full = None
            self._start_y_full = None
            return

    def _on_key(self, e: events.GenericEventArguments) -> None:
        """Handle keyboard shortcuts: delete ROI, reset view."""
        args = e.args or {}
        key = args.get("key", "")

        if key in ("Backspace", "Delete"):
            self.delete_selected_roi()
        elif key == "Enter":
            self.reset_view()

    def _on_wheel(self, e: events.GenericEventArguments) -> None:
        """Handle wheel zoom with configurable axes."""
        args = e.args or {}
        dx = args.get("deltaX", 0)
        dy = args.get("deltaY", 0)

        # Normalize dy, and for cases like Shift+wheel (horizontal scroll),
        # fall back to deltaX so we still zoom.
        if not isinstance(dy, (int, float)):
            dy = 0
        if dy == 0 and isinstance(dx, (int, float)) and dx != 0:
            dy = dx

        if not isinstance(dy, (int, float)) or dy == 0:
            return

        shift = bool(args.get("shiftKey", False))
        ctrl = bool(args.get("ctrlKey", False))

        # Base zoom factor: negative dy -> zoom in; positive -> out
        base_factor = (
            self.config.zoom_in_factor if dy < 0 else self.config.zoom_out_factor
        )

        axis = self.config.wheel_default_axis  # "both", "x", "y"
        if shift and self.config.wheel_shift_axis is not None:
            axis = self.config.wheel_shift_axis
        elif ctrl and self.config.wheel_ctrl_axis is not None:
            axis = self.config.wheel_ctrl_axis

        if axis == "x":
            factor_x = base_factor
            factor_y = 1.0
        elif axis == "y":
            factor_x = 1.0
            factor_y = base_factor
        else:  # "both" or unknown -> both axes
            factor_x = base_factor
            factor_y = base_factor

        vp = self.viewport
        cx = self._last_mouse_x_full
        cy = self._last_mouse_y_full

        if cx is None or cy is None:
            cx = 0.5 * (vp.x_min + vp.x_max)
            cy = 0.5 * (vp.y_min + vp.y_max)

        self.viewport.zoom_around(
            x_center=cx,
            y_center=cy,
            factor_x=factor_x,
            factor_y=factor_y,
        )
        self._update_image()
        logger.debug(
            f"Zoom: axis={axis}, factor_x={factor_x:.2f}, factor_y={factor_y:.2f}, "
            f"center=({cx:.1f}, {cy:.1f})"
        )

    # ------------- internals: ROI helpers -------------

    def _enforce_min_size(self, roi: _Roi) -> None:
        """Ensure ROI meets minimum size constraints in full-image pixels."""
        min_w = self.config.min_roi_width
        min_h = self.config.min_roi_height

        # Width
        width = roi.right - roi.left
        if width < min_w:
            cx = 0.5 * (roi.left + roi.right)
            half_w = min_w / 2.0
            roi.left = self._clamp_x(cx - half_w)
            roi.right = self._clamp_x(cx + half_w)

        # Height
        height = roi.bottom - roi.top
        if height < min_h:
            cy = 0.5 * (roi.top + roi.bottom)
            half_h = min_h / 2.0
            roi.top = self._clamp_y(cy - half_h)
            roi.bottom = self._clamp_y(cy + half_h)

    def _hit_test(
        self,
        x_full: float,
        y_full: float,
    ) -> tuple[Optional[str], Optional[str]]:
        """Hit-test ROIs at (x_full, y_full).

        Returns:
            (roi_id, mode) where mode is one of:
                - "body"
                - "resizing_left"
                - "resizing_right"
                - "resizing_top"
                - "resizing_bottom"
            or (None, None) if no ROI hit.
        """
        vp = self.viewport
        tol = self.config.edge_tolerance_px

        # Check most recently added first
        rois = list(self._rois.values())

        for roi in reversed(rois):
            # Quick reject in full coords: outside ROI box
            inside = (
                roi.left <= x_full <= roi.right
                and roi.top <= y_full <= roi.bottom
            )
            if not inside:
                continue

            # Convert ROI corners and point to display coords
            left_vx, top_vy = full_to_view(
                roi.left, roi.top, vp, self.DISPLAY_W, self.DISPLAY_H
            )
            right_vx, bottom_vy = full_to_view(
                roi.right, roi.bottom, vp, self.DISPLAY_W, self.DISPLAY_H
            )
            vx, vy = full_to_view(
                x_full, y_full, vp, self.DISPLAY_W, self.DISPLAY_H
            )

            on_left = abs(vx - left_vx) <= tol
            on_right = abs(vx - right_vx) <= tol
            on_top = abs(vy - top_vy) <= tol
            on_bottom = abs(vy - bottom_vy) <= tol

            if on_left:
                return roi.id, "resizing_left"
            if on_right:
                return roi.id, "resizing_right"
            if on_top:
                return roi.id, "resizing_top"
            if on_bottom:
                return roi.id, "resizing_bottom"

            # Inside but not near edges: body
            return roi.id, "body"

        return None, None

    def _create_new_roi(self, x_full: float, y_full: float) -> _Roi:
        """Create a new ROI starting at a single point."""
        roi_id = f"roi-{self._next_id}"
        self._next_id += 1
        roi = _Roi(
            id=roi_id,
            left=self._clamp_x(x_full),
            top=self._clamp_y(y_full),
            right=self._clamp_x(x_full),
            bottom=self._clamp_y(y_full),
        )
        self._rois[roi_id] = roi
        return roi

    def _clamp_x(self, x: float) -> float:
        return max(0.0, min(float(self.img_width), x))

    def _clamp_y(self, y: float) -> float:
        return max(0.0, min(float(self.img_height), y))
