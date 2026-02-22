from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from nicegui import ui

from nicewidgets.roi_image_widget.roi_image_widget import RoiImageWidget, RoiImageConfig
from nicewidgets.utils.logging import configure_logging

# Configure logging for this standalone example
# Logs will go to console and to ~/nicewidgets_example.log
configure_logging(level="DEBUG")


def create_demo_image(height: int = 120, width: int = 400) -> np.ndarray:
    """Simple demo image: sine waves + noise."""
    x = np.linspace(0, 4 * np.pi, width)
    img = np.zeros((height, width), dtype=float)
    for y in range(height):
        phase = 2 * np.pi * (y / height)
        img[y, :] = 0.5 + 0.5 * np.sin(x + phase)
    img += 0.05 * np.random.randn(height, width)
    img = np.clip(img, 0.0, 1.0)
    return img

def create_checkerboard_image(
    height: int = 160,
    width: int = 2000,
    block_h: int = 20,
    block_w: int = 40,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """
    Create a checkerboard test image for ROI widget debugging.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        block_h: Height of one checker tile.
        block_w: Width of one checker tile.
        smooth_sigma: Gaussian blur sigma. Set 0.0 for no smoothing.

    Returns:
        2D float32 NumPy array in [0,1].
    """
    img = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            by = (y // block_h)
            bx = (x // block_w)
            img[y, x] = 1.0 if (bx + by) % 2 == 0 else 0.0

    # Optional Gaussian smoothing for nicer zoom & interpolation
    if smooth_sigma > 0:
        img = gaussian_filter(img, smooth_sigma)

    # Normalize to [0,1] after smoothing
    img -= img.min()
    img /= (img.max() + 1e-8)

    return img

if __name__ in {"__main__", "__mp_main__"}:
    # img = create_demo_image()
    img = create_checkerboard_image()

    """
    And layout-wise, you can still use normal Tailwind stuff:

        .classes("w-full") → width tracks the column width.

        If you want to constrain height further, you could wrap it in a container with Tailwind like h-96 or max-h-[400px] — the aspect-ratio will still try to enforce the DISPLAY_W/H ratio, so the exact behavior will depend on how you combine those, but the logical mapping won’t break.

    So the mental model becomes:

        Visual size ≈ container width × aspect-ratio, where aspect-ratio = DISPLAY_W/DISPLAY_H you chose.

        Logical pixel grid = DISPLAY_W × DISPLAY_H you chose.
        This is what all zoom/pan/ROI math uses.
    """

    with ui.row().classes("w-full gap-6"):
        with ui.column().classes("items-start gap-2 w-3/4"):
            ui.label("RoiImageWidget demo").classes("text-lg font-bold")

            config = RoiImageConfig(
                wheel_default_axis="x",
                wheel_shift_axis=None,
                wheel_ctrl_axis="y",
                # display_width_px=1000,
                display_height_px=400,
            )

            widget = RoiImageWidget(
                img,
                rois=[
                    {"id": "roi-1", "left": 20, "top": 10, "right": 80, "bottom": 50},
                ],
                config=config,
            )

            # Register callbacks for ROI events
            def on_roi_created(roi: dict) -> None:
                ui.notify(f"ROI created: {roi['id']}", timeout=1.0)
            
            def on_roi_updated(roi: dict) -> None:
                # You can inspect roi dict here if desired
                pass
            
            widget.on_roi_created(on_roi_created)
            widget.on_roi_updated(on_roi_updated)

        with ui.column().classes("items-start gap-2 w-1/4"):
            ui.label("Contrast & colormap")

            vmin_slider = ui.slider(
                min=0.0,
                max=1.0,
                step=0.01,
                value=0.1,
                # label="vmin",
            )

            vmax_slider = ui.slider(
                min=0.0,
                max=1.0,
                step=0.01,
                value=0.9,
                # label="vmax",
            )

            def update_contrast():
                vmin = float(vmin_slider.value)
                vmax = float(vmax_slider.value)
                if vmin >= vmax:
                    return
                widget.set_contrast(vmin, vmax)

            vmin_slider.on_value_change(lambda e: update_contrast())
            vmax_slider.on_value_change(lambda e: update_contrast())

            cmap_select = ui.select(
                {
                    "gray": "gray",
                    "viridis": "viridis",
                    "magma": "magma",
                    "plasma": "plasma",
                },
                value="gray",
                label="Colormap",
            )

            @cmap_select.on_value_change
            def _(e):
                widget.set_cmap(str(e.value))

            ui.button("Auto contrast", on_click=lambda: widget.auto_contrast())

        with ui.column().classes("items-start gap-2 w-1/4"):
            ui.label("ROI Dump")

            json_area = ui.textarea(label="ROIs JSON").props("rows=6")

            def dump_rois():
                import json
                json_area.value = json.dumps(widget.get_rois(), indent=2)

            def load_rois():
                import json
                rois = json.loads(json_area.value)
                widget.set_rois(rois)

            ui.button("Dump ROIs", on_click=dump_rois)
            ui.button("Load ROIs", on_click=load_rois)
    ui.run()
