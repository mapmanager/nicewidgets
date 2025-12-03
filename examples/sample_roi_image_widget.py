from __future__ import annotations

import numpy as np
from nicegui import ui

from nicewidgets.roi_image_widget.roi_image_widget import RoiImageWidget


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


if __name__ in {"__main__", "__mp_main__"}:
    img = create_demo_image()

    with ui.row().classes("w-full gap-6"):
        with ui.column().classes("items-start gap-2 w-3/4"):
            ui.label("RoiImageWidget demo").classes("text-lg font-bold")

            widget = RoiImageWidget(
                img,
                rois=[
                    {"id": "roi-1", "left": 20, "top": 10, "right": 80, "bottom": 50},
                ],
                wheel_default_axis="both",
                wheel_shift_axis="x",
                wheel_ctrl_axis="y",
            )

            @widget.viewport_changed.connect
            def on_vp(vp_dict: dict) -> None:
                # Just show a very small toast as proof
                x_min = vp_dict["x_min"]
                x_max = vp_dict["x_max"]
                ui.notify(f"Viewport x=[{x_min:.1f}, {x_max:.1f}]", close_button=True, timeout=1.0)

            @widget.roi_created.connect
            def on_roi_created(roi: dict) -> None:
                ui.notify(f"ROI created: {roi['id']}", timeout=1.0)

            @widget.roi_updated.connect
            def on_roi_updated(roi: dict) -> None:
                # You can inspect roi dict here if desired
                pass

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

    ui.run()
