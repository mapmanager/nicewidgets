"""
Combined demo showing CustomAgGrid and RoiImageWidget together.

Demonstrates:
- CustomAgGrid with editable cells and row selection
- RoiImageWidget with ROI drawing and zoom/pan
- Both using callback pattern (no psygnal)
- Coordination between widgets

Run:
    cd /Users/cudmore/Sites/kymflow_outer/nicewidgets
    uv run python examples/combined_widgets_demo.py
"""

import numpy as np
from nicegui import ui

from nicewidgets.custom_ag_grid import CustomAgGrid, ColumnConfig, GridConfig
from nicewidgets.roi_image_widget import RoiImageWidget, RoiImageConfig


def create_sample_image(height: int = 200, width: int = 300) -> np.ndarray:
    """Create a simple test image with gradient and noise."""
    x = np.linspace(0, 2 * np.pi, width)
    y = np.linspace(0, 2 * np.pi, height)
    xx, yy = np.meshgrid(x, y)
    img = 0.5 + 0.3 * np.sin(xx) + 0.3 * np.cos(yy)
    img += 0.1 * np.random.randn(height, width)
    img = np.clip(img, 0.0, 1.0)
    return img


@ui.page("/")
def index():
    ui.label("NiceWidgets Combined Demo").classes("text-3xl font-bold mb-6")
    
    with ui.row().classes("w-full gap-6"):
        # Left column: Grid
        with ui.column().classes("flex-1"):
            ui.label("CustomAgGrid Example").classes("text-xl font-bold mb-2")
            
            # Sample data: list of images/ROIs
            grid_data = [
                {"id": 1, "name": "Image 1", "rois": 3, "status": "analyzed"},
                {"id": 2, "name": "Image 2", "rois": 5, "status": "pending"},
                {"id": 3, "name": "Image 3", "rois": 2, "status": "analyzed"},
            ]
            
            columns = [
                ColumnConfig(field="id", header="ID", editable=False),
                ColumnConfig(field="name", header="Name", editable=True),
                ColumnConfig(field="rois", header="ROIs", editable=False),
                ColumnConfig(
                    field="status",
                    header="Status",
                    editable=True,
                    editor="select",
                    choices=["pending", "analyzed", "reviewed"],
                ),
            ]
            
            grid_config = GridConfig(
                selection_mode="single",
                height="h-64",
                zebra_rows=True,
                hover_highlight=True,
            )
            
            grid = CustomAgGrid(
                data=grid_data,
                columns=columns,
                grid_config=grid_config,
            )
            
            # Info labels updated by callbacks
            grid_info = ui.label("No selection").classes("text-sm text-gray-500 mt-2")
            edit_info = ui.label("").classes("text-sm text-gray-500")
            
            # Register grid callbacks
            def on_row_selected(row_idx: int, row_data: dict) -> None:
                grid_info.text = f"Selected: {row_data['name']} (ID: {row_data['id']})"
            
            def on_cell_edited(row_idx: int, field: str, old, new, row_data: dict) -> None:
                edit_info.text = f"Edited {field}: {old} → {new}"
            
            grid.on_row_selected(on_row_selected)
            grid.on_cell_edited(on_cell_edited)
        
        # Right column: ROI Widget
        with ui.column().classes("flex-1"):
            ui.label("RoiImageWidget Example").classes("text-xl font-bold mb-2")
            
            img = create_sample_image()
            
            config = RoiImageConfig(
                wheel_default_axis="both",
                wheel_shift_axis="x",
                wheel_ctrl_axis="y",
                edge_tolerance_px=8.0,
                display_width_px=400,
                display_height_px=267,  # 3:2 aspect ratio
            )
            
            roi_widget = RoiImageWidget(
                img,
                rois=[
                    {"id": "roi-1", "left": 50, "top": 30, "right": 150, "bottom": 100},
                ],
                config=config,
            )
            
            # Info labels updated by callbacks
            roi_info = ui.label("No ROI selected").classes("text-sm text-gray-500 mt-2")
            roi_event = ui.label("").classes("text-sm text-gray-500")
            
            # Register ROI widget callbacks
            def on_roi_created(roi: dict) -> None:
                roi_event.text = f"Created: {roi['id']}"
            
            def on_roi_updated(roi: dict) -> None:
                roi_event.text = f"Updated: {roi['id']}"
            
            def on_roi_deleted(roi_id: str) -> None:
                roi_event.text = f"Deleted: {roi_id}"
                roi_info.text = "No ROI selected"
            
            def on_roi_selected(roi_id: str | None) -> None:
                if roi_id:
                    rois = roi_widget.get_rois()
                    roi = next((r for r in rois if r['id'] == roi_id), None)
                    if roi:
                        roi_info.text = (
                            f"Selected: {roi_id} - "
                            f"L={roi['left']}, T={roi['top']}, "
                            f"R={roi['right']}, B={roi['bottom']}"
                        )
                else:
                    roi_info.text = "No ROI selected"
            
            roi_widget.on_roi_created(on_roi_created)
            roi_widget.on_roi_updated(on_roi_updated)
            roi_widget.on_roi_deleted(on_roi_deleted)
            roi_widget.on_roi_selected(on_roi_selected)
    
    # Instructions
    with ui.card().classes("w-full p-4 mt-4"):
        ui.label("Instructions").classes("font-bold mb-2")
        ui.label("Grid: Click rows to select, double-click cells to edit")
        ui.label("ROI Widget: Click and drag to create ROIs, click edges to resize")
        ui.label("Both widgets use callback pattern (no psygnal!)")


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(port=8004, reload=True)

