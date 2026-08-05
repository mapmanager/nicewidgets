"""Combined-demo integration contracts for RasterViewerWidget."""

from inspect import signature
from pathlib import Path

from examples.raster_viewer_widget.page import build_raster_widget_demo_page


def test_main_demo_advertises_the_canvas_raster_widget_route() -> None:
    """Keep the canvas viewer distinct from the existing Plotly raster route."""
    source = (
        Path(__file__).resolve().parents[3] / "examples" / "main_demo" / "main.py"
    ).read_text()
    assert "'/raster-widget'," in source
    assert "'RasterViewerWidget'," in source
    assert "build_demo_route(build_raster_widget_demo_page)" in source
    assert "# from examples.raster_viewer.page import build_raster_demo_page" in source
    assert "# @ui.page('/raster')" in source


def test_raster_widget_page_builder_accepts_shared_shell_state() -> None:
    """Require the common embedded-layout and theme builder contract."""
    parameters = signature(build_raster_widget_demo_page).parameters
    assert parameters["embedded"].default is False
    assert parameters["dark_mode"].default is True
