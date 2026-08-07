"""Static safeguards for the instance-scoped browser component."""

from pathlib import Path

WEB_ASSETS = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "nicewidgets"
    / "raster_viewer_widget"
    / "web"
)


def test_component_and_viewer_do_not_use_demo_page_globals() -> None:
    """Verify the reusable browser implementation contains no demo singleton."""
    source = "\n".join(path.read_text() for path in WEB_ASSETS.glob("*.js"))
    assert "window.rasterViewerDemo" not in source
    assert "rasterViewerDemoHostId" not in source


def test_layout_radio_group_is_keyed_by_viewer_instance() -> None:
    """Verify equal datasets in two widgets cannot share a browser radio group."""
    source = (WEB_ASSETS / "raster-viewer.js").read_text()
    assert "rv-mode-${this.instanceId}" in source
    assert "rv-mode-${this.dataset.id}" not in source


def test_widget_config_exposes_slice_wheel_direction() -> None:
    """Verify Python config reaches the instance-scoped JavaScript viewer."""
    component = (WEB_ASSETS / "raster_viewer_component.js").read_text()
    assert "invertSliceWheel: {type: Boolean, default: true}" in component
    assert "invertSliceWheel: this.invertSliceWheel" in component
    assert "wheelZoomFactor: {type: Number, default: 1.06}" in component
    assert "wheelZoomFactor: this.wheelZoomFactor" in component
    assert "initialChannelToolbarsVisible: {type: Boolean, default: true}" in component
    assert "setChannelToolbarsVisible(this.initialChannelToolbarsVisible)" in component


def test_component_declares_the_complete_xy_plot_bridge() -> None:
    """Verify the namespaced Python API can reach every plot lifecycle method."""
    component = (WEB_ASSETS / "raster_viewer_component.js").read_text()
    for method in ("addXYPlot", "updateXYPlot", "removeXYPlot", "showXYPlot", "hideXYPlot"):
        assert f"async {method}" in component


def test_component_declares_recent_lifecycle_and_presentation_bridges() -> None:
    """Verify Python can reach every recently added public viewer method."""
    component = (WEB_ASSETS / "raster_viewer_component.js").read_text()
    for method in (
        "clear",
        "resetView",
        "resetXRange",
        "setTIndex",
        "setZIndex",
        "setPhysicalCalibration",
        "setYRange",
        "setPhysicalRange",
        "selectChannel",
        "setChannelDisplay",
        "setChannelToolbarsVisible",
    ):
        assert f"async {method}" in component


def test_single_channel_controls_are_fixed_canvas_chrome() -> None:
    """Verify one-channel options and Sliding-Z share canvas-owned chrome."""
    viewer = (WEB_ASSETS / "raster-viewer.js").read_text()
    stylesheet = (WEB_ASSETS / "raster-viewer.css").read_text()
    assert "overlayControls.className = 'rv-pane-overlay-controls'" in viewer
    assert "if (slidingZControls) overlayControls.append(slidingZControls)" in viewer
    assert "overlayControls.append(this.optionsMenu)" in viewer
    assert "wrap.append(overlayControls)" in viewer
    assert "menuPanel.append(resetButton)" in viewer
    assert ".rv-pane-overlay-controls" in stylesheet
    assert "position: absolute" in stylesheet
