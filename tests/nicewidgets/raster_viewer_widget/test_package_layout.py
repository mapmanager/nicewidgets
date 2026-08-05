"""Tests ensuring the installed distribution retains component web assets."""

from pathlib import Path

import nicewidgets.raster_viewer_widget as raster_viewer


def test_component_assets_live_inside_the_import_package() -> None:
    """Verify every runtime file needed by the Vue adapter is package-owned."""
    package_root = Path(raster_viewer.__file__).resolve().parent
    web_root = package_root / "web"
    required = {
        "raster_viewer_component.js",
        "raster-viewer.js",
        "raster-viewer.css",
        "tooltip.js",
        "xy-plot-overlay.js",
    }
    assert required <= {path.name for path in web_root.iterdir() if path.is_file()}
    assert (package_root / "py.typed").is_file()
    assert "ISC License" in (web_root / "LUCIDE_LICENSE.txt").read_text()
