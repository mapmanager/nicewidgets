# nicewidgets/assets/registry.py
from __future__ import annotations

from pathlib import Path

from nicegui import app, ui


def register_assets() -> None:
    """Serve nicewidgets static assets and link them in <head>.

    This avoids embedding large CSS/JS blobs as Python strings.
    """
    assets_dir = Path(__file__).resolve().parent
    app.add_static_files("/nicewidgets-assets", str(assets_dir))

    ui.add_head_html('<link rel="stylesheet" href="/nicewidgets-assets/aggrid_theme.css">')
