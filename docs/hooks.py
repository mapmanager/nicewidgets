"""MkDocs hooks for NiceWidgets documentation."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def on_config(config: Any) -> Any:
    """Expose the NiceWidgets package version to MkDocs templates."""
    pyproject_path = Path('pyproject.toml')
    version = 'unknown'

    if pyproject_path.exists():
        data = tomllib.loads(pyproject_path.read_text())
        version = data.get('project', {}).get('version', 'unknown')

    config.setdefault('extra', {})
    config['extra']['nicewidgets_version'] = version
    return config
