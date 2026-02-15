# tests/plot_pool_app/conftest.py
"""Pytest configuration for plot_pool_app tests."""
from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    # Ensure nicewidgets package is importable when running tests from repo root.
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
