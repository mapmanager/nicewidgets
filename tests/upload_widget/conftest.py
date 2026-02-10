# tests/upload_widget/conftest.py
from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    # Ensure `nicewidgets/src` is importable when running tests from the repo root.
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
