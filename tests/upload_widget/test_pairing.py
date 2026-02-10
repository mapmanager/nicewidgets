# tests/upload_widget/test_pairing.py
from __future__ import annotations

from pathlib import Path

import pytest

from nicewidgets.upload_widget.pairing import pair_by_stem


def test_pair_by_stem_pairs_and_sorts(tmp_path: Path) -> None:
    # Create dummy paths (files need not exist for pairing logic)
    paths = [
        tmp_path / "b.txt",
        tmp_path / "a.tif",
        tmp_path / "a.txt",
        tmp_path / "b.tif",
    ]
    pairs = pair_by_stem(paths, exts=(".tif", ".txt"))
    assert [(p1.name, p2.name) for p1, p2 in pairs] == [
        ("a.tif", "a.txt"),
        ("b.tif", "b.txt"),
    ]


def test_pair_by_stem_raises_on_missing_member(tmp_path: Path) -> None:
    paths = [tmp_path / "a.tif", tmp_path / "b.txt"]
    with pytest.raises(ValueError):
        pair_by_stem(paths, exts=(".tif", ".txt"))
