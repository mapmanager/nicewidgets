# nicewidgets/src/nicewidgets/upload_widget/pairing.py
"""Optional helpers for multi-file uploads."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def pair_by_stem(
    paths: Iterable[Path],
    *,
    exts: tuple[str, str] = (".tif", ".txt"),
) -> list[tuple[Path, Path]]:
    """Pair uploaded files by filename stem.

    Args:
        paths: Uploaded file paths.
        exts: Two extensions which define a pair (case-insensitive).

    Returns:
        List of (path_for_exts[0], path_for_exts[1]) pairs sorted by stem.

    Raises:
        ValueError: If any stem is missing one of the required extensions.
    """
    a_ext, b_ext = exts[0].lower(), exts[1].lower()
    by_stem: dict[str, dict[str, Path]] = {}

    for p in paths:
        p = Path(p)
        d = by_stem.setdefault(p.stem, {})
        d[p.suffix.lower()] = p

    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []

    for stem in sorted(by_stem.keys()):
        d = by_stem[stem]
        a = d.get(a_ext)
        b = d.get(b_ext)
        if a is None or b is None:
            missing.append(stem)
        else:
            pairs.append((a, b))

    if missing:
        raise ValueError(f"Missing required pair members for stems: {missing}")

    return pairs
