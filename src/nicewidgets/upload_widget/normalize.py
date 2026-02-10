# nicewidgets/src/nicewidgets/upload_widget/normalize.py
"""Upload normalization utilities for NiceGUI.

NiceGUI's ``ui.upload`` yields an event whose ``e.file`` is typically one of:

- **LargeFileUpload**: already written to disk; often exposes ``._path``.
- **SmallFileUpload**: in-memory; may expose async ``.save(path)`` / ``.read()`` or internal ``._data``.

This module converts either style into a real on-disk ``pathlib.Path``.

Key behaviors
-------------
- Always returns a readable file path on disk.
- When we must create a temp file, we **preserve the original filename suffix**
  (e.g. ``.tif``), because caller code commonly filters by ``Path.suffix``.
- If NiceGUI provides a temp path via ``._path`` **without a suffix**, we create
  a *new* temp file with the correct suffix and copy the bytes. This keeps
  caller behavior consistent across NiceGUI implementations.

Temp-file lifecycle
-------------------
We follow your chosen policy: **caller owns temp cleanup** (no auto-delete).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional


def _as_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    try:
        return Path(value)
    except Exception:
        return None


def _infer_suffix(upload_file: Any, *, suffix_hint: str | None) -> str:
    """Infer a suffix (including leading dot) for temp files."""
    if isinstance(suffix_hint, str) and suffix_hint:
        return suffix_hint

    name = getattr(upload_file, "name", None)
    if isinstance(name, str) and name:
        suf = Path(name).suffix
        if suf:
            return suf

    ctype = getattr(upload_file, "content_type", None)
    if isinstance(ctype, str):
        lc = ctype.lower()
        if "tif" in lc or "tiff" in lc:
            return ".tif"
        if "png" in lc:
            return ".png"
        if "jpeg" in lc or "jpg" in lc:
            return ".jpg"

    return ""


def safe_upload_file_summary(upload_file: Any) -> str:
    """Safe one-line summary without dumping bytes."""
    cls = type(upload_file).__name__
    name = getattr(upload_file, "name", None)
    ctype = getattr(upload_file, "content_type", None)

    p = _as_path(getattr(upload_file, "_path", None))
    has_path = bool(p and p.exists())

    data = getattr(upload_file, "_data", None)
    data_len = len(data) if isinstance(data, (bytes, bytearray)) else None

    has_save = callable(getattr(upload_file, "save", None))
    has_read = callable(getattr(upload_file, "read", None))

    return (
        f"{cls}(name={name!r}, content_type={ctype!r}, "
        f"has_path={has_path}, data_len={data_len}, has_save={has_save}, has_read={has_read})"
    )


def safe_upload_event_summary(e: Any) -> str:
    sender = getattr(e, "sender", None)
    sender_s = type(sender).__name__ if sender is not None else None
    f = getattr(e, "file", None)
    if isinstance(f, list):
        parts = ", ".join(safe_upload_file_summary(x) for x in f)
        file_s = f"[{parts}]"
    else:
        file_s = safe_upload_file_summary(f) if f is not None else "None"
    return f"UploadEventArguments(sender={sender_s}, file={file_s})"


def _mk_temp_path(*, suffix: str) -> Path:
    """Create a named temp file path with a stable suffix."""
    f = tempfile.NamedTemporaryFile(
        prefix="nicewidgets_upload_",
        suffix=suffix,
        delete=False,
    )
    try:
        return Path(f.name)
    finally:
        f.close()


async def normalize_uploaded_file(upload_file: Any, *, suffix_hint: str | None = None) -> Path:
    """Normalize a NiceGUI upload file object into a readable filesystem Path.

    Order:
      1) If ``._path`` exists and file exists:
         - if it already has a suffix, return it
         - if it lacks a suffix but we can infer one, copy to a new temp file with suffix
      2) Else if async ``save(path)`` exists, save to temp file with inferred suffix
      3) Else if async ``read()`` exists, write returned bytes to temp file with inferred suffix
      4) Else if ``._data`` exists, write to temp file with inferred suffix

    Raises:
        RuntimeError if no usable interface is available.
    """
    inferred_suffix = _infer_suffix(upload_file, suffix_hint=suffix_hint)

    # 1) NiceGUI large-upload path (already on disk)
    p = _as_path(getattr(upload_file, "_path", None))
    if p is not None and p.exists():
        # If the path already preserves a suffix, use it directly.
        if p.suffix:
            return p
        # If the path has no suffix but we can infer one, copy to a new temp file.
        if inferred_suffix:
            dst = _mk_temp_path(suffix=inferred_suffix)
            shutil.copyfile(p, dst)
            return dst
        return p

    # Helper to create a temp file path now (suffix preserved)
    def _new_tmp() -> Path:
        return _mk_temp_path(suffix=inferred_suffix)

    # 2) Prefer async save() if available.
    save = getattr(upload_file, "save", None)
    if callable(save):
        tmp_path = _new_tmp()
        res = save(tmp_path)
        if hasattr(res, "__await__"):
            await res
        if tmp_path.exists():
            return tmp_path

    # 3) Async read() -> bytes
    read = getattr(upload_file, "read", None)
    if callable(read):
        data = read()
        if hasattr(data, "__await__"):
            data = await data
        if isinstance(data, (bytes, bytearray)):
            tmp_path = _new_tmp()
            tmp_path.write_bytes(bytes(data))
            return tmp_path

    # 4) Internal bytes
    data2 = getattr(upload_file, "_data", None)
    if isinstance(data2, (bytes, bytearray)):
        tmp_path = _new_tmp()
        tmp_path.write_bytes(bytes(data2))
        return tmp_path

    content = getattr(upload_file, "content", None)
    if isinstance(content, (bytes, bytearray)):
        tmp_path = _new_tmp()
        tmp_path.write_bytes(bytes(content))
        return tmp_path

    raise RuntimeError(
        "Upload did not provide a readable temp file path (LargeFileUpload._path) "
        "and no usable save/read/data interface for SmallFileUpload."
    )
