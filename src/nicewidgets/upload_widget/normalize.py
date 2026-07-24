"""Normalize NiceGUI upload file objects into readable filesystem paths."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any


def safe_upload_file_summary(upload_file: Any) -> str:
    """Return a short upload-file summary without dumping byte content.

    Args:
        upload_file: NiceGUI upload file object or compatible test double.

    Returns:
        One-line description of the file object shape.
    """
    cls = type(upload_file).__name__
    name = getattr(upload_file, 'name', None)
    content_type = getattr(upload_file, 'content_type', None)
    path = _as_path(getattr(upload_file, '_path', None))
    data = getattr(upload_file, '_data', None)
    data_len = len(data) if isinstance(data, (bytes, bytearray)) else None
    return (
        f'{cls}(name={name!r}, content_type={content_type!r}, '
        f'has_path={bool(path and path.exists())}, data_len={data_len}, '
        f'has_save={callable(getattr(upload_file, "save", None))}, '
        f'has_read={callable(getattr(upload_file, "read", None))})'
    )


def safe_upload_event_summary(event: Any) -> str:
    """Return a short upload-event summary without dumping byte content.

    Args:
        event: NiceGUI upload event object or compatible test double.

    Returns:
        One-line description of the event sender and file payload.
    """
    sender = getattr(event, 'sender', None)
    sender_summary = type(sender).__name__ if sender is not None else None
    file_payload = getattr(event, 'file', None)
    if isinstance(file_payload, list):
        file_summary = '[' + ', '.join(safe_upload_file_summary(item) for item in file_payload) + ']'
    else:
        file_summary = safe_upload_file_summary(file_payload) if file_payload is not None else 'None'
    return f'UploadEventArguments(sender={sender_summary}, file={file_summary})'


async def normalize_uploaded_file(upload_file: Any, *, suffix_hint: str | None = None) -> Path:
    """Normalize a NiceGUI upload file object into a readable path on disk.

    Args:
        upload_file: NiceGUI ``LargeFileUpload``/``SmallFileUpload`` object or a
            compatible object exposing ``_path``, ``save()``, ``read()``,
            ``_data``, or ``content``.
        suffix_hint: Optional suffix, including a leading dot, to preserve when
            the upload object lacks a stable on-disk suffix.

    Returns:
        Readable filesystem path. Temporary files are owned by the caller.

    Raises:
        RuntimeError: If the object does not expose a readable path or bytes.
    """
    inferred_suffix = _infer_suffix(upload_file, suffix_hint=suffix_hint)

    path = _as_path(getattr(upload_file, '_path', None))
    if path is not None and path.exists():
        if path.suffix:
            return path
        if inferred_suffix:
            destination = _mk_temp_path(suffix=inferred_suffix)
            shutil.copyfile(path, destination)
            return destination
        return path

    save = getattr(upload_file, 'save', None)
    if callable(save):
        tmp_path = _mk_temp_path(suffix=inferred_suffix)
        result = save(tmp_path)
        if hasattr(result, '__await__'):
            await result
        if tmp_path.exists():
            return tmp_path

    read = getattr(upload_file, 'read', None)
    if callable(read):
        data = read()
        if hasattr(data, '__await__'):
            data = await data
        if isinstance(data, (bytes, bytearray)):
            tmp_path = _mk_temp_path(suffix=inferred_suffix)
            tmp_path.write_bytes(bytes(data))
            return tmp_path

    for attr_name in ('_data', 'content'):
        data = getattr(upload_file, attr_name, None)
        if isinstance(data, (bytes, bytearray)):
            tmp_path = _mk_temp_path(suffix=inferred_suffix)
            tmp_path.write_bytes(bytes(data))
            return tmp_path

    raise RuntimeError(
        'Upload did not provide a readable temp file path and no usable '
        'save/read/data interface for in-memory upload content.'
    )


def _as_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    try:
        return Path(value)
    except Exception:
        return None


def _infer_suffix(upload_file: Any, *, suffix_hint: str | None) -> str:
    """Infer a suffix, including leading dot, for temporary upload files."""
    if isinstance(suffix_hint, str) and suffix_hint:
        return suffix_hint

    name = getattr(upload_file, 'name', None)
    if isinstance(name, str) and name:
        suffix = Path(name).suffix
        if suffix:
            return suffix

    content_type = getattr(upload_file, 'content_type', None)
    if isinstance(content_type, str):
        lowered = content_type.lower()
        if 'tif' in lowered or 'tiff' in lowered:
            return '.tif'
        if 'png' in lowered:
            return '.png'
        if 'jpeg' in lowered or 'jpg' in lowered:
            return '.jpg'

    return ''


def _mk_temp_path(*, suffix: str) -> Path:
    """Create a named temporary file path with a stable suffix."""
    handle = tempfile.NamedTemporaryFile(
        prefix='nicewidgets_upload_',
        suffix=suffix,
        delete=False,
    )
    try:
        return Path(handle.name)
    finally:
        handle.close()
