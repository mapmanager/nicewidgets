"""Tests for upload normalization helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nicewidgets.upload_widget.normalize import normalize_uploaded_file


class PathUpload:
    """Upload double exposing a NiceGUI-like ``_path``."""

    def __init__(self, path: Path, *, name: str) -> None:
        self._path = path
        self.name = name


class AsyncReadUpload:
    """Upload double exposing async ``read`` bytes."""

    def __init__(self, data: bytes, *, name: str) -> None:
        self._data_to_read = data
        self.name = name

    async def read(self) -> bytes:
        """Return upload bytes."""
        return self._data_to_read


class DataUpload:
    """Upload double exposing in-memory ``_data`` bytes."""

    def __init__(self, data: bytes, *, name: str) -> None:
        self._data = data
        self.name = name


def test_normalize_returns_existing_path_when_suffix_is_present(tmp_path: Path) -> None:
    upload_path = tmp_path / 'large.oir'
    upload_path.write_bytes(b'large')

    result = asyncio.run(normalize_uploaded_file(PathUpload(upload_path, name='large.oir')))

    assert result == upload_path
    assert result.read_bytes() == b'large'


def test_normalize_copies_existing_path_without_suffix_to_suffix_temp(tmp_path: Path) -> None:
    upload_path = tmp_path / 'largeupload'
    upload_path.write_bytes(b'large')

    result = asyncio.run(normalize_uploaded_file(PathUpload(upload_path, name='large.oir')))

    try:
        assert result != upload_path
        assert result.suffix == '.oir'
        assert result.read_bytes() == b'large'
    finally:
        result.unlink(missing_ok=True)


def test_normalize_writes_async_read_bytes_to_suffix_temp() -> None:
    result = asyncio.run(normalize_uploaded_file(AsyncReadUpload(b'read-bytes', name='read.tif')))

    try:
        assert result.suffix == '.tif'
        assert result.read_bytes() == b'read-bytes'
    finally:
        result.unlink(missing_ok=True)


def test_normalize_writes_data_bytes_to_suffix_temp() -> None:
    result = asyncio.run(normalize_uploaded_file(DataUpload(b'data-bytes', name='data.czi')))

    try:
        assert result.suffix == '.czi'
        assert result.read_bytes() == b'data-bytes'
    finally:
        result.unlink(missing_ok=True)


def test_normalize_suffix_hint_overrides_filename_suffix() -> None:
    result = asyncio.run(
        normalize_uploaded_file(
            DataUpload(b'data-bytes', name='data.tif'),
            suffix_hint='.oir',
        )
    )

    try:
        assert result.suffix == '.oir'
        assert result.read_bytes() == b'data-bytes'
    finally:
        result.unlink(missing_ok=True)
