# tests/upload_widget/test_normalize.py
from __future__ import annotations

from pathlib import Path

import pytest

from nicewidgets.upload_widget.normalize import normalize_uploaded_file


class FakeLargeFileUpload:
    def __init__(self, path: Path) -> None:
        self._path = path
        self.name = path.name
        self.content_type = "application/octet-stream"


class FakeSmallFileUploadSave:
    def __init__(self, data: bytes, name: str = "x.bin") -> None:
        self._data = data
        self.name = name
        self.content_type = "application/octet-stream"

    async def save(self, path: Path) -> None:
        Path(path).write_bytes(self._data)


class FakeSmallFileUploadRead:
    def __init__(self, data: bytes, name: str = "y.bin") -> None:
        self._data = data
        self.name = name
        self.content_type = "application/octet-stream"

    async def read(self) -> bytes:
        return self._data


class FakeSmallFileUploadDataOnly:
    def __init__(self, data: bytes, name: str = "z.bin") -> None:
        self._data = data
        self.name = name
        self.content_type = "application/octet-stream"


@pytest.mark.asyncio
async def test_normalize_largefileupload_returns_existing_path(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("hello")
    up = FakeLargeFileUpload(p)

    out = await normalize_uploaded_file(up)

    assert out == p
    assert out.exists()
    assert out.read_text() == "hello"


@pytest.mark.asyncio
async def test_normalize_smallfileupload_save_writes_temp_file() -> None:
    data = b"abc123"
    up = FakeSmallFileUploadSave(data, name="file.dat")

    out = await normalize_uploaded_file(up)

    assert out.exists()
    assert out.read_bytes() == data
    # temp file should preserve suffix from name
    assert out.suffix == ".dat"


@pytest.mark.asyncio
async def test_normalize_smallfileupload_read_writes_temp_file() -> None:
    data = b"readme"
    up = FakeSmallFileUploadRead(data, name="readme.txt")

    out = await normalize_uploaded_file(up)

    assert out.exists()
    assert out.read_bytes() == data
    assert out.suffix == ".txt"


@pytest.mark.asyncio
async def test_normalize_smallfileupload_dataonly_writes_temp_file() -> None:
    data = b"rawdata"
    up = FakeSmallFileUploadDataOnly(data, name="raw.bin")

    out = await normalize_uploaded_file(up)

    assert out.exists()
    assert out.read_bytes() == data
    assert out.suffix == ".bin"
