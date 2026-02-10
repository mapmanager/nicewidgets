from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

import pytest

import nicewidgets.upload_widget.upload_widget as uw_mod
from nicewidgets.upload_widget.upload_widget import UploadWidget

pytestmark = pytest.mark.requires_nicegui


class _FakeElement:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.visible: bool = True

    def classes(self, *_args: Any, **_kwargs: Any) -> "_FakeElement":
        return self

    def props(self, *_args: Any, **_kwargs: Any) -> "_FakeElement":
        return self


class _FakeDialog(_FakeElement):
    def __enter__(self) -> "_FakeDialog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def open(self) -> None:
        self.visible = True

    def close(self) -> None:
        self.visible = False


class _FakeCard(_FakeElement):
    def __enter__(self) -> "_FakeCard":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeRow(_FakeElement):
    def __enter__(self) -> "_FakeRow":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeUploadControl(_FakeElement):
    def __init__(
        self,
        *,
        label: str,
        auto_upload: bool,
        multiple: bool,
        on_upload: Callable[..., Any],
        on_multi_upload: Callable[..., Any],
    ) -> None:
        super().__init__(text=label)
        self.label = label
        self.auto_upload = auto_upload
        self.multiple = multiple
        self.on_upload = on_upload
        self.on_multi_upload = on_multi_upload
        self._props_string: str = ""

    def props(self, s: str) -> "_FakeUploadControl":
        self._props_string = s
        return self

    def classes(self, *_args: Any, **_kwargs: Any) -> "_FakeUploadControl":
        return self


class _FakeUI:
    def __init__(self) -> None:
        self.last_upload: Optional[_FakeUploadControl] = None

    def label(self, text: str) -> _FakeElement:
        return _FakeElement(text=text)

    def spinner(self, size: str = "lg") -> _FakeElement:
        return _FakeElement(text=f"spinner:{size}")

    def row(self) -> _FakeRow:
        return _FakeRow()

    def button(self, text: str, on_click: Callable[..., Any]) -> _FakeElement:
        el = _FakeElement(text=text)
        el._on_click = on_click  # type: ignore[attr-defined]
        return el

    def dialog(self) -> _FakeDialog:
        return _FakeDialog()

    def card(self) -> _FakeCard:
        return _FakeCard()

    def upload(
        self,
        *,
        label: str,
        auto_upload: bool,
        multiple: bool,
        on_upload: Callable[..., Any],
        on_multi_upload: Callable[..., Any],
    ) -> _FakeUploadControl:
        ctrl = _FakeUploadControl(
            label=label,
            auto_upload=auto_upload,
            multiple=multiple,
            on_upload=on_upload,
            on_multi_upload=on_multi_upload,
        )
        self.last_upload = ctrl
        return ctrl


@pytest.fixture()
def headless_widget_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _FakeUI:
    fake_ui = _FakeUI()
    monkeypatch.setattr(uw_mod, "ui", fake_ui, raising=True)

    async def _normalize_stub(upload_file: Any, **_kwargs: Any) -> Path:
        name = getattr(upload_file, "name", "unnamed.bin")
        p = tmp_path / name
        p.write_bytes(b"")
        return p

    monkeypatch.setattr(uw_mod, "normalize_uploaded_file", _normalize_stub, raising=True)
    return fake_ui


@dataclass
class _FakeUploadFile:
    name: str


@dataclass
class _FakeUploadEvent:
    sender: Any = None
    file: Any = None


@pytest.mark.asyncio
async def test_widget_wires_callbacks_and_batches(headless_widget_env: _FakeUI) -> None:
    received: List[List[Path]] = []

    async def on_paths_ready(paths: List[Path], _cancel: Any) -> None:
        received.append(paths)

    widget = UploadWidget(
        label="Upload",
        accept=".tif,.tiff",
        on_paths_ready=on_paths_ready,
        multiple=True,
        fallback_batch_debounce_sec=None,
    )

    assert headless_widget_env.last_upload is not None
    assert headless_widget_env.last_upload.on_upload == widget._on_upload_one
    assert headless_widget_env.last_upload.on_multi_upload == widget._on_upload_batch_done

    # simulate per-file events
    await widget._on_upload_one(_FakeUploadEvent(file=_FakeUploadFile(name="a.tif")))
    await widget._on_upload_one(_FakeUploadEvent(file=_FakeUploadFile(name="b.tif")))

    # authoritative batch boundary (no list), flushes pending
    await widget._on_upload_batch_done(_FakeUploadEvent(file=None))

    assert len(received) == 1
    assert [p.name for p in received[0]] == ["a.tif", "b.tif"]


@pytest.mark.asyncio
async def test_widget_debounce_fallback_processes_without_multi_upload(headless_widget_env: _FakeUI) -> None:
    received: List[List[Path]] = []

    async def on_paths_ready(paths: List[Path], _cancel: Any) -> None:
        received.append(paths)

    widget = UploadWidget(
        label="Upload",
        accept=".tif,.tiff",
        on_paths_ready=on_paths_ready,
        multiple=True,
        fallback_batch_debounce_sec=0.05,
    )

    await widget._on_upload_one(_FakeUploadEvent(file=_FakeUploadFile(name="x.tif")))
    await widget._on_upload_one(_FakeUploadEvent(file=_FakeUploadFile(name="y.tif")))

    await asyncio.sleep(0.12)

    assert len(received) == 1
    assert sorted(p.name for p in received[0]) == ["x.tif", "y.tif"]