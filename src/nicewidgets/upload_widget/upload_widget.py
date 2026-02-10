# nicewidgets/src/nicewidgets/upload_widget/upload_widget.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Optional

from nicegui import ui

from nicewidgets.upload_widget.normalize import normalize_uploaded_file, safe_upload_event_summary
from nicewidgets.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CancelToken:
    """Cooperative cancellation token for post-upload processing."""
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


# Public type aliases (stable API)
OnProgress = Callable[[float, Optional[str]], None]
OnPathsReady = Callable[[List[Path], CancelToken], Awaitable[None]]


class UploadWidget:
    """Reusable NiceGUI upload widget with robust batch semantics and normalization."""

    def __init__(
        self,
        *,
        label: str,
        accept: str,
        on_paths_ready: OnPathsReady,
        multiple: bool = True,
        max_files: int = 20,
        on_progress: Optional[OnProgress] = None,
        fallback_batch_debounce_sec: Optional[float] = 0.25,
    ) -> None:
        self._label = label
        self._accept = accept
        self._multiple = multiple
        self._max_files = max_files
        self._on_paths_ready = on_paths_ready
        self._on_progress = on_progress
        self._fallback_batch_debounce_sec = fallback_batch_debounce_sec

        self._cancel = CancelToken(cancelled=False)
        self._pending_upload_files: List[Any] = []
        self._debounce_task: Optional[asyncio.Task[None]] = None

        self._build()

    def _build(self) -> None:
        """Build the NiceGUI UI. Must be called within a NiceGUI slot."""
        self._status = ui.label("").classes("text-sm text-gray-600")
        self._spinner = ui.spinner(size="lg").classes("mt-2")
        self._spinner.visible = False

        with ui.row().classes("items-center gap-2"):
            ui.button("Cancel processing", on_click=self.cancel).props("outline")

        self._upload = ui.upload(
            label=self._label,
            auto_upload=True,
            multiple=self._multiple,
            on_upload=self._on_upload_one,
            on_multi_upload=self._on_upload_batch_done,
        ).props(
            f'accept="{self._accept}" max-files="{self._max_files}"'
        ).classes("w-full")

    def cancel(self) -> None:
        self._cancel.cancel()
        self._status.text = "Cancelled"
        self._spinner.visible = False
        logger.info("cancel requested")

    def _progress(self, p: float, msg: str | None) -> None:
        if msg:
            self._status.text = msg
        if self._on_progress is not None:
            try:
                self._on_progress(float(p), msg)
            except Exception:
                logger.exception("on_progress callback failed")

    async def _normalize_batch(self, upload_files: List[Any]) -> List[Path]:
        paths: List[Path] = []
        n = len(upload_files)

        for i, f in enumerate(upload_files, start=1):
            if self._cancel.cancelled:
                return paths

            self._progress(0.20, f"Normalizing file {i}/{n}")

            # Force suffix preservation using the original filename when available.
            suffix_hint: str | None = None
            name = getattr(f, "name", None)
            if isinstance(name, str) and name:
                suf = Path(name).suffix
                if suf:
                    suffix_hint = suf

            try:
                p = await normalize_uploaded_file(f, suffix_hint=suffix_hint)
                paths.append(p)
                logger.debug("normalized %s -> %s", getattr(f, "name", "<unnamed>"), p)
            except Exception:
                logger.exception(
                    "upload normalize failed: %s",
                    safe_upload_event_summary(type("E", (), {"sender": None, "file": f})()),
                )

        return paths

    def _cancel_debounce_task(self) -> None:
        t = self._debounce_task
        self._debounce_task = None
        if t is not None and not t.done():
            t.cancel()

    def _schedule_debounce_flush(self) -> None:
        if self._fallback_batch_debounce_sec is None:
            return

        self._cancel_debounce_task()

        async def _flush_later() -> None:
            try:
                await asyncio.sleep(self._fallback_batch_debounce_sec)
                if self._pending_upload_files:
                    logger.info(
                        "debounce flush: %d pending file(s)",
                        len(self._pending_upload_files),
                    )
                    await self._flush_pending_as_batch(reason="debounce")
            except asyncio.CancelledError:
                return

        self._debounce_task = asyncio.create_task(_flush_later())

    async def _flush_pending_as_batch(self, *, reason: str) -> None:
        upload_files = list(self._pending_upload_files)
        self._pending_upload_files.clear()

        if not upload_files:
            logger.debug("flush (%s): no pending upload files", reason)
            return

        self._spinner.visible = True
        self._progress(0.00, "Upload received")

        paths = await self._normalize_batch(upload_files)

        if self._cancel.cancelled:
            self._spinner.visible = False
            logger.info("flush (%s): cancelled", reason)
            return

        try:
            await self._on_paths_ready(paths, self._cancel)
        except Exception:
            logger.exception("on_paths_ready failed")
        finally:
            self._progress(1.00, "Done")
            self._spinner.visible = False

    async def _on_upload_one(self, e: Any) -> None:
        if self._cancel.cancelled:
            return

        upload_file = getattr(e, "file", None)
        if upload_file is None:
            logger.warning("on_upload called with no file: %s", safe_upload_event_summary(e))
            return

        self._pending_upload_files.append(upload_file)
        logger.debug(
            "received file (%d pending): %s",
            len(self._pending_upload_files),
            getattr(upload_file, "name", "<unnamed>"),
        )

        self._schedule_debounce_flush()

    async def _on_upload_batch_done(self, e: Any) -> None:
        if self._cancel.cancelled:
            return

        self._cancel_debounce_task()

        f = getattr(e, "file", None)

        if isinstance(f, list) and f:
            upload_files = f
            self._pending_upload_files.clear()
            logger.info("upload batch: %d file(s) (from on_multi_upload)", len(upload_files))
        else:
            upload_files = list(self._pending_upload_files)
            self._pending_upload_files.clear()
            logger.info("upload batch: %d file(s) (from pending)", len(upload_files))

        if not upload_files:
            logger.debug("batch boundary: empty batch (ignoring)")
            return

        self._spinner.visible = True
        self._progress(0.00, "Upload received")

        paths = await self._normalize_batch(upload_files)

        if self._cancel.cancelled:
            self._spinner.visible = False
            logger.info("batch: cancelled")
            return

        try:
            await self._on_paths_ready(paths, self._cancel)
        except Exception:
            logger.exception("on_paths_ready failed")
        finally:
            self._progress(1.00, "Done")
            self._spinner.visible = False