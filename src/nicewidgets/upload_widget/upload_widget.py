"""Reusable NiceGUI upload widget with normalized path callbacks."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nicegui import ui

from nicewidgets.upload_widget.normalize import normalize_uploaded_file, safe_upload_event_summary
from nicewidgets.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CancelToken:
    """Cooperative cancellation token for post-upload processing.

    Attributes:
        cancelled: Whether cancellation has been requested.
    """

    cancelled: bool = False

    def cancel(self) -> None:
        """Request cooperative cancellation.

        Returns:
            None.
        """
        self.cancelled = True


OnProgress = Callable[[float, str | None], None]
OnPathsReady = Callable[[list[Path], CancelToken], Awaitable[None]]


class UploadWidget:
    """Reusable NiceGUI upload widget with batch semantics and path normalization.

    Args:
        label: User-facing upload label.
        accept: Browser ``accept`` attribute, such as ``".tif,.oir"``.
        on_paths_ready: Async callback invoked once with normalized paths after a
            file selection/drop completes.
        multiple: Whether the upload input accepts multiple files.
        max_files: Browser-side maximum file count.
        on_progress: Optional progress callback receiving ``(fraction, message)``.
        fallback_batch_debounce_sec: Optional debounce used when NiceGUI does not
            emit an ``on_multi_upload`` batch boundary.
        show_inline_status: When True, render an inline status label and spinner
            beside the uploader. When False, callers are expected to drive their
            own progress UI from ``on_progress``.
        extra_props: Additional Quasar props appended to the inner ``ui.upload``
            (e.g. ``'flat dense bordered hide-upload-btn'``).
        extra_classes: Additional CSS classes applied to the inner ``ui.upload``.
        reset_after_batch: When True, call the inner ``ui.upload.reset()`` after
            ``on_paths_ready`` returns to clear the queue UI for repeat use.
    """

    def __init__(
        self,
        *,
        label: str,
        accept: str,
        on_paths_ready: OnPathsReady,
        multiple: bool = False,
        max_files: int = 1,
        on_progress: OnProgress | None = None,
        fallback_batch_debounce_sec: float | None = 0.25,
        show_inline_status: bool = True,
        extra_props: str = '',
        extra_classes: str = '',
        reset_after_batch: bool = True,
    ) -> None:
        self._label = label
        self._accept = accept
        self._multiple = bool(multiple)
        self._max_files = int(max_files)
        self._on_paths_ready = on_paths_ready
        self._on_progress = on_progress
        self._fallback_batch_debounce_sec = fallback_batch_debounce_sec
        self._show_inline_status = bool(show_inline_status)
        self._extra_props = extra_props
        self._extra_classes = extra_classes
        self._reset_after_batch = bool(reset_after_batch)

        self._cancel = CancelToken(cancelled=False)
        self._pending_upload_files: list[Any] = []
        self._original_names_by_path: dict[str, str] = {}
        self._debounce_task: asyncio.Task[None] | None = None
        self._status: ui.label | None = None
        self._spinner: ui.spinner | None = None

        self._build()

    def get_original_filename(self, path: Path) -> str:
        """Return the upload event filename associated with a normalized path.

        Args:
            path: Normalized path previously supplied to ``on_paths_ready``.

        Returns:
            Original upload filename when known, otherwise ``path.name``.
        """
        return self._original_names_by_path.get(str(path), path.name)

    def cancel(self) -> None:
        """Request cancellation of pending post-upload processing.

        Returns:
            None.
        """
        self._cancel.cancel()
        if self._status is not None:
            self._status.text = 'Cancelled'
        if self._spinner is not None:
            self._spinner.visible = False
        logger.info('upload cancel requested')

    def reset_cancel(self) -> None:
        """Clear any prior cancellation so the widget can accept another upload.

        Returns:
            None.
        """
        self._cancel = CancelToken(cancelled=False)

    def _build(self) -> None:
        """Build the NiceGUI upload controls in the current slot."""
        if self._show_inline_status:
            self._status = ui.label('').classes('text-sm text-gray-600')
            self._spinner = ui.spinner(size='lg').classes('mt-2')
            self._spinner.visible = False

        props = f'accept="{self._accept}" max-files="{self._max_files}"'
        if self._extra_props:
            props = f'{props} {self._extra_props}'

        upload = ui.upload(
            label=self._label,
            auto_upload=True,
            multiple=self._multiple,
            on_upload=self._on_upload_one,
            on_multi_upload=self._on_upload_batch_done,
        ).props(props).classes('w-full')
        if self._extra_classes:
            upload = upload.classes(self._extra_classes)
        self._upload = upload

    def _set_spinner_visible(self, visible: bool) -> None:
        if self._spinner is not None:
            self._spinner.visible = visible

    def _progress(self, progress: float, message: str | None) -> None:
        if message and self._status is not None:
            self._status.text = message
        if self._on_progress is None:
            return
        try:
            self._on_progress(float(progress), message)
        except Exception:
            logger.exception('on_progress callback failed')

    def _reset_inner_upload(self) -> None:
        if not self._reset_after_batch:
            return
        try:
            self._upload.reset()
        except Exception:
            logger.debug('inner upload.reset() failed', exc_info=True)

    async def _normalize_batch(self, upload_files: list[Any]) -> list[Path]:
        paths: list[Path] = []
        total = len(upload_files)
        for index, upload_file in enumerate(upload_files, start=1):
            if self._cancel.cancelled:
                return paths

            self._progress(0.20, f'Normalizing file {index}/{total}')
            suffix_hint = _suffix_hint_from_name(getattr(upload_file, 'name', None))

            try:
                path = await normalize_uploaded_file(upload_file, suffix_hint=suffix_hint)
                paths.append(path)
                original_name = getattr(upload_file, 'name', None)
                if isinstance(original_name, str) and original_name:
                    self._original_names_by_path[str(path)] = original_name
                logger.debug('normalized upload %s -> %s', getattr(upload_file, 'name', '<unnamed>'), path)
            except Exception:
                logger.exception(
                    'upload normalize failed: %s',
                    safe_upload_event_summary(type('UploadEvent', (), {'sender': None, 'file': upload_file})()),
                )
        return paths

    def _cancel_debounce_task(self) -> None:
        task = self._debounce_task
        self._debounce_task = None
        if task is not None and not task.done():
            task.cancel()

    def _schedule_debounce_flush(self) -> None:
        if self._fallback_batch_debounce_sec is None:
            return

        self._cancel_debounce_task()

        async def _flush_later() -> None:
            try:
                await asyncio.sleep(self._fallback_batch_debounce_sec)
                if self._pending_upload_files:
                    await self._flush_pending_as_batch(reason='debounce')
            except asyncio.CancelledError:
                return

        self._debounce_task = asyncio.create_task(_flush_later())

    async def _flush_pending_as_batch(self, *, reason: str) -> None:
        upload_files = list(self._pending_upload_files)
        self._pending_upload_files.clear()

        if not upload_files:
            logger.debug('upload flush (%s): no pending files', reason)
            return

        self._set_spinner_visible(True)
        self._progress(0.00, 'Upload received')
        try:
            paths = await self._normalize_batch(upload_files)

            if self._cancel.cancelled:
                logger.info('upload flush (%s): cancelled', reason)
                return

            try:
                await self._on_paths_ready(paths, self._cancel)
            except Exception:
                logger.exception('on_paths_ready failed')
        finally:
            terminal_message = 'Cancelled' if self._cancel.cancelled else 'Done'
            self._progress(1.00, terminal_message)
            self._set_spinner_visible(False)
            self._reset_inner_upload()

    async def _on_upload_one(self, event: Any) -> None:
        if self._cancel.cancelled:
            return

        upload_file = getattr(event, 'file', None)
        if upload_file is None:
            logger.warning('on_upload called with no file: %s', safe_upload_event_summary(event))
            return

        self._pending_upload_files.append(upload_file)
        self._schedule_debounce_flush()

    async def _on_upload_batch_done(self, event: Any) -> None:
        if self._cancel.cancelled:
            return

        self._cancel_debounce_task()
        file_payload = getattr(event, 'file', None)

        if isinstance(file_payload, list) and file_payload:
            upload_files = file_payload
            self._pending_upload_files.clear()
        else:
            upload_files = list(self._pending_upload_files)
            self._pending_upload_files.clear()

        if not upload_files:
            logger.debug('upload batch boundary had no files')
            return

        self._set_spinner_visible(True)
        self._progress(0.00, 'Upload received')
        try:
            paths = await self._normalize_batch(upload_files)

            if self._cancel.cancelled:
                logger.info('upload batch cancelled')
                return

            try:
                await self._on_paths_ready(paths, self._cancel)
            except Exception:
                logger.exception('on_paths_ready failed')
        finally:
            terminal_message = 'Cancelled' if self._cancel.cancelled else 'Done'
            self._progress(1.00, terminal_message)
            self._set_spinner_visible(False)
            self._reset_inner_upload()


def _suffix_hint_from_name(name: object) -> str | None:
    if not isinstance(name, str) or not name:
        return None
    suffix = Path(name).suffix
    return suffix or None
