# nicewidgets/src/nicewidgets/upload_widget/tutorials/demo_upload_multi.py
"""Tutorial: Upload multiple files (batch semantics) and print their normalized paths.

Run:
    uv run python nicewidgets/src/nicewidgets/upload_widget/tutorials/demo_upload_multi.py
"""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from nicewidgets.upload_widget import UploadWidget, CancelToken
from nicewidgets.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def on_progress(p: float, msg: str | None) -> None:
    logger.info("progress %.2f %s", p, msg or "")
    print(f"[progress] {p:.2f} {msg or ''}")


async def on_paths_ready(paths: list[Path], cancel: CancelToken) -> None:
    logger.info("batch ready: %d file(s)", len(paths))
    print(f"[demo] received batch with {len(paths)} file(s):")
    for p in paths:
        print(f"  - {p} exists={p.exists()}")
    if cancel.cancelled:
        return
    ui.notify(f"Received {len(paths)} file(s)")


def main() -> None:
    configure_logging(level="INFO")
    ui.page_title("UploadWidget demo: multi-file (batch)")
    ui.label("UploadWidget demo: multi-file (batch)").classes("text-lg font-semibold")

    UploadWidget(
        label="Upload files",
        accept=".tif,.tiff,.txt,.csv,.json",
        multiple=True,
        max_files=20,
        on_paths_ready=on_paths_ready,
        on_progress=on_progress,
    )

    ui.run(native=False, reload=False)


if __name__ == "__main__":
    main()
