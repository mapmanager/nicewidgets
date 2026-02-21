# nicewidgets/src/nicewidgets/upload_widget/tutorials/demo_upload_tif.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from nicegui import run, ui

from nicewidgets.upload_widget.upload_widget import CancelToken, UploadWidget
from nicewidgets.utils.logging import get_logger, setup_logging

# Avoid duplicated log lines if NiceGUI reloads modules or setup_logging is called elsewhere.
_root = logging.getLogger()
if not _root.handlers:
    setup_logging(level="INFO")

logger = get_logger(__name__)


@dataclass(frozen=True)
class TifStats:
    path: Path
    shape: tuple[int, ...]
    dtype: str
    vmin: float
    vmax: float


def _read_tif_stats(path: Path) -> TifStats:
    """Blocking TIFF read (run via run.io_bound)."""
    import numpy as np
    import tifffile

    arr = tifffile.imread(str(path))
    shape = tuple(int(x) for x in arr.shape)
    dtype = str(arr.dtype)
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    return TifStats(path=path, shape=shape, dtype=dtype, vmin=vmin, vmax=vmax)


def on_progress(p: float, msg: str | None) -> None:
    # Minimal CLI signal.
    if p in (0.0, 1.0):
        logger.info("upload progress %.2f %s", p, msg or "")


async def _try_read_one(path: Path) -> Optional[TifStats]:
    try:
        return await run.io_bound(_read_tif_stats, path)
    except Exception:
        return None


async def on_paths_ready(paths: List[Path], cancel: CancelToken) -> None:
    logger.info("on_paths_ready: %d file(s)", len(paths))
    if not paths:
        stats_label.text = "No files received."
        table.rows = []
        table.update()
        return

    stats_label.text = f"Reading {len(paths)} file(s)…"

    rows: List[dict] = []
    last: Optional[TifStats] = None

    for p in paths:
        if cancel.cancelled:
            stats_label.text = "Cancelled."
            return

        info = await _try_read_one(p)
        if info is None:
            continue

        last = info
        rows.append(
            {
                "name": info.path.name,
                "shape": str(info.shape),
                "dtype": info.dtype,
                "min": f"{info.vmin:g}",
                "max": f"{info.vmax:g}",
            }
        )

    if not rows:
        stats_label.text = "No readable TIFFs in upload."
        ui.notify("No readable TIFFs found", type="warning")
        table.rows = []
        table.update()
        return

    table.rows = rows
    table.update()

    assert last is not None
    stats_label.text = (
        f"Last TIFF: {last.path.name} | shape={last.shape} | dtype={last.dtype} "
        f"| min={last.vmin:g} | max={last.vmax:g}"
    )
    ui.notify(f"Read {len(rows)} TIFF(s)")


def main() -> None:
    ui.page_title("UploadWidget demo: TIFF")

    ui.label("UploadWidget demo: TIFF").classes("text-lg font-semibold")
    ui.label("Install demo deps: uv run pip install tifffile numpy").classes("text-sm text-gray-600")

    global stats_label
    stats_label = ui.label("Drop a .tif/.tiff to see stats here.").classes("text-sm")

    UploadWidget(
        label="Upload .tif/.tiff",
        accept=".tif,.tiff",
        multiple=True,
        max_files=20,
        on_paths_ready=on_paths_ready,
        on_progress=on_progress,
        fallback_batch_debounce_sec=None,
    )

    global table
    table = ui.table(
        columns=[
            {"name": "name", "label": "File", "field": "name"},
            {"name": "shape", "label": "Shape", "field": "shape"},
            {"name": "dtype", "label": "Dtype", "field": "dtype"},
            {"name": "min", "label": "Min", "field": "min"},
            {"name": "max", "label": "Max", "field": "max"},
        ],
        rows=[],
    ).classes("w-full")

    _native = True
    ui.run(native=_native, reload=False)


if __name__ == "__main__":
    main()
