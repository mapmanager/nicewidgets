"""Plot pool app: standalone NiceGUI application for PlotPoolController.

Runs in native, web, or Docker modes via env vars. Uses @ui.page("/") pattern.

Run:
    uv run python -m nicewidgets.plot_pool_app.plot_pool_app

Env vars:
    PLOT_POOL_GUI_NATIVE: 1/0 (default 1)
    PLOT_POOL_GUI_RELOAD: 1/0 (default 0)
    HOST: bind host (default 127.0.0.1 native, 0.0.0.0 web)
    PORT: bind port (default find_open_port native, 8080 web)
"""

from __future__ import annotations

import os
import multiprocessing as mp
from multiprocessing import freeze_support

from nicegui import app, ui

from nicewidgets.utils import setUpGuiDefaults
from nicewidgets.utils.logging import configure_logging, get_logger
from nicewidgets.plot_pool_app import header, schema
from nicewidgets.plot_pool_widget.plot_pool_controller import PlotPoolController

logger = get_logger(__name__)

# Radon CSV only (no switching, no upload) for web deployment
RADON_CSV = "radon_report_db.csv"

configure_logging(level="DEBUG")

STORAGE_SECRET = "nicewidgets-plot-pool-session-secret"


def _env_bool(name: str, default: bool) -> bool:
    """Parse env var as bool; if unset/invalid returns default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    """Parse env var as int; if unset/invalid returns default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def configure_save_on_quit() -> None:
    """Configure native window to confirm before close. Safe to call before ui.run."""
    native = getattr(app, "native", None)
    if native is None:
        return
    native.window_args["confirm_close"] = True


def configure_native_window_args() -> None:
    """Set native window rect to fixed default (100, 100, 1200, 800). Safe to call before ui.run."""
    native = getattr(app, "native", None)
    if native is None:
        return
    x, y, w, h = 100, 100, 1200, 800
    if w < 200 or h < 200:
        return
    native.window_args.update({
        "x": x,
        "y": y,
        "width": w,
        "height": h,
    })


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

@ui.page("/")
def home() -> None:
    """Home page: header + PlotPoolController with radon data only (no CSV switching)."""

    setUpGuiDefaults('text-xs')

    ui.page_title("Plot Pool")

    header.build_plot_pool_header()

    with ui.column().classes("w-full h-screen flex flex-col gap-4 p-4"):
        main_container = ui.column().classes("w-full flex-1 min-h-0 overflow-auto")

        try:
            df = schema.load_csv_for_file(RADON_CSV)
            cfg = schema.get_config_for_csv(RADON_CSV)
            ctrl = PlotPoolController(df, config=cfg)
            ctrl.build(container=main_container)
        except FileNotFoundError:
            with main_container:
                ui.label(f"{RADON_CSV} not found in data directory.").classes(
                    "text-negative"
                )
        except Exception as e:
            logger.exception("Failed to load %s: %s", RADON_CSV, e)
            with main_container:
                ui.label(f"Failed to load: {e}").classes("text-negative")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(*, reload: bool | None = None, native_bool: bool | None = None) -> None:
    """Start the plot pool application.

    Defaults (no env vars, no args):
      - native=True
      - reload=False

    Env vars (used when arg is None):
      - PLOT_POOL_GUI_NATIVE: 1/0
      - PLOT_POOL_GUI_RELOAD: 1/0
      - HOST: bind host
      - PORT: bind port
    """
    native_bool = _env_bool("PLOT_POOL_GUI_NATIVE", True) if native_bool is None else native_bool
    reload = _env_bool("PLOT_POOL_GUI_RELOAD", False) if reload is None else reload

    from nicegui import native as native_module
    if native_bool:
        port = _env_int("PORT", native_module.find_open_port())
    else:
        port = _env_int("PORT", 8080)

    default_host = "127.0.0.1" if native_bool else "0.0.0.0"
    host = os.getenv("HOST", default_host)

    logger.info(
        "Starting Plot Pool app: port=%s reload=%s native=%s",
        port,
        reload,
        native_bool,
    )

    run_kwargs: dict = {
        "host": host,
        "port": port,
        "reload": reload,
        "native": native_bool,
        "storage_secret": STORAGE_SECRET,
        "title": "Plot Pool",
    }
    if native_bool:
        run_kwargs["window_size"] = (1200, 800)
    ui.run(**run_kwargs)


if __name__ == "__main__":
    freeze_support()
    current_process = mp.current_process()
    is_main_process = current_process.name == "MainProcess"

    logger.info(
        "plot_pool_app: __name__=%s process=%s is_main=%s",
        __name__,
        current_process.name,
        is_main_process,
    )

    native_bool = _env_bool("PLOT_POOL_GUI_NATIVE", True)
    if native_bool:
        configure_save_on_quit()
        configure_native_window_args()

    if is_main_process:
        main()
    else:
        logger.debug("Skipping GUI startup in worker process: %s", current_process.name)
