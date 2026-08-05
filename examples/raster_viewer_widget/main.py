"""Executable entry point for the NiceGUI raster-viewer development app."""

from __future__ import annotations

import logging
import os
from enum import StrEnum
from pathlib import Path

from nicegui import ui
from nicegui.helpers.network import find_free_port, is_port_open

from examples.raster_viewer_widget.gui_defaults import set_up_gui_defaults
from examples.raster_viewer_widget.logging_config import configure_logging
from examples.raster_viewer_widget.multi_widget_demo import MultiWidgetDemo
from examples.raster_viewer_widget.raster_demo import RasterViewerDemo

LOGGER = logging.getLogger(__name__)
RUN_MODE_ENVIRONMENT_VARIABLE = "RASTER_VIEWER_RUN_MODE"


class DemoRunMode(StrEnum):
    """Supported NiceGUI presentation modes."""

    WEB = "web"
    APP = "app"

    @classmethod
    def from_environment(cls) -> DemoRunMode:
        """Read the requested presentation mode from the environment.

        Returns:
            Configured mode, defaulting to web.

        Raises:
            ValueError: If the environment value is not ``web`` or ``app``.
        """
        raw_value = os.environ.get(RUN_MODE_ENVIRONMENT_VARIABLE, cls.WEB.value)
        try:
            return cls(raw_value)
        except ValueError as error:
            raise ValueError(
                f"{RUN_MODE_ENVIRONMENT_VARIABLE} must be 'web' or 'app', got {raw_value!r}"
            ) from error


def select_demo_port(preferred_port: int = 8080) -> int:
    """Select an available TCP port for the NiceGUI development server.

    Args:
        preferred_port: Familiar port to use when no local process is listening.

    Returns:
        The preferred port when available, otherwise a free ephemeral port
        selected by NiceGUI's network utility.

    Raises:
        ValueError: If ``preferred_port`` is outside the valid TCP port range.
    """
    if not 1 <= preferred_port <= 65535:
        raise ValueError("preferred_port must be between 1 and 65535")
    if not is_port_open("127.0.0.1", preferred_port):
        return preferred_port
    selected_port = find_free_port()
    LOGGER.warning(
        "Preferred port %d is in use; selected available port %d",
        preferred_port,
        selected_port,
    )
    return selected_port


def main() -> None:
    """Create and run the local NiceGUI development application."""
    configure_logging()
    set_up_gui_defaults()
    mode = DemoRunMode.from_environment()
    example_root = Path(__file__).resolve().parent
    demo = RasterViewerDemo(example_root)
    multi_widget_demo = MultiWidgetDemo(demo.datasets)
    ui.page("/multi-widget")(multi_widget_demo.build_page)
    if mode is DemoRunMode.APP:
        LOGGER.info("Starting Raster Viewer JS in native app mode")
        ui.run(
            root=demo.build_page,
            title="Raster Viewer JS",
            native=True,
            window_size=(1024, 1024),
            reload=False,
        )
        return
    port = select_demo_port()
    LOGGER.info("Starting Raster Viewer JS in web mode at http://localhost:%d", port)
    ui.run(
        root=demo.build_page,
        title="Raster Viewer JS",
        native=False,
        port=port,
        reload=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
