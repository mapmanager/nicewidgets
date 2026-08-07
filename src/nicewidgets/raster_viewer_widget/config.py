"""Initial presentation configuration for the NiceGUI raster viewer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ViewerTheme(StrEnum):
    """Supported viewer chrome themes."""

    DARK = "dark"
    LIGHT = "light"


class ViewerLayout(StrEnum):
    """Supported initial multi-channel layouts."""

    AUTO = "auto"
    SIDE = "side"
    STACK = "stack"
    SINGLE = "single"
    COMPOSITE = "composite"


class RoiHostMode(StrEnum):
    """Whether JS ROI chrome mutates locally or only emits host requests.

    Attributes:
        LOCAL: Add/delete/edit chrome mutates the in-viewer ROI list (demo/mock).
        DELEGATED: Chrome emits request events; the host validates and applies
            silent ``viewer.rois`` commands.
    """

    LOCAL = "local"
    DELEGATED = "delegated"


@dataclass(frozen=True, slots=True)
class RasterViewerConfig:
    """Configure initial viewer presentation without owning dataset state.

    Attributes:
        theme: Initial light or dark viewer chrome theme.
        layout: Initial channel layout; ``auto`` preserves viewer defaults.
        axes_visible: Whether image axes are initially visible.
        rois_visible: Whether ROI overlays are initially visible. Ignored when
            ``roi_chrome_enabled`` is false (overlays stay off).
        channel_toolbars_visible: Whether each pane initially shows its header
            toolbar containing channel controls and Copy view.
        roi_toolbar_visible: Whether the top-toolbar ROI strip (dropdown + CRUD)
            is initially visible. Ignored when ``roi_chrome_enabled`` is false.
        roi_chrome_enabled: When false, ROI overlays and ROI CRUD chrome are
            unavailable (no strip, no ROIs / ROI Toolbar options toggles). Use
            for display-only hosts such as a reference-image viewer.
        roi_host_mode: Local mock mutation vs host-delegated request callbacks.
        invert_slice_wheel: Whether Alt/Option+wheel up moves toward plane zero
            and wheel down moves toward the maximum. It targets Z when present,
            otherwise T, and does not change ordinary wheel zoom.
        wheel_zoom_factor: Multiplicative zoom applied for each wheel event.
            Values closer to ``1.0`` zoom more slowly; for example, use ``1.03``
            for gentler zoom. Valid values are greater than ``1.0`` and at most
            ``2.0``. The default ``1.06`` is intentionally gentler than the
            viewer's original hard-coded ``1.12``.
    """

    theme: ViewerTheme = ViewerTheme.DARK
    layout: ViewerLayout = ViewerLayout.AUTO
    axes_visible: bool = True
    rois_visible: bool = True
    channel_toolbars_visible: bool = True
    roi_toolbar_visible: bool = True
    roi_chrome_enabled: bool = True
    roi_host_mode: RoiHostMode = RoiHostMode.LOCAL
    invert_slice_wheel: bool = True
    wheel_zoom_factor: float = 1.06

    def __post_init__(self) -> None:
        """Validate numeric interaction configuration."""
        if not 1.0 < self.wheel_zoom_factor <= 2.0:
            raise ValueError("wheel_zoom_factor must be greater than 1.0 and at most 2.0")
