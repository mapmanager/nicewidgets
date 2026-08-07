"""Tests for public raster-viewer presentation configuration."""

import pytest

from nicewidgets.raster_viewer_widget.config import RasterViewerConfig


def test_wheel_zoom_factor_defaults_to_slower_zoom() -> None:
    """Verify the public default replaces the original aggressive factor."""
    config = RasterViewerConfig()
    assert config.wheel_zoom_factor == 1.06
    assert config.channel_toolbars_visible is True
    assert config.roi_toolbar_visible is True
    assert config.roi_chrome_enabled is True
    assert RasterViewerConfig(roi_chrome_enabled=False).roi_chrome_enabled is False
    assert config.roi_host_mode.value == "local"


@pytest.mark.parametrize("value", [1.0, 0.9, 2.01])
def test_wheel_zoom_factor_rejects_values_outside_safe_bounds(value: float) -> None:
    """Verify invalid multiplicative zoom factors fail during configuration."""
    with pytest.raises(ValueError, match="wheel_zoom_factor"):
        RasterViewerConfig(wheel_zoom_factor=value)
