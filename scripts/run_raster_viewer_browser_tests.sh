#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
uv run --with 'selenium>=4.27,<5' pytest \
  -p nicegui.testing.screen_plugin \
  browser_tests/raster_viewer_widget
