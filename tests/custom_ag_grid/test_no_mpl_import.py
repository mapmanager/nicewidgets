import sys

import pytest


def test_imports_without_matplotlib():
    """Verify custom_ag_grid imports do not import matplotlib as a side-effect.

    Matplotlib may be installed in the environment. This test ensures that
    importing custom_ag_grid does not pull matplotlib into sys.modules.
    """
    # Clear any existing matplotlib modules so we can detect new imports.
    matplotlib_modules = [k for k in list(sys.modules.keys()) if k.startswith("matplotlib")]
    for mod in matplotlib_modules:
        del sys.modules[mod]

    # Import what kymflow uses
    from nicewidgets.custom_ag_grid.config import ColumnConfig, GridConfig, SelectionMode
    from nicewidgets.custom_ag_grid.custom_ag_grid_v2 import CustomAgGrid_v2

    assert ColumnConfig is not None
    assert GridConfig is not None
    assert SelectionMode is not None
    assert CustomAgGrid_v2 is not None

    # Instantiate config objects
    col = ColumnConfig(field="test")
    assert col.field == "test"

    cfg = GridConfig()
    assert cfg.selection_mode in ("none", "single", "multiple")

    # The actual requirement: custom_ag_grid should not import matplotlib.
    assert not any(k.startswith("matplotlib") for k in sys.modules.keys())