"""Test that custom_ag_grid works without matplotlib."""
import sys

import pytest


def test_imports_without_matplotlib():
    """Verify custom_ag_grid imports work when matplotlib is not available.
    
    This test simulates an environment without matplotlib by removing it
    from sys.modules before importing. It verifies that the imports used
    by kymflow work correctly without matplotlib.
    """
    # Remove matplotlib if already imported (e.g., by other tests)
    matplotlib_modules = [k for k in sys.modules.keys() if k.startswith('matplotlib')]
    for mod in matplotlib_modules:
        del sys.modules[mod]
    
    # Now import what kymflow uses
    from nicewidgets.custom_ag_grid.config import ColumnConfig, GridConfig, SelectionMode
    from nicewidgets.custom_ag_grid.custom_ag_grid_v2 import CustomAgGrid_v2
    
    # Verify classes are usable (not None)
    assert ColumnConfig is not None
    assert GridConfig is not None
    assert CustomAgGrid_v2 is not None
    
    # Verify we can instantiate config objects
    col = ColumnConfig(field="test")
    assert col.field == "test"
    
    cfg = GridConfig()
    assert cfg.selection_mode in ("none", "single", "multiple")
    
    # Verify matplotlib is still not available
    with pytest.raises(ImportError):
        import matplotlib  # type: ignore[import-untyped]