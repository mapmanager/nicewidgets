from __future__ import annotations

from unittest.mock import MagicMock, patch

from nicegui import ui
from nicewidgets.custom_ag_grid.config import ColumnConfig, GridConfig
from nicewidgets.custom_ag_grid.custom_ag_grid_v2 import CustomAgGrid_v2


def _create_grid() -> CustomAgGrid_v2:
    rows = [{"id": f"row_{i:03d}", "name": f"Item {i}", "value": i * 10} for i in range(1, 5)]
    cols = [
        ColumnConfig("id", editable=False),
        ColumnConfig("name", editable=True),
        ColumnConfig("value", editable=True),
    ]
    cfg = GridConfig(selection_mode="single", row_id_field="id")
    # Use a simple container to host the grid; NiceGUI will create an app context
    with ui.row():
        grid = CustomAgGrid_v2(rows, cols, cfg)
    return grid


def test_update_row_updates_single_row_and_internal_cache() -> None:
    """Ensure CustomAgGrid_v2.update_row patches only the targeted row."""
    grid = _create_grid()
    original_rows = grid.rows
    assert len(original_rows) == 4

    target_id = original_rows[1]["id"]
    updated = dict(original_rows[1])
    updated["name"] = "Updated Name"
    updated["value"] = 999

    grid.update_row(target_id, updated)

    new_rows = grid.rows
    assert len(new_rows) == 4
    # Only the targeted row should differ
    for i, (before, after) in enumerate(zip(original_rows, new_rows)):
        if i == 1:
            assert after["id"] == target_id
            assert after["name"] == "Updated Name"
            assert after["value"] == 999
        else:
            assert before == after


def test_update_row_calls_run_row_method() -> None:
    """Test that update_row calls run_row_method with correct parameters."""
    grid = _create_grid()
    original_rows = grid.rows
    
    target_id = original_rows[1]["id"]
    updated = dict(original_rows[1])
    updated["name"] = "Updated Name"
    
    # Mock run_row_method to verify it's called
    with patch.object(grid._grid, "run_row_method") as mock_run:
        grid.update_row(target_id, updated)
        
        # Verify run_row_method was called with correct parameters
        mock_run.assert_called_once_with(target_id, "setData", updated)


def test_update_row_handles_missing_row_id() -> None:
    """Test that update_row handles missing row_id gracefully."""
    grid = _create_grid()
    original_rows = grid.rows.copy()
    
    # Try to update non-existent row
    updated = {"id": "nonexistent", "name": "Updated", "value": 999}
    
    # Should not crash
    grid.update_row("nonexistent", updated)
    
    # Rows should be unchanged
    assert grid.rows == original_rows


def test_update_row_handles_deleted_grid() -> None:
    """Test that update_row handles deleted grid gracefully."""
    grid = _create_grid()
    original_rows = grid.rows.copy()
    
    target_id = original_rows[1]["id"]
    updated = dict(original_rows[1])
    updated["name"] = "Updated Name"
    
    # Simulate grid deletion by raising RuntimeError with "deleted" message
    with patch.object(grid._grid, "run_row_method", side_effect=RuntimeError("Grid deleted")):
        # Should not raise exception
        grid.update_row(target_id, updated)
        
        # Internal cache should still be updated
        assert grid.rows[1]["name"] == "Updated Name"


def test_update_row_preserves_other_rows() -> None:
    """Test that update_row only changes the targeted row."""
    grid = _create_grid()
    original_rows = [dict(r) for r in grid.rows]  # Deep copy
    
    target_id = original_rows[2]["id"]
    updated = dict(original_rows[2])
    updated["name"] = "Updated Name"
    updated["value"] = 999
    
    grid.update_row(target_id, updated)
    
    new_rows = grid.rows
    # Verify only row 2 changed
    for i, (before, after) in enumerate(zip(original_rows, new_rows)):
        if i == 2:
            assert after["name"] == "Updated Name"
            assert after["value"] == 999
        else:
            assert before == after


def test_update_row_does_not_call_set_data() -> None:
    """Test that update_row does not call set_data."""
    grid = _create_grid()
    original_rows = grid.rows.copy()
    
    target_id = original_rows[1]["id"]
    updated = dict(original_rows[1])
    updated["name"] = "Updated Name"
    
    # Mock set_data to raise exception if called
    grid.set_data = MagicMock(side_effect=Exception("set_data should not be called"))
    
    # Should not raise exception (proves set_data not called)
    grid.update_row(target_id, updated)
    
    # Verify set_data was not called
    grid.set_data.assert_not_called()
    
    # Verify row was updated via update_row
    assert grid.rows[1]["name"] == "Updated Name"

