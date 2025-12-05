import pytest

from nicewidgets.custom_ag_grid.config import ColumnConfig, GridConfig
from nicewidgets.custom_ag_grid import CustomAgGrid


def test_convert_list_of_dicts() -> None:
    data = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]
    cols = [ColumnConfig("id"), ColumnConfig("name", editable=True)]
    grid = CustomAgGrid(data=data, columns=cols)

    rows = grid.to_rows()
    assert len(rows) == 2
    assert rows[0]["id"] == 1
    assert rows[1]["name"] == "Bob"


def test_grid_config_defaults() -> None:
    cfg = GridConfig()
    assert cfg.selection_mode == "none"
    assert cfg.zebra_rows is True
    assert cfg.hover_highlight is True
    assert cfg.tight_layout is True
