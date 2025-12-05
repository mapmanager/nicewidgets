from nicewidgets.custom_ag_grid.config import ColumnConfig, GridConfig
from nicewidgets.custom_ag_grid import CustomAgGrid


def test_select_editor_unique_values() -> None:
    data = [
        {"city": "A"},
        {"city": "B"},
        {"city": "A"},
    ]
    cols = [
        ColumnConfig("city", header="City", editable=True, editor="select", choices="unique")
    ]
    grid_cfg = GridConfig(selection_mode="single")
    grid = CustomAgGrid(data=data, columns=cols, grid_config=grid_cfg)

    # ColumnDefs are stored in custom_ag_grid.grid.options['columnDefs']
    col_defs = grid.grid.options["columnDefs"]
    assert len(col_defs) == 1
    col_def = col_defs[0]

    assert col_def["field"] == "city"
    assert col_def["editable"] is True
    assert ":cellEditor" in col_def
    assert col_def[":cellEditor"] == "'agSelectCellEditor'"

    values = col_def["cellEditorParams"]["values"]
    # Values should be sorted unique values from the data
    assert values == ["A", "B"]


def test_selection_mode_mapping() -> None:
    data = [{"id": 1}, {"id": 2}]
    cols = [ColumnConfig("id")]

    for mode, expected in [
        ("none", None),
        ("single", "single"),
        ("multiple", "multiple"),
    ]:
        cfg = GridConfig(selection_mode=mode)  # type: ignore[arg-type]
        grid = CustomAgGrid(data=data, columns=cols, grid_config=cfg)
        opts = grid.grid.options
        if expected is None:
            assert "rowSelection" not in opts
        else:
            assert opts["rowSelection"] == expected
