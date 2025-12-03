import pytest

from nicewidgets.custom_ag_grid.config import ColumnConfig
from nicewidgets.custom_ag_grid.grid import CustomAgGrid, HAS_PANDAS, HAS_POLARS


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
def test_convert_from_pandas() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {"id": [1, 2], "name": ["Alice", "Bob"], "city": ["A", "B"]}
    )
    cols = [ColumnConfig("id"), ColumnConfig("name"), ColumnConfig("city")]
    grid = CustomAgGrid(data=df, columns=cols)

    rows = grid.to_rows()
    assert rows == df.to_dict(orient="records")
    df2 = grid.to_pandas()
    assert list(df2.columns) == ["id", "name", "city"]
    assert len(df2) == 2


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_convert_from_polars() -> None:
    import polars as pl

    df = pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
    cols = [ColumnConfig("id"), ColumnConfig("name")]
    grid = CustomAgGrid(data=df, columns=cols)

    rows = grid.to_rows()
    assert rows == df.to_dicts()
    df2 = grid.to_polars()
    assert df2.shape == (2, 2)
