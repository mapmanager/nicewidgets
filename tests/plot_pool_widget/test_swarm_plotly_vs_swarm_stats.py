"""Unit test: compare Plotly swarm figure y-values with pandas groupby (first principles).

Steps:
1. Synthetic df with minimal swarm columns (from radon_report_db.csv style)
2. Simple PlotState (ycol, group_col, no color_grouping)
3. Call _figure_swarm() to generate plotly dict
4. From first principles: df.groupby(state.group_col)[state.ycol] with same transforms as plot
5. Assert plotly dict y-values per group match pandas ground truth (as multisets)

Note: plotly fig.to_dict() uses binary serialization (bdata/dtype) for arrays;
we decode that to extract values.
"""

from __future__ import annotations

import base64
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from nicewidgets.plot_pool_widget.dataframe_processor import DataFrameProcessor
from nicewidgets.plot_pool_widget.figure_generator import FigureGenerator
from nicewidgets.plot_pool_widget.plot_state import PlotState, PlotType
from nicewidgets.plot_pool_widget.pre_filter_conventions import PRE_FILTER_NONE


# -----------------------------------------------------------------------------
# 1. Synthetic df (minimal columns for swarm, from radon_report_db.csv style)
# -----------------------------------------------------------------------------

# Radon columns: roi_id, vel_mean, event_type-like (grandparent_folder), path
# Minimal for swarm: roi_id (pre_filter), group_col, ycol, unique_row_id

def make_synthetic_swarm_df(
    n_per_group: dict[str, int],
    *,
    ycol: str = "vel_mean",
    group_col: str = "event_type",
    pre_filter_col: str = "roi_id",
    unique_row_id_col: str = "path",
    seed: int = 42,
) -> pd.DataFrame:
    """Build synthetic df with minimal swarm columns (radon_report_db.csv style)."""
    np.random.seed(seed)
    rows = []
    row_id = 0
    for group_val, n in n_per_group.items():
        for _ in range(n):
            rows.append({
                pre_filter_col: 1,
                group_col: group_val,
                ycol: 10.0 + row_id * 0.1 + np.random.rand() * 0.01,
                unique_row_id_col: f"p{row_id}",
            })
            row_id += 1
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotly binary decode (fig.to_dict uses bdata/dtype, not plain lists)
# -----------------------------------------------------------------------------

def _decode_plotly_array(obj: Any) -> np.ndarray:
    """Decode plotly binary serialization (dtype + bdata) if present."""
    if isinstance(obj, dict) and "bdata" in obj and "dtype" in obj:
        b = base64.b64decode(obj["bdata"])
        dtype = np.dtype(obj["dtype"])
        return np.frombuffer(b, dtype=dtype).copy()
    return np.asarray(obj)


def extract_plotly_y_per_group(fig_dict: dict[str, Any]) -> dict[str, list[float]]:
    """Extract y-values per group from plotly figure data.

    Trace: scatter, mode=markers. customdata[:,0] = group, y = values.
    """
    result: dict[str, list[float]] = {}
    for trace in fig_dict.get("data", []):
        if trace.get("type") != "scatter" or trace.get("mode") != "markers":
            continue
        y_raw = trace.get("y")
        cd_raw = trace.get("customdata")
        if y_raw is None or cd_raw is None:
            continue
        y_flat = np.ravel(np.atleast_1d(_decode_plotly_array(y_raw)))
        cd = np.atleast_2d(_decode_plotly_array(cd_raw))
        if cd.shape[0] == 1 and cd.shape[1] > 1:
            cd = cd.T
        if len(y_flat) == 0:
            continue
        n = min(len(y_flat), cd.shape[0])
        for i in range(n):
            g = str(cd[i, 0])
            result.setdefault(g, []).append(float(y_flat[i]))
    return result


# -----------------------------------------------------------------------------
# 4. First principles: pandas groupby with same transforms as plot
# -----------------------------------------------------------------------------

def pandas_y_per_group(
    df_f: pd.DataFrame,
    state: PlotState,
) -> dict[str, list[float]]:
    """From first principles: groupby(group_col), apply same transforms as _figure_swarm.

    Transforms: pd.to_numeric, optional abs, optional remove_values.
    Drops rows with NaN in group_col or y.
    """
    y = pd.to_numeric(df_f[state.ycol], errors="coerce")
    if state.use_absolute_value:
        y = y.abs()
    if state.use_remove_values and state.remove_values_threshold is not None:
        y = y.copy()
        y[(y < -state.remove_values_threshold) | (y > state.remove_values_threshold)] = np.nan

    g = df_f[state.group_col].astype(str)
    tmp = pd.DataFrame({"g": g, "y": y}).dropna(subset=["g", "y"])

    result: dict[str, list[float]] = {}
    for grp_val, sub in tmp.groupby("g", sort=True):
        result[str(grp_val)] = sub["y"].tolist()
    return result


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _assert_multiset_equal(a: list[float], b: list[float], msg: str = "") -> None:
    """Assert same multiset (values and counts, order irrelevant)."""
    a_rounded = [round(v, 10) for v in a if not (isinstance(v, float) and np.isnan(v))]
    b_rounded = [round(v, 10) for v in b if not (isinstance(v, float) and np.isnan(v))]
    a_nan = sum(1 for v in a if isinstance(v, float) and np.isnan(v))
    b_nan = sum(1 for v in b if isinstance(v, float) and np.isnan(v))
    assert a_nan == b_nan, f"{msg} NaN counts differ: {a_nan} vs {b_nan}"
    assert Counter(a_rounded) == Counter(b_rounded), (
        f"{msg} multisets differ: {Counter(a_rounded)} vs {Counter(b_rounded)}"
    )


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------

def test_swarm_plotly_y_values_match_pandas_groupby() -> None:
    """Plotly swarm trace y-values per group equal pandas groupby (first principles)."""
    # 1. Synthetic df
    n_per_group = {"A": 5, "B": 7, "C": 3}
    df = make_synthetic_swarm_df(n_per_group)

    group_col = "event_type"
    ycol = "vel_mean"
    pre_filter_col = "roi_id"
    unique_row_id_col = "path"

    # 2. Testing PlotState (simple: ycol, group_col, no color_grouping)
    state = PlotState(
        pre_filter={pre_filter_col: PRE_FILTER_NONE},
        xcol=group_col,
        ycol=ycol,
        plot_type=PlotType.SWARM,
        group_col=group_col,
        color_grouping=None,
        show_raw=True,
    )

    # Filter df (same as plot pipeline)
    processor = DataFrameProcessor(
        df,
        pre_filter_columns=[pre_filter_col],
        unique_row_id_col=unique_row_id_col,
    )
    df_f = processor.filter_by_pre_filters(state.pre_filter)

    # 3. Call _figure_swarm
    gen = FigureGenerator(processor, unique_row_id_col=unique_row_id_col)
    fig_dict, _ = gen._figure_swarm(df_f, state)

    # 4. First principles: pandas groupby with same transforms
    pandas_per_group = pandas_y_per_group(df_f, state)

    # 5. Extract plotly values and compare
    plotly_per_group = extract_plotly_y_per_group(fig_dict)

    assert set(plotly_per_group.keys()) == set(pandas_per_group.keys()), (
        f"Group keys differ: plotly={set(plotly_per_group.keys())} vs pandas={set(pandas_per_group.keys())}"
    )
    for g in pandas_per_group.keys():
        _assert_multiset_equal(
            plotly_per_group[g],
            pandas_per_group[g],
            msg=f"Group {g!r}: ",
        )
