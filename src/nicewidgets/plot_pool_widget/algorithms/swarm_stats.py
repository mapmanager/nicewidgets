"""
Swarm plot statistics algorithm — pure pandas/numpy, self-contained.

Computes summary stats and ragged value table for swarm (strip) plots.
Matches the logic in FigureGenerator._figure_swarm(): groups by group_col,
optionally by color_grouping. Ignores color_grouping when None, "", or "(none)".

Does not use or extend group_plot.py. Intended for box/violin plots as well.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# Sentinel for "no filter" (matches pre_filter_conventions.PRE_FILTER_NONE).
PRE_FILTER_NONE = "(none)"

# Sentinel for "no color grouping" — treat as single group_col only.
COLOR_GROUPING_NONE = ("(none)", "", None)

# Stats columns for the summary table.
STATS_COLUMNS = ["count", "min", "max", "mean", "median", "std", "sem", "cv"]


# -----------------------------------------------------------------------------
# Step 1: Pre-filter the master dataframe
# -----------------------------------------------------------------------------


def filter_by_pre_filters(
    df: pd.DataFrame,
    pre_filter_columns: list[str],
    selections: dict[str, Any],
    unique_row_id_col: str,
) -> pd.DataFrame:
    """
    Filter the master dataframe by pre-filter column selections.

    Same logic as DataFrameProcessor.filter_by_pre_filters().
    For each column in pre_filter_columns, if the selection is not PRE_FILTER_NONE,
    keep only rows where df[col].astype(str) == str(selection). Selections are
    ANDed across columns. Rows with missing unique_row_id_col are then dropped.

    Args:
        df: Master (raw) dataframe.
        pre_filter_columns: Column names used for pre-filtering (e.g. ["roi_id"]).
        selections: Map column name -> selected value. PRE_FILTER_NONE means no filter.
        unique_row_id_col: Column that must be non-null after filtering.

    Returns:
        Filtered dataframe df_f.
    """
    df_f = df.copy()
    for col in pre_filter_columns:
        val = selections.get(col, PRE_FILTER_NONE)
        if val is None or val == PRE_FILTER_NONE:
            continue
        df_f = df_f[df_f[col].astype(str) == str(val)]
    df_f = df_f.dropna(subset=[unique_row_id_col])
    return df_f


# -----------------------------------------------------------------------------
# Step 2: Get y values with optional transformations
# -----------------------------------------------------------------------------


def get_y_values(
    df_f: pd.DataFrame,
    ycol: str,
    use_absolute: bool = False,
    use_remove_values: bool = False,
    remove_values_threshold: Optional[float] = None,
) -> pd.Series:
    """
    Get the y column as numeric series, with optional transformations.

    Same logic as DataFrameProcessor.get_y_values(). Converts column to numeric
    (coerce errors to NaN), optionally applies abs(), and optionally sets values
    outside [-threshold, +threshold] to NaN.
    """
    y = pd.to_numeric(df_f[ycol], errors="coerce")
    if use_absolute:
        y = y.abs()
    if use_remove_values and remove_values_threshold is not None:
        y = y.copy()
        y[(y < -remove_values_threshold) | (y > remove_values_threshold)] = np.nan
    return y


# -----------------------------------------------------------------------------
# Helper: check if color_grouping should be used
# -----------------------------------------------------------------------------


def _use_color_grouping(
    color_grouping: Optional[str],
    df_f: pd.DataFrame,
) -> bool:
    """True if we should group by (group_col, color_grouping)."""
    if color_grouping is None or color_grouping == "" or color_grouping == PRE_FILTER_NONE:
        return False
    return color_grouping in df_f.columns


# -----------------------------------------------------------------------------
# Step 3: Build tmp frame (same structure as _figure_swarm)
# -----------------------------------------------------------------------------


def _build_swarm_tmp(
    df_f: pd.DataFrame,
    group_col: str,
    ycol: str,
    color_grouping: Optional[str],
    use_absolute: bool = False,
    use_remove_values: bool = False,
    remove_values_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build tmp dataframe with x (group_col), y, and optionally color.

    Matches _figure_swarm: dropna(subset=["x"]). y may contain NaN;
    we dropna on y when computing stats so count reflects plotted points.
    """
    x_cat = df_f[group_col].astype(str)
    y = get_y_values(
        df_f, ycol,
        use_absolute=use_absolute,
        use_remove_values=use_remove_values,
        remove_values_threshold=remove_values_threshold,
    )
    tmp_data: dict[str, pd.Series] = {"x": x_cat, "y": y}
    if _use_color_grouping(color_grouping, df_f):
        tmp_data["color"] = df_f[color_grouping].astype(str)
    tmp = pd.DataFrame(tmp_data).dropna(subset=["x"])
    # Drop rows with NaN y so stats reflect only plotted values
    tmp = tmp.dropna(subset=["y"])
    tmp["y"] = pd.to_numeric(tmp["y"], errors="coerce")
    return tmp


# -----------------------------------------------------------------------------
# Step 4: Stats table per (group_col, color_grouping) or per group_col
# -----------------------------------------------------------------------------


def swarm_full_stats_table(
    df_f: pd.DataFrame,
    group_col: str,
    ycol: str,
    color_grouping: Optional[str] = None,
    use_absolute: bool = False,
    use_remove_values: bool = False,
    remove_values_threshold: Optional[float] = None,
    cv_epsilon: float = 1e-10,
) -> pd.DataFrame:
    """
    Compute full stats table per group for swarm plot.

    Groups by (group_col, color_grouping) when color_grouping is set and valid;
    otherwise groups by group_col only. Ignores color_grouping when None, "", or "(none)".

    Stats: count, min, max, mean, median, std, sem, cv.
    std and sem use ddof=1. cv = std/mean with NaN when |mean| < cv_epsilon.

    Returns:
        DataFrame with bookkeeping columns (group_col, color_grouping if used) before
        stats columns. No compound group_key; group dimensions are explicit columns.
    """
    tmp = _build_swarm_tmp(
        df_f, group_col, ycol, color_grouping,
        use_absolute=use_absolute,
        use_remove_values=use_remove_values,
        remove_values_threshold=remove_values_threshold,
    )
    if len(tmp) == 0:
        cols = [group_col]
        if _use_color_grouping(color_grouping, df_f):
            cols.append(color_grouping)
        return pd.DataFrame(columns=cols + STATS_COLUMNS)

    use_color = _use_color_grouping(color_grouping, df_f)
    if use_color:
        tmp["_group_key"] = list(zip(tmp["x"].astype(str), tmp["color"].astype(str)))
    else:
        tmp["_group_key"] = list(zip(tmp["x"].astype(str), [None] * len(tmp)))

    grp = tmp.groupby("_group_key", sort=True)["y"]
    count = grp.count()
    min_ = grp.min()
    max_ = grp.max()
    mean_ = grp.mean()
    median_ = grp.median()
    std_ = grp.std(ddof=1)
    sem_ = grp.sem(ddof=1)
    cv_ = (std_ / mean_).where(np.abs(mean_) >= cv_epsilon, np.nan)

    stats_df = pd.DataFrame({
        "count": count,
        "min": min_,
        "max": max_,
        "mean": mean_,
        "median": median_,
        "std": std_,
        "sem": sem_,
        "cv": cv_,
    })

    # Add bookkeeping columns (group_col, color_grouping) before stats
    group_col_vals = [k[0] for k in stats_df.index]
    stats_df.insert(0, group_col, group_col_vals)
    if use_color:
        color_vals = [k[1] for k in stats_df.index]
        stats_df.insert(1, color_grouping, color_vals)
    stats_df = stats_df.reset_index(drop=True)
    return stats_df


# -----------------------------------------------------------------------------
# Step 5: Values per group (ragged dict for TSV)
# -----------------------------------------------------------------------------


def swarm_values_per_group(
    df_f: pd.DataFrame,
    group_col: str,
    ycol: str,
    color_grouping: Optional[str] = None,
    use_absolute: bool = False,
    use_remove_values: bool = False,
    remove_values_threshold: Optional[float] = None,
) -> Dict[str, List[float]]:
    """
    Return dict mapping group key -> list of y values (what goes into each swarm column).

    Same grouping logic as swarm_full_stats_table. Keys sorted for stable output.
    """
    tmp = _build_swarm_tmp(
        df_f, group_col, ycol, color_grouping,
        use_absolute=use_absolute,
        use_remove_values=use_remove_values,
        remove_values_threshold=remove_values_threshold,
    )
    if len(tmp) == 0:
        return {}

    if _use_color_grouping(color_grouping, df_f):
        tmp["group_key"] = tmp["x"].astype(str) + "_" + tmp["color"].astype(str)
    else:
        tmp["group_key"] = tmp["x"].astype(str)

    result: Dict[str, List[float]] = {}
    for key, sub in tmp.groupby("group_key", sort=True):
        result[str(key)] = sub["y"].tolist()
    return result


# -----------------------------------------------------------------------------
# dict_of_lists_to_tsv (user-provided snippet)
# -----------------------------------------------------------------------------


def dict_of_lists_to_tsv(data: Dict[str, List[float]]) -> str:
    """
    Convert dict[str, list[float]] to TSV string with columns = keys
    and rows padded with "" for unequal lengths.

    Input:
        data = {
            "condA": [1.1, 2.2, 3.3],
            "condB": [4.4],
            "condC": [5.5, 6.6],
        }

    Output:
        TSV string suitable for copy/paste into Excel.
    """
    if not data:
        return ""

    max_len = max(len(values) for values in data.values())
    padded = {
        key: values + [""] * (max_len - len(values))
        for key, values in data.items()
    }
    df = pd.DataFrame(padded)
    return df.to_csv(sep="\t", index=False)


# -----------------------------------------------------------------------------
# Full TSV report
# -----------------------------------------------------------------------------


def swarm_report(
    df_master: pd.DataFrame,
    *,
    pre_filter_columns: list[str],
    pre_filter: dict[str, Any],
    unique_row_id_col: str,
    group_col: str,
    ycol: str,
    color_grouping: Optional[str] = None,
    use_absolute_value: bool = False,
    use_remove_values: bool = False,
    remove_values_threshold: Optional[float] = None,
    cv_epsilon: float = 1e-10,
) -> str:
    """
    Produce full TSV report: parameters, stats table, ragged values table.

    Args:
        df_master: Master (raw) dataframe.
        pre_filter_columns: Column names for pre-filtering.
        pre_filter: Map column -> selected value (PRE_FILTER_NONE = no filter).
        unique_row_id_col: Row ID column.
        group_col: X-axis grouping column for swarm.
        ycol: Y column.
        color_grouping: Optional nested grouping; None/""/"(none)" = ignore.
        use_absolute_value, use_remove_values, remove_values_threshold, cv_epsilon:
            Same as PlotState for swarm.

    Returns:
        Multi-section TSV string suitable for print() or copy/paste.
    """
    df_f = filter_by_pre_filters(
        df_master, pre_filter_columns, pre_filter, unique_row_id_col
    )

    stats_df = swarm_full_stats_table(
        df_f, group_col, ycol, color_grouping,
        use_absolute=use_absolute_value,
        use_remove_values=use_remove_values,
        remove_values_threshold=remove_values_threshold,
        cv_epsilon=cv_epsilon,
    )
    values_dict = swarm_values_per_group(
        df_f, group_col, ycol, color_grouping,
        use_absolute=use_absolute_value,
        use_remove_values=use_remove_values,
        remove_values_threshold=remove_values_threshold,
    )

    lines: list[str] = []

    # Parameters
    lines.append("# Parameters")
    lines.append(f"pre_filter\t{pre_filter}")
    lines.append(f"group_col\t{group_col}")
    lines.append(f"ycol\t{ycol}")
    lines.append(f"color_grouping\t{color_grouping if color_grouping else '(none)'}")
    lines.append("")

    # Stats table
    lines.append("# Stats (one row per group)")
    if len(stats_df) > 0:
        lines.append(stats_df.to_csv(sep="\t"))
    else:
        lines.append("(no data)")
    lines.append("")

    # Ragged values table
    lines.append("# Values (ragged, one col per group)")
    lines.append(dict_of_lists_to_tsv(values_dict))

    return "\n".join(lines)


def swarm_report_from_state(
    df_master: pd.DataFrame,
    state: "PlotState",
    *,
    unique_row_id_col: str,
    pre_filter_columns: Optional[list[str]] = None,
) -> str:
    """
    Produce full TSV report from a PlotState. Unpacks state and calls swarm_report().

    Args:
        df_master: Master (raw) dataframe.
        state: PlotState with plot configuration (pre_filter, group_col, ycol, etc.).
        unique_row_id_col: Row ID column.
        pre_filter_columns: Column names for pre-filtering. Defaults to list(state.pre_filter.keys()).

    Returns:
        Multi-section TSV string suitable for print() or copy/paste.
    """
    if pre_filter_columns is None:
        pre_filter_columns = list(state.pre_filter.keys())
    group_col = state.group_col if state.group_col is not None else state.xcol
    return swarm_report(
        df_master,
        pre_filter_columns=pre_filter_columns,
        pre_filter=state.pre_filter,
        unique_row_id_col=unique_row_id_col,
        group_col=group_col,
        ycol=state.ycol,
        color_grouping=state.color_grouping,
        use_absolute_value=state.use_absolute_value,
        use_remove_values=state.use_remove_values,
        remove_values_threshold=state.remove_values_threshold,
        cv_epsilon=state.cv_epsilon,
    )


# -----------------------------------------------------------------------------
# Example: load CSV and run
# -----------------------------------------------------------------------------


def _example_from_csv(
    csv_path: str,
    pre_filter_columns: list[str],
    pre_filter: dict[str, Any],
    unique_row_id_col: str,
    group_col: str,
    ycol: str,
    color_grouping: Optional[str] = None,
) -> None:
    """Load CSV, run swarm_report, print result."""
    df = pd.read_csv(csv_path)
    report = swarm_report(
        df,
        pre_filter_columns=pre_filter_columns,
        pre_filter=pre_filter,
        unique_row_id_col=unique_row_id_col,
        group_col=group_col,
        ycol=ycol,
        color_grouping=color_grouping,
    )
    print(report)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent.parent.parent.parent.parent
    _src = _root / "src"
    if _src.exists() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from nicewidgets.plot_pool_widget.plot_state import PlotState, PlotType
    _data_dir = _root / "data"
    _csv = _data_dir / "kym_event_report.csv"
    if not _csv.exists():
        print(f"Example CSV not found: {_csv}", file=sys.stderr)
        sys.exit(1)

    state = PlotState(
        pre_filter={"roi_id": PRE_FILTER_NONE},
        xcol="event_type",
        ycol="score_peak",
        plot_type=PlotType.SWARM,
        group_col="event_type",
        color_grouping=None,
    )
    df = pd.read_csv(str(_csv))
    report = swarm_report_from_state(
        df, state, unique_row_id_col="kym_event_id"
    )
    print("=" * 60)
    print("Swarm stats example (group_col only)")
    print("=" * 60)
    print(report)
