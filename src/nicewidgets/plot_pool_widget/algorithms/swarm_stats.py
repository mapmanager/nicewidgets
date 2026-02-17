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

from nicewidgets.plot_pool_widget.plot_state import PlotState, PlotType


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


def get_x_values(
    df_f: pd.DataFrame,
    xcol: str,
    use_absolute: bool = False,
    use_remove_values: bool = False,
    remove_values_threshold: Optional[float] = None,
) -> pd.Series:
    """
    Get the x column as numeric series, with optional transformations.

    Same logic as DataFrameProcessor.get_x_values(). Converts column to numeric
    (coerce errors to NaN), optionally applies abs(), and optionally sets values
    outside [-threshold, +threshold] to NaN.
    """
    x = pd.to_numeric(df_f[xcol], errors="coerce")
    if use_absolute:
        x = x.abs()
    if use_remove_values and remove_values_threshold is not None:
        x = x.copy()
        x[(x < -remove_values_threshold) | (x > remove_values_threshold)] = np.nan
    return x


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
# Step 3: Prepare swarm data (filtering + normalization in one place)
# -----------------------------------------------------------------------------


def prepare_swarm_tmp(
    df_f: pd.DataFrame,
    group_col: str,
    ycol: str,
    color_grouping: Optional[str],
    *,
    use_absolute: bool = False,
    use_remove_values: bool = False,
    remove_values_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build tmp dataframe with x (group_col), y, and optionally color.

    Applies filtering and normalization (abs, remove outliers) to y values.
    Single place for this logic — used by swarm_full_stats_table and swarm_values_per_group.

    Matches _figure_swarm: dropna(subset=["x"]). Drops rows with NaN y so stats
    reflect only plotted values.

    Returns:
        DataFrame with columns x, y, and optionally color.
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
    tmp = tmp.dropna(subset=["y"])
    tmp["y"] = pd.to_numeric(tmp["y"], errors="coerce")
    return tmp


# -----------------------------------------------------------------------------
# Step 4: Stats table per (group_col, color_grouping) or per group_col
# -----------------------------------------------------------------------------


def swarm_full_stats_table(
    tmp: pd.DataFrame,
    group_col: str,
    color_grouping: Optional[str],
    *,
    cv_epsilon: float = 1e-10,
) -> pd.DataFrame:
    """
    Compute full stats table per group for swarm plot.

    Expects tmp from prepare_swarm_tmp (columns x, y, and optionally color).
    Groups by (group_col, color_grouping) when tmp has "color"; else by group_col.

    Stats: count, min, max, mean, median, std, sem, cv.
    std and sem use ddof=1. cv = std/mean with NaN when |mean| < cv_epsilon.

    Returns:
        DataFrame with bookkeeping columns (group_col, color_grouping if used) before
        stats columns. No compound group_key; group dimensions are explicit columns.
    """
    if len(tmp) == 0:
        cols = [group_col]
        if "color" in tmp.columns:
            cols.append(color_grouping or "color")
        return pd.DataFrame(columns=cols + STATS_COLUMNS)

    use_color = "color" in tmp.columns
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
    tmp: pd.DataFrame,
) -> Dict[str, List[float]]:
    """
    Return dict mapping group key -> list of y values (what goes into each swarm column).

    Expects tmp from prepare_swarm_tmp (columns x, y, and optionally color).
    Same grouping logic as swarm_full_stats_table. Keys sorted for stable output.
    """
    if len(tmp) == 0:
        return {}

    if "color" in tmp.columns:
        tmp["group_key"] = tmp["x"].astype(str) + "_" + tmp["color"].astype(str)
    else:
        tmp["group_key"] = tmp["x"].astype(str)

    result: Dict[str, List[float]] = {}
    for key, sub in tmp.groupby("group_key", sort=True):
        result[str(key)] = sub["y"].tolist()
    return result


# -----------------------------------------------------------------------------
# Scatter: prepare_scatter_tmp, scatter_full_stats_table, scatter_values_per_group
# -----------------------------------------------------------------------------


def prepare_scatter_tmp(
    df_f: pd.DataFrame,
    xcol: str,
    ycol: str,
    group_col: Optional[str],
    *,
    use_absolute: bool = False,
    use_remove_values: bool = False,
    remove_values_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build tmp dataframe with x, y, and g (group).

    Applies same transforms to x and y as get_x_values/get_y_values.
    Drops rows where x or y is NaN.
    Uses group_col for g; when group_col is None, g = "(all)".
    """
    x = get_x_values(
        df_f, xcol,
        use_absolute=use_absolute,
        use_remove_values=use_remove_values,
        remove_values_threshold=remove_values_threshold,
    )
    y = get_y_values(
        df_f, ycol,
        use_absolute=use_absolute,
        use_remove_values=use_remove_values,
        remove_values_threshold=remove_values_threshold,
    )
    if group_col and group_col in df_f.columns:
        g = df_f[group_col].astype(str)
    else:
        g = pd.Series(["(all)"] * len(df_f), index=df_f.index)
    tmp = pd.DataFrame({"x": x, "y": y, "g": g}).dropna(subset=["x", "y"])
    tmp = tmp.dropna(subset=["g"])
    return tmp


def scatter_full_stats_table(
    tmp: pd.DataFrame,
    *,
    cv_epsilon: float = 1e-10,
) -> pd.DataFrame:
    """
    Compute stats per group for scatter: count, x/y min/max/mean, y median/std/sem/cv.
    Expects tmp from prepare_scatter_tmp (columns x, y, g).
    """
    if len(tmp) == 0:
        return pd.DataFrame(columns=["group", "count", "x_min", "x_max", "x_mean", "y_min", "y_max", "y_mean", "y_median", "y_std", "y_sem", "y_cv"])

    grp = tmp.groupby("g", sort=True)
    count = grp.size()
    x_min = grp["x"].min()
    x_max = grp["x"].max()
    x_mean = grp["x"].mean()
    y_min = grp["y"].min()
    y_max = grp["y"].max()
    y_mean = grp["y"].mean()
    y_median = grp["y"].median()
    y_std = grp["y"].std(ddof=1)
    y_sem = grp["y"].sem(ddof=1)
    y_cv = (y_std / y_mean).where(np.abs(y_mean) >= cv_epsilon, np.nan)

    return pd.DataFrame({
        "group": count.index,
        "count": count.values,
        "x_min": x_min.values,
        "x_max": x_max.values,
        "x_mean": x_mean.values,
        "y_min": y_min.values,
        "y_max": y_max.values,
        "y_mean": y_mean.values,
        "y_median": y_median.values,
        "y_std": y_std.values,
        "y_sem": y_sem.values,
        "y_cv": y_cv.values,
    })


def scatter_values_per_group(tmp: pd.DataFrame) -> Dict[str, List[float]]:
    """
    Return dict of x,y lists per group for scatter. No group: {"x": [...], "y": [...]}.
    With group: {"x_A": [...], "y_A": [...], "x_B": [...], "y_B": [...]}.
    """
    if len(tmp) == 0:
        return {}

    result: Dict[str, List[float]] = {}
    for gval, sub in tmp.groupby("g", sort=True):
        lab = str(gval)
        result[f"x_{lab}"] = sub["x"].tolist()
        result[f"y_{lab}"] = sub["y"].tolist()
    # Single group "(all)": use "x" and "y" keys (no suffix) for consistency with spec
    if len(result) == 1 and "(all)" in result:
        x_vals = result.pop("x_(all)")
        y_vals = result.pop("y_(all)")
        result["x"] = x_vals
        result["y"] = y_vals
    return result


# -----------------------------------------------------------------------------
# Histogram values per group (x = positions, y = counts or cumulative proportion)
# -----------------------------------------------------------------------------


def histogram_values_per_group(
    df_f: pd.DataFrame,
    xcol: str,
    group_col: Optional[str],
    nbins: int,
    *,
    cumulative: bool = False,
    use_absolute: bool = False,
    use_remove_values: bool = False,
    remove_values_threshold: Optional[float] = None,
) -> Dict[str, List[float]]:
    """
    Return dict of x and y lists for histogram, suitable for dict_of_lists_to_tsv.

    Matches the plotting logic in FigureGenerator._figure_histogram and
    _figure_cumulative_histogram. Uses only group_col (not color_grouping).

    Chosen options:
    - Regular histogram: x = bin centers, y = raw counts.
      (Alternative: x = bin edges.)
    - Cumulative histogram: x = bin edges (for step plot), y = cumulative
      proportions in [0, 1].
      (Alternative: x = bin centers.)
    - nbins: parameter (same as _figure_histogram / _figure_cumulative_histogram).

    No group_col: keys "x_(all)", "y_(all)".
    With group_col: keys "x_{group}", "y_{group}".
    """
    x = get_x_values(
        df_f, xcol,
        use_absolute=use_absolute,
        use_remove_values=use_remove_values,
        remove_values_threshold=remove_values_threshold,
    ).dropna()
    if len(x) == 0:
        return {}

    result: Dict[str, List[float]] = {}

    if group_col is None or group_col not in df_f.columns:
        groups = [("(all)", x)]
    else:
        g = df_f.loc[x.index, group_col].astype(str)
        tmp = pd.DataFrame({"x": x, "g": g}).dropna(subset=["g"])
        if len(tmp) == 0:
            return {}
        groups = [(str(k), sub["x"]) for k, sub in tmp.groupby("g", sort=True)]

    for group_label, x_vals in groups:
        x_arr = x_vals.values
        if len(x_arr) == 0:
            continue
        counts, bin_edges = np.histogram(x_arr, bins=nbins)

        if cumulative:
            cumsum = np.cumsum(counts)
            total = cumsum[-1]
            y_vals = (cumsum / total if total > 0 else cumsum).tolist()
            # x: bin edges for step plot (alternative: bin centers)
            x_vals_out = bin_edges.tolist()
        else:
            # x: bin centers (alternative: bin_edges)
            bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
            x_vals_out = bin_centers
            y_vals = counts.tolist()

        result[f"x_{group_label}"] = x_vals_out
        result[f"y_{group_label}"] = y_vals

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
    state: PlotState,
    *,
    unique_row_id_col: str,
    pre_filter_columns: Optional[list[str]] = None,
) -> str:
    """
    Produce full TSV report: parameters, stats table, ragged values table.

    Args:
        df_master: Master (raw) dataframe.
        state: PlotState with plot configuration (pre_filter, group_col, ycol, etc.).
        unique_row_id_col: Row ID column.
        pre_filter_columns: Column names for pre-filtering. Defaults to list(state.pre_filter.keys()).

    Returns:
        Multi-section TSV string suitable for print() or copy/paste.
    """
    plot_type = state.plot_type

    if pre_filter_columns is None:
        pre_filter_columns = list(state.pre_filter.keys())
    xcol = state.xcol
    ycol = state.ycol
    group_col = state.group_col  # Group/Color
    color_grouping = state.color_grouping  # Group/Nesting
    nbins = state.histogram_bins

    df_f = filter_by_pre_filters(
        df_master, pre_filter_columns, state.pre_filter, unique_row_id_col
    )

    if plot_type in (PlotType.HISTOGRAM, PlotType.CUMULATIVE_HISTOGRAM):
        values_dict = histogram_values_per_group(
            df_f,
            xcol,
            group_col,
            nbins,
            cumulative=(plot_type == PlotType.CUMULATIVE_HISTOGRAM),
            use_absolute=state.use_absolute_value,
            use_remove_values=state.use_remove_values,
            remove_values_threshold=state.remove_values_threshold,
        )
        # Simple x-value stats per group for histogram (count, min, max, mean)
        x_series = get_x_values(
            df_f, xcol,
            use_absolute=state.use_absolute_value,
            use_remove_values=state.use_remove_values,
            remove_values_threshold=state.remove_values_threshold,
        ).dropna()
        if len(x_series) == 0:
            stats_df = pd.DataFrame(columns=["group", "count", "min", "max", "mean"])
        elif group_col and group_col in df_f.columns:
            g = df_f.loc[x_series.index, group_col].astype(str)
            tmp = pd.DataFrame({"x": x_series, "g": g}).dropna(subset=["g"])
            grp = tmp.groupby("g", sort=True)["x"]
            stats_df = pd.DataFrame({
                "group": grp.count().index,
                "count": grp.count().values,
                "min": grp.min().values,
                "max": grp.max().values,
                "mean": grp.mean().values,
            })
        else:
            stats_df = pd.DataFrame([{
                "group": "(all)",
                "count": len(x_series),
                "min": float(x_series.min()),
                "max": float(x_series.max()),
                "mean": float(x_series.mean()),
            }])
    elif plot_type == PlotType.SCATTER:
        tmp = prepare_scatter_tmp(
            df_f, xcol, ycol, group_col,
            use_absolute=state.use_absolute_value,
            use_remove_values=state.use_remove_values,
            remove_values_threshold=state.remove_values_threshold,
        )
        stats_df = scatter_full_stats_table(tmp, cv_epsilon=state.cv_epsilon)
        values_dict = scatter_values_per_group(tmp)
    else:
        # Swarm/box/violin require group_col; use synthetic "(all)" if None so prepare_swarm_tmp does not fail.
        effective_group_col = group_col if group_col else "group"
        if not group_col:
            df_f = df_f.copy()
            df_f["group"] = "(all)"
        tmp = prepare_swarm_tmp(
            df_f, effective_group_col, ycol, color_grouping,
            use_absolute=state.use_absolute_value,
            use_remove_values=state.use_remove_values,
            remove_values_threshold=state.remove_values_threshold,
        )
        stats_df = swarm_full_stats_table(
            tmp, effective_group_col, color_grouping, cv_epsilon=state.cv_epsilon
        )
        values_dict = swarm_values_per_group(tmp)

    lines: list[str] = []

    # Parameters
    lines.append("# Parameters")
    lines.append(f"plot_type\t{state.plot_type.value}")
    lines.append(f"pre_filter\t{state.pre_filter}")

    if plot_type in (PlotType.SCATTER, PlotType.HISTOGRAM, PlotType.CUMULATIVE_HISTOGRAM):
        lines.append(f"xcol\t{xcol}")

    if plot_type in (PlotType.HISTOGRAM, PlotType.CUMULATIVE_HISTOGRAM):
        lines.append(f"histogram_bins\t{nbins}")

    if plot_type not in (PlotType.HISTOGRAM, PlotType.CUMULATIVE_HISTOGRAM):
        lines.append(f"ycol\t{ycol}")

    lines.append(f"group_col\t{group_col}")
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
    state: PlotState,
    *,
    unique_row_id_col: str,
    pre_filter_columns: Optional[list[str]] = None,
) -> str:
    """
    Produce full TSV report from a PlotState. Calls swarm_report().

    Args:
        df_master: Master (raw) dataframe.
        state: PlotState with plot configuration (pre_filter, group_col, ycol, etc.).
        unique_row_id_col: Row ID column.
        pre_filter_columns: Column names for pre-filtering. Defaults to list(state.pre_filter.keys()).

    Returns:
        Multi-section TSV string suitable for print() or copy/paste.
    """
    return swarm_report(
        df_master,
        state,
        unique_row_id_col=unique_row_id_col,
        pre_filter_columns=pre_filter_columns,
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
    state = PlotState(
        pre_filter=pre_filter,
        xcol=group_col,
        ycol=ycol,
        plot_type=PlotType.SWARM,
        group_col=group_col,
        color_grouping=color_grouping,
    )
    report = swarm_report(
        df,
        state,
        unique_row_id_col=unique_row_id_col,
        pre_filter_columns=pre_filter_columns,
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
        df, state, unique_row_id_col="_unique_row_id"
    )
    print("=" * 60)
    print("Swarm stats example (group_col only)")
    print("=" * 60)
    print(report)
