"""
Group plot algorithm — pure pandas/numpy reference.

This module is a "methods" reference for the Grouped plot type. It reproduces
the exact algorithm used in FigureGenerator._figure_grouped() using only
pandas and numpy (no Plotly, no nicewidgets). It shows:

  1. How the master (raw) dataframe is pre-filtered.
  2. How the y column is extracted and optionally transformed.
  3. How grouping and the chosen y-stat produce the final plot data.

The result is the aggregated series (group labels → stat value) that would
be plotted on x vs y. Run this file to see the algorithm on example data.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


# Sentinel for "no filter" on a pre-filter column (matches pre_filter_conventions.PRE_FILTER_NONE).
PRE_FILTER_NONE = "(none)"


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
        # Compare as strings so UI string selection matches numeric columns
        df_f = df_f[df_f[col].astype(str) == str(val)]
    df_f = df_f.dropna(subset=[unique_row_id_col])
    return df_f


# -----------------------------------------------------------------------------
# Step 2: Extract y values with optional absolute value and remove-values filter
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
        y[(y < -remove_values_threshold) | (y > remove_values_threshold)] = np.nan
    return y


# -----------------------------------------------------------------------------
# Step 3: Group by group_col and compute the chosen y-stat
# -----------------------------------------------------------------------------


def grouped_aggregate(
    df_f: pd.DataFrame,
    group_col: str,
    ycol: str,
    ystat: str,
    use_absolute: bool = False,
    use_remove_values: bool = False,
    remove_values_threshold: Optional[float] = None,
    cv_epsilon: float = 1e-10,
) -> pd.Series:
    """
    Compute the aggregated stat per group (same logic as _figure_grouped).

    Returns a Series with index = group labels, values = aggregated y (one point per group).
    """
    # Extract group labels (strings) and y values (with optional transforms)
    g = df_f[group_col].astype(str)
    y = get_y_values(
        df_f, ycol,
        use_absolute=use_absolute,
        use_remove_values=use_remove_values,
        remove_values_threshold=remove_values_threshold,
    )

    # Build temporary frame: one row per row of df_f, columns [group, y]; drop rows with no group
    tmp = pd.DataFrame({"group": g, "y": y}).dropna(subset=["group"])

    if ystat == "count":
        # Count of (non-NaN) y values per group
        agg = tmp.groupby("group", dropna=False)["y"].count()
        return agg

    # For all other stats, y must be numeric
    tmp["y"] = pd.to_numeric(tmp["y"], errors="coerce")

    if ystat == "cv":
        # Coefficient of variation: std / mean. NaN when |mean| < cv_epsilon.
        grp = tmp.groupby("group", dropna=False)["y"]
        mean_ = grp.mean()
        std_ = grp.std(ddof=1)
        cv = std_ / mean_
        agg = cv.where(np.abs(mean_) >= cv_epsilon, np.nan)
        return agg

    if ystat == "sem":
        # Standard error of the mean: sample std / sqrt(n), ddof=1
        agg = tmp.groupby("group", dropna=False)["y"].sem(ddof=1)
        return agg

    # mean, median, sum, std, min, max (and any other groupby method)
    agg = getattr(tmp.groupby("group", dropna=False)["y"], ystat)()
    return agg


# -----------------------------------------------------------------------------
# Full stats table: one group_col, one ycol, multiple agg stats
# -----------------------------------------------------------------------------


def grouped_full_stats_table(
    df_f: pd.DataFrame,
    group_col: str,
    ycol: str,
    use_absolute: bool = False,
    use_remove_values: bool = False,
    remove_values_threshold: Optional[float] = None,
    cv_epsilon: float = 1e-10,
) -> pd.DataFrame:
    """
    For one group column and one y column, compute a full stats table per group.

    Stats: count, min, max, mean, std, sem, CV. Same preprocessing as
    get_y_values (numeric coerce, optional abs, optional remove-values).
    std and sem use ddof=1; CV = std/mean with NaN when |mean| < cv_epsilon.

    Returns:
        DataFrame with index = group labels, columns = count, min, max, mean, std, sem, cv.
    """
    g = df_f[group_col].astype(str)
    y = get_y_values(
        df_f, ycol,
        use_absolute=use_absolute,
        use_remove_values=use_remove_values,
        remove_values_threshold=remove_values_threshold,
    )
    tmp = pd.DataFrame({"group": g, "y": y}).dropna(subset=["group"])
    tmp["y"] = pd.to_numeric(tmp["y"], errors="coerce")

    grp = tmp.groupby("group", dropna=False)["y"]
    count = grp.count()
    min_ = grp.min()
    max_ = grp.max()
    mean_ = grp.mean()
    std_ = grp.std(ddof=1)
    sem_ = grp.sem(ddof=1)
    cv_ = (std_ / mean_).where(np.abs(mean_) >= cv_epsilon, np.nan)

    return pd.DataFrame({
        "count": count,
        "min": min_,
        "max": max_,
        "mean": mean_,
        "std": std_,
        "sem": sem_,
        "cv": cv_,
    })


# -----------------------------------------------------------------------------
# Full pipeline: raw data → pre-filter → grouped aggregation
# -----------------------------------------------------------------------------


def group_plot_algorithm(
    df_master: pd.DataFrame,
    *,
    pre_filter_columns: list[str],
    unique_row_id_col: str,
    pre_filter_selections: dict[str, Any],
    group_col: str,
    ycol: str,
    ystat: str,
    use_absolute_value: bool = False,
    use_remove_values: bool = False,
    remove_values_threshold: Optional[float] = None,
    cv_epsilon: float = 1e-10,
) -> pd.Series:
    """
    Run the full Grouped-plot algorithm from master df to final plot data.

    Returns the aggregated Series (index = group labels, values = stat(y) per group).
    This is exactly the data that _figure_grouped() turns into a Plotly trace (x=index, y=values).
    """
    # Step 1: Pre-filter the master dataframe
    df_f = filter_by_pre_filters(
        df_master,
        pre_filter_columns,
        pre_filter_selections,
        unique_row_id_col,
    )

    # Step 2 & 3: Extract y (with optional abs/remove-values) and aggregate by group
    agg = grouped_aggregate(
        df_f,
        group_col=group_col,
        ycol=ycol,
        ystat=ystat,
        use_absolute=use_absolute_value,
        use_remove_values=use_remove_values,
        remove_values_threshold=remove_values_threshold,
        cv_epsilon=cv_epsilon,
    )
    return agg


# -----------------------------------------------------------------------------
# Example: load CSV and run algorithm (no Plotly)
# -----------------------------------------------------------------------------


def _example_from_csv(
    csv_path: str,
    pre_filter_columns: list[str],
    unique_row_id_col: str,
    pre_filter: dict[str, Any],
    group_col: str,
    ycol: str,
    ystat: str,
    print_intermediate: bool = True,
) -> None:
    """Load a CSV and run the group-plot algorithm; print the result table and optional intermediate steps."""
    # Load raw data (master df)
    df_master = pd.read_csv(csv_path)
    # Optional: add unique_row_id if schema expects it (e.g. radon_report_db)
    if unique_row_id_col not in df_master.columns and "path" in df_master.columns and "roi_id" in df_master.columns:
        df_master[unique_row_id_col] = df_master["path"].astype(str) + "|" + df_master["roi_id"].astype(str)

    if print_intermediate:
        print("--- Step 0: Master dataframe (raw) ---")
        print(f"Shape: {df_master.shape}")
        cols = [c for c in [group_col, ycol, unique_row_id_col] if c in df_master.columns]
        if cols:
            print(df_master[cols].head(10).to_string(index=False))
        else:
            print(df_master.head(10).to_string(index=False))
        print()

    # Step 1: Pre-filter
    df_f = filter_by_pre_filters(
        df_master,
        pre_filter_columns,
        pre_filter,
        unique_row_id_col,
    )
    if print_intermediate:
        print("--- Step 1: After pre-filter ---")
        print(f"Shape: {df_f.shape}")
        cols = [c for c in [group_col, ycol, unique_row_id_col] if c in df_f.columns]
        if cols:
            print(df_f[cols].head(20).to_string(index=False))
        else:
            print(df_f.head(20).to_string(index=False))
        print()

    # Step 2: Y values (display-only df so user can follow along)
    y_series = get_y_values(df_f, ycol)
    if print_intermediate:
        step2_display = pd.DataFrame({
            group_col: df_f[group_col].values,
            ycol: df_f[ycol].values,
            "y (computed)": y_series.values,
        })
        print("--- Step 2: Y values (raw column + computed series) ---")
        print(step2_display.head(20).to_string(index=False))
        print()

    # Step 3: Per-row (group, y) before aggregation (display-only)
    if print_intermediate:
        g = df_f[group_col].astype(str)
        tmp_display = pd.DataFrame({"group": g, "y": y_series}).dropna(subset=["group"])
        tmp_display["y"] = pd.to_numeric(tmp_display["y"], errors="coerce")
        print("--- Step 3: Per-row (group, y) before aggregation ---")
        print(tmp_display.head(20).to_string(index=False))
        print()

    # Run full pipeline to get aggregated result
    agg = group_plot_algorithm(
        df_master,
        pre_filter_columns=pre_filter_columns,
        unique_row_id_col=unique_row_id_col,
        pre_filter_selections=pre_filter,
        group_col=group_col,
        ycol=ycol,
        ystat=ystat,
    )

    # Result: one row per group, ready to be plotted as x=index, y=values
    result_table = pd.DataFrame({"group": agg.index.astype(str), "value": agg.values})
    print("\n--- Final grouped plot data ---")
    print(f"Parameters used: group_col = {group_col!r}, ycol = {ycol!r}, ystat = {ystat!r}")
    print("Table (x = group, y = value):")
    print(result_table.to_string(index=False))

    # Full stats table: one group_col, one ycol, all agg stats (count, min, max, mean, std, sem, CV)
    full_stats = grouped_full_stats_table(df_f, group_col=group_col, ycol=ycol)
    print("\n--- Full stats table (group_col = {0!r}, ycol = {1!r}) ---".format(group_col, ycol))
    print(full_stats.to_string())


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Resolve path to nicewidgets/data (this file is in .../plot_pool_widget/algorithms/group_plot.py)
    _root = Path(__file__).resolve().parent.parent.parent.parent.parent
    _data_dir = _root / "data"
    _csv = _data_dir / "kym_event_report.csv"
    if not _csv.exists():
        print(f"Example CSV not found: {_csv}", file=sys.stderr)
        sys.exit(1)

    # Example parameters (matches typical UI choices for kym_event_report)
    _example_from_csv(
        str(_csv),
        pre_filter_columns=["roi_id"],
        unique_row_id_col="kym_event_id",
        pre_filter={"roi_id": PRE_FILTER_NONE},  # no filter = use all rows
        group_col="event_type",
        ycol="score_peak",
        ystat="mean",
    )
