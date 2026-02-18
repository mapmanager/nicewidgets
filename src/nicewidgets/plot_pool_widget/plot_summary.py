"""Plot summary builders for text-summarizable report data.

Builds (params, summary_table, columnar) from intermediate results produced
during figure generation. Used by FigureGenerator; does not depend on Plotly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from nicewidgets.plot_pool_widget.plot_state import PlotState

# Stats columns for summary table (full set per plan).
STATS_COLUMNS = ["count", "min", "max", "mean", "median", "std", "sem", "cv"]

# Params keys that are purely visual (excluded from text report).
PARAMS_VISUAL_KEYS = frozenset({
    "show_mean", "show_std_sem", "std_sem_type", "mean_line_width", "error_line_width",
    "show_raw", "point_size", "show_legend",
})


@dataclass
class PlotSummary:
    """Structured summary of what is plotted (for text report or inspection).

    Attributes:
        params: PlotState as dict (state.to_dict()).
        summary_table: One row per group; columns = group keys + stats (count, min, max, mean, etc.).
        columnar: Long-format table: one row per data point (or per bin for hist); supports ragged row counts per group.
    """
    params: dict[str, Any]
    summary_table: pd.DataFrame
    columnar: pd.DataFrame


def _format_swarm_columnar_to_tsv(summary: PlotSummary) -> str:
    """Format swarm (and box/violin) columnar as multiple column sets: one per (group_col, color_grouping).

    First column: group_col name (row0), color_grouping name (row1), then blank. Each block has
    group value only in first cell of block, color value only in first cell, then headers
    (file_stem, x_jitter, ycol), then data rows. One empty column between blocks.
    row_id is not included in output (columnar still holds it for other uses).
    """
    col = summary.columnar
    group_col = col.columns[0]
    use_color = len(col.columns) >= 5 and col.columns[1] != "y" and col.columns[1] != "x_jitter"
    color_col = col.columns[1] if use_color else None
    has_x_jitter = "x_jitter" in col.columns
    ycol_name = summary.params.get("ycol", "y")
    group_col_name = summary.params.get("group_col") or group_col
    color_grouping_name = summary.params.get("color_grouping") if use_color else None

    # Group by (group_col, color) or group_col
    if use_color:
        groups = list(col.groupby([group_col, color_col], sort=True))
    else:
        groups = [(k, sub) for k, sub in col.groupby(group_col, sort=True)]
    if not groups:
        return "(none)"

    # Each group: list of rows for this block (file_stem, x_jitter?, y per point)
    sets_data: List[tuple[str, str, List[tuple[str, Any, Any]]]] = []
    for key, sub in groups:
        group_val = key[0] if isinstance(key, tuple) else str(key)
        color_val = key[1] if isinstance(key, tuple) else "(none)"
        file_stems = sub["file_stem"].astype(str).tolist()
        ys = sub["y"].tolist()
        x_jitters = sub["x_jitter"].tolist() if has_x_jitter else [None] * len(file_stems)
        points = list(zip(file_stems, x_jitters, ys))
        sets_data.append((group_val, color_val, points))

    max_rows = max(len(s[2]) for s in sets_data)
    sep = "\t"
    block_width = 3 if has_x_jitter else 2

    # Build column cells: first column (labels), then per-block columns + separator
    row0_cells: List[str] = []
    row1_cells: List[str] = []
    row2_cells: List[str] = []
    data_rows: List[List[str]] = []

    for group_val, color_val, points in sets_data:
        # Block: group value only in first cell, then blanks
        row0_cells.append(group_val)
        row0_cells.extend([""] * (block_width - 1))
        row1_cells.append(color_val)
        row1_cells.extend([""] * (block_width - 1))
        if has_x_jitter:
            row2_cells.extend(["file_stem", "x_jitter", ycol_name])
        else:
            row2_cells.extend(["file_stem", ycol_name])
        row0_cells.append("")  # separator column
        row1_cells.append("")
        row2_cells.append("")

    for r in range(max_rows):
        row_cells: List[str] = []
        for group_val, color_val, points in sets_data:
            if r < len(points):
                fs, xj, yv = points[r]
                if has_x_jitter:
                    row_cells.extend([fs, str(xj), str(yv)])
                else:
                    row_cells.extend([fs, str(yv)])
            else:
                row_cells.extend([""] * block_width)
            row_cells.append("")  # separator column
        data_rows.append(row_cells)

    # First column: group_col name, color_grouping name (or blank), then blank for header and data rows
    lines: List[str] = []
    row0_line = group_col_name + sep + sep.join(row0_cells)
    row1_line = (color_grouping_name or "") + sep + sep.join(row1_cells)
    row2_line = sep + sep.join(row2_cells)
    lines.append(row0_line)
    lines.append(row1_line)
    lines.append(row2_line)
    for row_cells in data_rows:
        lines.append(sep + sep.join(row_cells))
    return "\n".join(lines)


def _format_scatter_columnar_to_tsv(summary: PlotSummary) -> str:
    """Format scatter columnar as multiple column sets: one per group_col value (color_grouping ignored).

    First column: group_col name (row0), then blank. Each block has group value in first cell only,
    then headers (xcol, ycol, file_stem), then data rows. One empty column between blocks.
    row_id is not included in output.
    """
    col = summary.columnar
    group_col_name = summary.params.get("group_col")
    if not group_col_name or group_col_name not in col.columns:
        return col.to_csv(path_or_buf=None, sep="\t", index=False).rstrip("\n")
    xcol_name = summary.params.get("xcol", "x")
    ycol_name = summary.params.get("ycol", "y")
    # Columnar has xcol, ycol, row_id, file_stem, group_col (and maybe color_grouping); we group by group_col only
    groups = list(col.groupby(group_col_name, sort=True))
    if not groups:
        return "(none)"
    sep = "\t"
    block_width = 3  # xcol, ycol, file_stem
    row0_cells: List[str] = []
    row2_cells: List[str] = []
    data_rows: List[List[str]] = []
    sets_data: List[tuple[str, List[tuple[Any, Any, str]]]] = []
    for key, sub in groups:
        group_val = str(key)
        points = list(zip(
            sub[xcol_name].tolist(),
            sub[ycol_name].tolist(),
            sub["file_stem"].astype(str).tolist(),
        ))
        sets_data.append((group_val, points))
    max_rows = max(len(s[1]) for s in sets_data)
    for group_val, points in sets_data:
        row0_cells.append(group_val)
        row0_cells.extend([""] * (block_width - 1))
        row0_cells.append("")
        row2_cells.extend([xcol_name, ycol_name, "file_stem"])
        row2_cells.append("")
    for r in range(max_rows):
        row_cells: List[str] = []
        for group_val, points in sets_data:
            if r < len(points):
                xv, yv, fs = points[r]
                row_cells.extend([str(xv), str(yv), fs])
            else:
                row_cells.extend([""] * block_width)
            row_cells.append("")
        data_rows.append(row_cells)
    lines: List[str] = []
    lines.append(group_col_name + sep + sep.join(row0_cells))
    lines.append(sep + sep.join(row2_cells))
    for row_cells in data_rows:
        lines.append(sep + sep.join(row_cells))
    return "\n".join(lines)


def _format_hist_columnar_to_tsv(summary: PlotSummary, value_col: str) -> str:
    """Format histogram or cumulative histogram columnar as multiple column sets: one per group_col.

    value_col is 'count' for histogram, 'cumulative_proportion' for cumulative histogram.
    First column: group_col name (row0), then blank. Each block has group value in first cell only,
    then headers (bin_center, value_col), then data rows. One empty column between blocks.
    Color_grouping is ignored (one block per group_col value only).
    """
    col = summary.columnar
    group_col_name = summary.params.get("group_col")
    if not group_col_name or group_col_name not in col.columns:
        return col.to_csv(path_or_buf=None, sep="\t", index=False).rstrip("\n")
    if value_col not in col.columns:
        return col.to_csv(path_or_buf=None, sep="\t", index=False).rstrip("\n")
    # Group by group_col; for cum hist with color in columnar, take first row per (group, bin_center)
    groups = []
    for key, sub in col.groupby(group_col_name, sort=True):
        sub = sub.drop_duplicates(subset=["bin_center"], keep="first")
        points = list(zip(
            sub["bin_center"].tolist(),
            sub[value_col].tolist(),
        ))
        groups.append((str(key), points))
    if not groups:
        return "(none)"
    sep = "\t"
    block_width = 2  # bin_center, value_col
    row0_cells: List[str] = []
    row2_cells: List[str] = []
    data_rows: List[List[str]] = []
    max_rows = max(len(g[1]) for g in groups)
    for group_val, points in groups:
        row0_cells.append(group_val)
        row0_cells.extend([""] * (block_width - 1))
        row0_cells.append("")
        row2_cells.extend(["bin_center", value_col])
        row2_cells.append("")
    for r in range(max_rows):
        row_cells: List[str] = []
        for group_val, points in groups:
            if r < len(points):
                bc, v = points[r]
                row_cells.extend([str(bc), str(v)])
            else:
                row_cells.extend([""] * block_width)
            row_cells.append("")
        data_rows.append(row_cells)
    lines = [
        group_col_name + sep + sep.join(row0_cells),
        sep + sep.join(row2_cells),
    ]
    for row_cells in data_rows:
        lines.append(sep + sep.join(row_cells))
    return "\n".join(lines)


def format_plot_summary_to_str(summary: PlotSummary) -> str:
    """Convert PlotSummary to a plain-text string (params, summary table, columnar)."""
    lines: list[str] = []
    # (1) Params (tab between name and value; exclude purely visual keys)
    lines.append("=== Params ===")
    for k, v in summary.params.items():
        if k in PARAMS_VISUAL_KEYS:
            continue
        lines.append(f"{k}\t{v}")
    # (2) Summary table (tab-separated so Excel pastes into columns)
    lines.append("")
    lines.append("=== Summary table ===")
    if summary.summary_table is not None and len(summary.summary_table) > 0:
        lines.append(summary.summary_table.to_csv(path_or_buf=None, sep="\t", index=False).rstrip("\n"))
    else:
        lines.append("(none)")
    # (3) Columnar (tab-separated; swarm/box/violin use multi-set columns per group)
    lines.append("")
    lines.append("=== Columnar ===")
    if summary.columnar is not None and len(summary.columnar) > 0:
        plot_type = summary.params.get("plot_type")
        if plot_type in ("swarm", "box_plot", "violin"):
            lines.append(_format_swarm_columnar_to_tsv(summary))
        elif plot_type == "scatter":
            lines.append(_format_scatter_columnar_to_tsv(summary))
        elif plot_type == "histogram":
            lines.append(_format_hist_columnar_to_tsv(summary, "count"))
        elif plot_type == "cumulative_histogram":
            lines.append(_format_hist_columnar_to_tsv(summary, "cumulative_proportion"))
        else:
            # Long-format table for grouped, etc.
            lines.append(summary.columnar.to_csv(path_or_buf=None, sep="\t", index=False).rstrip("\n"))
    else:
        lines.append("(none)")
    return "\n".join(lines)


def stats_row_for_series(y: pd.Series, cv_epsilon: float = 1e-10) -> dict[str, Any]:
    """Compute count, min, max, mean, median, std, sem, cv for a series.

    std/sem use ddof=1. cv = std/mean with NaN when |mean| < cv_epsilon.
    """
    y = pd.to_numeric(y, errors="coerce").dropna()
    n = len(y)
    if n == 0:
        return {k: np.nan for k in STATS_COLUMNS}
    return {
        "count": n,
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "median": float(y.median()),
        "std": float(y.std(ddof=1)) if n > 1 else np.nan,
        "sem": float(y.sem(ddof=1)) if n > 1 else np.nan,
        "cv": float(y.std(ddof=1) / y.mean()) if n > 1 and np.abs(y.mean()) >= cv_epsilon else np.nan,
    }


def build_swarm_summary(
    state: PlotState,
    tmp: pd.DataFrame,
    group_col: str,
    color_grouping: Optional[str] = None,
    *,
    cv_epsilon: Optional[float] = None,
) -> PlotSummary:
    """Build summary from swarm/box/violin-style tmp (x, y, row_id, file_stem, optional color).

    tmp must have columns: x (categorical), y, row_id, file_stem; optionally "color".
    Summary table: one row per (x, color) when color present, else per x. Full stats.
    Columnar: long format with x_cat, (color_group), y, row_id, file_stem.
    """
    if cv_epsilon is None:
        cv_epsilon = state.cv_epsilon
    params = state.to_dict()
    use_color = "color" in tmp.columns and color_grouping
    if len(tmp) == 0:
        summary_cols = [group_col]
        if use_color:
            summary_cols.append(color_grouping or "color")
        summary_table = pd.DataFrame(columns=summary_cols + STATS_COLUMNS)
        columnar_cols = [group_col, "y", "row_id", "file_stem"]
        if use_color:
            columnar_cols.insert(1, color_grouping or "color")
        columnar = pd.DataFrame(columns=columnar_cols)
        return PlotSummary(params=params, summary_table=summary_table, columnar=columnar)

    # Summary table: group by (x, color) or x
    if use_color:
        grp = tmp.groupby([tmp["x"].astype(str), tmp["color"].astype(str)], sort=True)["y"]
    else:
        grp = tmp.groupby(tmp["x"].astype(str), sort=True)["y"]
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
    stats_df = stats_df.reset_index()
    # Index columns may be named level_0/level_1 or x/color depending on pandas; rename by position
    stats_df = stats_df.rename(columns={stats_df.columns[0]: group_col})
    if use_color:
        stats_df = stats_df.rename(columns={stats_df.columns[1]: color_grouping or "color"})
    summary_table = stats_df[[group_col] + ([color_grouping or "color"] if use_color else []) + STATS_COLUMNS]

    # Columnar: long format, one row per point. Include x_jitter when present (swarm).
    base_cols = ["x", "y", "row_id", "file_stem"]
    if "x_jitter" in tmp.columns:
        base_cols = ["x", "x_jitter", "y", "row_id", "file_stem"]
    columnar = tmp[[c for c in base_cols if c in tmp.columns]].copy()
    columnar = columnar.rename(columns={"x": group_col})
    if use_color:
        columnar.insert(1, color_grouping or "color", tmp["color"].values)
    columnar[group_col] = columnar[group_col].astype(str)
    return PlotSummary(params=params, summary_table=summary_table, columnar=columnar)


def build_scatter_summary(
    state: PlotState,
    tmp: pd.DataFrame,
    xcol: str,
    ycol: str,
    group_col: Optional[str] = None,
    color_grouping: Optional[str] = None,
    *,
    cv_epsilon: Optional[float] = None,
) -> PlotSummary:
    """Build summary from scatter-style tmp (x, y, row_id, file_stem; optional color).

    tmp has x, y, row_id, file_stem; optionally "color" (from group_col) and/or "symbol" (from color_grouping).
    Summary table: one row global or one per group. Columnar: long format (x, y, row_id, file_stem, group columns).
    """
    if cv_epsilon is None:
        cv_epsilon = state.cv_epsilon
    params = state.to_dict()
    if len(tmp) == 0:
        summary_table = pd.DataFrame(columns=[xcol, ycol, "count"] + STATS_COLUMNS)
        columnar = pd.DataFrame(columns=["x", "y", "row_id", "file_stem"])
        return PlotSummary(params=params, summary_table=summary_table, columnar=columnar)

    use_group = "color" in tmp.columns and group_col
    if use_group:
        rows: List[dict[str, Any]] = []
        for _, sub in tmp.groupby("color", sort=True):
            st = stats_row_for_series(sub["y"], cv_epsilon=cv_epsilon)
            rows.append({group_col or "group": sub["color"].iloc[0], **st})
        summary_table = pd.DataFrame(rows)
        summary_table = summary_table[[group_col or "group"] + STATS_COLUMNS]
    else:
        st = stats_row_for_series(tmp["y"], cv_epsilon=cv_epsilon)
        summary_table = pd.DataFrame([st])

    columnar = tmp[["x", "y", "row_id", "file_stem"]].copy()
    columnar = columnar.rename(columns={"x": xcol, "y": ycol})
    if use_group:
        columnar.insert(columnar.shape[1], group_col or "group", tmp["color"].values)
    if "symbol" in tmp.columns and color_grouping:
        columnar[color_grouping] = tmp["symbol"].values
    return PlotSummary(params=params, summary_table=summary_table, columnar=columnar)


def build_histogram_summary(
    state: PlotState,
    x: pd.Series,
    group_series: Optional[pd.Series] = None,
    group_col: Optional[str] = None,
    *,
    n_bins: Optional[int] = None,
) -> PlotSummary:
    """Build summary from histogram data (x values, optional group).

    Computes counts and bin_edges per group; summary_table one row per trace; columnar (bin_center, count, group?).
    """
    params = state.to_dict()
    n_bins = n_bins or state.histogram_bins
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        summary_table = pd.DataFrame(columns=["n", "min", "max", "count"] + STATS_COLUMNS)
        columnar = pd.DataFrame(columns=["bin_center", "count"])
        return PlotSummary(params=params, summary_table=summary_table, columnar=columnar)

    if group_series is None:
        counts, bin_edges = np.histogram(x.values, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        summary_table = pd.DataFrame([{
            "n": len(x),
            "min": float(x.min()),
            "max": float(x.max()),
            "count": int(counts.sum()),
        }])
        columnar = pd.DataFrame({"bin_center": bin_centers, "count": counts})
        return PlotSummary(params=params, summary_table=summary_table, columnar=columnar)

    tmp = pd.DataFrame({"x": x, "g": group_series.loc[x.index].astype(str)}).dropna(subset=["g"])
    if len(tmp) == 0:
        summary_table = pd.DataFrame(columns=[group_col or "group", "n", "min", "max", "count"])
        columnar = pd.DataFrame(columns=["bin_center", "count", group_col or "group"])
        return PlotSummary(params=params, summary_table=summary_table, columnar=columnar)

    summary_rows = []
    columnar_dfs = []
    for group_val, sub in tmp.groupby("g", sort=True):
        x_vals = sub["x"].values
        counts, bin_edges = np.histogram(x_vals, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        summary_rows.append({
            (group_col or "group"): group_val,
            "n": len(sub),
            "min": float(np.min(x_vals)),
            "max": float(np.max(x_vals)),
            "count": int(counts.sum()),
        })
        df = pd.DataFrame({"bin_center": bin_centers, "count": counts})
        df[group_col or "group"] = group_val
        columnar_dfs.append(df)
    summary_table = pd.DataFrame(summary_rows)
    columnar = pd.concat(columnar_dfs, ignore_index=True)
    return PlotSummary(params=params, summary_table=summary_table, columnar=columnar)


def build_cumulative_histogram_summary(
    state: PlotState,
    x: pd.Series,
    group_series: Optional[pd.Series] = None,
    color_series: Optional[pd.Series] = None,
    group_col: Optional[str] = None,
    color_grouping: Optional[str] = None,
    *,
    n_bins: Optional[int] = None,
) -> PlotSummary:
    """Build summary from cumulative histogram data (x, optional group/color).

    Columnar: bin_center, cumulative_proportion per trace; long format with group/color column when present.
    """
    params = state.to_dict()
    n_bins = n_bins or state.histogram_bins
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        summary_table = pd.DataFrame(columns=["n", "min", "max"])
        columnar = pd.DataFrame(columns=["bin_center", "cumulative_proportion"])
        return PlotSummary(params=params, summary_table=summary_table, columnar=columnar)

    if group_series is None:
        counts, bin_edges = np.histogram(x.values, bins=n_bins)
        cumsum = np.cumsum(counts)
        cumsum_norm = cumsum / cumsum[-1] if cumsum[-1] > 0 else cumsum
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        summary_table = pd.DataFrame([{"n": len(x), "min": float(x.min()), "max": float(x.max())}])
        columnar = pd.DataFrame({
            "bin_center": bin_centers,
            "cumulative_proportion": cumsum_norm,
        })
        return PlotSummary(params=params, summary_table=summary_table, columnar=columnar)

    tmp_dict = {"x": x, "g": group_series.loc[x.index].astype(str)}
    if color_series is not None:
        tmp_dict["color"] = color_series.loc[x.index].astype(str)
    tmp = pd.DataFrame(tmp_dict).dropna(subset=["g"])
    if len(tmp) == 0:
        summary_table = pd.DataFrame(columns=[group_col or "group", "n", "min", "max"])
        columnar = pd.DataFrame(columns=["bin_center", "cumulative_proportion", group_col or "group"])
        return PlotSummary(params=params, summary_table=summary_table, columnar=columnar)

    groupby_cols = ["g", "color"] if "color" in tmp.columns else ["g"]
    summary_rows = []
    columnar_dfs = []
    for key, sub in tmp.groupby(groupby_cols, sort=True):
        group_val = key[0] if isinstance(key, tuple) else key
        color_val = key[1] if isinstance(key, tuple) and len(key) > 1 else ""
        x_vals = sub["x"].values
        if len(x_vals) == 0:
            continue
        counts, bin_edges = np.histogram(x_vals, bins=n_bins)
        cumsum = np.cumsum(counts)
        cumsum_norm = cumsum / cumsum[-1] if cumsum[-1] > 0 else cumsum
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        row = {(group_col or "group"): group_val, "n": len(sub), "min": float(np.min(x_vals)), "max": float(np.max(x_vals))}
        if color_grouping:
            row[color_grouping] = color_val
        summary_rows.append(row)
        df = pd.DataFrame({
            "bin_center": bin_centers,
            "cumulative_proportion": cumsum_norm,
        })
        df[group_col or "group"] = group_val
        if color_grouping:
            df[color_grouping] = color_val
        columnar_dfs.append(df)
    summary_table = pd.DataFrame(summary_rows)
    columnar = pd.concat(columnar_dfs, ignore_index=True) if columnar_dfs else pd.DataFrame()
    return PlotSummary(params=params, summary_table=summary_table, columnar=columnar)
