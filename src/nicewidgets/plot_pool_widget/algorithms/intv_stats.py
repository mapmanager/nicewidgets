"""
Interval statistics algorithm — pure pandas/numpy.

Computes inter-event-interval (iei) and instantaneous frequency (inst_freq)
for time-series event columns (e.g. t_start) after filtering the master
dataframe by roi_id, rel_path, and event_type.

Assumptions (documented):
  1. Ordering: Within a filtered group (roi_id, rel_path), events are assumed
     to be already ordered chronologically. The algorithm does not sort.
  2. Zero IEI (Step 2.5): Successive events may share the same timestamp (iei=0),
     assumed to be detection errors (two events cannot occur at the same time).
     We filter them out before aggregation. n_original = count before filtering;
     count = count after filtering.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# Aggregate statistics to compute over iei and inst_freq.
AGG_STATS = ["count", "min", "max", "mean", "std", "sem", "cv"]


# -----------------------------------------------------------------------------
# rel_path parsing: split by '/' into grandparent, parent, tif_file
# -----------------------------------------------------------------------------


def parse_rel_path(rel_path: str) -> dict[str, str]:
    """
    Split rel_path by '/' into grandparent, parent, tif_file.

    Example: '14d Saline/20251020/20251020_A100_0013.tif' →
      grandparent='14d Saline', parent='20251020', tif_file='20251020_A100_0013.tif'

    If fewer than 3 parts: grandparent=first (or ''), parent='' or middle,
    tif_file=last component. Empty rel_path → all ''.
    """
    parts = rel_path.strip().split("/")
    if len(parts) == 0:
        return {"grandparent": "", "parent": "", "tif_file": ""}
    if len(parts) == 1:
        return {"grandparent": parts[0], "parent": "", "tif_file": parts[0]}
    if len(parts) == 2:
        return {"grandparent": parts[0], "parent": "", "tif_file": parts[1]}
    return {
        "grandparent": parts[0],
        "parent": parts[1],
        "tif_file": parts[-1],
    }


# -----------------------------------------------------------------------------
# Step 1: Filter master dataframe by roi_id, rel_path, event_type
# -----------------------------------------------------------------------------


def filter_for_intv_stats(
    df: pd.DataFrame,
    roi_id: Any,
    rel_path: str,
    event_type: str,
) -> pd.DataFrame:
    """
    Filter the master dataframe for interval statistics.

    Requires exactly one roi_id, one rel_path, and one event_type.
    Selections are ANDed. All three filters are required (no "none" option).

    Args:
        df: Master (raw) dataframe.
        roi_id: Selected roi_id value.
        rel_path: Selected rel_path value.
        event_type: Selected event_type value.

    Returns:
        Filtered dataframe.
    """
    df_f = df.copy()
    df_f = df_f[df_f["roi_id"].astype(str) == str(roi_id)]
    df_f = df_f[df_f["rel_path"].astype(str) == str(rel_path)]
    df_f = df_f[df_f["event_type"].astype(str) == str(event_type)]
    return df_f.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Step 2: Extract time series, compute iei and inst_freq
# -----------------------------------------------------------------------------


def compute_iei_and_inst_freq(ts: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Compute iei and inst_freq from a time series of event times.

    Convention: iei[i] and inst_freq[i] are w.r.t. the previous event.
    First event gets nan for both. iei = diff(ts), inst_freq = 1/iei.
    Zero iei yields inst_freq = inf (filtered in Step 2.5).

    Args:
        ts: Numeric series of event times (e.g. t_start), in chronological order.

    Returns:
        (iei_series, inst_freq_series) both aligned to original index.
    """
    ts = pd.to_numeric(ts, errors="coerce")
    iei = ts.diff()  # first value is nan
    inst_freq = 1.0 / iei  # inf when iei=0
    return iei, inst_freq


# -----------------------------------------------------------------------------
# Step 2.5: Filter out iei=0 (detection errors)
# -----------------------------------------------------------------------------


def filter_zero_iei(
    iei: pd.Series,
    inst_freq: pd.Series,
) -> tuple[pd.Series, pd.Series, int]:
    """
    Remove iei=0 (and corresponding inst_freq) before aggregation.

    We assume iei=0 indicates detection errors (two events cannot occur at
    the same time). Set iei=0 and corresponding inst_freq to NaN so they
    are excluded from aggregates.

    Args:
        iei: Inter-event-interval series.
        inst_freq: Instantaneous frequency series.

    Returns:
        (iei_filtered, inst_freq_filtered, n_original)
        n_original = count of non-NaN iei before filtering (excluding first nan).
    """
    iei_f = iei.copy()
    inst_freq_f = inst_freq.copy()
    n_original = iei.notna().sum()
    mask = iei == 0
    iei_f = iei_f.where(~mask, np.nan)
    inst_freq_f = inst_freq_f.where(~mask, np.nan)
    return iei_f, inst_freq_f, int(n_original)


# -----------------------------------------------------------------------------
# Step 3: Aggregate statistics over iei and inst_freq
# -----------------------------------------------------------------------------


def aggregate_intv_stats(
    iei: pd.Series,
    inst_freq: pd.Series,
    n_original: int,
    cv_epsilon: float = 1e-10,
) -> pd.DataFrame:
    """
    Compute count, min, max, mean, std, sem, cv for iei and inst_freq.

    NaN values are dropped when computing aggregates (pandas default).
    cv = std/mean, with NaN when |mean| < cv_epsilon.
    n_original is added as a column (original interval count before filtering iei=0).

    Args:
        n_original: Number of non-NaN iei before Step 2.5 filtering.

    Returns:
        DataFrame with index ['iei', 'inst_freq'], columns = AGG_STATS + n_original.
    """
    def _agg(s: pd.Series) -> dict[str, float]:
        valid = s.dropna()
        n = len(valid)
        if n == 0:
            return {k: np.nan for k in AGG_STATS}
        count = valid.count()
        min_ = valid.min()
        max_ = valid.max()
        mean_ = valid.mean()
        std_ = valid.std(ddof=1)
        sem_ = valid.sem(ddof=1)
        cv_ = np.nan if np.abs(mean_) < cv_epsilon else (std_ / mean_)
        return {
            "count": count,
            "min": min_,
            "max": max_,
            "mean": mean_,
            "std": std_,
            "sem": sem_,
            "cv": cv_,
        }

    rows = [
        _agg(iei),
        _agg(inst_freq),
    ]
    table = pd.DataFrame(rows, index=["iei", "inst_freq"], columns=AGG_STATS)
    table["n_original"] = n_original
    return table


# -----------------------------------------------------------------------------
# Full pipeline: master df → filtered → iei/inst_freq → aggregate table
# -----------------------------------------------------------------------------


def intv_stats(
    df_master: pd.DataFrame,
    time_col: str,
    roi_id: Any,
    rel_path: str,
    event_type: str,
    cv_epsilon: float = 1e-10,
) -> dict[str, Any]:
    """
    Compute interval statistics for a time-series event column.

    Filters by roi_id, rel_path, event_type; computes iei and inst_freq;
    returns aggregate stats table plus metadata.

    Args:
        df_master: Master (raw) dataframe.
        time_col: Column name for event times (e.g. 't_start', 't_peak').
        roi_id: Selected roi_id.
        rel_path: Selected rel_path.
        event_type: Selected event_type.
        cv_epsilon: Epsilon for cv (std/mean) when mean is near zero.

    Returns:
        Dict with:
          - 'metadata': str describing filters and original column.
          - 'table': DataFrame with index ['iei','inst_freq'], columns = AGG_STATS,
            n_original, plus extra: original_column, roi_id, grandparent, parent,
            tif_file, event_type.
          - 'iei': Series of iei per event (first is nan, iei=0 masked to nan).
          - 'inst_freq': Series of inst_freq per event (first is nan, iei=0 masked).
    """
    # Step 1: Filter
    df_f = filter_for_intv_stats(df_master, roi_id, rel_path, event_type)

    # Extract time series (assumption: already ordered within roi_id, rel_path)
    ts = df_f[time_col] if time_col in df_f.columns else pd.Series(dtype=float)

    # Step 2: Compute iei and inst_freq
    iei, inst_freq = compute_iei_and_inst_freq(ts)

    # Step 2.5: Filter out iei=0 (detection errors)
    iei, inst_freq, n_original = filter_zero_iei(iei, inst_freq)

    # Step 3: Aggregate (using filtered iei, inst_freq)
    table = aggregate_intv_stats(iei, inst_freq, n_original=n_original, cv_epsilon=cv_epsilon)

    # Add context columns (rel_path parsed into grandparent, parent, tif_file)
    parsed = parse_rel_path(rel_path)
    table["original_column"] = time_col
    table["roi_id"] = roi_id
    table["grandparent"] = parsed["grandparent"]
    table["parent"] = parsed["parent"]
    table["tif_file"] = parsed["tif_file"]
    table["event_type"] = event_type

    # Metadata string (use str() for cleaner display of numeric roi_id)
    metadata = (
        f"Interval statistics for {time_col!r}\n"
        f"  - filtering by roi_id = {str(roi_id)!r}\n"
        f"  - filtering by rel_path = {rel_path!r}\n"
        f"  - filtering by event_type = {event_type!r}"
    )

    return {
        "metadata": metadata,
        "table": table,
        "iei": iei,
        "inst_freq": inst_freq,
    }


# -----------------------------------------------------------------------------
# Batch: run intv_stats across all unique (rel_path, event_type) after roi_id filter
# -----------------------------------------------------------------------------


def intv_stats_batch(
    df_master: pd.DataFrame,
    time_col: str,
    roi_id: Any,
    cv_epsilon: float = 1e-10,
) -> pd.DataFrame:
    """
    Run intv_stats for all unique (rel_path, event_type) in the roi-filtered df.

    Pre-filters by roi_id, then iterates over each unique (rel_path, event_type),
    runs intv_stats, and concatenates the result tables into one summary DataFrame.

    Args:
        df_master: Master (raw) dataframe.
        time_col: Column name for event times (e.g. 't_start').
        roi_id: Selected roi_id (required for batch).
        cv_epsilon: Epsilon for cv when mean is near zero.

    Returns:
        DataFrame with all per-(rel_path, event_type) results concatenated.
        Columns: grandparent, parent, tif_file, event_type, roi_id, original_column,
        stat_type (iei or inst_freq from index), count, min, max, mean, std, sem, cv.
    """
    df_roi = df_master[df_master["roi_id"].astype(str) == str(roi_id)]
    pairs = (
        df_roi[["rel_path", "event_type"]]
        .drop_duplicates()
        .itertuples(index=False)
    )
    tables = []
    for rel_path, event_type in pairs:
        result = intv_stats(
            df_master,
            time_col=time_col,
            roi_id=roi_id,
            rel_path=rel_path,
            event_type=str(event_type),
            cv_epsilon=cv_epsilon,
        )
        t = result["table"].copy()
        t["stat_type"] = t.index
        t = t.reset_index(drop=True)
        tables.append(t)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


# -----------------------------------------------------------------------------
# Example: load CSV and run (no Plotly)
# -----------------------------------------------------------------------------


def _example_from_csv(
    csv_path: str,
    time_col: str,
    roi_id: Any,
    rel_path: str,
    event_type: str,
) -> None:
    """Load a CSV and run the intv_stats algorithm; print metadata and table."""
    df = pd.read_csv(csv_path)
    result = intv_stats(
        df,
        time_col=time_col,
        roi_id=roi_id,
        rel_path=rel_path,
        event_type=event_type,
    )
    print(result["metadata"])
    print()
    print("Table:")
    print(result["table"].to_string())
    print()
    print("Per-event iei (first 15):")
    print(result["iei"].head(15).to_string())
    print()
    print("Per-event inst_freq (first 15):")
    print(result["inst_freq"].head(15).to_string())


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent.parent.parent.parent.parent
    _data_dir = _root / "data"
    _csv = _data_dir / "kym_event_report.csv"
    if not _csv.exists():
        print(f"Example CSV not found: {_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(_csv)

    # Single example: first distinct (roi_id, rel_path, event_type)
    first = df.drop_duplicates(subset=["roi_id", "rel_path", "event_type"]).iloc[0]
    print("=" * 60)
    print("Single example")
    print("=" * 60)
    _example_from_csv(
        str(_csv),
        time_col="t_start",
        roi_id=first["roi_id"],
        rel_path=first["rel_path"],
        event_type=first["event_type"],
    )

    # Batch: all unique (rel_path, event_type) after roi_id filter
    roi_id = first["roi_id"]
    print()
    print("=" * 60)
    print(f"Batch summary (roi_id={roi_id}, all rel_path × event_type)")
    print("=" * 60)
    batch = intv_stats_batch(df, time_col="t_start", roi_id=roi_id)
    print(batch.to_string())
