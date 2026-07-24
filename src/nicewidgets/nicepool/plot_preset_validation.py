"""Runtime validation for NicePool plot presets.

Saved presets are user-owned JSON and can outlive the DataFrame schema that
created them. This module repairs stale preset values against the current
runtime DataFrame before the controller applies them to NiceGUI controls.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from nicewidgets.nicepool.dataframe_processor import DataFrameProcessor
from nicewidgets.nicepool.plot_state import PlotState, PlotType
from nicewidgets.nicepool.pre_filter_conventions import PRE_FILTER_NONE, default_pre_filter

VALID_LAYOUTS: tuple[str, ...] = ("1x1", "1x2", "2x1", "2x2")
VALID_YSTATS: tuple[str, ...] = ("mean", "median", "sum", "count", "std", "sem", "min", "max", "cv")
VALID_STD_SEM_TYPES: tuple[str, ...] = ("std", "sem")


def sanitize_layout(layout: object, *, default: str = "1x1") -> str:
    """Return a supported NicePool layout string.

    Args:
        layout: Candidate layout value loaded from JSON.
        default: Fallback layout.

    Returns:
        Valid layout string.
    """
    candidate = str(layout)
    return candidate if candidate in VALID_LAYOUTS else default


def plot_count_for_layout(layout: str) -> int:
    """Return the number of plot slots for a valid layout string.

    Args:
        layout: Layout string such as ``"1x2"``.

    Returns:
        Number of visible plot slots.
    """
    safe_layout = sanitize_layout(layout)
    rows, columns = safe_layout.split("x")
    return int(rows) * int(columns)


def safe_plot_state_from_dict(data: object, *, default_state: PlotState) -> PlotState:
    """Deserialize a plot state dictionary with fallback to a default state.

    Args:
        data: Candidate serialized plot state.
        default_state: Fallback state used when deserialization fails.

    Returns:
        PlotState instance.
    """
    if not isinstance(data, dict):
        return PlotState.from_dict(default_state.to_dict())
    try:
        return PlotState.from_dict(data)
    except Exception:
        return PlotState.from_dict(default_state.to_dict())


def sanitize_plot_state(
    state: PlotState,
    *,
    df: pd.DataFrame,
    data_processor: DataFrameProcessor,
    pre_filter_columns: list[str],
    default_state: PlotState,
) -> PlotState:
    """Repair a PlotState against the current DataFrame and runtime options.

    Args:
        state: PlotState to validate.
        df: Current source DataFrame.
        data_processor: DataFrameProcessor for current pre-filter values.
        pre_filter_columns: Active pre-filter columns.
        default_state: Fallback state for invalid columns and options.

    Returns:
        Sanitized PlotState safe to bind into the control panel.
    """
    columns = {str(column) for column in df.columns}

    xcol = state.xcol if state.xcol in columns else default_state.xcol
    ycol = state.ycol if state.ycol in columns else default_state.ycol
    if xcol not in columns:
        xcol = _first_column(df)
    if ycol not in columns:
        ycol = _first_numeric_column(df) or _first_column(df)

    group_col = state.group_col if state.group_col in columns else None
    color_grouping = state.color_grouping if state.color_grouping in columns else None

    plot_type = state.plot_type if isinstance(state.plot_type, PlotType) else default_state.plot_type
    ystat = state.ystat if state.ystat in VALID_YSTATS else default_state.ystat
    if ystat not in VALID_YSTATS:
        ystat = "mean"
    std_sem_type = state.std_sem_type if state.std_sem_type in VALID_STD_SEM_TYPES else default_state.std_sem_type
    if std_sem_type not in VALID_STD_SEM_TYPES:
        std_sem_type = "std"

    pre_filter = _sanitize_pre_filter(
        state.pre_filter,
        data_processor=data_processor,
        pre_filter_columns=pre_filter_columns,
    )

    return PlotState(
        pre_filter=pre_filter,
        xcol=xcol,
        ycol=ycol,
        plot_type=plot_type,
        group_col=group_col,
        color_grouping=color_grouping,
        ystat=ystat,
        cv_epsilon=_clamp_float(state.cv_epsilon, minimum=1e-20, maximum=1.0, default=0.01),
        histogram_bins=_clamp_int(state.histogram_bins, minimum=5, maximum=500, default=50),
        use_absolute_value=bool(state.use_absolute_value),
        swarm_jitter_amount=_clamp_float(state.swarm_jitter_amount, minimum=0.0, maximum=1.0, default=0.35),
        swarm_group_offset=_clamp_float(state.swarm_group_offset, minimum=0.0, maximum=1.0, default=0.3),
        use_remove_values=bool(state.use_remove_values),
        remove_values_threshold=_optional_float(state.remove_values_threshold),
        show_mean=bool(state.show_mean),
        show_std_sem=bool(state.show_std_sem),
        std_sem_type=std_sem_type,
        mean_line_width=_clamp_int(state.mean_line_width, minimum=1, maximum=10, default=2),
        error_line_width=_clamp_int(state.error_line_width, minimum=1, maximum=10, default=2),
        show_raw=bool(state.show_raw),
        point_size=_clamp_int(state.point_size, minimum=1, maximum=20, default=6),
        show_legend=bool(state.show_legend),
    )


def sanitize_preset_payload(
    payload: dict[str, Any],
    *,
    df: pd.DataFrame,
    data_processor: DataFrameProcessor,
    pre_filter_columns: list[str],
    default_state: PlotState,
) -> tuple[str, list[PlotState]]:
    """Convert a raw preset payload into a valid layout and plot states.

    Args:
        payload: Raw named preset dictionary from JSON.
        df: Current source DataFrame.
        data_processor: Current DataFrame processor.
        pre_filter_columns: Active pre-filter columns.
        default_state: Fallback state for missing or stale values.

    Returns:
        Tuple of valid layout and sanitized plot states.
    """
    layout = sanitize_layout(payload.get("layout", "1x1"))
    raw_states = payload.get("plot_states", [])
    if not isinstance(raw_states, list):
        raw_states = []
    needed = plot_count_for_layout(layout)
    states: list[PlotState] = []
    for index in range(needed):
        raw_state = raw_states[index] if index < len(raw_states) else default_state.to_dict()
        state = safe_plot_state_from_dict(raw_state, default_state=default_state)
        states.append(
            sanitize_plot_state(
                state,
                df=df,
                data_processor=data_processor,
                pre_filter_columns=pre_filter_columns,
                default_state=default_state,
            )
        )
    return layout, states


def _sanitize_pre_filter(
    values: dict[str, Any],
    *,
    data_processor: DataFrameProcessor,
    pre_filter_columns: list[str],
) -> dict[str, str]:
    result = default_pre_filter(pre_filter_columns)
    for column in pre_filter_columns:
        raw_value = values.get(column, PRE_FILTER_NONE)
        if raw_value is None or raw_value == PRE_FILTER_NONE:
            result[column] = PRE_FILTER_NONE
            continue
        try:
            options = {str(value) for value in data_processor.get_pre_filter_values(column)}
        except Exception:
            options = set()
        value = str(raw_value)
        result[column] = value if value in options else PRE_FILTER_NONE
    return result


def _first_column(df: pd.DataFrame) -> str:
    return str(df.columns[0]) if len(df.columns) else ""


def _first_numeric_column(df: pd.DataFrame) -> str | None:
    numeric = df.select_dtypes(include="number").columns
    return str(numeric[0]) if len(numeric) else None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_float(value: object, *, minimum: float, maximum: float, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, numeric))


def _clamp_int(value: object, *, minimum: int, maximum: int, default: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, numeric))
