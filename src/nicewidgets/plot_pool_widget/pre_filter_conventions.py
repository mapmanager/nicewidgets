"""Pre-filter conventions for PlotPoolController.

Single source of truth for sentinel value and rules so the codebase stays
consistent across PlotState, DataFrameProcessor, PoolControlPanel, etc.
"""

# Sentinel value meaning "no filter" for a pre-filter column.
# Used in: UI dropdown options, PlotState.pre_filter, DataFrameProcessor.filter_by_pre_filters.
PRE_FILTER_NONE = "(none)"


def default_pre_filter(pre_filter_columns: list[str]) -> dict[str, str]:
    """Build initial pre_filter dict for new/default plot states.

    Rule: All columns start with no filter (PRE_FILTER_NONE).
    This yields an unfiltered dataframe until the user selects values.

    Args:
        pre_filter_columns: Column names used for pre-filtering.

    Returns:
        Dict mapping each column to PRE_FILTER_NONE.
    """
    return {col: PRE_FILTER_NONE for col in pre_filter_columns}


def is_filtered(selections: dict[str, object]) -> bool:
    """True if any selection applies a filter (is not PRE_FILTER_NONE)."""
    return any(v != PRE_FILTER_NONE for v in selections.values())


def format_pre_filter_display(selections: dict[str, object]) -> str:
    """Short label for trace names / tooltips: active pre-filter or 'All'."""
    if not is_filtered(selections):
        return "All"
    parts = [f"{k}={v}" for k, v in selections.items() if v != PRE_FILTER_NONE]
    return ", ".join(parts)
