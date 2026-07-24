"""Plot configuration and data errors for NicePool figure generation."""

from __future__ import annotations

import textwrap

import pandas as pd

from nicewidgets.nicepool.plot_helpers import is_categorical_column
from nicewidgets.nicepool.plot_state import PlotType


class PlotConfigurationError(ValueError):
    """Raised when plot state is invalid for the current dataframe."""


class PlotDataError(ValueError):
    """Raised when filtered data cannot satisfy the requested plot."""


def _value_type_bucket(value: object) -> str:
    """Return a coarse type bucket for categorical comparability checks."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return type(value).__name__


def _categorical_value_to_label(value: object) -> str | None:
    """Convert one categorical cell value to a plot label, or None when missing."""
    if value is None or pd.isna(value):
        return None
    return str(value)


def require_comparable_categorical_column(
    series: pd.Series,
    *,
    column_name: str,
    role: str,
) -> None:
    """Require that non-missing values in a categorical column share one type family.

    Args:
        series: Column values after filtering.
        column_name: Column name for user-facing errors.
        role: Short role label such as ``"group axis"`` or ``"color grouping"``.

    Raises:
        PlotDataError: When non-missing values mix incompatible scalar types.
    """
    non_null = series.dropna()
    if non_null.empty:
        return
    buckets = {_value_type_bucket(value) for value in non_null}
    if len(buckets) == 1:
        return
    if buckets <= {"int", "float"}:
        return
    bucket_text = ", ".join(sorted(buckets))
    raise PlotDataError(
        f"Column {column_name!r} has mixed value types ({bucket_text}) and cannot be used as "
        f"{role}. Use consistent metadata types (for example all numbers or all text) or choose "
        "another column."
    )


def prepare_categorical_column(
    series: pd.Series,
    *,
    column_name: str,
    role: str,
) -> tuple[pd.Series, list[str]]:
    """Normalize one categorical column to string labels and sorted unique categories.

    Args:
        series: Column values after filtering.
        column_name: Column name for user-facing errors.
        role: Short role label such as ``"group axis"`` or ``"color grouping"``.

    Returns:
        Tuple of (label series aligned to ``series.index``, sorted unique labels).

    Raises:
        PlotDataError: When values are mixed-type or all missing after normalization.
    """
    require_comparable_categorical_column(series, column_name=column_name, role=role)
    labels = series.map(_categorical_value_to_label)
    labeled = labels.dropna()
    if labeled.empty:
        raise PlotDataError(
            f"No valid values in column {column_name!r} for {role} after filters. "
            "Check metadata, pre-filters, or choose another column."
        )
    unique_labels = sorted(labeled.unique())
    return labels, unique_labels


_PLOT_TYPE_LABELS: dict[PlotType, str] = {
    PlotType.SCATTER: "Scatter plot",
    PlotType.SWARM: "Swarm plot",
    PlotType.BOX_PLOT: "Box plot",
    PlotType.VIOLIN: "Violin plot",
    PlotType.GROUPED: "Grouped plot",
    PlotType.HISTOGRAM: "Histogram",
    PlotType.CUMULATIVE_HISTOGRAM: "Cumulative histogram",
}


def plot_type_label(plot_type: PlotType) -> str:
    """Return a user-facing label for a plot type.

    Args:
        plot_type: Plot type enum value.

    Returns:
        Short label suitable for notifications.
    """
    return _PLOT_TYPE_LABELS.get(plot_type, plot_type.value.replace("_", " ").title())


def require_group_col(group_col: str | None, *, plot_type: PlotType) -> None:
    """Require a group column for plot types that need categorical x-axis grouping.

    Args:
        group_col: Selected group column, if any.
        plot_type: Requested plot type.

    Raises:
        PlotConfigurationError: When ``group_col`` is missing.
    """
    if group_col:
        return
    label = plot_type_label(plot_type)
    raise PlotConfigurationError(
        f"{label} requires a Group column. Select a categorical column such as parent or roi_id "
        "in the control panel."
    )


def require_categorical_group_col(
    df: pd.DataFrame,
    group_col: str | None,
    *,
    plot_type: PlotType,
) -> None:
    """Require a categorical group column for box, violin, and swarm plots.

    Args:
        df: Filtered dataframe used for plotting.
        group_col: Selected group column, if any.
        plot_type: Requested plot type.

    Raises:
        PlotConfigurationError: When ``group_col`` is missing or not categorical.
    """
    require_group_col(group_col, plot_type=plot_type)
    assert group_col is not None
    if group_col not in df.columns:
        label = plot_type_label(plot_type)
        raise PlotConfigurationError(
            f"{label} group column {group_col!r} is not in the current data. "
            "Choose another Group column or refresh the pool data."
        )
    if is_categorical_column(df, group_col):
        return
    label = plot_type_label(plot_type)
    raise PlotConfigurationError(
        f"{label} requires a categorical Group column. {group_col!r} has too many unique values. "
        "Choose a low-cardinality column such as parent or roi_id, or switch to Scatter plot."
    )


def require_histogram_x_values(x: pd.Series, *, xcol: str, plot_type: PlotType) -> None:
    """Require non-empty numeric x values for histogram plot types.

    Args:
        x: Candidate x values after filtering and coercion.
        xcol: Column name shown in the error message.
        plot_type: Histogram or cumulative histogram plot type.

    Raises:
        PlotDataError: When no valid x values remain.
    """
    if len(x) > 0:
        return
    label = plot_type_label(plot_type)
    raise PlotDataError(
        f"{label} has no valid values for column {xcol!r} after filters. "
        "Widen pre-filters, choose another x column, or check remove-values settings."
    )


def _format_plot_error_text(message: str, *, width: int = 40) -> str:
    """Wrap plot-area error text for Plotly annotation display.

    Plotly annotation ``text`` supports ``<br>`` line breaks. Long single-line
    messages overflow narrow plot panels, so wrap at a fixed character width.

    Args:
        message: User-facing error message.
        width: Maximum characters per line before wrapping.

    Returns:
        Message with Plotly-friendly line breaks.
    """
    lines = textwrap.wrap(
        message.strip(),
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )
    if not lines:
        return message
    return "<br>".join(lines)


def empty_plotly_figure(message: str, *, wrap_width: int = 40) -> dict:
    """Return a minimal Plotly figure dict that displays an error message.

    Args:
        message: User-facing text to show in the plot area.
        wrap_width: Maximum characters per line in the plot annotation.

    Returns:
        Plotly figure dictionary with no data traces.
    """
    display_text = _format_plot_error_text(message, width=wrap_width)
    return {
        "data": [],
        "layout": {
            "annotations": [
                {
                    "text": display_text,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "align": "center",
                    "xanchor": "center",
                    "yanchor": "middle",
                }
            ],
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "margin": {"l": 20, "r": 20, "t": 20, "b": 20},
        },
    }
