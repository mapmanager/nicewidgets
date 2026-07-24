"""DataFrame conversion and filtering helpers for NicePool."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


ALL_FILTER_VALUE = "__all__"


def dataframe_to_rows(df: pd.DataFrame, *, unique_row_id_col: str) -> list[dict[str, Any]]:
    """Convert a DataFrame into TableWidget-compatible row dictionaries.

    Args:
        df: Source DataFrame.
        unique_row_id_col: Required unique row-id column.

    Returns:
        List of row dictionaries with JSON-friendly missing values.

    Raises:
        ValueError: If the row-id column is missing, empty, or non-unique.
    """
    if unique_row_id_col not in df.columns:
        raise ValueError(f"DataFrame is missing unique row id column {unique_row_id_col!r}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(df.to_dict(orient="records")):
        row = {str(key): _normalize_value(value) for key, value in raw_row.items()}
        raw_id = row.get(unique_row_id_col)
        if raw_id is None or str(raw_id) == "":
            raise ValueError(f"Row {index} has an empty row id at {unique_row_id_col!r}")
        row_id = str(raw_id)
        if row_id in seen:
            raise ValueError(f"Duplicate row id {row_id!r} at {unique_row_id_col!r}")
        seen.add(row_id)
        row[unique_row_id_col] = row_id
        rows.append(row)
    return rows


def filter_dataframe(df: pd.DataFrame, filters: Mapping[str, object]) -> pd.DataFrame:
    """Return rows matching active categorical filter values.

    Args:
        df: Source DataFrame.
        filters: Mapping from column name to selected value. Values equal to
            ``ALL_FILTER_VALUE`` or None are ignored.

    Returns:
        Filtered DataFrame preserving original row order.
    """
    filtered = df
    for column, value in filters.items():
        if value in (None, ALL_FILTER_VALUE):
            continue
        if column not in filtered.columns:
            continue
        filtered = filtered.loc[filtered[column].map(_filter_value) == str(value)]
    return filtered


def unique_filter_values(df: pd.DataFrame, columns: Sequence[str]) -> dict[str, list[str]]:
    """Return sorted unique values for categorical filter controls.

    Args:
        df: Source DataFrame.
        columns: Candidate filter columns.

    Returns:
        Mapping from column name to sorted string values.
    """
    values: dict[str, list[str]] = {}
    for column in columns:
        if column not in df.columns:
            values[column] = []
            continue
        unique = sorted({_filter_value(value) for value in df[column].tolist()})
        values[column] = unique
    return values


def _normalize_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def _filter_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)
