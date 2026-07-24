"""DataFrame processing for pool plotting application.

This module provides the DataFrameProcessor class for core data manipulation
operations, separating data processing logic from UI/plotting concerns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nicewidgets.nicepool.pre_filter_conventions import PRE_FILTER_NONE


class DataFrameProcessor:
    """Processes DataFrames for plotting operations.

    Encapsulates core DataFrame manipulation operations including filtering,
    value extraction, and statistical calculations. This class separates
    data processing logic from UI/plotting code for better testability
    and reusability.

    Attributes:
        df: The source DataFrame.
        pre_filter_columns: Column names used for pre-filtering.
        unique_row_id_col: Column name containing unique row identifiers.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        pre_filter_columns: list[str],
        unique_row_id_col: str = "path",
    ) -> None:
        """Initialize DataFrameProcessor with dataframe and column configuration.

        Args:
            df: DataFrame containing plot data with required columns.
            pre_filter_columns: Column names for pre-filtering. Each column must
                exist in df. Empty values are allowed so NicePool can initialize
                with an empty DataFrame and later receive data via set_dataframe().
            unique_row_id_col: Column name containing unique row identifiers.

        Raises:
            ValueError: If required columns are missing.
        """
        self.df = df
        self.pre_filter_columns = list(pre_filter_columns)
        self.unique_row_id_col = unique_row_id_col

        if self.unique_row_id_col not in df.columns:
            raise ValueError(f"df must contain required unique id column {unique_row_id_col!r}")

        for col in self.pre_filter_columns:
            if col not in df.columns:
                raise ValueError(f"df must contain pre_filter column {col!r}")

    def get_pre_filter_values(self, column: str) -> list[Any]:
        """Get sorted list of unique values for a pre-filter column.

        Args:
            column: Must be in pre_filter_columns.

        Returns:
            Sorted list of unique values with type preserved from dataframe.
            Empty columns return an empty list.
        """
        if column not in self.pre_filter_columns:
            raise ValueError(f"Unknown pre_filter column {column!r}")
        s = self.df[column].dropna()
        vals = sorted(set(s.tolist()), key=lambda x: (str(x), x))
        return vals

    def filter_by_pre_filters(self, selections: dict[str, Any]) -> pd.DataFrame:
        """Filter dataframe by pre-filter column selections.

        Args:
            selections: Map column name to selected value. PRE_FILTER_NONE means
                no filter for that column.

        Returns:
            Filtered dataframe using AND across columns, with rows missing
            unique_row_id_col removed.
        """
        df_f = self.df
        for col in self.pre_filter_columns:
            val = selections.get(col, PRE_FILTER_NONE)
            if val is None or val == PRE_FILTER_NONE:
                continue
            df_f = df_f[df_f[col].astype(str) == str(val)]
        df_f = df_f.dropna(subset=[self.unique_row_id_col])
        return df_f

    def build_row_id_index(self, df_f: pd.DataFrame) -> dict[str, int]:
        """Build mapping from unique_row_id to iloc index in filtered dataframe.

        Args:
            df_f: Filtered dataframe to build index for.

        Returns:
            Dictionary mapping unique_row_id as string to iloc index.
        """
        row_ids = df_f[self.unique_row_id_col].astype(str).tolist()
        return {rid: i for i, rid in enumerate(row_ids)}

    def get_y_values(
        self,
        df_f: pd.DataFrame,
        ycol: str,
        use_absolute: bool = False,
        use_remove_values: bool = False,
        remove_values_threshold: float | None = None,
    ) -> pd.Series:
        """Get y column values with optional numeric transformations.

        Args:
            df_f: Filtered dataframe.
            ycol: Column name for y values.
            use_absolute: If True, apply absolute value to numeric values.
            use_remove_values: If True, remove values outside the threshold.
            remove_values_threshold: Threshold for remove values.

        Returns:
            Series of y values with transformations applied.
        """
        y = pd.to_numeric(df_f[ycol], errors="coerce")
        if use_absolute:
            y = y.abs()
        if use_remove_values and remove_values_threshold is not None:
            y[(y < -remove_values_threshold) | (y > remove_values_threshold)] = np.nan
        return y

    def get_x_values(
        self,
        df_f: pd.DataFrame,
        xcol: str,
        use_absolute: bool = False,
        use_remove_values: bool = False,
        remove_values_threshold: float | None = None,
    ) -> pd.Series:
        """Get x column values for plotting.

        Numeric columns optionally receive absolute-value and threshold
        transformations. Non-numeric columns are returned unchanged.

        Args:
            df_f: Filtered dataframe.
            xcol: Column name for x values.
            use_absolute: If True and column is numeric, apply absolute value.
            use_remove_values: If True, remove values outside the threshold.
            remove_values_threshold: Threshold for remove values.

        Returns:
            Series of x values.
        """
        if xcol not in df_f.columns:
            return pd.Series(dtype=float)
        col = df_f[xcol]
        kind = getattr(col.dtype, "kind", None)
        if kind in ("i", "u", "f"):
            x = pd.to_numeric(col, errors="coerce")
            if use_absolute:
                x = x.abs()
            if use_remove_values and remove_values_threshold is not None:
                x[(x < -remove_values_threshold) | (x > remove_values_threshold)] = np.nan
            return x
        return col

    def calculate_group_stats(
        self,
        df_f: pd.DataFrame,
        group_col: str,
        ycol: str,
        use_absolute: bool = False,
        xcol: str | None = None,
        include_x: bool = False,
        use_remove_values: bool = False,
        remove_values_threshold: float | None = None,
    ) -> dict[str, dict[str, float]]:
        """Calculate mean, standard deviation, and SEM within each group.

        Args:
            df_f: Filtered dataframe with group column and y values.
            group_col: Column name for grouping.
            ycol: Column name for y values.
            use_absolute: If True, apply absolute value before calculation.
            xcol: Column name for x values when include_x is True.
            include_x: If True, also calculate stats for x values.
            use_remove_values: If True, remove values outside the threshold.
            remove_values_threshold: Threshold for remove values.

        Returns:
            Dictionary mapping group value to summary statistics.
        """
        if not group_col:
            return {}

        y = self.get_y_values(df_f, ycol, use_absolute, use_remove_values, remove_values_threshold)
        g = df_f[group_col].astype(str)

        if include_x:
            if not xcol:
                raise ValueError("xcol is required when include_x=True")
            x = self.get_x_values(df_f, xcol, use_absolute, use_remove_values, remove_values_threshold)
            tmp = pd.DataFrame({"x": x, "y": y, "g": g}).dropna(subset=["y", "g", "x"])
        else:
            tmp = pd.DataFrame({"y": y, "g": g}).dropna(subset=["y", "g"])

        stats = {}
        for group_value, sub in tmp.groupby("g", sort=True):
            y_values = sub["y"].values
            if len(y_values) == 0:
                continue

            mean_val = float(np.mean(y_values))
            std_val = float(np.std(y_values, ddof=1))
            sem_val = std_val / np.sqrt(len(y_values)) if len(y_values) > 1 else 0.0

            group_stats = {
                "mean": mean_val,
                "std": std_val,
                "sem": sem_val,
            }

            if include_x:
                x_values = sub["x"].values
                if len(x_values) > 0:
                    x_mean_val = float(np.mean(x_values))
                    x_std_val = float(np.std(x_values, ddof=1))
                    x_sem_val = x_std_val / np.sqrt(len(x_values)) if len(x_values) > 1 else 0.0
                    group_stats.update(
                        {
                            "x_mean": x_mean_val,
                            "x_std": x_std_val,
                            "x_sem": x_sem_val,
                        }
                    )

            stats[str(group_value)] = group_stats

        return stats