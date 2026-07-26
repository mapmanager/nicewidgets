"""Synthetic sample data for the NicePool demo.

Pure data module: no NiceGUI imports. NicePool consumes a
:class:`pandas.DataFrame`, so this module authors rows as ``list[dict]`` (the
same shape that ``AcqImageList`` analysis pools return from
``get_schema_rows()``) and converts them with :func:`rows_to_dataframe`.

The column schema deliberately mirrors ``acqstore``'s ``VelocityAnalysisPool``
(identity + pre-filter columns, experiment metadata, and a few velocity/heart
metrics). In v2 the demo will swap :class:`SampleDataCatalog.get_dataframe` for
``images.velocity_analysis_pool.get_dataframe()`` with no widget-side changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

UNIQUE_ROW_ID_COL = "pool_row_id"

PRE_FILTER_COLUMNS: tuple[str, ...] = ("accept", "channel", "roi_id")

NUMERIC_COLUMNS: tuple[str, ...] = (
    "velocity_mean",
    "velocity_std",
    "hr_bpm",
    "event_count",
)


def build_pool_row_id(file_id: str, *, channel: int, roi_id: int) -> str:
    """Return a stable unique row id, matching the acqstore pool convention.

    Args:
        file_id: Stable acquisition-file identifier (here, the file path).
        channel: Zero-based channel index.
        roi_id: ROI identifier.

    Returns:
        Stable string suitable for the NicePool ``unique_row_id_col`` contract.
    """
    return f"{file_id}|channel={int(channel)}|roi_id={int(roi_id)}"


@dataclass(frozen=True)
class DemoExperiment:
    """One synthetic experiment (a ``grandparent`` folder of acquisitions).

    Args:
        grandparent: Top-level experiment label used for grouping in plots.
        condition: Experimental condition applied to every file in the group.
        file_count: Number of acquisition files to synthesize.
        channels: Channel indices present in every file.
        roi_ids: ROI ids measured per channel.
        velocity_center: Mean velocity around which per-row values are drawn.
    """

    grandparent: str
    condition: str
    file_count: int
    channels: tuple[int, ...]
    roi_ids: tuple[int, ...]
    velocity_center: float


class SampleDataCatalog:
    """Provide deterministic synthetic velocity-pool DataFrames.

    The catalog exposes one or more named datasets. Each dataset expands a set
    of :class:`DemoExperiment` definitions into one pool row per
    file/channel/ROI, matching the ``VelocityAnalysisPool`` row grain.
    """

    def __init__(self, *, seed: int = 0) -> None:
        self._seed = seed
        self._datasets: dict[str, tuple[DemoExperiment, ...]] = {
            "Velocity pool (3 experiments)": (
                DemoExperiment(
                    grandparent="ctrl_2025",
                    condition="control",
                    file_count=4,
                    channels=(0, 1),
                    roi_ids=(1, 2),
                    velocity_center=1200.0,
                ),
                DemoExperiment(
                    grandparent="drugA_2025",
                    condition="drugA",
                    file_count=3,
                    channels=(0, 1),
                    roi_ids=(1,),
                    velocity_center=1850.0,
                ),
                DemoExperiment(
                    grandparent="drugB_2025",
                    condition="drugB",
                    file_count=3,
                    channels=(0, 1),
                    roi_ids=(1, 2),
                    velocity_center=650.0,
                ),
            ),
            "Velocity pool (small)": (
                DemoExperiment(
                    grandparent="ctrl_pilot",
                    condition="control",
                    file_count=2,
                    channels=(0,),
                    roi_ids=(1,),
                    velocity_center=1000.0,
                ),
                DemoExperiment(
                    grandparent="drug_pilot",
                    condition="drugA",
                    file_count=2,
                    channels=(0,),
                    roi_ids=(1,),
                    velocity_center=1700.0,
                ),
            ),
        }

    @property
    def names(self) -> list[str]:
        """Return selectable dataset names."""
        return list(self._datasets)

    def get_rows(self, name: str) -> list[dict[str, object]]:
        """Return one dataset as ``list[dict]`` pool rows.

        Args:
            name: Dataset name from :attr:`names`.

        Returns:
            List of row dictionaries, one per file/channel/ROI.
        """
        experiments = self._datasets[name]
        rng = np.random.default_rng(self._seed + abs(hash(name)) % 10_000)
        rows: list[dict[str, object]] = []
        pool_row = 0
        for experiment in experiments:
            for file_index in range(experiment.file_count):
                file_stem = f"{experiment.grandparent}_f{file_index:02d}"
                path = f"/data/{experiment.grandparent}/{file_stem}.tif"
                for channel in experiment.channels:
                    for roi_id in experiment.roi_ids:
                        rows.append(
                            self._make_row(
                                experiment=experiment,
                                pool_row=pool_row,
                                file_stem=file_stem,
                                path=path,
                                channel=channel,
                                roi_id=roi_id,
                                rng=rng,
                            )
                        )
                        pool_row += 1
        return rows

    def get_dataframe(self, name: str) -> pd.DataFrame:
        """Return one dataset as a NicePool-ready DataFrame.

        Args:
            name: Dataset name from :attr:`names`.

        Returns:
            DataFrame with a unique ``pool_row_id`` column.
        """
        return rows_to_dataframe(self.get_rows(name))

    @staticmethod
    def _make_row(
        *,
        experiment: DemoExperiment,
        pool_row: int,
        file_stem: str,
        path: str,
        channel: int,
        roi_id: int,
        rng: np.random.Generator,
    ) -> dict[str, object]:
        """Build one synthetic pool row for a file/channel/ROI selection."""
        velocity_mean = float(rng.normal(experiment.velocity_center, 180.0))
        velocity_std = float(abs(rng.normal(experiment.velocity_center * 0.15, 40.0)))
        hr_bpm = float(rng.normal(310.0, 25.0))
        event_count = int(max(0, round(rng.normal(12.0, 4.0))))
        return {
            "pool_row_id": build_pool_row_id(path, channel=channel, roi_id=roi_id),
            "pool_row": pool_row,
            "name": f"{file_stem}.tif",
            "path": path,
            "parent": experiment.grandparent,
            "grandparent": experiment.grandparent,
            "condition": experiment.condition,
            # accept is a categorical pre-filter; most rows accepted, a few not.
            "accept": bool(rng.random() > 0.2),
            "channel": int(channel),
            "roi_id": int(roi_id),
            "velocity_mean": velocity_mean,
            "velocity_std": velocity_std,
            "hr_bpm": hr_bpm,
            "event_count": event_count,
        }


def rows_to_dataframe(rows: Sequence[dict[str, object]]) -> pd.DataFrame:
    """Convert pool rows to a DataFrame, enforcing the row-id contract.

    Args:
        rows: Pool rows as produced by :meth:`SampleDataCatalog.get_rows`.

    Returns:
        DataFrame built from ``rows``.

    Raises:
        ValueError: If ``pool_row_id`` values are missing or not unique.
    """
    df = pd.DataFrame(list(rows))
    if UNIQUE_ROW_ID_COL not in df.columns:
        raise ValueError(f"rows are missing required column {UNIQUE_ROW_ID_COL!r}")
    if not df[UNIQUE_ROW_ID_COL].is_unique:
        raise ValueError(f"{UNIQUE_ROW_ID_COL!r} values must be unique")
    return df
