"""Schema and data loading for plot pool app.

Provides CSV discovery, loading, and PlotPoolConfig per file.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from nicewidgets.plot_pool_widget.plot_pool_controller import PlotPoolConfig

# Default CSV to load on first visit
DEFAULT_CSV = "radon_report_db.csv"


def get_data_dir() -> Path:
    """Resolve nicewidgets/data/ directory.

    Works when run from kymflow_outer/ or nicewidgets/ project root.
    Package layout: nicewidgets/src/nicewidgets/plot_pool_app/schema.py
    Data: nicewidgets/data/
    """
    # schema.py -> plot_pool_app -> nicewidgets -> src -> nicewidgets project root
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = pkg_root / "data"
    return data_dir


def get_data_csv_files() -> list[str]:
    """List .csv filenames in nicewidgets/data/ (sorted)."""
    data_dir = get_data_dir()
    if not data_dir.exists():
        return []
    files = sorted(f.name for f in data_dir.iterdir() if f.suffix.lower() == ".csv")
    return files


def load_csv_for_file(filename: str) -> pd.DataFrame:
    """Load CSV from nicewidgets/data/ and apply schema prep.

    For radon_report_db.csv: adds unique_row_id (path|roi_id) if not present.
    For kym_event_report.csv: kym_event_id already exists.
    """
    data_dir = get_data_dir()
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    # Radon schema: add unique_row_id if path and roi_id exist, row_id/unique_row_id don't
    if "unique_row_id" not in df.columns and "row_id" not in df.columns:
        if "kym_event_id" not in df.columns and "path" in df.columns and "roi_id" in df.columns:
            df["unique_row_id"] = df["path"].astype(str) + "|" + df["roi_id"].astype(str)
    return df


def get_config_for_csv(filename: str) -> PlotPoolConfig:
    """Return PlotPoolConfig for a given CSV filename."""
    cfg = _CSV_SCHEMA.get(filename)
    if cfg is not None:
        return cfg
    # Generic fallback for unknown CSVs: try common patterns
    return PlotPoolConfig(
        pre_filter_columns=["roi_id"],
        unique_row_id_col="unique_row_id",
        db_type="default",
        app_name="nicewidgets",
    )


_CSV_SCHEMA: dict[str, PlotPoolConfig] = {
    "radon_report_db.csv": PlotPoolConfig(
        pre_filter_columns=["roi_id"],
        unique_row_id_col="unique_row_id",
        db_type="radon_db",
        app_name="nicewidgets",
    ),
    "kym_event_report.csv": PlotPoolConfig(
        pre_filter_columns=["roi_id"],
        unique_row_id_col="kym_event_id",
        db_type="kym_event_db",
        app_name="nicewidgets",
    ),
}
