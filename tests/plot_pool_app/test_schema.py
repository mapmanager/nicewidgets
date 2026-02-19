"""Unit tests for plot_pool_app schema module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nicewidgets.plot_pool_app.schema import (
    DEFAULT_CSV,
    get_config_for_csv,
    get_data_dir,
    get_data_csv_files,
    load_csv_for_file,
)


def test_default_csv_constant():
    """DEFAULT_CSV is radon_report_db.csv."""
    assert DEFAULT_CSV == "radon_report_db.csv"


def test_get_data_dir_returns_path_ending_in_data():
    """get_data_dir returns a Path whose last component is 'data'."""
    data_dir = get_data_dir()
    assert isinstance(data_dir, Path)
    assert data_dir.name == "data"


def test_get_data_dir_resolves_to_nicewidgets_data():
    """get_data_dir resolves to nicewidgets/data/ relative to package root."""
    data_dir = get_data_dir()
    assert "nicewidgets" in str(data_dir)
    assert data_dir.exists(), f"Expected data dir to exist: {data_dir}"


def test_get_data_csv_files_returns_sorted_list():
    """get_data_csv_files returns sorted list of .csv filenames."""
    files = get_data_csv_files()
    assert isinstance(files, list)
    assert files == sorted(files)
    for f in files:
        assert f.endswith(".csv")


def test_get_data_csv_files_includes_known_files():
    """get_data_csv_files includes radon_report_db and kym_event_db when data dir exists."""
    data_dir = get_data_dir()
    if not data_dir.exists():
        pytest.skip("nicewidgets/data not found (run from repo with data)")
    files = get_data_csv_files()
    assert "radon_report_db.csv" in files
    assert "kym_event_db.csv" in files


def test_load_csv_for_file_radon_adds_unique_row_id():
    """load_csv_for_file adds _unique_row_id for radon_report_db.csv if not present."""
    data_dir = get_data_dir()
    if not (data_dir / "radon_report_db.csv").exists():
        pytest.skip("radon_report_db.csv not found")
    df = load_csv_for_file("radon_report_db.csv")
    assert "_unique_row_id" in df.columns
    assert df["_unique_row_id"].dtype == object
    # Check format: path|roi_id
    sample = df["_unique_row_id"].iloc[0]
    assert "|" in str(sample)


def test_load_csv_for_file_kym_event_keeps_unique_row_id():
    """load_csv_for_file has _unique_row_id for kym_event_db (from file or path|roi_id)."""
    data_dir = get_data_dir()
    if not (data_dir / "kym_event_db.csv").exists():
        pytest.skip("kym_event_db.csv not found")
    df = load_csv_for_file("kym_event_db.csv")
    assert "_unique_row_id" in df.columns
    # Should NOT add unique_row_id (schema uses _unique_row_id)
    assert "unique_row_id" not in df.columns


def test_load_csv_for_file_missing_raises():
    """load_csv_for_file raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_csv_for_file("nonexistent_file.csv")


def test_get_config_for_csv_radon():
    """get_config_for_csv returns correct config for radon_report_db.csv."""
    cfg = get_config_for_csv("radon_report_db.csv")
    assert cfg.pre_filter_columns == ["roi_id", "accepted"]
    assert cfg.unique_row_id_col == "_unique_row_id"
    assert cfg.db_type == "radon_db"
    assert cfg.app_name == "nicewidgets"


def test_get_config_for_csv_kym_event():
    """get_config_for_csv returns correct config for kym_event_db.csv."""
    cfg = get_config_for_csv("kym_event_db.csv")
    assert cfg.pre_filter_columns == ["roi_id"]
    assert cfg.unique_row_id_col == "_unique_row_id"
    assert cfg.db_type == "kym_event_db"
    assert cfg.app_name == "nicewidgets"


def test_get_config_for_csv_unknown_returns_fallback():
    """get_config_for_csv returns generic fallback for unknown filenames."""
    cfg = get_config_for_csv("unknown.csv")
    assert cfg.pre_filter_columns == ["roi_id"]
    assert cfg.unique_row_id_col == "_unique_row_id"
    assert cfg.db_type == "default"
    assert cfg.app_name == "nicewidgets"
