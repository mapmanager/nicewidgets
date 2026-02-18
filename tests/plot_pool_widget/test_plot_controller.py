"""Basic smoke tests for PlotPoolController and plot generation.

These tests verify that different plot configurations can be generated
without runtime errors. They don't verify visual correctness, just that
the plotting functions execute successfully.
"""

import pytest
import pandas as pd
import numpy as np

from nicewidgets.plot_pool_widget.plot_pool_controller import PlotPoolConfig, PlotPoolController
from nicewidgets.plot_pool_widget.plot_state import PlotState, PlotType
from nicewidgets.plot_pool_widget.pre_filter_conventions import PRE_FILTER_NONE


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    np.random.seed(42)
    n_rows = 100

    data = {
        "roi_id": np.random.choice([1, 2, 3], n_rows),
        "path": [f"path_{i}" for i in range(n_rows)],
        "grandparent_folder": np.random.choice(["A", "B", "C"], n_rows),
        "img_mean": np.random.randn(n_rows) * 10 + 50,
        "vel_mean": np.random.randn(n_rows) * 5 + 20,
        "other_numeric": np.random.randn(n_rows) * 3,
    }

    return pd.DataFrame(data)


@pytest.fixture
def plot_controller(sample_dataframe, tmp_path):
    """Create a PlotPoolController instance for testing. Uses a temp config path to avoid touching user config."""
    cfg = PlotPoolConfig(
        pre_filter_columns=["roi_id"],
        unique_row_id_col="path",
        db_type="test",
        config_path=tmp_path / "pool_plot_config_test.json",
    )
    return PlotPoolController(sample_dataframe, config=cfg)


def test_scatter_plot_basic(plot_controller, sample_dataframe):
    """Test basic scatter plot generation."""
    state = PlotState(
        pre_filter={"roi_id": PRE_FILTER_NONE},
        xcol="img_mean",
        ycol="vel_mean",
        plot_type=PlotType.SCATTER,
    )
    plot_controller.plot_states[0] = state

    df_f = plot_controller.data_processor.filter_by_pre_filters(state.pre_filter)

    # Generate figure
    figure_dict, _ = plot_controller.figure_generator.make_figure(df_f, state)

    assert "data" in figure_dict
    assert len(figure_dict["data"]) > 0


def test_scatter_plot_with_grouping(plot_controller, sample_dataframe):
    """Test scatter plot with group/color."""
    state = PlotState(
        pre_filter={"roi_id": PRE_FILTER_NONE},
        xcol="img_mean",
        ycol="vel_mean",
        plot_type=PlotType.SCATTER,
        group_col="grandparent_folder",
    )
    plot_controller.plot_states[0] = state

    df_f = plot_controller.data_processor.filter_by_pre_filters(state.pre_filter)
    figure_dict, _ = plot_controller.figure_generator.make_figure(df_f, state)

    assert "data" in figure_dict
    assert len(figure_dict["data"]) > 0


def test_box_plot_basic(plot_controller, sample_dataframe):
    """Test basic box plot generation."""
    state = PlotState(
        pre_filter={"roi_id": PRE_FILTER_NONE},
        xcol="img_mean",  # Not used, but required
        ycol="vel_mean",
        plot_type=PlotType.BOX_PLOT,
        group_col="grandparent_folder",
    )
    plot_controller.plot_states[0] = state

    df_f = plot_controller.data_processor.filter_by_pre_filters(state.pre_filter)
    figure_dict, _ = plot_controller.figure_generator.make_figure(df_f, state)

    assert "data" in figure_dict
    assert len(figure_dict["data"]) > 0


def test_box_plot_with_nesting(plot_controller, sample_dataframe):
    """Test box plot with nested grouping."""
    state = PlotState(
        pre_filter={"roi_id": PRE_FILTER_NONE},
        xcol="img_mean",
        ycol="vel_mean",
        plot_type=PlotType.BOX_PLOT,
        group_col="grandparent_folder",
        color_grouping="roi_id",
    )
    plot_controller.plot_states[0] = state

    df_f = plot_controller.data_processor.filter_by_pre_filters(state.pre_filter)
    figure_dict, _ = plot_controller.figure_generator.make_figure(df_f, state)

    assert "data" in figure_dict
    assert len(figure_dict["data"]) > 0


def test_violin_plot_basic(plot_controller, sample_dataframe):
    """Test basic violin plot generation."""
    state = PlotState(
        pre_filter={"roi_id": PRE_FILTER_NONE},
        xcol="img_mean",
        ycol="vel_mean",
        plot_type=PlotType.VIOLIN,
        group_col="grandparent_folder",
    )
    plot_controller.plot_states[0] = state

    df_f = plot_controller.data_processor.filter_by_pre_filters(state.pre_filter)
    figure_dict, _ = plot_controller.figure_generator.make_figure(df_f, state)

    assert "data" in figure_dict
    assert len(figure_dict["data"]) > 0


def test_violin_plot_with_nesting(plot_controller, sample_dataframe):
    """Test violin plot with nested grouping."""
    state = PlotState(
        pre_filter={"roi_id": PRE_FILTER_NONE},
        xcol="img_mean",
        ycol="vel_mean",
        plot_type=PlotType.VIOLIN,
        group_col="grandparent_folder",
        color_grouping="roi_id",
    )
    plot_controller.plot_states[0] = state

    df_f = plot_controller.data_processor.filter_by_pre_filters(state.pre_filter)
    figure_dict, _ = plot_controller.figure_generator.make_figure(df_f, state)

    assert "data" in figure_dict
    assert len(figure_dict["data"]) > 0


def test_swarm_plot_basic(plot_controller, sample_dataframe):
    """Test basic swarm plot generation."""
    state = PlotState(
        pre_filter={"roi_id": PRE_FILTER_NONE},
        xcol="img_mean",
        ycol="vel_mean",
        plot_type=PlotType.SWARM,
        group_col="grandparent_folder",
    )
    plot_controller.plot_states[0] = state

    df_f = plot_controller.data_processor.filter_by_pre_filters(state.pre_filter)
    figure_dict, _ = plot_controller.figure_generator.make_figure(df_f, state)

    assert "data" in figure_dict
    assert len(figure_dict["data"]) > 0


def test_swarm_plot_with_nesting(plot_controller, sample_dataframe):
    """Test swarm plot with nested grouping."""
    state = PlotState(
        pre_filter={"roi_id": PRE_FILTER_NONE},
        xcol="img_mean",
        ycol="vel_mean",
        plot_type=PlotType.SWARM,
        group_col="grandparent_folder",
        color_grouping="roi_id",
    )
    plot_controller.plot_states[0] = state

    df_f = plot_controller.data_processor.filter_by_pre_filters(state.pre_filter)
    figure_dict, _ = plot_controller.figure_generator.make_figure(df_f, state)

    assert "data" in figure_dict
    assert len(figure_dict["data"]) > 0


def test_histogram_plot_basic(plot_controller, sample_dataframe):
    """Test basic histogram plot generation."""
    state = PlotState(
        pre_filter={"roi_id": PRE_FILTER_NONE},
        xcol="img_mean",
        ycol="vel_mean",
        plot_type=PlotType.HISTOGRAM,
    )
    plot_controller.plot_states[0] = state

    df_f = plot_controller.data_processor.filter_by_pre_filters(state.pre_filter)
    figure_dict, _ = plot_controller.figure_generator.make_figure(df_f, state)

    assert "data" in figure_dict
    assert len(figure_dict["data"]) > 0


def test_cumulative_histogram_plot_basic(plot_controller, sample_dataframe):
    """Test basic cumulative histogram plot generation."""
    state = PlotState(
        pre_filter={"roi_id": PRE_FILTER_NONE},
        xcol="img_mean",
        ycol="vel_mean",
        plot_type=PlotType.CUMULATIVE_HISTOGRAM,
    )
    plot_controller.plot_states[0] = state

    df_f = plot_controller.data_processor.filter_by_pre_filters(state.pre_filter)
    figure_dict, _ = plot_controller.figure_generator.make_figure(df_f, state)

    assert "data" in figure_dict
    assert len(figure_dict["data"]) > 0


def test_plot_with_prefilters(plot_controller, sample_dataframe):
    """Test plot generation with pre-filters enabled."""
    state = PlotState(
        pre_filter={"roi_id": 1},
        xcol="img_mean",
        ycol="vel_mean",
        plot_type=PlotType.SCATTER,
        use_absolute_value=True,
        use_remove_values=True,
        remove_values_threshold=100.0,
    )
    plot_controller.plot_states[0] = state

    df_f = plot_controller.data_processor.filter_by_pre_filters(state.pre_filter)
    figure_dict, _ = plot_controller.figure_generator.make_figure(df_f, state)

    assert "data" in figure_dict
    assert len(figure_dict["data"]) > 0


def test_plot_with_mean_std(plot_controller, sample_dataframe):
    """Test plot generation with mean/std traces."""
    state = PlotState(
        pre_filter={"roi_id": PRE_FILTER_NONE},
        xcol="img_mean",
        ycol="vel_mean",
        plot_type=PlotType.SCATTER,
        group_col="grandparent_folder",
        show_mean=True,
        show_std_sem=True,
        std_sem_type="std",
    )
    plot_controller.plot_states[0] = state

    df_f = plot_controller.data_processor.filter_by_pre_filters(state.pre_filter)
    figure_dict, _ = plot_controller.figure_generator.make_figure(df_f, state)

    assert "data" in figure_dict
    assert len(figure_dict["data"]) > 0


def test_config_filename_default(sample_dataframe, tmp_path):
    """_config_filename returns pool_plot_config.json when db_type is 'default'."""
    cfg = PlotPoolConfig(
        pre_filter_columns=["roi_id"],
        unique_row_id_col="path",
        db_type="default",
        config_path=tmp_path / "pool_plot_config.json",
    )
    ctrl = PlotPoolController(sample_dataframe, config=cfg)
    assert ctrl._config_filename() == "pool_plot_config.json"


def test_config_filename_custom(sample_dataframe, tmp_path):
    """_config_filename returns pool_plot_config_{db_type}.json for non-default db_type."""
    cfg = PlotPoolConfig(
        pre_filter_columns=["roi_id"],
        unique_row_id_col="path",
        db_type="radon_db",
        config_path=tmp_path / "pool_plot_config_radon_db.json",
    )
    ctrl = PlotPoolController(sample_dataframe, config=cfg)
    assert ctrl._config_filename() == "pool_plot_config_radon_db.json"


def test_validate_plot_state_columns_fallback(sample_dataframe, tmp_path):
    """Loaded state with missing xcol/ycol gets fallback to first numeric columns."""
    cfg = PlotPoolConfig(
        pre_filter_columns=["roi_id"],
        unique_row_id_col="path",
        db_type="test",
        config_path=tmp_path / "pool_plot_config_test.json",
    )
    ctrl = PlotPoolController(sample_dataframe, config=cfg)
    # sample_dataframe has numeric columns like img_mean, vel_mean, other_numeric
    state = PlotState(
        pre_filter={"roi_id": PRE_FILTER_NONE},
        xcol="nonexistent_x",
        ycol="nonexistent_y",
        plot_type=PlotType.SCATTER,
    )
    validated = ctrl._validate_plot_state_columns(state)
    assert validated.xcol != "nonexistent_x"
    assert validated.ycol != "nonexistent_y"
    assert validated.xcol in ctrl.df.columns
    assert validated.ycol in ctrl.df.columns


def test_validate_plot_state_columns_pre_filter_removed(sample_dataframe, tmp_path):
    """Loaded state with pre_filter key not in df has that key removed."""
    cfg = PlotPoolConfig(
        pre_filter_columns=["roi_id"],
        unique_row_id_col="path",
        db_type="test",
        config_path=tmp_path / "pool_plot_config_test.json",
    )
    ctrl = PlotPoolController(sample_dataframe, config=cfg)
    state = PlotState(
        pre_filter={"roi_id": 1, "not_a_column": "x"},
        xcol="img_mean",
        ycol="vel_mean",
        plot_type=PlotType.SCATTER,
    )
    validated = ctrl._validate_plot_state_columns(state)
    assert "not_a_column" not in validated.pre_filter
    assert "roi_id" in validated.pre_filter
