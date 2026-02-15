"""
Plot Pool — Marimo notebook

Select CSV from nicewidgets/data/, then configure and plot. Uses nicewidgets
FigureGenerator with widgets: pre-filter, abs value, plot type, groups/color,
groups/nesting. Swarm/Box/Violin use Groups/Color for x-axis; Scatter/Histogram
use X column for x-axis.

Run:
  cd nicewidgets && uv run marimo edit notebooks/plot_pool_marimo.py
  cd nicewidgets && uv run marimo run notebooks/plot_pool_marimo.py

Requires: pip install nicewidgets marimo  (or uv add marimo)
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import sys
    from pathlib import Path

    # Ensure nicewidgets is importable when run from kymflow_outer or nicewidgets/
    _nb_path = Path(__file__).resolve()
    _nicewidgets_root = _nb_path.parent.parent
    _src = _nicewidgets_root / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    import pandas as pd
    import plotly.graph_objects as go

    from nicewidgets.plot_pool_app.schema import get_config_for_csv, get_data_csv_files, load_csv_for_file
    from nicewidgets.plot_pool_widget.dataframe_processor import DataFrameProcessor
    from nicewidgets.plot_pool_widget.figure_generator import FigureGenerator
    from nicewidgets.plot_pool_widget.plot_helpers import categorical_candidates, numeric_columns
    from nicewidgets.plot_pool_widget.plot_state import PlotState, PlotType
    from nicewidgets.plot_pool_widget.pre_filter_conventions import PRE_FILTER_NONE

    csv_files = get_data_csv_files()
    default_csv = "kym_event_report.csv" if "kym_event_report.csv" in csv_files else (csv_files[0] if csv_files else "kym_event_report.csv")

    return (
        PRE_FILTER_NONE,
        PlotState,
        PlotType,
        csv_files,
        default_csv,
        get_config_for_csv,
        go,
        load_csv_for_file,
        mo,
        numeric_columns,
        categorical_candidates,
    )


@app.cell
def _(csv_files, default_csv, mo):
    csv_file_select = mo.ui.dropdown(
        options=csv_files,
        value=default_csv,
        label="CSV file",
    )
    mo.vstack([mo.md("### Plot Pool"), csv_file_select], gap=1)
    return (csv_file_select,)


@app.cell(hide_code=True)
def _(
    PRE_FILTER_NONE,
    PlotState,
    PlotType,
    categorical_candidates,
    get_config_for_csv,
    load_csv_for_file,
    csv_file_select,
    go,
    numeric_columns,
):
    # Load CSV and apply schema (unique_row_id, pre_filter, etc.)
    _filename = csv_file_select.value or "kym_event_report.csv"
    df = load_csv_for_file(_filename)
    cfg = get_config_for_csv(_filename)
    pre_filter_columns = cfg.pre_filter_columns
    unique_row_id_col = cfg.unique_row_id_col

    data_processor = DataFrameProcessor(
        df, pre_filter_columns=pre_filter_columns, unique_row_id_col=unique_row_id_col
    )
    figure_generator = FigureGenerator(data_processor, unique_row_id_col=unique_row_id_col)

    cat_cols = categorical_candidates(df)
    num_cols = numeric_columns(df)
    xcol_options = ["(none)"] + num_cols + [c for c in cat_cols if c not in num_cols]
    ycol_options = ["(none)"] + num_cols + [c for c in cat_cols if c not in num_cols]
    group_options = ["(none)"] + cat_cols
    pre_filter_roi_options = [PRE_FILTER_NONE] + [
        str(v) for v in data_processor.get_pre_filter_values(pre_filter_columns[0])
    ]

    # Per-file defaults: t_peak/score_peak for kym_event, else first numeric columns
    default_xcol = "t_peak" if "t_peak" in df.columns else (num_cols[0] if num_cols else "")
    default_ycol = "score_peak" if "score_peak" in df.columns else (
        num_cols[1] if len(num_cols) > 1 else (num_cols[0] if num_cols else "")
    )

    return (
        PRE_FILTER_NONE,
        PlotState,
        PlotType,
        data_processor,
        default_xcol,
        default_ycol,
        figure_generator,
        go,
        group_options,
        pre_filter_columns,
        pre_filter_roi_options,
        unique_row_id_col,
        xcol_options,
        ycol_options,
    )


@app.cell
def _(
    PRE_FILTER_NONE,
    PlotType,
    default_xcol,
    default_ycol,
    group_options,
    mo,
    pre_filter_roi_options,
    xcol_options,
    ycol_options,
):
    pre_filter_roi = mo.ui.dropdown(
        options=pre_filter_roi_options,
        value=PRE_FILTER_NONE,
        label="Pre filter roi",
    )
    abs_value_checkbox = mo.ui.checkbox(value=False, label="Absolute value")
    plot_type_select = mo.ui.dropdown(
        options={pt.value.replace("_", " ").title(): pt.value for pt in PlotType},
        value=PlotType.SCATTER.value,
        label="Plot type",
    )
    groups_color = mo.ui.dropdown(
        options=group_options,
        value="(none)",
        label="Groups/Color (x-axis for Swarm/Box/Violin)",
    )
    groups_nesting = mo.ui.dropdown(
        options=group_options,
        value="(none)",
        label="Groups/Nesting (x-axis for Swarm/Box/Violin)",
    )
    _valid_xcols = [c for c in xcol_options if c != "(none)"]
    _valid_ycols = [c for c in ycol_options if c != "(none)"]
    xcol_select = mo.ui.dropdown(
        options={c: c for c in _valid_xcols},
        value=default_xcol if default_xcol in _valid_xcols else (_valid_xcols[0] if _valid_xcols else ""),
        label="X column (Scatter/Histogram)",
    )
    ycol_select = mo.ui.dropdown(
        options={c: c for c in _valid_ycols},
        value=default_ycol if default_ycol in _valid_ycols else (_valid_ycols[0] if _valid_ycols else ""),
        label="Y column",
    )
    plot_btn = mo.ui.run_button(label="Plot")

    mo.vstack(
        [
            pre_filter_roi,
            abs_value_checkbox,
            plot_type_select,
            groups_color,
            groups_nesting,
            xcol_select,
            ycol_select,
            plot_btn,
        ],
        gap=1,
    )
    return (
        abs_value_checkbox,
        groups_color,
        groups_nesting,
        plot_btn,
        plot_type_select,
        pre_filter_roi,
        xcol_select,
        ycol_select,
    )


@app.cell
def _(fig, df_f, mo):
    # Plot display (appears below controls, above gate cell)
    mo.vstack([mo.md(f"**{len(df_f)}** rows (filtered)"), mo.ui.plotly(fig)], gap=1)
    return


@app.cell(hide_code=True)
def _(
    PRE_FILTER_NONE,
    PlotState,
    PlotType,
    abs_value_checkbox,
    data_processor,
    figure_generator,
    go,
    groups_color,
    groups_nesting,
    mo,
    plot_btn,
    plot_type_select,
    pre_filter_columns,
    pre_filter_roi,
    xcol_select,
    ycol_select,
):
    # Gate plot generation behind Plot button
    mo.stop(not plot_btn.value, mo.md("Click **Plot** to generate the figure."))

    # Build PlotState from widget values
    roi_val = pre_filter_roi.value
    pre_filter = {col: roi_val if col == "roi_id" else PRE_FILTER_NONE for col in pre_filter_columns}

    xcol = xcol_select.value or ""
    ycol = ycol_select.value or ""
    group_col = groups_color.value if groups_color.value != "(none)" else None
    color_grouping = groups_nesting.value if groups_nesting.value != "(none)" else None

    try:
        pt_val = str(plot_type_select.value)
        plot_type = PlotType(pt_val) if pt_val in [p.value for p in PlotType] else PlotType.SCATTER
    except (ValueError, TypeError):
        plot_type = PlotType.SCATTER

    state = PlotState(
        pre_filter=pre_filter,
        xcol=xcol,
        ycol=ycol,
        plot_type=plot_type,
        group_col=group_col,
        color_grouping=color_grouping,
        use_absolute_value=abs_value_checkbox.value,
    )

    # Filter data and generate figure
    df_f = data_processor.filter_by_pre_filters(state.pre_filter)
    fig_dict = figure_generator.make_figure(df_f, state)
    fig = go.Figure(fig_dict)
    return df_f, fig


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
