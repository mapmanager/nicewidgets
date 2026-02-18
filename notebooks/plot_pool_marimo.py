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
        DataFrameProcessor,
        FigureGenerator,
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
    DataFrameProcessor,
    FigureGenerator,
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
    pre_filter_columns,
    pre_filter_roi_options,
    xcol_options,
    ycol_options,
):
    # --- NiceGUI pool_control_panel layout mapping ---
    # ui.select -> mo.ui.dropdown, ui.checkbox -> mo.ui.checkbox
    # ui.number -> mo.ui.number(start, stop, step), ui.row -> mo.hstack

    _roi_col = pre_filter_columns[0]
    pre_filter_roi = mo.ui.dropdown(
        options=pre_filter_roi_options,
        value=PRE_FILTER_NONE,
        label=_roi_col,
    )
    abs_value_checkbox = mo.ui.checkbox(value=False, label="Absolute Value")
    use_remove_values_checkbox = mo.ui.checkbox(value=False, label="Remove Values")
    remove_values_threshold = mo.ui.number(
        start=0.0, stop=1000.0, step=0.1, value=0.0,
        label="Remove +/-",
    )

    plot_type_select = mo.ui.dropdown(
        options={pt.value.replace("_", " ").title(): pt.value for pt in PlotType},
        value="Scatter",
        label="Plot type",
    )
    groups_color = mo.ui.dropdown(
        options=group_options,
        value="(none)",
        label="Group/Color",
    )
    groups_nesting = mo.ui.dropdown(
        options=group_options,
        value="(none)",
        label="Group/Nesting",
    )
    ystat_select = mo.ui.dropdown(
        options=["mean", "median", "sum", "count", "std", "sem", "min", "max", "cv"],
        value="mean",
        label="Y stat (grouped)",
    )
    cv_epsilon = mo.ui.number(
        start=1e-20, stop=1.0, step=1e-12, value=1e-10,
        label="CV ε (|μ| < this → NaN)",
    )

    swarm_jitter = mo.ui.number(start=0.0, stop=1.0, step=0.05, value=0.35, label="Swarm Jitter")
    swarm_offset = mo.ui.number(start=0.0, stop=1.0, step=0.05, value=0.3, label="Swarm Offset")

    show_mean_checkbox = mo.ui.checkbox(value=False, label="Mean")
    show_std_sem_checkbox = mo.ui.checkbox(value=False, label="+/-")
    std_sem_select = mo.ui.dropdown(options=["std", "sem"], value="std", label="")

    _valid_xcols = [c for c in xcol_options if c != "(none)"]
    _valid_ycols = [c for c in ycol_options if c != "(none)"]
    xcol_select = mo.ui.dropdown(
        options={c: c for c in _valid_xcols},
        value=default_xcol if default_xcol in _valid_xcols else (_valid_xcols[0] if _valid_xcols else ""),
        label="X column",
    )
    ycol_select = mo.ui.dropdown(
        options={c: c for c in _valid_ycols},
        value=default_ycol if default_ycol in _valid_ycols else (_valid_ycols[0] if _valid_ycols else ""),
        label="Y column",
    )

    mean_line_width = mo.ui.number(start=1, stop=10, step=1, value=2, label="Mean Line Width")
    error_line_width = mo.ui.number(start=1, stop=10, step=1, value=2, label="Error Line Width")
    show_raw_checkbox = mo.ui.checkbox(value=True, label="Raw")
    point_size = mo.ui.number(start=1, stop=20, step=1, value=6, label="Point Size")
    show_legend_checkbox = mo.ui.checkbox(value=True, label="Legend")

    plot_btn = mo.ui.run_button(label="Plot")

    mo.vstack(
        [
            mo.md("**Pre Filter**"),
            pre_filter_roi,
            abs_value_checkbox,
            mo.hstack([use_remove_values_checkbox, remove_values_threshold], justify="start", gap=1),
            mo.md("**Plot**"),
            plot_type_select,
            groups_color,
            groups_nesting,
            ystat_select,
            cv_epsilon,
            mo.hstack([swarm_jitter, swarm_offset], justify="start", gap=1),
            mo.hstack([show_mean_checkbox, show_std_sem_checkbox, std_sem_select], justify="start", gap=1),
            xcol_select,
            ycol_select,
            mo.md("**Plot Options**"),
            mo.hstack([mean_line_width, error_line_width], justify="start", gap=1),
            mo.hstack([show_raw_checkbox, point_size], justify="start", gap=1),
            show_legend_checkbox,
            plot_btn,
        ],
        gap=1,
    )
    return (
        abs_value_checkbox,
        cv_epsilon,
        error_line_width,
        groups_color,
        groups_nesting,
        mean_line_width,
        plot_btn,
        plot_type_select,
        point_size,
        pre_filter_roi,
        remove_values_threshold,
        show_legend_checkbox,
        show_mean_checkbox,
        show_raw_checkbox,
        show_std_sem_checkbox,
        std_sem_select,
        swarm_jitter,
        swarm_offset,
        use_remove_values_checkbox,
        xcol_select,
        ycol_select,
        ystat_select,
    )


@app.cell
def _(fig, df_f, mo):
    mo.vstack([mo.md(f"**{len(df_f)}** rows (filtered)"), mo.ui.plotly(fig)], gap=1)
    return


@app.cell(hide_code=True)
def _(
    PRE_FILTER_NONE,
    PlotState,
    PlotType,
    abs_value_checkbox,
    cv_epsilon,
    data_processor,
    error_line_width,
    figure_generator,
    go,
    groups_color,
    groups_nesting,
    mean_line_width,
    mo,
    plot_btn,
    plot_type_select,
    point_size,
    pre_filter_columns,
    pre_filter_roi,
    remove_values_threshold,
    show_legend_checkbox,
    show_mean_checkbox,
    show_raw_checkbox,
    show_std_sem_checkbox,
    std_sem_select,
    swarm_jitter,
    swarm_offset,
    use_remove_values_checkbox,
    xcol_select,
    ycol_select,
    ystat_select,
):
    mo.stop(not plot_btn.value, mo.md("Click **Plot** to generate the figure."))

    roi_val = pre_filter_roi.value
    pre_filter = {
        col: str(roi_val) if roi_val is not None else PRE_FILTER_NONE
        for col in pre_filter_columns
    }

    xcol = xcol_select.value or ""
    ycol = ycol_select.value or ""
    gv = str(groups_color.value) if groups_color.value else "(none)"
    group_col = None if gv == "(none)" else gv
    cgv = str(groups_nesting.value) if groups_nesting.value else "(none)"
    color_grouping = None if cgv == "(none)" else cgv

    try:
        pt_val = str(plot_type_select.value)
        plot_type = PlotType(pt_val) if pt_val in [p.value for p in PlotType] else PlotType.SCATTER
    except (ValueError, TypeError):
        plot_type = PlotType.SCATTER

    _rvt = remove_values_threshold.value

    state = PlotState(
        pre_filter=pre_filter,
        xcol=xcol,
        ycol=ycol,
        plot_type=plot_type,
        group_col=group_col,
        color_grouping=color_grouping,
        ystat=str(ystat_select.value or "mean"),
        cv_epsilon=float(cv_epsilon.value) if cv_epsilon.value is not None else 1e-10,
        use_absolute_value=bool(abs_value_checkbox.value),
        swarm_jitter_amount=float(swarm_jitter.value or 0.35),
        swarm_group_offset=float(swarm_offset.value or 0.3),
        use_remove_values=bool(use_remove_values_checkbox.value),
        remove_values_threshold=float(_rvt) if _rvt is not None else None,
        show_mean=bool(show_mean_checkbox.value),
        show_std_sem=bool(show_std_sem_checkbox.value),
        std_sem_type=str(std_sem_select.value or "std"),
        mean_line_width=int(mean_line_width.value or 2),
        error_line_width=int(error_line_width.value or 2),
        show_raw=bool(show_raw_checkbox.value),
        point_size=int(point_size.value or 6),
        show_legend=bool(show_legend_checkbox.value),
    )

    df_f = data_processor.filter_by_pre_filters(state.pre_filter)
    fig_dict, _ = figure_generator.make_figure(df_f, state)
    fig = go.Figure(fig_dict)
    return df_f, fig


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
