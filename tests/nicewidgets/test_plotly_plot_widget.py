"""Tests for the reusable PlotlyPlotWidget public API helpers."""

from __future__ import annotations

from typing import Any

import pytest

from nicewidgets.plotly_plot.models import (
    MeasurementChangeEvent,
    PlotlyAxisRange,
    PlotlyScatterData,
    PlotlySeriesMenuItem,
    PlotlyTraceData,
)
from nicewidgets.plotly_layout_margins import PlotlyLayoutMarginsProfile
from nicewidgets.plotly_plot.display_options import PlotlyPlotDisplayOptions
from nicewidgets.plotly_plot.event_overlay import PlotlyEventOverlay
from nicewidgets.plotly_plot.widget import (
    PlotlyPlotWidget,
    build_plotly_figure_dict,
    extract_rect_selection_x_range_from_relayout,
    resolve_plot_layout_margins,
)


class _FakeClient:
    """Capture JavaScript pushed by the widget during tests."""

    def __init__(self) -> None:
        """Create an empty JavaScript call recorder."""
        self.calls: list[str] = []

    def run_javascript(self, js: str, *, timeout: float | None = None) -> None:
        """Record JavaScript instead of sending it to a browser.

        Args:
            js: JavaScript source.
            timeout: Optional NiceGUI timeout argument.
        """
        self.calls.append(js)


class _FakePlotlyElement:
    """Small stand-in for NiceGUI's Plotly element."""

    def __init__(self, figure: dict[str, Any]) -> None:
        """Create a fake element.

        Args:
            figure: Figure dictionary passed to ``ui.plotly``.
        """
        self.figure = figure
        self.id = 123
        self.client = _FakeClient()
        self.handlers: dict[str, Any] = {}

    def classes(self, *_args: str, **_kwargs: Any) -> _FakePlotlyElement:
        """Return self for chaining."""
        return self

    def on(self, event_name: str, handler: Any) -> None:
        """Record event handlers registered by the widget.

        Args:
            event_name: NiceGUI event name.
            handler: Callback registered for the event.
        """
        self.handlers[event_name] = handler

    def update(self) -> None:
        """Record NiceGUI figure sync calls."""
        return None


class _FakeUiElement:
    """Small stand-in for generic NiceGUI container elements."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.visible = True

    def classes(self, *_args: str, **_kwargs: Any) -> _FakeUiElement:
        """Return self for chaining."""
        return self

    def set_visibility(self, visible: bool) -> None:
        """Record visibility changes."""
        self.visible = bool(visible)

    def __enter__(self) -> _FakeUiElement:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


class _FakeUiLabel:
    """Small stand-in for NiceGUI labels."""

    def __init__(self, text: str = "") -> None:
        self.text = text

    def classes(self, *_args: str, **_kwargs: Any) -> _FakeUiLabel:
        """Return self for chaining."""
        return self


@pytest.fixture
def fake_plotly(monkeypatch: pytest.MonkeyPatch) -> list[_FakePlotlyElement]:
    """Patch NiceGUI UI factories and return created fake Plotly elements."""
    created: list[_FakePlotlyElement] = []

    def plotly_factory(figure: dict[str, Any]) -> _FakePlotlyElement:
        element = _FakePlotlyElement(figure)
        created.append(element)
        return element

    monkeypatch.setattr("nicewidgets.plotly_plot.widget.ui.plotly", plotly_factory)
    monkeypatch.setattr("nicewidgets.plotly_plot.widget.ui.element", _FakeUiElement)
    monkeypatch.setattr("nicewidgets.plotly_plot.widget.ui.label", _FakeUiLabel)
    monkeypatch.setattr(
        "nicewidgets.plotly_plot.widget.ui.add_head_html",
        lambda *_args, **_kwargs: None,
    )
    return created


class _RelayoutEvent:
    """Fake NiceGUI event object with Plotly relayout args."""

    def __init__(self, args: dict[str, Any]) -> None:
        """Create a fake relayout event.

        Args:
            args: Plotly relayout payload.
        """
        self.args = args


def test_trace_data_validates_lengths() -> None:
    """Continuous traces should reject mismatched x/y lengths."""
    with pytest.raises(ValueError):
        PlotlyTraceData.from_sequences(name="trace", x=[0.0, 1.0], y=[1.0])


def test_scatter_data_validates_lengths() -> None:
    """Scatter overlays should reject mismatched x/y lengths."""
    with pytest.raises(ValueError):
        PlotlyScatterData.from_sequences(name="peaks", x=[0.0], y=[1.0, 2.0])


def test_axis_range_validates_bounds() -> None:
    """Axis ranges should reject partial or inverted bounds."""
    with pytest.raises(ValueError):
        PlotlyAxisRange(x_min=2.0, x_max=1.0)
    with pytest.raises(ValueError):
        PlotlyAxisRange(x_min=2.0, x_max=None)


def test_build_plotly_figure_dict_includes_config_and_shapes() -> None:
    """Figure dict should include NiceGUI-compatible Plotly config."""
    figure = build_plotly_figure_dict(
        x_label="time",
        y_label="df/f0",
        x_range=PlotlyAxisRange(0.0, 1.0),
        shapes=[{"type": "line"}],
    )

    assert "title" not in figure["layout"]
    assert figure["layout"]["xaxis"]["range"] == [0.0, 1.0]
    assert figure["layout"]["xaxis"]["autorange"] is False
    assert figure["layout"]["shapes"] == [{"type": "line"}]
    assert figure["layout"]["legend"]["orientation"] == "h"
    assert figure["layout"]["margin"]["b"] == 40
    assert figure["layout"]["paper_bgcolor"] == "white"
    assert figure["config"]["editable"] is True
    assert figure["config"]["scrollZoom"] is True
    assert figure["config"]["edits"]["titleText"] is False


def test_resolve_plot_layout_margins_bottom_by_axis_and_legend() -> None:
    """Bottom margin should follow axis-label and legend visibility."""
    assert resolve_plot_layout_margins(
        show_axis_labels=False,
        show_legend=False,
    )["b"] == 8
    assert resolve_plot_layout_margins(
        show_axis_labels=False,
        show_legend=True,
    )["b"] == 40
    assert resolve_plot_layout_margins(
        show_axis_labels=True,
        show_legend=False,
    )["b"] == 40
    assert resolve_plot_layout_margins(
        show_axis_labels=True,
        show_legend=True,
    )["b"] == 72


def test_resolve_plot_layout_margins_uses_profile_when_provided() -> None:
    """A layout profile should bypass default legend and dual-axis margin rules."""
    profile = PlotlyLayoutMarginsProfile(
        with_axis_labels={"l": 60, "r": 24, "t": 10, "b": 40},
        compact={"l": 8, "r": 8, "t": 8, "b": 8},
        stabilize_axis_automargin=True,
    )

    assert resolve_plot_layout_margins(
        show_axis_labels=True,
        show_legend=True,
        has_yaxis2=True,
        layout_margins_profile=profile,
    ) == {"l": 60, "r": 24, "t": 10, "b": 40}


def test_layout_margins_profile_pins_automargin_and_margins(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Stack margin profiles should apply fixed margins and disable automargin."""
    profile = PlotlyLayoutMarginsProfile(
        with_axis_labels={"l": 60, "r": 24, "t": 10, "b": 40},
        compact={"l": 8, "r": 8, "t": 8, "b": 8},
        stabilize_axis_automargin=True,
    )
    widget = PlotlyPlotWidget(layout_margins_profile=profile)

    assert widget.figure["layout"]["margin"] == {"l": 8, "r": 8, "t": 8, "b": 8}
    assert widget.figure["layout"]["xaxis"]["automargin"] is False
    assert widget.figure["layout"]["yaxis"]["automargin"] is False

    widget.set_x_axis_labels_visible(True)

    assert widget.figure["layout"]["margin"] == {"l": 60, "r": 24, "t": 10, "b": 40}


def test_build_plotly_figure_dict_applies_dark_theme() -> None:
    """Dark theme should set Plotly layout colors."""
    figure = build_plotly_figure_dict(theme="dark")

    assert figure["layout"]["paper_bgcolor"] == "#111827"
    assert figure["layout"]["font"]["color"] == "#f9fafb"


def test_widget_add_update_remove_trace(fake_plotly: list[_FakePlotlyElement]) -> None:
    """Named continuous trace API should keep the figure dict synchronized."""
    widget = PlotlyPlotWidget()

    widget.add_trace(name="df/f0", x=[0.0, 1.0], y=[2.0, 3.0])
    assert widget.figure["data"][0]["name"] == "df/f0"
    assert widget.figure["data"][0]["mode"] == "lines"

    widget.update_trace(name="df/f0", x=[0.0, 2.0], y=[4.0, 5.0])
    assert widget.figure["data"][0]["x"] == [0.0, 2.0]
    assert "Plotly.restyle" in fake_plotly[0].client.calls[-1]

    widget.remove_trace("df/f0")
    assert widget.figure["data"] == []
    assert "Plotly.deleteTraces" in fake_plotly[0].client.calls[-1]


def test_widget_add_update_remove_scatter(fake_plotly: list[_FakePlotlyElement]) -> None:
    """Named scatter overlay API should keep the figure dict synchronized."""
    widget = PlotlyPlotWidget()

    widget.plot_scatter(name="peaks", x=[0.5], y=[1.5])
    assert widget.figure["data"][0]["name"] == "peaks"
    assert widget.figure["data"][0]["mode"] == "markers"

    widget.update_scatter(name="peaks", x=[0.25, 0.75], y=[1.0, 2.0])
    assert widget.figure["data"][0]["x"] == [0.25, 0.75]
    assert "Plotly.restyle" in fake_plotly[0].client.calls[-1]

    widget.remove_scatter("peaks")
    assert widget.figure["data"] == []
    assert "Plotly.deleteTraces" in fake_plotly[0].client.calls[-1]


def test_widget_set_and_reset_x_axis_limits(fake_plotly: list[_FakePlotlyElement]) -> None:
    """Programmatic x-axis range API should update layout and use relayout."""
    widget = PlotlyPlotWidget()

    widget.set_x_axis_limits(1.0, 2.0)
    assert widget.figure["layout"]["xaxis"]["range"] == [1.0, 2.0]
    assert widget.figure["layout"]["xaxis"]["autorange"] is False
    assert "Plotly.relayout" in fake_plotly[0].client.calls[-1]

    widget.reset_x_axis_limits()
    assert "range" not in widget.figure["layout"]["xaxis"]
    assert widget.figure["layout"]["xaxis"]["autorange"] is True


def test_set_series_reapplies_x_axis_limits_when_range_unchanged(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """``set_series`` should pin line-trace x limits when logical range is automatic."""
    widget = PlotlyPlotWidget()
    fake_plotly[0].client.calls.clear()

    widget.set_series(
        traces=[PlotlyTraceData.from_sequences(name="signal", x=[0.0, 10.0], y=[0.0, 1.0])],
        scatters=[PlotlyScatterData.from_sequences(name="Onsets", x=[0.0, 5.0], y=[0.5, 0.6])],
    )

    assert widget.figure["layout"]["xaxis"]["range"] == [0.0, 10.0]
    assert widget.figure["layout"]["xaxis"]["autorange"] is False
    assert any("xaxis.range" in call for call in fake_plotly[0].client.calls)


def test_reset_x_axis_limits_uses_line_trace_extent_not_scatter_padding(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Automatic x reset should derive limits from line traces, not scatters."""
    widget = PlotlyPlotWidget()
    widget.set_series(
        traces=[PlotlyTraceData.from_sequences(name="signal", x=[0.0, 10.0], y=[0.0, 1.0])],
        scatters=[PlotlyScatterData.from_sequences(name="Onsets", x=[0.5, 5.0], y=[0.5, 0.6])],
    )

    widget.reset_x_axis_limits()

    assert widget.x_range_limits == (None, None)
    assert widget.figure["layout"]["xaxis"]["range"] == [0.0, 10.0]
    assert widget.figure["layout"]["xaxis"]["autorange"] is False


def test_scatter_trace_clips_markers_on_axis(fake_plotly: list[_FakePlotlyElement]) -> None:
    """Scatter overlays should clip marker radius at axis edges."""
    widget = PlotlyPlotWidget()
    widget.plot_scatter(name="Onsets", x=[0.0], y=[1.0])

    assert widget.figure["data"][0]["cliponaxis"] is True


def test_set_y_label_updates_layout_when_axis_labels_visible(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Primary y-axis title updates should relayout when decorations are on."""
    widget = PlotlyPlotWidget()
    widget.set_y_axis_labels_visible(True)

    widget.set_y_label("df/f0")

    assert widget.figure["layout"]["yaxis"]["title"]["text"] == "df/f0"
    assert any("yaxis.title.text" in call for call in fake_plotly[0].client.calls)


def test_widget_emits_user_x_range_callback(fake_plotly: list[_FakePlotlyElement]) -> None:
    """User relayout x-range events should notify the parent callback."""
    ranges: list[tuple[float | None, float | None]] = []
    widget = PlotlyPlotWidget(on_x_range_changed=lambda x0, x1: ranges.append((x0, x1)))

    widget._on_plotly_relayout(_RelayoutEvent({"xaxis.range[0]": 2.0, "xaxis.range[1]": 5.0}))
    widget._on_plotly_relayout(_RelayoutEvent({"xaxis.autorange": True}))

    assert ranges == [(2.0, 5.0), (None, None)]
    assert widget.figure["layout"]["xaxis"]["autorange"] is True


def test_widget_suppresses_relayout_echo_after_programmatic_set_x_limits(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Programmatic x limits should not re-fire ``on_x_range_changed`` on relayout echo."""
    ranges: list[tuple[float | None, float | None]] = []
    widget = PlotlyPlotWidget(on_x_range_changed=lambda x0, x1: ranges.append((x0, x1)))

    widget.set_x_axis_limits(2.0, 8.0)
    widget._on_plotly_relayout(_RelayoutEvent({"xaxis.range[0]": 2.0, "xaxis.range[1]": 8.0}))
    widget._on_plotly_relayout(
        _RelayoutEvent({"xaxis.range[0]": 2.0 + 1e-12, "xaxis.range[1]": 8.0 - 1e-12})
    )

    assert ranges == []


def test_measurement_line_drag_updates_state_and_callbacks(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Dragged single measurement lines should update position and emit payloads."""
    events: list[MeasurementChangeEvent] = []
    widget = PlotlyPlotWidget(on_measurement_changed=events.append)
    line = widget.add_measurement_line(name="manual-f0", orientation="horizontal", value=10.0)

    widget._on_plotly_relayout(_RelayoutEvent({"shapes[0].y0": 12.5, "shapes[0].y1": 12.5}))

    assert line.position == 12.5
    assert events[-1].name == "manual-f0"
    assert events[-1].kind == "line"
    assert events[-1].orientation == "horizontal"
    assert events[-1].position == 12.5


def test_measurement_line_style_and_legend_options(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Measurement lines should honor editable/color/dash/legend shape options."""
    widget = PlotlyPlotWidget(
        display_options=PlotlyPlotDisplayOptions(theme="dark"),
    )
    line = widget.add_measurement_line(
        name="auto-f0",
        orientation="horizontal",
        value=1.5,
        editable=False,
        color="#38bdf8",
        dash="dot",
        show_legend=True,
        legend_label="Auto F0",
    )

    shape = widget.figure["layout"]["shapes"][0]
    assert line.editable is False
    assert shape["editable"] is False
    assert shape["line"]["color"] == "#38bdf8"
    assert shape["line"]["dash"] == "dot"
    assert shape["name"] == "Auto F0"
    assert shape["showlegend"] is True


def test_non_editable_measurement_line_ignores_relayout_drags(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Non-editable measurement lines should not emit drag callbacks."""
    events: list[MeasurementChangeEvent] = []
    widget = PlotlyPlotWidget(on_measurement_changed=events.append)
    line = widget.add_measurement_line(
        name="auto-f0",
        orientation="horizontal",
        value=2.0,
        editable=False,
    )

    widget._on_plotly_relayout(_RelayoutEvent({"shapes[0].y0": 9.0, "shapes[0].y1": 9.0}))

    assert line.position == 2.0
    assert events == []


def test_measurement_line_without_legend_omits_shape_name(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Measurement shapes should omit legend keys when show_legend is False."""
    widget = PlotlyPlotWidget()
    widget.add_measurement_line(
        name="manual-f0",
        orientation="horizontal",
        value=1.0,
        show_legend=False,
    )

    shape = widget.figure["layout"]["shapes"][0]
    assert "name" not in shape
    assert "showlegend" not in shape


def test_horizontal_measurement_drag_keeps_line_axis_aligned(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Endpoint-style drags should be normalized back to a full horizontal line."""
    events: list[MeasurementChangeEvent] = []
    widget = PlotlyPlotWidget(on_measurement_changed=events.append)
    line = widget.add_measurement_line(
        name="manual-f0",
        orientation="horizontal",
        value=10.0,
        show_legend=False,
    )

    widget._on_plotly_relayout(
        _RelayoutEvent(
            {
                "shapes[0].y0": 12.0,
                "shapes[0].y1": 8.0,
                "shapes[0].x0": 0.2,
                "shapes[0].x1": 0.8,
            }
        )
    )

    shape = widget.figure["layout"]["shapes"][0]
    assert line.position == 10.0  # mean of dragged endpoints
    assert shape["y0"] == 10.0
    assert shape["y1"] == 10.0
    assert shape["x0"] == 0
    assert shape["x1"] == 1
    assert shape["xref"] == "paper"
    assert events[-1].position == 10.0


def test_horizontal_measurement_single_endpoint_drag_moves_whole_line(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Dragging only y1 (Plotly vertex) should move the full H-line to that y."""
    events: list[MeasurementChangeEvent] = []
    widget = PlotlyPlotWidget(on_measurement_changed=events.append)
    line = widget.add_measurement_line(
        name="manual-f0",
        orientation="horizontal",
        value=10.0,
        show_legend=False,
    )

    widget._on_plotly_relayout(_RelayoutEvent({"shapes[0].y1": 15.0}))

    shape = widget.figure["layout"]["shapes"][0]
    assert line.position == 15.0
    assert shape["y0"] == 15.0
    assert shape["y1"] == 15.0
    assert shape["x0"] == 0
    assert shape["x1"] == 1
    assert events[-1].position == 15.0


def test_measurement_pair_drag_updates_delta(fake_plotly: list[_FakePlotlyElement]) -> None:
    """Dragged pair lines should update positions and report absolute delta."""
    events: list[MeasurementChangeEvent] = []
    widget = PlotlyPlotWidget(on_measurement_changed=events.append)
    pair = widget.add_measurement_pair(
        name="window",
        orientation="vertical",
        value1=1.0,
        value2=4.0,
    )

    widget._on_plotly_relayout(_RelayoutEvent({"shapes[1].x0": 6.0, "shapes[1].x1": 6.0}))

    assert pair.position1 == 1.0
    assert pair.position2 == 6.0
    assert pair.delta == 5.0
    assert events[-1].kind == "pair"
    assert events[-1].position1 == 1.0
    assert events[-1].position2 == 6.0
    assert events[-1].delta == 5.0


def test_widget_set_series_replaces_data_in_one_update(fake_plotly: list[_FakePlotlyElement]) -> None:
    """Batch series replacement should update traces and pin automatic x limits."""
    widget = PlotlyPlotWidget()
    widget.add_trace(name="old", x=[0.0], y=[1.0])

    traces = [
        PlotlyTraceData.from_sequences(name="df/f0", x=[0.0, 1.0], y=[2.0, 3.0]),
        PlotlyTraceData.from_sequences(name="derivative", x=[0.0, 1.0], y=[0.1, 0.2]),
    ]
    scatters = [
        PlotlyScatterData.from_sequences(name="peaks", x=[0.5], y=[1.5]),
    ]
    calls_before = len(fake_plotly[0].client.calls)
    widget.set_series(traces=traces, scatters=scatters)

    assert [trace["name"] for trace in widget.figure["data"]] == ["df/f0", "derivative", "peaks"]
    new_calls = fake_plotly[0].client.calls[calls_before:]
    assert any("Plotly.deleteTraces" in call and "Plotly.addTraces" in call for call in new_calls)
    assert any("xaxis.range" in call for call in new_calls)


def test_extract_rect_selection_parses_flat_keys() -> None:
    """Box-select relayout should parse flat selections[0].x0/x1 keys."""
    args = {"selections[0].x0": 1.5, "selections[0].x1": 3.5}
    assert extract_rect_selection_x_range_from_relayout(args) == (1.5, 3.5)


def test_extract_rect_selection_parses_list_form() -> None:
    """Box-select relayout should parse selections list payloads."""
    args = {
        "selections": [{"type": "rect", "x0": 2.0, "x1": 4.0, "y0": 0, "y1": 1}],
    }
    assert extract_rect_selection_x_range_from_relayout(args) == (2.0, 4.0)


def test_begin_select_x_range_echo_does_not_emit_x_range_changed(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Dragmode relayout echo after arming should not fire on_x_range_changed."""
    ranges: list[tuple[float | None, float | None]] = []
    widget = PlotlyPlotWidget(on_x_range_changed=lambda x0, x1: ranges.append((x0, x1)))

    widget.begin_select_x_range()
    widget._on_plotly_relayout(_RelayoutEvent({"dragmode": "select"}))

    assert ranges == []
    assert widget.figure["layout"]["dragmode"] == "select"


def test_box_select_emits_on_x_range_selected_once(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """A completed box-select should call on_x_range_selected and disarm."""
    selected: list[tuple[float, float]] = []
    widget = PlotlyPlotWidget(on_x_range_selected=lambda x0, x1: selected.append((x0, x1)))

    widget.begin_select_x_range()
    widget._on_plotly_relayout(
        _RelayoutEvent({"selections[0].x0": 1.0, "selections[0].x1": 2.5})
    )

    assert selected == [(1.0, 2.5)]
    assert widget.figure["layout"]["dragmode"] == "zoom"
    assert widget.figure["layout"]["selections"] == []


def test_doubleclick_resets_x_range_and_emits_auto(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Double-click should reset x-axis and emit (None, None)."""
    ranges: list[tuple[float | None, float | None]] = []
    widget = PlotlyPlotWidget(on_x_range_changed=lambda x0, x1: ranges.append((x0, x1)))

    widget.set_x_axis_limits(1.0, 5.0)
    widget._on_plotly_doubleclick(_RelayoutEvent({}))

    assert widget.figure["layout"]["xaxis"]["autorange"] is True
    assert ranges == [(None, None)]


def test_set_legend_visible_updates_bottom_margin(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Legend toggle should shrink or restore bottom margin."""
    widget = PlotlyPlotWidget()

    assert widget.figure["layout"]["margin"]["b"] == 40

    widget.set_legend_visible(False)
    assert widget.figure["layout"]["showlegend"] is False
    assert widget.figure["layout"]["margin"]["b"] == 8

    widget.set_legend_visible(True)
    assert widget.figure["layout"]["showlegend"] is True
    assert widget.figure["layout"]["margin"]["b"] == 40


def test_set_legend_visible_preserves_bottom_horizontal_layout(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Legend toggle should preserve bottom-centered horizontal legend layout."""
    widget = PlotlyPlotWidget()

    widget.set_legend_visible(False)
    assert widget.figure["layout"]["showlegend"] is False

    widget.set_legend_visible(True)
    legend = widget.figure["layout"]["legend"]
    assert widget.figure["layout"]["showlegend"] is True
    assert legend["orientation"] == "h"
    assert legend["x"] == 0.5
    assert legend["y"] == -0.15


def test_event_overlays_render_as_non_editable_rects(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Event overlays should append rect shapes below measurement shapes."""
    widget = PlotlyPlotWidget()
    widget.add_measurement_line(name="m", orientation="horizontal", value=1.0)
    widget.events.set_events(
        [PlotlyEventOverlay(id="7", x0=1.0, x1=2.0, event_type="user")]
    )

    shapes = widget.figure["layout"]["shapes"]
    assert len(shapes) == 2
    assert shapes[1]["type"] == "rect"
    assert shapes[1]["name"] == "event:7"
    assert shapes[1]["editable"] is False
    assert shapes[1]["yref"] == "paper"


def test_event_overlays_survive_set_series(fake_plotly: list[_FakePlotlyElement]) -> None:
    """Replacing series should keep event overlay shapes."""
    widget = PlotlyPlotWidget()
    widget.add_trace(name="old", x=[0.0], y=[1.0])
    widget.events.set_events([PlotlyEventOverlay(id="1", x0=0.5, x1=1.5)])
    widget.set_series(traces=[PlotlyTraceData.from_sequences(name="new", x=[0.0, 1.0], y=[1.0, 2.0])])

    event_shapes = [s for s in widget.figure["layout"]["shapes"] if s.get("name", "").startswith("event:")]
    assert len(event_shapes) == 1


def test_select_event_updates_highlight_style(fake_plotly: list[_FakePlotlyElement]) -> None:
    """Selected event should use the selected highlight style."""
    widget = PlotlyPlotWidget()
    widget.events.set_events([PlotlyEventOverlay(id="a", x0=1.0, x1=2.0, event_type="user")])
    widget.events.select_event("a")

    shape = widget.figure["layout"]["shapes"][0]
    assert shape["line"]["width"] == 4


def test_widget_set_series_preserves_measurement_shapes(fake_plotly: list[_FakePlotlyElement]) -> None:
    """Replacing series data should not remove measurement layout shapes."""
    widget = PlotlyPlotWidget()
    widget.add_measurement_line(name="f0", orientation="horizontal", value=1.0)
    widget.set_series(
        traces=[PlotlyTraceData.from_sequences(name="signal", x=[0.0, 1.0], y=[1.0, 2.0])],
    )

    assert len(widget.figure["layout"]["shapes"]) == 1
    assert widget.figure["data"][0]["name"] == "signal"


def test_widget_set_dark_mode_updates_layout_and_relayouts(fake_plotly: list[_FakePlotlyElement]) -> None:
    """Dark-mode toggles should update layout colors and relayout once."""
    widget = PlotlyPlotWidget(display_options=PlotlyPlotDisplayOptions(theme="light"))

    widget.set_dark_mode(True)

    assert widget.figure["layout"]["paper_bgcolor"] == "#111827"
    assert "Plotly.relayout" in fake_plotly[0].client.calls[-1]
    assert '"paper_bgcolor": "#111827"' in fake_plotly[0].client.calls[-1]


def test_host_can_replace_default_h_full_with_fixed_height(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hosts must be able to drop ``h-full`` and apply an explicit height.

    Percentage ``h-full`` collapses when the parent height is content-sized.
    Demo/host code removes ``h-full`` then adds ``h-96`` (see layout guide).
    """

    class _TrackingElement(_FakeUiElement):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            super().__init__(*_args, **_kwargs)
            self.class_tokens: set[str] = set()

        def classes(self, *args: Any, **kwargs: Any) -> _TrackingElement:
            if args:
                self.class_tokens.update(str(args[0]).split())
            remove = kwargs.get("remove")
            if remove:
                self.class_tokens.difference_update(str(remove).split())
            add = kwargs.get("add")
            if add:
                self.class_tokens.update(str(add).split())
            return self

    created: list[_FakePlotlyElement] = []

    def plotly_factory(figure: dict[str, Any]) -> _FakePlotlyElement:
        element = _FakePlotlyElement(figure)
        created.append(element)
        return element

    monkeypatch.setattr("nicewidgets.plotly_plot.widget.ui.plotly", plotly_factory)
    monkeypatch.setattr("nicewidgets.plotly_plot.widget.ui.element", _TrackingElement)
    monkeypatch.setattr("nicewidgets.plotly_plot.widget.ui.label", _FakeUiLabel)
    monkeypatch.setattr(
        "nicewidgets.plotly_plot.widget.ui.add_head_html",
        lambda *_args, **_kwargs: None,
    )

    widget = PlotlyPlotWidget()
    assert "h-full" in widget.container.class_tokens

    widget.container.classes(remove="h-full")
    widget.container.classes(add="w-full h-96")

    assert "h-full" not in widget.container.class_tokens
    assert "h-96" in widget.container.class_tokens
    assert created  # plot element constructed


def test_register_series_menu_items_preserves_visibility_across_refresh(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Registered series visibility should persist until page reload."""
    widget = PlotlyPlotWidget()
    widget.register_series_menu_items(
        [
            PlotlySeriesMenuItem("Derivative of df/f0", "Derivative of df/f0", default_visible=False),
            PlotlySeriesMenuItem("Peak width 50", "Peak width 50", default_visible=True),
        ]
    )
    traces = [
        PlotlyTraceData.from_sequences(name="df/f0 signal", x=[0.0, 1.0], y=[1.0, 2.0]),
        PlotlyTraceData.from_sequences(name="Derivative of df/f0", x=[0.0, 1.0], y=[0.1, 0.2]),
        PlotlyTraceData.from_sequences(name="Peak width 50", x=[0.5, 1.0], y=[0.3, 0.3]),
    ]
    widget.set_series(traces=traces)
    assert widget.figure["data"][1]["visible"] is False
    assert widget.figure["data"][2]["visible"] is True

    widget.toggle_series_visible("Derivative of df/f0")
    widget.set_series(traces=traces)
    assert widget.is_series_visible("Derivative of df/f0") is True
    assert widget.figure["data"][1]["visible"] is True


def test_toggle_series_visible_updates_existing_scatter(fake_plotly: list[_FakePlotlyElement]) -> None:
    """Scatter menu toggles should restyle an existing overlay trace."""
    widget = PlotlyPlotWidget()
    widget.register_series_menu_items(
        [PlotlySeriesMenuItem("Onsets", "Onsets", default_visible=True, kind="scatter")]
    )
    widget.set_series(
        scatters=[PlotlyScatterData.from_sequences(name="Onsets", x=[0.5], y=[1.0])],
    )

    widget.toggle_series_visible("Onsets")

    assert widget.is_series_visible("Onsets") is False
    assert widget.figure["data"][0]["visible"] is False
    assert "Plotly.restyle" in fake_plotly[0].client.calls[-1]


def test_right_axis_trace_creates_yaxis2(fake_plotly: list[_FakePlotlyElement]) -> None:
    """A right-axis trace should create ``layout.yaxis2`` and bind the trace."""
    widget = PlotlyPlotWidget(y2_label="rate (1/s)")

    widget.add_trace(name="signal", x=[0.0, 1.0], y=[1.0, 2.0])
    assert "yaxis2" not in widget.figure["layout"]

    widget.add_trace(name="derivative", x=[0.0, 1.0], y=[0.1, 0.2], y_axis="right")

    assert widget.figure["layout"]["yaxis2"]["overlaying"] == "y"
    assert widget.figure["layout"]["yaxis2"]["side"] == "right"
    assert widget.figure["data"][1]["yaxis"] == "y2"


def test_remove_last_right_axis_trace_removes_yaxis2(fake_plotly: list[_FakePlotlyElement]) -> None:
    """Removing the last right-axis trace should remove ``layout.yaxis2``."""
    widget = PlotlyPlotWidget()
    widget.add_trace(name="derivative", x=[0.0, 1.0], y=[0.1, 0.2], y_axis="right")
    assert "yaxis2" in widget.figure["layout"]

    widget.remove_trace("derivative")

    assert "yaxis2" not in widget.figure["layout"]


def test_set_series_with_mixed_axes(fake_plotly: list[_FakePlotlyElement]) -> None:
    """Batch series replacement should configure mixed-axis traces and yaxis2."""
    widget = PlotlyPlotWidget(y2_label="d/dt")
    traces = [
        PlotlyTraceData.from_sequences(name="df/f0", x=[0.0, 1.0], y=[1.0, 2.0]),
        PlotlyTraceData.from_sequences(
            name="derivative",
            x=[0.0, 1.0],
            y=[0.1, 0.2],
            y_axis="right",
        ),
    ]

    widget.set_series(traces=traces)

    assert widget.figure["data"][1]["yaxis"] == "y2"
    assert widget.figure["layout"]["yaxis2"]["title"]["text"] == ""


def test_right_axis_scatter_creates_yaxis2(fake_plotly: list[_FakePlotlyElement]) -> None:
    """A right-axis scatter should create ``layout.yaxis2``."""
    widget = PlotlyPlotWidget()
    widget.plot_scatter(name="markers", x=[0.5], y=[1.0], y_axis="right")

    assert widget.figure["data"][0]["yaxis"] == "y2"
    assert "yaxis2" in widget.figure["layout"]


def test_right_axis_measurement_requires_existing_yaxis2(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Right-axis measurements should fail before a right-axis trace exists."""
    widget = PlotlyPlotWidget()

    with pytest.raises(ValueError, match="right-axis trace or scatter"):
        widget.add_measurement_line(
            name="threshold",
            orientation="horizontal",
            value=1.0,
            y_axis="right",
        )


def test_right_axis_measurement_uses_y2_and_reports_axis_on_drag(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Right-axis horizontal measurements should bind to ``y2`` and report axis."""
    events: list[MeasurementChangeEvent] = []
    widget = PlotlyPlotWidget(on_measurement_changed=events.append)
    widget.add_trace(name="derivative", x=[0.0, 1.0], y=[0.1, 0.2], y_axis="right")
    widget.add_measurement_line(
        name="threshold",
        orientation="horizontal",
        value=1.0,
        y_axis="right",
    )

    assert widget.figure["layout"]["shapes"][0]["yref"] == "y2"
    widget._on_plotly_relayout(_RelayoutEvent({"shapes[0].y0": 1.5, "shapes[0].y1": 1.5}))

    assert events[-1].position == 1.5
    assert events[-1].y_axis == "right"


def test_axis_labels_toggle_updates_bottom_margin(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Axis-label toggle should adjust bottom margin when legend is visible."""
    widget = PlotlyPlotWidget()

    assert widget.figure["layout"]["margin"]["b"] == 40

    widget.set_x_axis_labels_visible(True)
    assert widget.figure["layout"]["margin"]["b"] == 72

    widget.set_x_axis_labels_visible(False)
    assert widget.figure["layout"]["margin"]["b"] == 40


def test_axis_labels_on_keeps_plot_grid_lines_off(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Axis labels should not enable internal horizontal/vertical grid lines."""
    widget = PlotlyPlotWidget()

    widget.set_x_axis_labels_visible(True)
    widget.set_y_axis_labels_visible(True)

    assert widget.figure["layout"]["xaxis"]["showgrid"] is False
    assert widget.figure["layout"]["yaxis"]["showgrid"] is False


def test_axis_labels_toggle_updates_yaxis2_and_dual_margin(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Axis-label toggle should decorate ``yaxis2`` and widen the right margin."""
    widget = PlotlyPlotWidget(y2_label="rate (1/s)")
    widget.add_trace(name="derivative", x=[0.0, 1.0], y=[0.1, 0.2], y_axis="right")

    widget.set_y_axis_labels_visible(True)

    assert widget.figure["layout"]["yaxis2"]["title"]["text"] == "rate (1/s)"
    assert widget.figure["layout"]["margin"]["r"] == 60


def test_hidden_right_axis_trace_hides_yaxis2_decorations(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Invisible right-axis traces should not show y2 decorations or dual margin."""
    widget = PlotlyPlotWidget(y2_label="rate (1/s)")
    widget.register_series_menu_items(
        [PlotlySeriesMenuItem("derivative", "derivative", default_visible=False)]
    )
    traces = [
        PlotlyTraceData.from_sequences(name="df/f0", x=[0.0, 1.0], y=[1.0, 2.0]),
        PlotlyTraceData.from_sequences(
            name="derivative",
            x=[0.0, 1.0],
            y=[0.1, 0.2],
            y_axis="right",
            visible=False,
        ),
    ]

    widget.set_series(traces=traces)
    widget.set_y_axis_labels_visible(True)

    assert widget.figure["layout"]["yaxis2"]["title"]["text"] == ""
    assert widget.figure["layout"]["yaxis2"]["showticklabels"] is False
    assert widget.figure["layout"]["margin"]["r"] == 24


def test_toggle_right_axis_visibility_updates_yaxis2_decorations(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Toggling a right-axis trace should show and hide y2 decorations."""
    widget = PlotlyPlotWidget(y2_label="rate (1/s)")
    widget.register_series_menu_items(
        [PlotlySeriesMenuItem("derivative", "derivative", default_visible=False)]
    )
    widget.set_series(
        traces=[
            PlotlyTraceData.from_sequences(name="df/f0", x=[0.0, 1.0], y=[1.0, 2.0]),
            PlotlyTraceData.from_sequences(
                name="derivative",
                x=[0.0, 1.0],
                y=[0.1, 0.2],
                y_axis="right",
            ),
        ]
    )
    widget.set_y_axis_labels_visible(True)

    assert widget.figure["layout"]["yaxis2"]["title"]["text"] == ""
    assert widget.figure["layout"]["margin"]["r"] == 24

    widget.toggle_series_visible("derivative")

    assert widget.figure["layout"]["yaxis2"]["title"]["text"] == "rate (1/s)"
    assert widget.figure["layout"]["margin"]["r"] == 60


def test_init_axis_label_visibility_kwargs(fake_plotly: list[_FakePlotlyElement]) -> None:
    """Constructor display options should set independent x/y axis label visibility."""
    widget = PlotlyPlotWidget(
        display_options=PlotlyPlotDisplayOptions(
            show_x_axis_labels=True, show_y_axis_labels=False
        )
    )

    assert widget.display_options.show_x_axis_labels is True
    assert widget.display_options.show_y_axis_labels is False
    assert widget.figure["layout"]["xaxis"]["showticklabels"] is True
    assert widget.figure["layout"]["yaxis"]["showticklabels"] is False


def test_x_and_y_axis_labels_toggle_independently(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """X and y axis decorations should toggle independently."""
    widget = PlotlyPlotWidget()

    widget.set_x_axis_labels_visible(True)
    assert widget.figure["layout"]["xaxis"]["showticklabels"] is True
    assert widget.figure["layout"]["yaxis"]["showticklabels"] is False

    widget.set_y_axis_labels_visible(True)
    assert widget.figure["layout"]["yaxis"]["showticklabels"] is True

    widget.set_x_axis_labels_visible(False)
    assert widget.figure["layout"]["xaxis"]["showticklabels"] is False
    assert widget.figure["layout"]["yaxis"]["showticklabels"] is True


def test_axis_label_font_size_is_explicit(fake_plotly: list[_FakePlotlyElement]) -> None:
    """Axis title and tick labels should use the widget default font size."""
    widget = PlotlyPlotWidget()
    widget.set_x_axis_labels_visible(True)
    widget.set_y_axis_labels_visible(True)

    for axis_name in ("xaxis", "yaxis"):
        axis = widget.figure["layout"][axis_name]
        assert axis["title"]["font"]["size"] == 11
        assert axis["tickfont"]["size"] == 11


def test_init_show_legend_false_builds_without_legend(fake_plotly: list[_FakePlotlyElement]) -> None:
    """Initial legend visibility should come from the constructor display options."""
    widget = PlotlyPlotWidget(display_options=PlotlyPlotDisplayOptions(show_legend=False))

    assert widget.display_options.show_legend is False
    assert widget.figure["layout"]["showlegend"] is False
    assert widget.figure["layout"]["margin"]["b"] == 8


def test_set_placeholder_text_shows_and_hides_overlay(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Placeholder text should toggle the centered overlay."""
    widget = PlotlyPlotWidget()

    widget.set_placeholder_text("No data")
    assert widget.placeholder_text == "No data"
    assert widget._placeholder_container.visible is True

    widget.set_placeholder_text(None)
    assert widget.placeholder_text is None
    assert widget._placeholder_container.visible is False


def test_set_series_with_data_clears_placeholder(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Replacing series with data should hide any visible placeholder."""
    widget = PlotlyPlotWidget()
    widget.set_placeholder_text("No data")

    widget.set_series(
        traces=[PlotlyTraceData.from_sequences(name="trace", x=[0.0], y=[1.0])]
    )

    assert widget.placeholder_text is None
    assert widget._placeholder_container.visible is False


def test_display_options_to_dict_from_dict_round_trip() -> None:
    """Display options should survive a to_dict/from_dict round trip."""
    options = PlotlyPlotDisplayOptions(
        show_x_axis_labels=True,
        show_y_axis_labels=True,
        show_plotly_toolbar=True,
        show_hover_info=True,
        show_legend=False,
        theme="dark",
    )

    restored = PlotlyPlotDisplayOptions.from_dict(options.to_dict())

    assert restored == options


def test_display_options_from_dict_ignores_unknown_and_defaults_missing() -> None:
    """from_dict should ignore unknown keys and default missing ones."""
    restored = PlotlyPlotDisplayOptions.from_dict(
        {"show_legend": False, "unexpected": 123}
    )

    assert restored.show_legend is False
    assert restored.show_x_axis_labels is False
    assert restored.theme == "light"


def test_set_series_visible_state_stores_pending_for_unloaded_series(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Setting state for an unloaded series should store visibility without raising."""
    widget = PlotlyPlotWidget()

    widget.set_series_visible_state("not-loaded", False)

    assert widget.is_series_visible("not-loaded") is False


def test_set_series_visible_state_restyles_loaded_series(
    fake_plotly: list[_FakePlotlyElement],
) -> None:
    """Setting state for a loaded series should update its rendered visibility."""
    widget = PlotlyPlotWidget()
    widget.set_series(
        traces=[PlotlyTraceData.from_sequences(name="trace", x=[0.0, 1.0], y=[1.0, 2.0])]
    )

    widget.set_series_visible_state("trace", False)

    assert widget.is_series_visible("trace") is False
    index = widget._series_index("trace", "trace")
    assert widget.figure["data"][index]["visible"] is False


def test_on_build_context_menu_invoked_before_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom context-menu callback should run after display toggles, before Copy."""
    from nicewidgets.plotly_plot.context_menu import PlotlyPlotContextMenu

    labels: list[str] = []

    def fake_menu_item(label: str, on_click: Any = None) -> None:
        _ = on_click
        labels.append(label)

    monkeypatch.setattr(
        "nicewidgets.plotly_plot.context_menu.ui.menu_item",
        fake_menu_item,
    )
    monkeypatch.setattr(
        "nicewidgets.plotly_plot.context_menu.ui.separator",
        lambda: labels.append("---"),
    )

    class _StubWidget:
        display_options = PlotlyPlotDisplayOptions()
        series_menu_items: tuple[PlotlySeriesMenuItem, ...] = ()
        on_build_context_menu = staticmethod(
            lambda _w: fake_menu_item("Custom Action")
        )

        def copy_plot_to_clipboard(self) -> None:
            return None

        def set_x_axis_labels_visible(self, _visible: bool) -> None:
            return None

        def set_y_axis_labels_visible(self, _visible: bool) -> None:
            return None

        def set_plotly_toolbar_visible(self, _visible: bool) -> None:
            return None

        def set_hover_info_visible(self, _visible: bool) -> None:
            return None

        def set_legend_visible(self, _visible: bool) -> None:
            return None

    menu = PlotlyPlotContextMenu(get_widget=lambda: _StubWidget())  # type: ignore[arg-type, return-value]
    menu.build()

    assert "Custom Action" in labels
    assert labels.index("Custom Action") < labels.index("Copy To Clipboard")
    assert labels[labels.index("Custom Action") - 1] == "---"
