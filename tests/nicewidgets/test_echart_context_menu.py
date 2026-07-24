"""Tests for :class:`EChartWidget` display options, context menu, and default x-zoom cursor.

The widget's ``__init__`` requires NiceGUI's ``ui.echart`` / ``ui.context_menu``,
so most tests construct the widget via ``__new__`` and exercise pure-Python
helpers directly. The behavior we care about (display options, option dict
shape, context-menu builder labels, zoom-cursor payload) is isolated from the
NiceGUI runtime.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from nicewidgets.echart_widget.context_menu import EChartWidgetContextMenu
from nicewidgets.echart_widget.display_options import EChartDisplayOptions
from nicewidgets.echart_widget.models import EChartLineData
from nicewidgets.echart_widget.widget import EChartWidget, build_line_options


def _make_widget(
    *,
    display_options: EChartDisplayOptions | None = None,
    line_data: EChartLineData | None = None,
) -> EChartWidget:
    """Build an :class:`EChartWidget` without invoking ``__init__``."""
    w = EChartWidget.__new__(EChartWidget)
    w._line_data = line_data
    w._display_options = display_options or EChartDisplayOptions()
    return w


def test_display_options_default_shows_toolbar() -> None:
    """Default :class:`EChartDisplayOptions` ships with the toolbar visible."""
    options = EChartDisplayOptions()
    assert options.show_toolbar is True


def test_display_options_default_hides_hover_info() -> None:
    """Default :class:`EChartDisplayOptions` ships with the hover tooltip hidden."""
    options = EChartDisplayOptions()
    assert options.show_hover_info is False


def test_empty_options_includes_toolbox_for_data_zoom_action() -> None:
    """The empty option dict must carry a ``toolbox.dataZoom`` feature.

    ECharts' ``dataZoomSelect`` cursor (used to enable click+drag x-axis zoom)
    requires the ``dataZoom`` feature to be present in the chart options,
    even when the toolbox icons themselves are hidden.
    """
    opts = EChartWidget._empty_options()
    assert "toolbox" in opts
    feature = opts["toolbox"]["feature"]
    assert "dataZoom" in feature


def test_apply_display_options_show_toolbar_propagates() -> None:
    """:meth:`_apply_display_options_to_options` flips ``toolbox.show``."""
    w_off = _make_widget(display_options=EChartDisplayOptions(show_toolbar=False))
    w_on = _make_widget(display_options=EChartDisplayOptions(show_toolbar=True))

    line = EChartLineData.from_sequences(
        x=[0.0, 1.0], y=[2.0, 3.0], x_label="x", y_label="y"
    )
    opts_off = build_line_options(line)
    opts_on = build_line_options(line)
    w_off._apply_display_options_to_options(opts_off)
    w_on._apply_display_options_to_options(opts_on)

    assert opts_off["toolbox"]["show"] is False
    assert opts_on["toolbox"]["show"] is True


def test_apply_display_options_show_hover_info_propagates() -> None:
    """:meth:`_apply_display_options_to_options` flips ``tooltip.show``.

    The ``trigger='axis'`` portion of the tooltip configuration is preserved
    so the axis pointer behavior remains stable across toggles.
    """
    w_off = _make_widget(display_options=EChartDisplayOptions(show_hover_info=False))
    w_on = _make_widget(display_options=EChartDisplayOptions(show_hover_info=True))

    line = EChartLineData.from_sequences(
        x=[0.0, 1.0], y=[2.0, 3.0], x_label="x", y_label="y"
    )
    opts_off = build_line_options(line)
    opts_on = build_line_options(line)
    w_off._apply_display_options_to_options(opts_off)
    w_on._apply_display_options_to_options(opts_on)

    assert opts_off["tooltip"]["show"] is False
    assert opts_off["tooltip"]["trigger"] == "axis"
    assert opts_on["tooltip"]["show"] is True
    assert opts_on["tooltip"]["trigger"] == "axis"


def test_apply_display_options_creates_missing_toolbox_and_tooltip() -> None:
    """Missing ``toolbox`` / ``tooltip`` keys are created so toggling does not raise."""
    w = _make_widget(
        display_options=EChartDisplayOptions(show_toolbar=True, show_hover_info=False)
    )
    options: dict[str, object] = {}
    w._apply_display_options_to_options(options)
    assert options["toolbox"]["show"] is True
    assert "feature" in options["toolbox"]
    assert options["tooltip"]["show"] is False
    assert options["tooltip"]["trigger"] == "axis"


def test_context_menu_labels_reflect_toggle_state() -> None:
    """Menu labels use a ``✓`` prefix only when the option is enabled."""
    on = EChartWidgetContextMenu._toggle_label("Show Toolbar", True)
    off = EChartWidgetContextMenu._toggle_label("Show Toolbar", False)
    assert on.startswith("✓ ")
    assert on.endswith("Show Toolbar")
    assert not off.startswith("✓ ")
    assert off == "Show Toolbar"


class _RecordingMenuItem:
    """Lightweight ``ui.menu_item`` replacement that records constructor args."""

    def __init__(self, label: str, *, on_click=None) -> None:
        self.label = label
        self.on_click = on_click


class _RecordingSeparator:
    """Lightweight ``ui.separator`` replacement."""


@pytest.fixture
def recording_ui(monkeypatch) -> Iterator[list[object]]:
    """Patch ``ui.menu_item`` and ``ui.separator`` to record build order."""
    recorded: list[object] = []

    def fake_menu_item(label, *, on_click=None):
        item = _RecordingMenuItem(label, on_click=on_click)
        recorded.append(item)
        return item

    def fake_separator():
        sep = _RecordingSeparator()
        recorded.append(sep)
        return sep

    monkeypatch.setattr(
        "nicewidgets.echart_widget.context_menu.ui.menu_item", fake_menu_item
    )
    monkeypatch.setattr(
        "nicewidgets.echart_widget.context_menu.ui.separator", fake_separator
    )
    yield recorded


def test_context_menu_builds_toggles_separator_and_clipboard_entries(
    recording_ui,
) -> None:
    """``build()`` produces all toggle items, a separator, and a Copy item."""
    widget = _make_widget(
        display_options=EChartDisplayOptions(
            show_toolbar=False,
            show_hover_info=True,
            show_axis_labels=True,
            show_horizontal_lines=False,
            show_vertical_lines=True,
        )
    )
    menu = EChartWidgetContextMenu(get_widget=lambda: widget)

    menu.build()

    assert len(recording_ui) == 7
    (
        toolbar_item,
        hover_item,
        axis_item,
        horiz_item,
        vert_item,
        separator,
        copy_item,
    ) = recording_ui
    assert isinstance(separator, _RecordingSeparator)
    assert toolbar_item.label == "Show Toolbar"
    assert hover_item.label == "✓ Hover Info"
    assert axis_item.label == "✓ Axis Labels"
    assert horiz_item.label == "Horizontal Lines"
    assert vert_item.label == "✓ Vertical Lines"
    assert copy_item.label == "Copy To Clipboard"


def test_context_menu_toolbar_item_flips_show_toolbar(recording_ui) -> None:
    """Clicking the toolbar menu item calls ``set_toolbar_visible`` with the inverse."""
    calls: list[bool] = []

    class _FakeWidget:
        display_options = EChartDisplayOptions(show_toolbar=False)

        def set_toolbar_visible(self, visible: bool) -> None:
            calls.append(bool(visible))

        def set_hover_info_visible(self, visible: bool) -> None:
            return None

        async def copy_plot_to_clipboard(self) -> None:
            return None

    fake = _FakeWidget()
    menu = EChartWidgetContextMenu(get_widget=lambda: fake)
    menu.build()

    toolbar_item = recording_ui[0]
    toolbar_item.on_click()
    assert calls == [True]


def test_context_menu_hover_item_flips_show_hover_info(recording_ui) -> None:
    """Clicking the hover-info menu item calls ``set_hover_info_visible`` with the inverse."""
    calls: list[bool] = []

    class _FakeWidget:
        display_options = EChartDisplayOptions(show_toolbar=False, show_hover_info=True)

        def set_toolbar_visible(self, visible: bool) -> None:
            return None

        def set_hover_info_visible(self, visible: bool) -> None:
            calls.append(bool(visible))

        async def copy_plot_to_clipboard(self) -> None:
            return None

    fake = _FakeWidget()
    menu = EChartWidgetContextMenu(get_widget=lambda: fake)
    menu.build()

    hover_item = recording_ui[1]
    hover_item.on_click()
    assert calls == [False]


def test_x_zoom_cursor_payload_is_data_zoom_select() -> None:
    """:meth:`_activate_x_zoom_cursor` dispatches the ``dataZoomSelect`` action."""
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeContainer:
        def run_chart_method(self, method: str, payload: dict[str, object]) -> None:
            calls.append((method, payload))

    w = _make_widget()
    w.container = _FakeContainer()  # type: ignore[assignment]
    w._activate_x_zoom_cursor()

    assert calls == [
        (
            "dispatchAction",
            {
                "type": "takeGlobalCursor",
                "key": "dataZoomSelect",
                "dataZoomSelectActive": True,
            },
        )
    ]


class _FakeContainer:
    """Records ``run_chart_method`` calls and supports the ``apply`` lifecycle."""

    def __init__(self) -> None:
        self.options: dict[str, object] = {}
        self.calls: list[tuple[str, object]] = []

    def update(self) -> None:
        return None

    def run_chart_method(self, method: str, payload: object = None) -> None:
        self.calls.append((method, payload))


def test_display_options_defaults_axis_labels_on_grid_lines_off() -> None:
    """New display toggles default to axis labels on, grid lines off."""
    options = EChartDisplayOptions()
    assert options.show_axis_labels is True
    assert options.show_horizontal_lines is False
    assert options.show_vertical_lines is False


def test_apply_display_options_hides_axis_decorations_and_shows_grid() -> None:
    """Disabling axis labels hides decorations on both axes; grid flags map per axis."""
    w = _make_widget(
        display_options=EChartDisplayOptions(
            show_axis_labels=False,
            show_horizontal_lines=True,
            show_vertical_lines=True,
        )
    )
    line = EChartLineData.from_sequences(
        x=[0.0, 1.0], y=[2.0, 3.0], x_label="x", y_label="y"
    )
    opts = build_line_options(line)
    w._apply_display_options_to_options(opts)

    for axis in ("xAxis", "yAxis"):
        assert opts[axis]["axisLabel"]["show"] is False
        assert opts[axis]["axisTick"]["show"] is False
        assert opts[axis]["axisLine"]["show"] is False
        assert opts[axis]["name"] == ""
    assert opts["xAxis"]["splitLine"]["show"] is True  # vertical lines
    assert opts["yAxis"]["splitLine"]["show"] is True  # horizontal lines


def test_apply_display_options_axis_labels_on_preserves_name_and_formatter() -> None:
    """With axis labels on (default), the axis name and tick formatter are kept."""
    w = _make_widget()
    line = EChartLineData.from_sequences(
        x=[0.0, 1.0], y=[2.0, 3.0], x_label="x", y_label="y"
    )
    opts = build_line_options(line)
    w._apply_display_options_to_options(opts)

    assert opts["xAxis"]["axisLabel"]["show"] is True
    assert ":formatter" in opts["xAxis"]["axisLabel"]
    assert opts["xAxis"]["name"] == "x"
    assert opts["xAxis"]["splitLine"]["show"] is False
    assert opts["yAxis"]["splitLine"]["show"] is False


def test_apply_rearms_x_zoom_cursor_after_redraw() -> None:
    """Each ``apply`` re-arms the click+drag x-zoom cursor (the runtime fix)."""
    w = _make_widget()
    w._selecting_x = False
    w.container = _FakeContainer()  # type: ignore[assignment]

    w.apply()

    assert any(
        method == "dispatchAction" and payload.get("key") == "dataZoomSelect"
        for method, payload in w.container.calls
    )


def test_apply_skips_cursor_rearm_during_brush_selection() -> None:
    """``apply`` must not clobber an active one-shot brush selection cursor."""
    w = _make_widget()
    w._selecting_x = True
    w.container = _FakeContainer()  # type: ignore[assignment]

    w.apply()

    assert not any(method == "dispatchAction" for method, _ in w.container.calls)


def test_set_zoom_axis_y_switches_feature_and_rearms() -> None:
    """Shift mode points the rubber band at the y-axis and re-arms the cursor."""
    w = _make_widget()
    w.container = _FakeContainer()  # type: ignore[assignment]

    w._set_zoom_axis(y_axis=True)

    set_options = [p for m, p in w.container.calls if m == "setOption"]
    assert set_options
    assert set_options[0]["toolbox"]["feature"]["dataZoom"] == {
        "xAxisIndex": "none",
        "yAxisIndex": 0,
    }
    assert any(
        m == "dispatchAction" and p.get("key") == "dataZoomSelect"
        for m, p in w.container.calls
    )


def test_set_zoom_axis_x_switches_feature_back() -> None:
    """Releasing Shift returns the rubber band to the x-axis."""
    w = _make_widget()
    w.container = _FakeContainer()  # type: ignore[assignment]

    w._set_zoom_axis(y_axis=False)

    set_options = [p for m, p in w.container.calls if m == "setOption"]
    assert set_options[0]["toolbox"]["feature"]["dataZoom"] == {
        "xAxisIndex": 0,
        "yAxisIndex": "none",
    }


class _FakeKeyAction:
    def __init__(self, *, keydown: bool, keyup: bool) -> None:
        self.keydown = keydown
        self.keyup = keyup


class _FakeKeyEvent:
    def __init__(self, key: str, *, keydown: bool = False, keyup: bool = False) -> None:
        self.key = key
        self.action = _FakeKeyAction(keydown=keydown, keyup=keyup)


def test_shift_keydown_enables_y_zoom_keyup_restores_x_zoom() -> None:
    """Shift down switches to y-axis zoom; Shift up restores x-axis zoom."""
    w = _make_widget()
    w.container = _FakeContainer()  # type: ignore[assignment]

    w._on_zoom_modifier_key(_FakeKeyEvent("Shift", keydown=True))
    w._on_zoom_modifier_key(_FakeKeyEvent("Shift", keyup=True))

    dataizoom_features = [
        p["toolbox"]["feature"]["dataZoom"]
        for m, p in w.container.calls
        if m == "setOption"
    ]
    assert dataizoom_features == [
        {"xAxisIndex": "none", "yAxisIndex": 0},
        {"xAxisIndex": 0, "yAxisIndex": "none"},
    ]


def test_non_shift_key_does_not_switch_zoom_axis() -> None:
    """Keys other than Shift leave the zoom axis untouched."""
    w = _make_widget()
    w.container = _FakeContainer()  # type: ignore[assignment]

    w._on_zoom_modifier_key(_FakeKeyEvent("a", keydown=True))

    assert not any(m == "setOption" for m, _ in w.container.calls)


def test_context_menu_new_toggle_items_flip_setters(recording_ui) -> None:
    """Clicking the axis/grid menu items calls the matching setters with inverses."""
    calls: dict[str, bool] = {}

    class _FakeWidget:
        display_options = EChartDisplayOptions(
            show_axis_labels=True,
            show_horizontal_lines=False,
            show_vertical_lines=False,
        )

        def set_toolbar_visible(self, visible: bool) -> None:
            return None

        def set_hover_info_visible(self, visible: bool) -> None:
            return None

        def set_axis_labels_visible(self, visible: bool) -> None:
            calls["axis"] = bool(visible)

        def set_horizontal_lines_visible(self, visible: bool) -> None:
            calls["horiz"] = bool(visible)

        def set_vertical_lines_visible(self, visible: bool) -> None:
            calls["vert"] = bool(visible)

        async def copy_plot_to_clipboard(self) -> None:
            return None

    menu = EChartWidgetContextMenu(get_widget=lambda: _FakeWidget())
    menu.build()

    recording_ui[2].on_click()  # Axis Labels
    recording_ui[3].on_click()  # Horizontal Lines
    recording_ui[4].on_click()  # Vertical Lines

    assert calls == {"axis": False, "horiz": True, "vert": True}


def test_cancel_select_x_range_reactivates_x_zoom_cursor() -> None:
    """Cancelling brush mode restores the default click+drag x-zoom cursor."""
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeContainer:
        def run_chart_method(self, method: str, payload: dict[str, object]) -> None:
            calls.append((method, payload))

    w = _make_widget()
    w._selecting_x = True
    w._pending_x_range = (1.0, 2.0)
    w.container = _FakeContainer()  # type: ignore[assignment]

    w.cancel_select_x_range()

    assert any(
        payload.get("key") == "dataZoomSelect"
        for method, payload in calls
        if method == "dispatchAction"
    )
    assert w._selecting_x is False
    assert w._pending_x_range is None
