"""Minimal NiceGUI host for the framework-neutral NicePool Custom Element."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from pprint import pprint
from typing import Any

import pandas as pd
from nicegui import app, events, ui

_HERE = Path(__file__).resolve().parent
_ASSET_DIRECTORY = _HERE / "dist-element"
_ASSET_URL = "/nicepool-web-assets"
_MISSING = object()


def dataframe_to_dataset(frame: pd.DataFrame, row_id_column: str) -> dict[str, Any]:
    """Convert a simple pandas table to the NicePool JSON data contract.

    Args:
        frame: Dataframe containing scalar string, number, boolean, or missing values.
        row_id_column: Column containing stable, unique row identifiers.

    Returns:
        A JSON-compatible dataset accepted by ``nice-pool.setData``.

    Raises:
        ValueError: If the requested row-ID column is absent.
    """
    if row_id_column not in frame.columns:
        raise ValueError(f"Unknown row-ID column: {row_id_column!r}")
    records = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
    return {"rowIdColumn": row_id_column, "rows": records}


class NicePoolWebView:
    """Thin NiceGUI bridge around the public ``<nice-pool>`` browser API."""

    def __init__(
        self,
        dataset: dict[str, Any],
        *,
        on_selection: Callable[[dict[str, Any]], None] | None = None,
        on_state: Callable[[dict[str, Any]], None] | None = None,
        on_theme: Callable[[str], None] | None = None,
    ) -> None:
        """Create and asynchronously seed a NicePool Custom Element.

        Args:
            dataset: JSON-compatible NicePool dataset.
            on_selection: Optional callback for user-initiated selection changes.
            on_state: Optional callback for user-initiated workspace-state changes.
            on_theme: Optional callback for user-initiated theme changes.
        """
        self._dataset = dataset
        self.element = ui.element("nice-pool").style("width: 100%;")
        if on_selection is not None:
            self.element.on(
                "nicepool-selection-change",
                lambda event: on_selection(self._event_detail(event)),
                ["detail"],
            )
        if on_state is not None:
            self.element.on(
                "nicepool-state-change",
                lambda event: on_state(self._event_detail(event)),
                ["detail"],
            )
        if on_theme is not None:
            self.element.on(
                "nicepool-theme-change",
                lambda event: on_theme(str(self._event_value(event))),
                ["detail"],
            )

    @staticmethod
    def _event_detail(event: events.GenericEventArguments) -> dict[str, Any]:
        detail = NicePoolWebView._event_value(event)
        return detail if isinstance(detail, dict) else {}

    @staticmethod
    def _event_value(event: events.GenericEventArguments) -> Any:
        return event.args.get("detail") if isinstance(event.args, dict) else None

    async def _call(self, method: str, argument: Any = _MISSING) -> Any:
        argument_source = "" if argument is _MISSING else json.dumps(argument, allow_nan=False)
        call_source = f"element.{method}({argument_source})"
        return await self.element.client.run_javascript(
            f"""
            await customElements.whenDefined('nice-pool');
            const element = getHtmlElement({self.element.id});
            return {call_source};
            """,
            timeout=10.0,
        )

    async def set_data(self, dataset: dict[str, Any]) -> None:
        """Replace the browser dataset and all dataset-dependent state."""
        self._dataset = dataset
        await self._call("setData", dataset)

    async def reload_data(self) -> None:
        """Reload the current dataset, resetting plot state and selection."""
        await self._call("setData", self._dataset)

    async def get_state(self) -> dict[str, Any]:
        """Return the complete browser workspace state."""
        result = await self._call("getState")
        return result if isinstance(result, dict) else {}

    async def set_state(self, state: dict[str, Any]) -> None:
        """Atomically replace the complete browser workspace state."""
        await self._call("setState", state)

    async def set_primary_selection(self, row_id: str | None) -> None:
        """Replace browser selection with one primary row, or clear it."""
        await self._call("setPrimarySelection", row_id)

    async def set_selection(self, primary_row_id: str | None, selected_row_ids: list[str]) -> None:
        """Replace browser selection with an explicit primary and row set."""
        await self._call(
            "setSelection",
            {"primaryRowId": primary_row_id, "selectedRowIds": selected_row_ids},
        )

    async def set_theme(self, dark: bool) -> None:
        """Apply one light/dark theme to both widget controls and Plotly."""
        await self._call("setTheme", "dark" if dark else "light")

    async def set_plot_presets(self, presets: list[dict[str, Any]]) -> None:
        """Replace the saved plot presets exposed by the browser control."""
        await self._call("setPlotPresets", presets)


def _sample_frame(*, comparison: bool = False) -> pd.DataFrame:
    rows = []
    conditions = ("control", "treated", "recovery")
    row_count = 96 if comparison else 180
    for index in range(row_count):
        condition = conditions[index % len(conditions)]
        common = {
            "pool_row_id": f"py-row-{index + 1:04d}",
            "accept": "no" if index % 11 == 0 else "yes",
            "condition": condition,
            "velocity": None if index % 53 == 0 else ((index % 31) - 15) / (3 if comparison else 5),
        }
        if comparison:
            rows.append(
                {
                    **common,
                    "batch": f"batch-{index % 4 + 1}",
                    "temperature": 20 + (index % 13) / 2,
                    "signal_strength": 40 + (index % 19) * 1.5,
                }
            )
        else:
            rows.append(
                {
                    **common,
                    "time": index / 10,
                    "amplitude": 8 + (index % 17) / 4 + (2 if condition == "treated" else 0),
                }
            )
    return pd.DataFrame(rows)


def _configure_assets() -> None:
    if not (_ASSET_DIRECTORY / "nicepool-element.js").exists():
        raise RuntimeError("NicePool web assets are missing; run `npm run build` in nicepool_web first.")
    app.add_static_files(_ASSET_URL, _ASSET_DIRECTORY)


_configure_assets()


@ui.page("/")
def index_page() -> None:
    """Render a small integration harness for Python-to-browser APIs."""
    asset_version = (_ASSET_DIRECTORY / "nicepool-element.js").stat().st_mtime_ns
    ui.add_head_html(f'<script type="module" src="{_ASSET_URL}/nicepool-element.js?v={asset_version}"></script>')
    datasets = {
        "baseline": dataframe_to_dataset(_sample_frame(), "pool_row_id"),
        "comparison": dataframe_to_dataset(_sample_frame(comparison=True), "pool_row_id"),
    }
    dark_mode = ui.dark_mode(value=True)

    ui.label("NicePool Web · NiceGUI bridge").classes("text-h5")
    selection_status = ui.label("Browser selection: none")
    state_status = ui.label("Browser layout: unknown")

    def selection_changed(selection: dict[str, Any]) -> None:
        selection_status.set_text(f"Browser selection: {selection.get('selectedRowIds', [])}")

    def state_changed(state: dict[str, Any]) -> None:
        state_status.set_text(f"Browser layout: {state.get('layout', 'unknown')}")

    def theme_changed_from_widget(theme: str) -> None:
        enabled = theme == "dark"
        if enabled:
            dark_mode.enable()
        else:
            dark_mode.disable()
        theme_switch.set_value(enabled)

    view = NicePoolWebView(
        datasets["baseline"],
        on_selection=selection_changed,
        on_state=state_changed,
        on_theme=theme_changed_from_widget,
    )

    async def theme_changed(event: events.ValueChangeEventArguments) -> None:
        enabled = bool(event.value)
        if enabled:
            dark_mode.enable()
        else:
            dark_mode.disable()
        await view.set_theme(enabled)

    theme_switch = ui.switch("Dark theme", value=True, on_change=theme_changed)

    async def use_two_plots() -> None:
        state = await view.get_state()
        state.update(layout="1x2", activePlotIndex=0)
        await view.set_state(state)
        state_status.set_text("Browser layout: 1x2 (set by Python)")

    async def read_state() -> None:
        state = await view.get_state()
        print("NicePool browser state read by Python:")
        pprint(state, sort_dicts=False, width=120)
        state_status.set_text(f"Browser layout: {state.get('layout', 'unknown')} (read by Python)")

    async def install_demo_preset() -> None:
        state = await view.get_state()
        plot_state = dict(state["plots"][0])
        plot_state.update(
            plotType="swarm",
            yColumn="velocity",
            groupColumn="condition",
            colorColumn="accept",
        )
        await view.set_plot_presets(
            [{"schemaVersion": 1, "name": "Velocity by condition", "plotState": plot_state}]
        )

    async def load_dataset(dataset_name: str) -> None:
        await view.set_data(datasets[dataset_name])
        await install_demo_preset()
        selection_status.set_text("Browser selection: none (cleared by setData)")
        state_status.set_text(f"Browser dataset: {dataset_name} (new default state)")

    async def dataset_changed(event: events.ValueChangeEventArguments) -> None:
        await load_dataset(str(event.value))

    async def reload_current_dataset() -> None:
        await view.reload_data()
        await install_demo_preset()
        selection_status.set_text("Browser selection: none (cleared by setData)")
        state_status.set_text("Current browser dataset reloaded with default state")

    async def initialize_demo() -> None:
        await load_dataset("baseline")

    async def select_first_row() -> None:
        await view.set_primary_selection("py-row-0001")

    async def clear_from_python() -> None:
        await view.set_primary_selection(None)

    async def select_three_rows() -> None:
        await view.set_selection("py-row-0001", ["py-row-0001", "py-row-0002", "py-row-0003"])

    with ui.row().classes("items-center"):
        ui.select(
            {"baseline": "Baseline dataframe (180 rows)", "comparison": "Comparison dataframe (96 rows)"},
            value="baseline",
            label="Python dataframe",
            on_change=dataset_changed,
        )
        ui.button("Select first row from Python", on_click=select_first_row)
        ui.button("Select three rows from Python", on_click=select_three_rows)
        ui.button("Clear from Python", on_click=clear_from_python)
        ui.button("Set 1×2 from Python", on_click=use_two_plots)
        ui.button("Read state in Python", on_click=read_state)
        ui.button("Reload current dataset", on_click=reload_current_dataset)

    ui.timer(0, initialize_demo, once=True)


if __name__ == "__main__":
    ui.run(title="NicePool Web NiceGUI Demo", reload=False)
