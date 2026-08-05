"""Focused real-browser acceptance tests for independent raster widgets."""

import numpy as np
import pytest
from nicegui import ui
from nicegui.testing import Screen
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

from nicewidgets.raster_viewer_widget import (
    LineEndpoints,
    LineRoi,
    NumPyRasterSource,
    RasterViewerWidget,
    RectRoi,
    RectRoiBounds,
    XYPlot,
    XYPlotMode,
)


@pytest.mark.nicegui_main_file("")
def test_two_widgets_render_independently_and_replace_one_source(screen: Screen) -> None:
    """Exercise mounting, tooltips, and dataset replacement in a real browser."""
    first = np.arange(20, dtype=np.uint16).reshape(4, 5)
    replacement = np.arange(2 * 3 * 6 * 7, dtype=np.uint16).reshape(2, 3, 6, 7)
    second = np.arange(12, dtype=np.uint16).reshape(3, 4)
    widgets: dict[str, RasterViewerWidget] = {}

    @ui.page("/")
    def page() -> None:
        with ui.row().classes("w-full h-96"):
            widgets["first"] = RasterViewerWidget.from_array(
                first,
                dims=("Y", "X"),
                physical_units=(1.0, 1.0),
                physical_units_labels=("px", "px"),
                source_id="first",
                rois=(
                    RectRoi(1, "0", RectRoiBounds(0, 2, 0, 3)),
                    LineRoi(2, "1", LineEndpoints(0, 0, 3, 4)),
                ),
            )
            widgets["second"] = RasterViewerWidget.from_array(
                second,
                dims=("Y", "X"),
                physical_units=(1.0, 1.0),
                physical_units_labels=("px", "px"),
                source_id="second",
            )

        async def replace_first() -> None:
            await widgets["first"].load_source(
                NumPyRasterSource.from_array(
                    replacement,
                    dims=("T", "Z", "Y", "X"),
                    physical_units=(0.5, 1.0, 1.0, 1.0),
                    physical_units_labels=("s", "slice", "px", "px"),
                    source_id="replacement",
                )
            )
            status.set_text("replacement loaded")

        async def add_plot() -> None:
            """Add one plot only to the first widget instance."""
            await widgets["first"].xy_plots.add(
                XYPlot(
                    plot_id="acceptance",
                    x=(0.5, 2.0, 3.5),
                    y=(0.5, 3.0, 4.5),
                    mode=XYPlotMode.LINES_MARKERS,
                )
            )
            status.set_text("plot added")

        async def select_line() -> None:
            """Select the initial line through the namespaced ROI API."""
            selected = await widgets["first"].rois.select(2)
            status.set_text(f"line selected: {selected}")

        async def navigate_replacement() -> None:
            """Exercise named plane and runtime calibration APIs."""
            await widgets["first"].set_t_index(1)
            await widgets["first"].set_z_index(2)
            await widgets["first"].set_physical_calibration(
                (0.25, 1.0, 0.2, 0.3), ("s", "slice", "um", "um")
            )
            status.set_text("planes and calibration updated")

        async def clear_first() -> None:
            """Exercise the empty viewer lifecycle after replacement."""
            await widgets["first"].clear_source()
            status.set_text("first cleared")

        ui.button("Replace first", on_click=replace_first)
        ui.button("Add plot", on_click=add_plot)
        ui.button("Select line", on_click=select_line)
        ui.button("Navigate replacement", on_click=navigate_replacement)
        ui.button("Clear first", on_click=clear_first)
        status = ui.label("initial sources loaded")

    screen.open("/")
    screen.wait_for(lambda: screen.selenium.execute_script(
        "return document.querySelectorAll('.rv-root').length === 2"
    ))
    screen.wait_for(lambda: screen.selenium.execute_script(
        "return [...document.querySelectorAll('.rv-root')]"
        ".every(root => root.querySelector('canvas') && root.querySelector('[data-rv-tooltip]'))"
    ))
    option_buttons = screen.selenium.find_elements(By.CSS_SELECTOR, '[aria-label="Viewer options"]')
    assert len(option_buttons) == 2
    ActionChains(screen.selenium).move_to_element(option_buttons[0]).perform()
    screen.wait_for(lambda: screen.selenium.execute_script(
        "return [...document.querySelectorAll('.rv-tooltip')]"
        ".some(item => !item.hidden && item.textContent === 'Viewer options')"
    ))
    option_buttons[0].click()
    reset_buttons = screen.selenium.find_elements(
        By.CSS_SELECTOR,
        '.rv-options-menu[open] [aria-label="Reset view"]',
    )
    assert len(reset_buttons) == 1
    reset_buttons[0].click()
    screen.click("Select line")
    screen.wait_for("line selected: True")
    screen.click("Add plot")
    screen.wait_for("plot added")
    screen.wait_for(lambda: screen.selenium.execute_script(
        "const canvases = [...document.querySelectorAll('.rv-xy-plot-canvas')];"
        "return canvases.length === 2 && canvases.some(canvas => {"
        "const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;"
        "return data.some((value, index) => index % 4 === 3 && value > 0);});"
    ))
    screen.wait_for(lambda: screen.selenium.execute_script(
        "return [...document.querySelectorAll('.rv-xy-plot-canvas')]"
        ".every(canvas => getComputedStyle(canvas).pointerEvents === 'none')"
    ))
    screen.click("Replace first")
    screen.wait_for("replacement loaded")
    screen.wait_for(lambda: screen.selenium.execute_script(
        "const root = document.querySelectorAll('.rv-root')[0];"
        "return [...root.querySelectorAll('.rv-slice-dimension')]"
        ".map(item => item.textContent).join(',') === 'T,Z'"
    ))
    screen.click("Navigate replacement")
    screen.wait_for("planes and calibration updated")
    screen.wait_for(lambda: screen.selenium.execute_script(
        "return document.querySelectorAll('.rv-root').length === 2"
    ))
    screen.wait_for(lambda: screen.selenium.execute_script(
        "const canvas = document.querySelector('.rv-root .rv-xy-plot-canvas');"
        "const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;"
        "return !data.some((value, index) => index % 4 === 3 && value > 0);"
    ))
    screen.click("Clear first")
    screen.wait_for("first cleared")
    screen.wait_for(lambda: screen.selenium.execute_script(
        "const root = document.querySelectorAll('.rv-root')[0];"
        "return root.querySelectorAll("
        "'.rv-raster-canvas, .rv-xy-plot-canvas, .rv-roi-canvas'"
        ").length === 0 && root.querySelector('.rv-range-popover').hidden"
    ))
