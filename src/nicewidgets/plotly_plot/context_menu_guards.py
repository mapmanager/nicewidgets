"""Desktop pywebview pointer guards for Plotly plot context menus."""

from __future__ import annotations

_PLOTLY_PLOT_CONTEXT_MENU_GUARD_FLAG = "csPlotlyPlotContextMenuGuard"


def pywebview_plotly_plot_context_menu_guard_js(*, plot_id: int) -> str:
    """Return JavaScript that installs idempotent plot-surface pointer guards.

    Args:
        plot_id: NiceGUI Plotly element id.

    Returns:
        JavaScript source executed in the browser client.
    """
    flag = _PLOTLY_PLOT_CONTEXT_MENU_GUARD_FLAG
    return f"""
const host = getElement({plot_id}).$el;
if (!host) return;
const plotDiv = host.querySelector('.js-plotly-plot') || host;
if (!plotDiv) return;
if (plotDiv.dataset.{flag} === '1') return;
plotDiv.dataset.{flag} = '1';
const capture = {{ capture: true }};
const blockNonPrimary = (ev) => {{
  if (typeof ev.button === 'number' && ev.button !== 0) {{
    ev.stopImmediatePropagation();
  }}
}};
const suppressNativeMenu = (ev) => {{
  ev.preventDefault();
}};
for (const name of ['pointerdown', 'pointerup', 'mousedown', 'mouseup']) {{
  plotDiv.addEventListener(name, blockNonPrimary, capture);
}}
plotDiv.addEventListener('contextmenu', suppressNativeMenu, capture);
"""
