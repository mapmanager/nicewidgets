"""Namespaced NiceGUI API for X/Y plot overlays."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .xy_plot import XYPlot

MethodRunner = Callable[..., Awaitable[Any]]


class XYPlotApi:
    """Manage the X/Y plots owned by one raster viewer widget instance."""

    def __init__(self, run_method: MethodRunner) -> None:
        """Create an API bound to one component method runner.

        Args:
            run_method: Async callable accepting a JavaScript method name and
                its positional arguments.
        """
        self._run_method = run_method

    async def add(self, plot: XYPlot) -> str:
        """Add a new plot and return its ID.

        Args:
            plot: Complete plot definition with an unused ``plot_id``.

        Returns:
            Installed plot identifier.

        Raises:
            RuntimeError: If JavaScript rejects a duplicate identifier or an
                otherwise invalid plot.
        """
        return str(await self._run_method("addXYPlot", plot.to_json()))

    async def update(self, plot: XYPlot) -> bool:
        """Fully replace an existing plot addressed by ``plot.plot_id``.

        Args:
            plot: Complete replacement definition.

        Returns:
            True after successful replacement.
        """
        return bool(await self._run_method("updateXYPlot", plot.to_json()))

    async def remove(self, plot_id: str) -> bool:
        """Remove one plot by ID and return whether it existed.

        Args:
            plot_id: Identifier supplied when the plot was added.

        Returns:
            True when a plot was removed.
        """
        return bool(await self._run_method("removeXYPlot", plot_id))

    async def show(self, plot_id: str) -> bool:
        """Show an existing plot without changing its data or style.

        Args:
            plot_id: Identifier of the plot to show.

        Returns:
            True when the plot exists.
        """
        return bool(await self._run_method("showXYPlot", plot_id))

    async def hide(self, plot_id: str) -> bool:
        """Hide an existing plot without removing it.

        Args:
            plot_id: Identifier of the plot to hide.

        Returns:
            True when the plot exists.
        """
        return bool(await self._run_method("hideXYPlot", plot_id))
