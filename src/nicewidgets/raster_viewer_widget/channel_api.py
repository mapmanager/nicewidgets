"""Namespaced channel-selection and presentation API."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .models import RasterChannelDisplay

type RunViewerMethod = Callable[..., Awaitable[Any]]


class ChannelApi:
    """Control active-channel and channel display state in one viewer."""

    def __init__(self, run_method: RunViewerMethod) -> None:
        """Store the component method adapter.

        Args:
            run_method: Awaitable adapter accepting a JavaScript method and arguments.
        """
        self._run_method = run_method

    async def select(self, channel_id: str) -> str:
        """Select one channel without emitting a user-originated callback.

        Args:
            channel_id: Dataset-local channel identifier.

        Returns:
            Applied channel identifier.
        """
        return str(await self._run_method("selectChannel", channel_id))

    async def set_display(
        self, channel_id: str, display: RasterChannelDisplay
    ) -> dict[str, object]:
        """Apply channel LUT, contrast range, and visibility in one operation.

        Args:
            channel_id: Dataset-local channel identifier.
            display: Complete desired channel presentation state.

        Returns:
            Browser-reported applied display snapshot.
        """
        result = await self._run_method("setChannelDisplay", channel_id, display.to_json())
        return result if isinstance(result, dict) else {}
