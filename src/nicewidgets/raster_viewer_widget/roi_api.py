"""Namespaced NiceGUI API for typed ROI lifecycle operations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from .roi import Roi, RoiCreate

MethodRunner = Callable[..., Awaitable[Any]]


class RoiApi:
    """Manage ROIs owned by one raster viewer instance."""

    def __init__(self, run_method: MethodRunner) -> None:
        """Bind the API to one component method runner.

        Args:
            run_method: Async component method bridge owned by one widget.
        """
        self._run_method = run_method

    async def set(self, rois: Sequence[Roi]) -> int:
        """Replace the complete committed ROI snapshot.

        Args:
            rois: Authoritative mixed-shape collection.

        Returns:
            Number of installed ROIs.
        """
        return int(await self._run_method("setRois", [roi.to_json() for roi in rois]))

    async def add(self, roi: Roi) -> bool:
        """Add a committed ROI, rejecting a duplicate ID.

        Args:
            roi: Complete typed committed ROI.

        Returns:
            True after successful installation.
        """
        return bool(await self._run_method("addRoi", roi.to_json()))

    async def update(self, roi: Roi) -> bool:
        """Replace a committed ROI, rejecting a missing ID.

        Args:
            roi: Complete typed replacement with an existing ID.

        Returns:
            True after successful replacement.
        """
        return bool(await self._run_method("updateRoi", roi.to_json()))

    async def remove(self, roi_id: int) -> bool:
        """Remove one committed ROI by stable ID.

        Args:
            roi_id: Dataset-local integer identity.

        Returns:
            Whether a matching ROI was removed.
        """
        return bool(await self._run_method("removeRoi", roi_id))

    async def select(self, roi_id: int | None) -> bool:
        """Select one committed ROI or clear selection with ``None``.

        Args:
            roi_id: Existing identity, or None to clear selection.

        Returns:
            Whether the requested selection was valid and applied.
        """
        return bool(await self._run_method("selectRoi", roi_id))

    async def clear_selection(self) -> bool:
        """Clear the current committed ROI selection.

        Returns:
            True after selection is cleared.
        """
        return await self.select(None)

    async def begin_create(self, specification: RoiCreate) -> bool:
        """Enter transactional creation mode with a typed draft.

        Args:
            specification: Rectangle or line initial geometry and metadata.

        Returns:
            False when another ROI transaction is already active.
        """
        return bool(await self._run_method("beginRoiCreate", specification.to_json()))

    async def begin_edit(self, roi_id: int) -> bool:
        """Enter transactional edit mode for one committed ROI.

        Args:
            roi_id: Existing committed ROI identity.

        Returns:
            Whether editing started.
        """
        return bool(await self._run_method("beginRoiEdit", roi_id))

    async def commit_edit(self) -> bool:
        """Emit one proposal for the active browser-local draft.

        Returns:
            Whether an active draft emitted a proposal.
        """
        return bool(await self._run_method("commitRoiEdit"))

    async def cancel_edit(self) -> bool:
        """Discard the active browser-local draft.

        Returns:
            Whether an active transaction was cancelled.
        """
        return bool(await self._run_method("cancelRoiEdit"))

    async def complete_commit(self, roi: Roi) -> bool:
        """Install the authoritative committed ROI and leave edit mode.

        Args:
            roi: Python-validated committed rectangle or line.

        Returns:
            True after installation and transaction completion.
        """
        return bool(await self._run_method("completeRoiCommit", roi.to_json()))
