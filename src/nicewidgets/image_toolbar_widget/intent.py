"""User-intent payloads emitted by ``ImageToolbarWidget`` (frozen dataclasses, no host-application imports)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageToolbarSelectChannelIntent:
    """User selected a channel in the toolbar."""

    channel: int | None


@dataclass(frozen=True)
class ImageToolbarSelectRoiIntent:
    """User selected an ROI in the toolbar."""

    roi_id: int | None


@dataclass(frozen=True)
class ImageToolbarRoiAddRequestIntent:
    """User requested adding an ROI."""


@dataclass(frozen=True)
class ImageToolbarRoiDeleteRequestIntent:
    """User requested deleting the given ROI."""

    roi_id: int


@dataclass(frozen=True)
class ImageToolbarRoiEditStartIntent:
    """User entered ROI edit mode for ``roi_id``."""

    roi_id: int


@dataclass(frozen=True)
class ImageToolbarRoiEditCancelIntent:
    """User cancelled ROI edit (``roi_id`` is the ROI being edited, if known)."""

    roi_id: int | None


@dataclass(frozen=True)
class ImageToolbarRoiEditSubmitIntent:
    """User accepted ROI edits for ``roi_id``."""

    roi_id: int


@dataclass(frozen=True)
class ImageToolbarRoiApplyFullWidthIntent:
    """User requested full-width adjustment while editing ``roi_id``."""

    roi_id: int


@dataclass(frozen=True)
class ImageToolbarRoiApplyFullHeightIntent:
    """User requested full-height adjustment while editing ``roi_id``."""

    roi_id: int


type ImageToolbarIntent = (
    ImageToolbarSelectChannelIntent
    | ImageToolbarSelectRoiIntent
    | ImageToolbarRoiAddRequestIntent
    | ImageToolbarRoiDeleteRequestIntent
    | ImageToolbarRoiEditStartIntent
    | ImageToolbarRoiEditCancelIntent
    | ImageToolbarRoiEditSubmitIntent
    | ImageToolbarRoiApplyFullWidthIntent
    | ImageToolbarRoiApplyFullHeightIntent
)
