"""Pure validation for ``ImageToolbarWidget`` programmatic APIs (unit-testable, no NiceGUI)."""

from __future__ import annotations

from collections.abc import Sequence


def _as_str_list(options: Sequence[str]) -> list[str]:
    return [str(x) for x in options]


def _as_int_list(options: Sequence[int]) -> list[int]:
    out: list[int] = []
    for x in options:
        if isinstance(x, bool) or not isinstance(x, int):
            raise TypeError(f'ROI option must be int, got {type(x).__name__}')
        out.append(x)
    return out


def validate_set_file_ext_args(
    channel: int | None,
    roi_id: int | None,
    channel_options: Sequence[str],
    roi_options: Sequence[int],
) -> None:
    """Validate arguments for ``set_file_ext``.

    Rules:

    - ``channel_options`` empty: ``channel`` must be ``None``.
    - ``channel_options`` non-empty: ``channel`` must be non-``None`` and ``str(channel)``
      must appear in ``channel_options``.
    - ``roi_options`` empty: ``roi_id`` must be ``None``.
    - ``roi_options`` non-empty: ``roi_id`` must be non-``None`` and must appear in
      ``roi_options``.

    Raises:
        ValueError: If any rule is violated.
    """
    ch = _as_str_list(channel_options)
    ro = _as_int_list(roi_options)
    if len(ch) == 0:
        if channel is not None:
            raise ValueError('channel must be None when channel_options is empty')
    else:
        if channel is None:
            raise ValueError('channel is required when channel_options is non-empty')
        if str(channel) not in ch:
            raise ValueError(f'channel {channel!r} (as str) not in channel_options')
    if len(ro) == 0:
        if roi_id is not None:
            raise ValueError('roi_id must be None when roi_options is empty')
    else:
        if roi_id is None:
            raise ValueError('roi_id is required when roi_options is non-empty')
        if roi_id not in ro:
            raise ValueError(f'roi_id {roi_id!r} not in roi_options')


def validate_set_roi_options_and_selection_ext(roi_options: Sequence[int], roi_id: int | None) -> None:
    """Validate arguments for ``set_roi_options_and_selection_ext``.

    Same non-empty / empty rules as ROI slice of ``validate_set_file_ext_args``.

    Raises:
        ValueError: If arguments are inconsistent.
    """
    ro = _as_int_list(roi_options)
    if len(ro) == 0:
        if roi_id is not None:
            raise ValueError('roi_id must be None when roi_options is empty')
    else:
        if roi_id is None:
            raise ValueError('roi_id is required when roi_options is non-empty')
        if roi_id not in ro:
            raise ValueError(f'roi_id {roi_id!r} not in roi_options')


def validate_scalar_in_options(value_str: str | None, options: Sequence[str], *, field: str) -> None:
    """Ensure a string select value is allowed for the current string option list.

    Raises:
        ValueError: If violated.
    """
    opts = _as_str_list(options)
    if len(opts) == 0:
        if value_str is not None:
            raise ValueError(f'{field} must be None when the option list is empty')
    else:
        if value_str is None:
            raise ValueError(f'{field} is required when the option list is non-empty')
        if value_str not in opts:
            raise ValueError(f'{field} {value_str!r} not in the current option list')


def validate_scalar_int_in_options(value: int | None, options: Sequence[int], *, field: str) -> None:
    """Ensure an int ROI select value is allowed for the current ROI option list.

    Raises:
        ValueError: If violated.
    """
    opts = _as_int_list(options)
    if len(opts) == 0:
        if value is not None:
            raise ValueError(f'{field} must be None when the option list is empty')
    else:
        if value is None:
            raise ValueError(f'{field} is required when the option list is non-empty')
        if value not in opts:
            raise ValueError(f'{field} {value!r} not in the current option list')


def validate_options_update_preserves_selection(
    *,
    field: str,
    current_value_str: str | None,
    new_options: Sequence[str],
) -> None:
    """For ``set_channel_options_ext``: fail if current value missing from new options."""
    if current_value_str is None:
        return
    opts = _as_str_list(new_options)
    if current_value_str not in opts:
        raise ValueError(f'current {field}={current_value_str!r} not in new {field}_options')


def validate_options_update_preserves_int_selection(
    *,
    field: str,
    current_value: int | None,
    new_options: Sequence[int],
) -> None:
    """For ``set_roi_options_ext``: fail if current ROI missing from new options."""
    if current_value is None:
        return
    opts = _as_int_list(new_options)
    if current_value not in opts:
        raise ValueError(f'current {field}={current_value!r} not in new {field}_options')
