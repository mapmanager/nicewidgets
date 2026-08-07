"""NiceGUI image toolbar: file label, channel/ROI selects, ROI CRUD intents.

DEPRECATED (ROI chrome): Prefer the canvas ``raster_viewer_widget`` JS ROI
toolbar (dropdown + add/delete/edit) for new work. CloudScope still uses this
widget via ``ImageToolbarView`` during migration; do not remove the package
until that wiring is retired. Demos under ``examples/raster_viewer_widget/``
no longer mount this toolbar.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from enum import Enum, auto

from nicegui import ui
from nicegui.events import ValueChangeEventArguments

from nicewidgets.compact_select_styles import (
    COMPACT_SELECT_CLASS,
    COMPACT_SELECT_PROPS,
    ensure_compact_select_styles,
)
from nicewidgets.image_toolbar_widget.intent import (
    ImageToolbarIntent,
    ImageToolbarRoiAddRequestIntent,
    ImageToolbarRoiApplyFullHeightIntent,
    ImageToolbarRoiApplyFullWidthIntent,
    ImageToolbarRoiDeleteRequestIntent,
    ImageToolbarRoiEditCancelIntent,
    ImageToolbarRoiEditStartIntent,
    ImageToolbarRoiEditSubmitIntent,
    ImageToolbarSelectChannelIntent,
    ImageToolbarSelectRoiIntent,
)
from nicewidgets.image_toolbar_widget.validation import (
    validate_options_update_preserves_int_selection,
    validate_options_update_preserves_selection,
    validate_scalar_in_options,
    validate_scalar_int_in_options,
    validate_set_file_ext_args,
    validate_set_roi_options_and_selection_ext,
)


class ImageToolbarMode(Enum):
    """Toolbar ROI interaction mode."""

    IDLE = auto()
    EDITING = auto()


class ImageToolbarWidget:
    """Toolbar with file label, channel/ROI selects, and ROI edit controls.

    DEPRECATED for ROI CRUD/select chrome: new hosts should use the JS ROI
    strip inside ``RasterViewerWidget`` (``RoiHostMode.DELEGATED`` + request
    hooks). Channel/file labeling may still be useful until those move.

    The widget does not own a layout container. Child controls are created in
    the caller's active NiceGUI slot so the parent fully controls layout (sit
    on a shared row with other widgets, etc.). This matches the composition
    pattern used by other ``nicewidgets`` (e.g. :class:`EChartWidget`).

    Programmatic updates use ``*_ext`` methods and never invoke ``on_intent``.
    User gestures emit frozen intent objects via ``on_intent``.

    ROI identifiers are ``int`` (aligned with backend ROI ids). Channel options remain
    string keys in the select; ROI options are integer keys in the select.
    """

    def __init__(
        self,
        *,
        on_intent: Callable[[ImageToolbarIntent], None] | None = None,
        widget_name: str = 'image_toolbar_widget',
    ) -> None:
        """Create the toolbar and build child controls.

        Args:
            on_intent: Called for user-driven intents. If ``None``, ROI action buttons
                are disabled (selects still emit selection intents when enabled).
            widget_name: Host-visible identifier for logging or debugging.
        """
        self._widget_name = widget_name
        self._on_intent = on_intent
        self._suppress_intent = False
        self._enabled = True
        self._mode = ImageToolbarMode.IDLE

        self._file_id: str | None = None
        self._channel_options: list[str] = []
        self._roi_options: list[int] = []
        self._channel: int | None = None
        self._roi_id: int | None = None

        ensure_compact_select_styles()

        with ui.row().classes('items-center gap-1'):
            ui.label('Channel')
            self._channel_select = ui.select(
                options=[],
                value=None,
                on_change=self._on_channel_change,
            ).props(f'name=channel {COMPACT_SELECT_PROPS}').classes(f'w-24 {COMPACT_SELECT_CLASS}')

        with ui.row().classes('items-center gap-1'):
            ui.label('ROI')
            self._roi_select = ui.select(
                options=[],
                value=None,
                on_change=self._on_roi_change,
            ).props(f'name=roi {COMPACT_SELECT_PROPS}').classes(f'w-24 {COMPACT_SELECT_CLASS}')

        self._add_btn = ui.button(icon='add', on_click=self._on_add_click).props('flat round')
        self._add_btn.tooltip('Add ROI')
        self._delete_btn = ui.button(icon='remove', on_click=self._on_delete_click).props('flat round')
        self._delete_btn.tooltip('Delete ROI')
        self._edit_btn = ui.button(icon='edit', on_click=self._on_edit_click).props('flat round')
        self._edit_btn.tooltip('Edit ROI')
        self._full_width_btn = ui.button(icon='code', on_click=self._on_full_width_click).props('flat round')
        self._full_width_btn.tooltip('Full width')
        self._full_height_btn = ui.button(icon='height', on_click=self._on_full_height_click).props('flat round')
        self._full_height_btn.tooltip('Full height')
        self._ok_btn = ui.button('OK', on_click=self._on_ok_click).props('flat')
        self._ok_btn.tooltip('Submit ROI edit')
        self._cancel_btn = ui.button('Cancel', on_click=self._on_cancel_click).props('flat')
        self._cancel_btn.tooltip('Cancel ROI edit')

        self._apply_roi_mode_to_ui()

    @contextmanager
    def _intent_suppressed(self) -> object:
        prev = self._suppress_intent
        self._suppress_intent = True
        try:
            yield
        finally:
            self._suppress_intent = prev

    def _emit(self, intent: ImageToolbarIntent) -> None:
        if self._suppress_intent or not self._enabled or self._on_intent is None:
            return
        self._on_intent(intent)

    def _format_file_label(self, file_id: str | None) -> str:
        return file_id if file_id is not None else '—'

    def _channel_as_str(self, channel: int | None) -> str | None:
        if channel is None:
            return None
        return str(channel)

    def _parse_channel_str(self, s: str | None) -> int | None:
        if s is None:
            return None
        try:
            return int(s, 10)
        except ValueError as e:
            raise ValueError(f'channel select value must be int-compatible str, got {s!r}') from e

    @staticmethod
    def _parse_roi_select_value(v: object) -> int | None:
        """Parse NiceGUI select value to ``int | None`` (reject bool)."""
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError(f'invalid roi select value type bool: {v!r}')
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            if not v.is_integer():
                raise ValueError(f'roi select value must be whole number, got {v!r}')
            return int(v)
        if isinstance(v, str):
            return int(v, 10)
        raise ValueError(f'unsupported roi select value type {type(v).__name__}: {v!r}')

    def _current_roi_for_actions(self) -> int | None:
        return self._roi_id

    def _apply_roi_mode_to_ui(self) -> None:
        """Show/hide and enable ROI action buttons from mode, options, and flags."""
        editing = self._mode == ImageToolbarMode.EDITING
        has_roi = bool(self._roi_options) and self._roi_id is not None
        hosted = self._on_intent is not None
        base = self._enabled

        self._channel_select.set_enabled(base and not editing)
        self._roi_select.set_enabled(base and not editing and bool(self._roi_options))

        for b in (self._add_btn, self._delete_btn, self._edit_btn):
            b.set_visibility(not editing)
        for b in (self._full_width_btn, self._full_height_btn, self._ok_btn, self._cancel_btn):
            b.set_visibility(editing)

        self._add_btn.set_enabled(base and not editing and hosted)
        self._delete_btn.set_enabled(base and not editing and has_roi and hosted)
        self._edit_btn.set_enabled(base and not editing and has_roi and hosted)
        self._full_width_btn.set_enabled(base and editing and has_roi and hosted)
        self._full_height_btn.set_enabled(base and editing and has_roi and hosted)
        self._ok_btn.set_enabled(base and editing and has_roi and hosted)
        self._cancel_btn.set_enabled(base and editing and hosted)

    def set_enabled_ext(self, enabled: bool) -> None:
        """Enable or disable all toolbar interaction (programmatic updates still work)."""
        self._enabled = enabled
        self._apply_roi_mode_to_ui()

    def set_file_ext(
        self,
        file_id: str | None,
        channel: int | None,
        roi_id: int | None,
        *,
        channel_options: Sequence[str],
        roi_options: Sequence[int],
    ) -> None:
        """Set file label, both option lists, and channel/ROI selection. Does not emit intents.

        Resets ROI edit mode to idle. Both option lists are always required.

        Validation rules:

        - Channel: empty ``channel_options`` requires ``channel is None``;
          non-empty requires a non-``None`` ``channel`` whose ``str(channel)``
          appears in ``channel_options``.
        - ROI: empty ``roi_options`` requires ``roi_id is None``; non-empty
          requires a non-``None`` ``roi_id`` present in ``roi_options``.

        Args:
            file_id: Display identifier for the loaded file, or ``None``.
            channel: Current channel, or ``None`` when no channels exist.
            roi_id: Current ROI id, or ``None`` when no ROIs exist.
            channel_options: All selectable channels as strings (e.g. ``['0', '1']``).
            roi_options: All selectable ROI ids as ints.

        Raises:
            ValueError: If selections are inconsistent with option lists.
        """
        validate_set_file_ext_args(channel, roi_id, channel_options, roi_options)
        with self._intent_suppressed():
            self._mode = ImageToolbarMode.IDLE
            self._file_id = file_id
            # self._file_label.set_text(self._format_file_label(file_id))
            self._channel_options = list(channel_options)
            self._roi_options = list(roi_options)
            self._channel = channel
            self._roi_id = roi_id
            self._channel_select.set_options(self._channel_options)
            self._channel_select.value = self._channel_as_str(channel)
            self._roi_select.set_options(self._roi_options)
            self._roi_select.value = roi_id
        self._apply_roi_mode_to_ui()

    def set_channel_options_ext(self, options: Sequence[str]) -> None:
        """Replace channel select options without emitting intents.

        Args:
            options: New channel option strings.

        Raises:
            ValueError: If a channel is currently selected and its string form
                is missing from ``options``. A ``None`` selection is allowed;
                follow with :meth:`set_channel_ext` if a selection is needed.
        """
        cur = self._channel_as_str(self._channel)
        validate_options_update_preserves_selection(field='channel', current_value_str=cur, new_options=options)
        with self._intent_suppressed():
            self._channel_options = list(options)
            self._channel_select.set_options(self._channel_options)
            self._channel_select.value = cur
        self._apply_roi_mode_to_ui()

    def set_roi_options_ext(self, options: Sequence[int]) -> None:
        """Replace ROI select options without emitting intents.

        Args:
            options: New ROI option ids.

        Raises:
            ValueError: If an ROI is currently selected and missing from
                ``options``. A ``None`` selection is allowed; follow with
                :meth:`set_roi_ext` if a selection is needed.
        """
        cur = self._roi_id
        validate_options_update_preserves_int_selection(field='roi', current_value=cur, new_options=options)
        with self._intent_suppressed():
            self._roi_options = list(options)
            self._roi_select.set_options(self._roi_options)
            self._roi_select.value = cur
        self._apply_roi_mode_to_ui()

    def set_roi_options_and_selection_ext(self, roi_options: Sequence[int], roi_id: int | None) -> None:
        """Atomically set ROI options and selection. Does not emit intents.

        Use after host-side ROI add/delete so options and selection stay
        consistent in one update.

        Args:
            roi_options: All selectable ROI ids.
            roi_id: Selected ROI id; must be ``None`` when ``roi_options`` is
                empty, and present in ``roi_options`` otherwise.

        Raises:
            ValueError: If ``roi_id`` is inconsistent with ``roi_options``.
        """
        validate_set_roi_options_and_selection_ext(roi_options, roi_id)
        with self._intent_suppressed():
            self._roi_options = list(roi_options)
            self._roi_id = roi_id
            self._roi_select.set_options(self._roi_options)
            self._roi_select.value = roi_id
        self._apply_roi_mode_to_ui()

    def set_channel_ext(self, channel: int | None) -> None:
        """Set channel selection without emitting intents.

        Args:
            channel: New channel. With empty ``channel_options`` the value must
                be ``None``; otherwise ``str(channel)`` must be in the options.

        Raises:
            ValueError: If the value is inconsistent with current options.
        """
        validate_scalar_in_options(self._channel_as_str(channel), self._channel_options, field='channel value')
        with self._intent_suppressed():
            self._channel = channel
            self._channel_select.value = self._channel_as_str(channel)
        self._apply_roi_mode_to_ui()

    def set_roi_ext(self, roi_id: int | None) -> None:
        """Set ROI selection without emitting intents.

        Args:
            roi_id: New ROI id. With empty ``roi_options`` the value must be
                ``None``; otherwise it must be in the options.

        Raises:
            ValueError: If the value is inconsistent with current options.
        """
        validate_scalar_int_in_options(roi_id, self._roi_options, field='roi_id')
        with self._intent_suppressed():
            self._roi_id = roi_id
            self._roi_select.value = roi_id
        self._apply_roi_mode_to_ui()

    def get_file_id(self) -> str | None:
        """Return last programmatic or initial file id (for tests)."""
        return self._file_id

    def get_channel(self) -> int | None:
        """Return current channel."""
        return self._channel

    def get_roi_id(self) -> int | None:
        """Return current ROI id."""
        return self._roi_id

    def get_channel_options(self) -> list[str]:
        """Return a copy of channel option strings."""
        return list(self._channel_options)

    def get_roi_options(self) -> list[int]:
        """Return a copy of ROI option ints."""
        return list(self._roi_options)

    def _on_channel_change(self, e: ValueChangeEventArguments) -> None:
        if self._suppress_intent or not self._enabled:
            return
        ch = self._parse_channel_str(e.value)
        self._channel = ch
        self._emit(ImageToolbarSelectChannelIntent(channel=ch))

    def _on_roi_change(self, e: ValueChangeEventArguments) -> None:
        if self._suppress_intent or not self._enabled:
            return
        rid = self._parse_roi_select_value(e.value)
        self._roi_id = rid
        self._emit(ImageToolbarSelectRoiIntent(roi_id=rid))

    def _on_add_click(self) -> None:
        if not self._enabled:
            return
        self._emit(ImageToolbarRoiAddRequestIntent())

    def _on_delete_click(self) -> None:
        if not self._enabled:
            return
        name = self._current_roi_for_actions()
        if name is None:
            return
        self._emit(ImageToolbarRoiDeleteRequestIntent(roi_id=name))

    def _on_edit_click(self) -> None:
        if not self._enabled:
            return
        name = self._current_roi_for_actions()
        if name is None:
            return
        self._mode = ImageToolbarMode.EDITING
        self._apply_roi_mode_to_ui()
        self._emit(ImageToolbarRoiEditStartIntent(roi_id=name))

    def _on_full_width_click(self) -> None:
        if not self._enabled:
            return
        name = self._current_roi_for_actions()
        if name is None:
            return
        self._emit(ImageToolbarRoiApplyFullWidthIntent(roi_id=name))

    def _on_full_height_click(self) -> None:
        if not self._enabled:
            return
        name = self._current_roi_for_actions()
        if name is None:
            return
        self._emit(ImageToolbarRoiApplyFullHeightIntent(roi_id=name))

    def _on_ok_click(self) -> None:
        if not self._enabled:
            return
        name = self._current_roi_for_actions()
        if name is None:
            return
        self._emit(ImageToolbarRoiEditSubmitIntent(roi_id=name))
        self._mode = ImageToolbarMode.IDLE
        self._apply_roi_mode_to_ui()

    def _on_cancel_click(self) -> None:
        if not self._enabled:
            return
        name = self._current_roi_for_actions()
        self._emit(ImageToolbarRoiEditCancelIntent(roi_id=name))
        self._mode = ImageToolbarMode.IDLE
        self._apply_roi_mode_to_ui()
