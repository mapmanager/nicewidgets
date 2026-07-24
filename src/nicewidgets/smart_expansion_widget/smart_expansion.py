"""Lifecycle-aware wrapper around NiceGUI ``ui.expansion``."""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType

from nicegui import ui
from nicegui.events import ValueChangeEventArguments


class SmartExpansion:
    """Wrap ``ui.expansion`` with open/close lifecycle callbacks.

    The widget is host-application-agnostic. Callers wire ``on_open`` and ``on_close``
    to connect or disconnect expensive content from their own MVC/event systems.

    Content is built once and kept in the DOM. Callbacks fire when the expansion
    opens or closes, including after programmatic ``open()`` / ``close()`` calls.

    Args:
        text: Expansion header title.
        icon: Optional Material icon name for the header.
        caption: Optional caption (sub-label) text.
        group: Optional accordion group name for coordinated open/close.
        initially_open: Whether the expansion starts open.
        on_open: Optional callback invoked when the expansion opens.
        on_close: Optional callback invoked when the expansion closes.
    """

    def __init__(
        self,
        text: str,
        *,
        icon: str | None = None,
        caption: str | None = None,
        group: str | None = None,
        initially_open: bool = False,
        on_open: Callable[[], None] | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._on_open = on_open
        self._on_close = on_close
        self._is_open = bool(initially_open)
        self._initial_state_applied = False
        self._expansion = ui.expansion(
            text,
            icon=icon,
            caption=caption,
            group=group,
            value=initially_open,
            on_value_change=self._on_value_change,
        )
        self._expansion.classes('w-full')

    @property
    def expansion(self) -> ui.expansion:
        """Return the underlying NiceGUI expansion element.

        Returns:
            Wrapped ``ui.expansion`` instance.
        """
        return self._expansion

    @property
    def is_open(self) -> bool:
        """Return whether the expansion is currently open.

        Returns:
            True when the expansion value is open.
        """
        return self._is_open

    def open(self) -> None:
        """Open the expansion programmatically.

        Returns:
            None.
        """
        self._expansion.open()

    def close(self) -> None:
        """Close the expansion programmatically.

        Returns:
            None.
        """
        self._expansion.close()

    def apply_initial_state(self) -> None:
        """Invoke open/close callbacks for the initial expansion value.

        NiceGUI does not reliably fire ``on_value_change`` for the constructor
        ``value``. Call this once after child content has been built.

        Returns:
            None.
        """
        if self._initial_state_applied:
            return
        self._initial_state_applied = True
        opened = bool(self._expansion.value)
        if opened:
            if self._on_open is not None:
                self._on_open()
        elif self._on_close is not None:
            self._on_close()
        self._is_open = opened

    def __enter__(self) -> ui.expansion:
        """Enter the expansion content context manager.

        Returns:
            Underlying ``ui.expansion`` for building child content.
        """
        return self._expansion.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the expansion content context manager.

        Args:
            exc_type: Exception type, if any.
            exc_val: Exception value, if any.
            exc_tb: Exception traceback, if any.

        Returns:
            None.
        """
        self._expansion.__exit__(exc_type, exc_val, exc_tb)

    def _on_value_change(self, event: ValueChangeEventArguments) -> None:
        """Dispatch lifecycle callbacks when expansion value changes.

        Args:
            event: NiceGUI value-change event for the expansion open state.

        Returns:
            None.
        """
        opened = bool(event.value)
        self._dispatch_open_state(opened)

    def _dispatch_open_state(self, opened: bool) -> None:
        """Invoke ``on_open`` or ``on_close`` when state actually changes.

        Args:
            opened: Desired open state.

        Returns:
            None.
        """
        if opened == self._is_open:
            return
        self._is_open = opened
        if opened:
            if self._on_open is not None:
                self._on_open()
            return
        if self._on_close is not None:
            self._on_close()
