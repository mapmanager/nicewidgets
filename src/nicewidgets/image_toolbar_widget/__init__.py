"""Image toolbar NiceGUI widget: channel/ROI selects and ROI action buttons.

Import from submodules (the package root does not re-export symbols):

```python
from nicewidgets.image_toolbar_widget.image_toolbar_widget import ImageToolbarWidget
from nicewidgets.image_toolbar_widget.intent import ImageToolbarIntent
```

Contract summary
----------------

- **User gestures emit intents.** All user actions invoke one callback,
  ``on_intent: Callable[[ImageToolbarIntent], None] | None``.
  :data:`~nicewidgets.image_toolbar_widget.intent.ImageToolbarIntent` is a
  union of frozen dataclasses (channel/ROI selection, ROI CRUD, and the ROI
  edit lifecycle). If ``on_intent`` is ``None``, ROI action buttons are
  disabled; selection intents are still emitted while the toolbar is enabled.
- **Programmatic updates use ``*_ext`` methods** which update UI and internal
  state without calling ``on_intent`` (no feedback loops).
- **Channel options are ``list[str]``** in the select (e.g. ``"0"``, ``"1"``);
  ``set_channel_ext`` takes ``int | None`` and matches via ``str(channel)``.
- **ROI options and ids are ``int``**, matching backend ROI identifiers.

See :class:`~nicewidgets.image_toolbar_widget.image_toolbar_widget.ImageToolbarWidget`
for per-method validation rules, and ``examples/raster_viewer/`` in the
repository for a runnable host wiring example.
"""
