"""Backend-first NiceGUI + Plotly raster viewer.

Main classes
------------

Backend (``nicewidgets.raster_viewer.backend``):

- ``BackendImage``: holds one full-resolution 2D numpy array plus grid metadata
- ``ImagePyramid``: precomputes multiresolution levels from a ``BackendImage``
- ``RasterViewService``: chooses level and payload mode, returns render responses

Frontend (``nicewidgets.raster_viewer.frontend``):

- ``PlotlyRasterViewer``: thin adapter; hosts call ``build()`` once, then
  ``await set_data(...)`` per dataset and public setters for contrast, ROIs,
  overlays, and axis ranges

Multi-channel (``nicewidgets.raster_viewer.multichannel``):

- models + additive RGB composition (Phase 1)
- ``MultiChannelRasterView`` coordinator: single / mosaic + linked viewport
  (Phase 2; ``examples/raster_viewer`` is the first consumer)
- composite RGB pane via Plotly ``image`` HxWx3 (Phase 3; not PNG)

Design goals
------------

- keep the backend public API small and easy to test
- keep the NiceGUI/Plotly frontend thin
- fully reset the viewer when a new dataset is selected
- let application code own file lists, channel selection, and caching policy

Why ``set_data()`` does a full reset
------------------------------------

``set_data()`` assumes the new array may have no overlap with the previous
one. It creates a fresh backend image, pyramid, and render service, resets the
bounds, and assigns a new Plotly ``uirevision`` token. Use
``swap_slice_plane()`` instead for Z/T/channel scrubs within one file, where
the zoom viewport, contrast, and overlays should be preserved.

Where caching belongs
---------------------

If loading files or building pyramids is expensive, keep that cache in your
application layer, keyed by your own ``(file, channel)`` identity, and pass
prebuilt pyramids via ``set_data_from_pyramid()`` / ``swap_slice_plane()``.
The viewer stays simple: it receives one 2D numpy array and the backend
service owns render policy for the currently selected array.

Demo
----

Run the NiceGUI demo from the repository root:

```bash
uv run python examples/raster_viewer/nicegui_raster_demo.py
```
"""
