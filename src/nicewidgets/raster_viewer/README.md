# raster_viewer

Backend-first source tree for a NiceGUI + Plotly raster viewer.

## Design goals

- keep the backend public API small and easy to test
- keep the NiceGUI/Plotly frontend thin
- fully reset the viewer when a new dataset is selected
- let application code own file lists, channel selection, and caching policy

## Main classes

### Backend

- `BackendImage`: holds one full-resolution 2D numpy array
- `ImagePyramid`: precomputes pyramid levels from a `BackendImage`
- `RasterViewService`: chooses level + payload mode and returns render responses

### Frontend

- `PlotlyRasterViewer`: thin adapter with one important public method:
  - `await set_data(data: np.ndarray)`

## Why `set_data()` does a full reset

`set_data()` assumes the new array may have no overlap with the previous one.
It creates a fresh backend image, fresh pyramid, fresh render service, resets the
bounds, and assigns a new Plotly `uirevision` token.

## Where caching belongs

If loading files or building pyramids is expensive and you want to reuse work
across repeated file/channel selections, keep that cache in your application
layer, keyed by your own `(file, channel)` identity.

That keeps the viewer simple:

- app layer chooses file/channel and may cache data or services
- viewer receives one 2D numpy array via `set_data()`
- backend service owns render policy for the currently selected array

## Demo

Run the small NiceGUI demo from the repository root:

```bash
uv run python examples/raster_viewer/nicegui_raster_demo.py
```

It uses 3 dummy generators and passes each generated numpy array into the
viewer via `set_data()`.
