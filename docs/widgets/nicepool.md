# NicePool

`NicePool` is a DataFrame-driven plot pool: pre-filter dropdowns, plot-type
controls, optional table, named presets, and linked selection between table
rows and Plotly points.

## Embed

```python
import pandas as pd
from nicewidgets.nicepool import NicePool, NicePoolConfig

df = pd.DataFrame(
    [
        {'pool_row_id': 'a', 'accept': True, 'channel': 0, 'roi_id': 1, 'velocity_mean': 1.2},
        {'pool_row_id': 'b', 'accept': True, 'channel': 1, 'roi_id': 1, 'velocity_mean': 2.4},
    ]
)

pool = NicePool(
    df,
    config=NicePoolConfig(
        unique_row_id_col='pool_row_id',
        show_table_widget=True,
        enable_config_persistence=False,
        dark_mode=False,
    ),
    on_row_selected=lambda row_id, row: print(row_id, row),
)
pool.build()
```

DataFrame contract (also in the class docstring):

- unique `unique_row_id_col` (default `pool_row_id`)
- optional categorical pre-filters (`accept`, `channel`, `roi_id` auto-detected)
- at least one numeric column for the y-axis

Demo: `examples/nicepool/` (also `/nicepool` in the combined demo).

## Configuration

Pass a `NicePoolConfig` for filters, table visibility, presets, persistence,
and initial plot layout. Prefer `initial_plot_config` for deterministic first
paint without reading disk.

## API

::: nicewidgets.nicepool.nice_pool.NicePool
    options:
      show_root_heading: true
      heading_level: 3

::: nicewidgets.nicepool.config.NicePoolConfig
    options:
      show_root_heading: true
      heading_level: 3
