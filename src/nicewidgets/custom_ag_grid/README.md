# CustomAgGrid Hover Highlight Toggle

## Problem
Setting `hover_highlight=False` still showed a row hover highlight. The reason is
that our custom CSS only **adds** a hover color when the container has the
`.aggrid-hover` class, but AG Grid's theme applies its own default hover styling.
So even when our hover class is absent, the theme hover remains visible.

## Fix
We applied a two-part fix so `hover_highlight=False` reliably disables hover:

1. **CSS override when hover is disabled**
   - Added a new container class: `.aggrid-no-hover`.
   - Added CSS that neutralizes the theme hover styling when this class is present.

2. **AG Grid option**
   - When `hover_highlight=False`, we set:
     - `suppressRowHoverHighlight: True`
   - This is passed through the grid options so AG Grid also disables hover.

## Where the changes live
- `custom_ag_grid.py` and `custom_ag_grid_v2.py`
  - Add `.aggrid-no-hover` when `hover_highlight=False`
  - Set `suppressRowHoverHighlight=True` in grid options
- `theme.py`
  - Added CSS rules for `.aggrid-no-hover` to override the theme's hover style

## Usage
```python
grid_cfg = GridConfig(
    hover_highlight=False,
    # other config...
)
```

This now disables hover reliably in both `CustomAgGrid` and `CustomAgGrid_v2`.
