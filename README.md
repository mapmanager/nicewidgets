# custom_ag_grid

A small, reusable wrapper around [NiceGUI](https://nicegui.io/)'s `ui.aggrid`
that makes it easy to configure and reuse AG Grid tables in Python.

`CustomAgGrid`:

- Accepts data as `list[dict]`, `pandas.DataFrame`, or `polars.DataFrame`
- Uses a declarative `ColumnConfig` / `GridConfig` API
- Provides simple, type-hinted configuration for editors (text, select)
- Exposes psygnal signals for cell edits and row selection
- Adds optional styling helpers (zebra rows, hover highlight, tight layout)

---

## Installation

You can install with `pip`:

```bash
pip install -e .
```

Or with [`uv`](https://github.com/astral-sh/uv) (from the project root):

```bash
uv pip install -e .
```

This library depends on:

- `nicegui`
- `psygnal`

`pandas` and `polars` are **optional**; they are only required if you
use the corresponding conversion helpers (`to_pandas()`, `to_polars()`)
or pass those types into the constructor.

---

## Quick Start

### Basic usage with `list[dict]` data

```python
from nicegui import ui
from custom_ag_grid.config import ColumnConfig, GridConfig
from custom_ag_grid.grid import CustomAgGrid

rows = [
    {"id": 1, "name": "Alice", "city": "Sacramento", "score": 88.5},
    {"id": 2, "name": "Bob", "city": "Baltimore", "score": 92.0},
]

columns = [
    ColumnConfig("id", header="ID"),
    ColumnConfig("name", header="Name", editable=True),
    ColumnConfig("city", header="City", editable=True, editor="select", choices="unique"),
    ColumnConfig("score", header="Score"),
]

grid_cfg = GridConfig(
    selection_mode="single",
    height="18rem",
    zebra_rows=True,
    hover_highlight=True,
    tight_layout=True,
)

with ui.header().classes("py-2 px-4"):
    ui.label("CustomAgGrid demo")

grid = CustomAgGrid(data=rows, columns=columns, grid_config=grid_cfg)

def on_edit(row: int, field: str, old: object, new: object, row_data: dict) -> None:
    print("EDIT:", row, field, old, "->", new, row_data)

def on_select(row: int, row_data: dict) -> None:
    print("SELECT:", row, row_data)

grid.cell_edited.connect(on_edit)
grid.row_selected.connect(on_select)

ui.run()
```

### Using pandas or Polars

```python
import pandas as pd
from custom_ag_grid.config import ColumnConfig, GridConfig
from custom_ag_grid.grid import CustomAgGrid

df = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Carol"],
        "city": ["Sacramento", "Baltimore", "Montreal"],
        "score": [88.5, 92.0, 76.5],
    }
)

columns = [
    ColumnConfig("id", header="ID"),
    ColumnConfig("name", header="Name", editable=True),
    ColumnConfig("city", header="City", editable=True, editor="select", choices="unique"),
    ColumnConfig("score", header="Score"),
]

grid_cfg = GridConfig(selection_mode="single", height="18rem")

grid = CustomAgGrid(data=df, columns=columns, grid_config=grid_cfg)
```

Later, you can extract the edited data back out:

```python
df_updated = grid.to_pandas()
rows_updated = grid.to_rows()
```

For Polars, use `grid.to_polars()` (with `polars` installed).

---

## API overview

### `ColumnConfig`

```python
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

EditorType = Literal["auto", "text", "select"]

@dataclass
class ColumnConfig:
    field: str
    header: str | None = None
    editable: bool = False
    editor: EditorType = "auto"
    choices: Iterable[Any] | Literal["unique"] | None = None
    sortable: bool = True
    filterable: bool = True
    resizable: bool = True
    extra_grid_options: dict[str, Any] = field(default_factory=dict)
```

Key points:

- `editor="auto"`  
  Uses AG Grid’s default editor for the column.

- `editor="select", choices="unique"`  
  Builds a dropdown editor whose values are the unique values of this column
  in the current data.

- `editor="select", choices=[...]`  
  Uses the given list as the dropdown options.

### `GridConfig`

```python
from dataclasses import dataclass, field
from typing import Literal, Any

SelectionMode = Literal["none", "single", "multiple"]

@dataclass
class GridConfig:
    selection_mode: SelectionMode = "none"
    height: str = "20rem"
    theme_class: str = "ag-theme-alpine"
    zebra_rows: bool = True
    hover_highlight: bool = True
    tight_layout: bool = True
    row_height: int = 28
    header_height: int = 30
    stop_editing_on_focus_loss: bool = True
    extra_grid_options: dict[str, Any] = field(default_factory=dict)
```

Style-related flags:

- `zebra_rows`  
  Adds `.aggrid-zebra` to the container, enabling zebra striping via CSS.

- `hover_highlight`  
  Adds `.aggrid-hover` to the container, enabling row hover highlight.

- `tight_layout`  
  Adds `.aggrid-tight` to the container, reducing padding and font size.

### `CustomAgGrid` signals

```python
from psygnal import Signal

class CustomAgGrid:
    cell_edited = Signal(int, str, object, object, dict)
    row_selected = Signal(int, dict)
```

Usage:

```python
def on_edit(row: int, field: str, old: object, new: object, row: dict) -> None:
    print("Cell edited:", row, field, old, "->", new)

grid.cell_edited.connect(on_edit)
```

---

## Data conversion helpers

`CustomAgGrid` stores data internally as `list[dict[str, Any]]` and provides
helpers to convert back to tabular libraries when needed:

```python
rows = grid.to_rows()       # list[dict[str, Any]]
df   = grid.to_pandas()     # pandas.DataFrame (requires pandas)
pldf = grid.to_polars()     # polars.DataFrame (requires polars)
```

If `pandas` or `polars` are not installed, the corresponding methods will
raise `ImportError`.

---

## Development

- Type hints throughout; intended to be mypy/pyright-friendly.
- Google-style docstrings.
- Tests live in `tests/` and can be run with:

  ```bash
  pytest
  ```

You can install the package in editable mode while developing:

```bash
pip install -e .
# or
uv pip install -e .
```
