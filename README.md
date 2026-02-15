# nicewidgets

A collection of reusable [NiceGUI](https://nicegui.io/) widgets for building interactive web applications in Python.

## Overview

`nicewidgets` provides high-level, configurable widgets built on top of NiceGUI:

- **CustomAgGrid**: AG Grid wrapper with editable cells, row selection, and declarative configuration
- **RoiImageWidget**: Interactive image viewer with ROI drawing, zoom, and pan capabilities
- **Logging utilities**: Simple logging setup for standalone use or library integration

This package is designed to be used as a standalone library or integrated into larger applications.

---

## Requirements

- **Python**: 3.11 or higher
- **Package manager**: [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- **Core dependencies**: `nicegui`, `psygnal`, `numpy`, `Pillow`, `matplotlib`
- **Optional**: `pandas`, `polars` (for data frame support in CustomAgGrid)

---

## Installation from Source

### 1. Clone the repository

```bash
git clone https://github.com/mapmanager/nicewidgets.git
cd nicewidgets
```

### 2. Set up environment

#### Using `uv` (recommended)

```bash
# Create virtual environment
uv venv

# Install package in editable mode
uv pip install -e .
```

#### Using standard Python `venv`

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install package in editable mode
pip install -e .
```

---

## Examples

See the `examples/` directory for working demonstrations:

- `examples/example_script.py` - Basic CustomAgGrid usage
- `examples/sample_roi_image_widget.py` - RoiImageWidget with zoom, pan, and ROI drawing

### Running Examples

Some examples require additional dependencies (like `scipy`). Install them with:

```bash
# Using uv
uv pip install -e ".[examples]"

# Using pip
pip install -e ".[examples]"
```

Then run an example:

```bash
python examples/sample_roi_image_widget.py
```

---

## Plot Pool App

The plot pool app is a standalone NiceGUI application for exploring CSV data with linked Plotly visualizations.

```bash
uv run python -m nicewidgets.plot_pool_app.plot_pool_app
```

See `src/nicewidgets/plot_pool_app/` for the full app and demo scripts.

---

## Running Tests

Tests use `pytest` and are located in the `tests/` directory.

### Install test dependencies

```bash
# Using uv
uv pip install -e ".[test]"

# Using pip
pip install -e ".[test]"
```

### Run all tests

```bash
pytest
```

### Run tests with coverage

```bash
pytest --cov=nicewidgets --cov-report=html
```

---

## Development

### Project Structure

```
nicewidgets/
├── src/nicewidgets/           # Source code
│   ├── custom_ag_grid/        # AG Grid wrapper
│   ├── roi_image_widget/      # ROI image viewer
│   ├── plot_pool_widget/      # Plot pool controller and table view
│   ├── plot_pool_app/         # Plot pool demo apps (demo_pool_app, demo_df_table_app)
│   └── utils/                 # Logging and utilities
├── tests/                     # Unit tests
├── examples/                  # Example scripts
└── pyproject.toml            # Project configuration
```

### Configuration

The project uses `pyproject.toml` for all configuration:

- **Build system**: `hatchling`
- **Dependencies**: Defined in `[project.dependencies]`
- **Optional extras**: `test`, `docs`, `dev`, `pandas`, `polars`
- **Test configuration**: `[tool.pytest.ini_options]`

### Logging

For standalone scripts, configure logging:

```python
from nicewidgets.utils.logging import setup_logging

# Console + file logging
setup_logging(level="DEBUG", log_file="~/my_app.log")

# Console only
setup_logging(level="INFO", log_file=None)
```

When imported by other applications, `nicewidgets` automatically uses the parent application's logging configuration.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes with tests
4. Ensure tests pass (`pytest`)
5. Commit your changes (`git commit -m "Add my feature"`)
6. Push to your branch (`git push origin feature/my-feature`)
7. Open a Pull Request

### Large File Check (Git Hooks)

A pre-commit and pre-push check blocks files over 60MB to avoid accidental pushes of large binaries.

**Install hooks (run once after cloning):**

```bash
./scripts/install-git-hooks.sh
```

- **pre-commit**: Checks staged files before `git commit`
- **pre-push**: Checks all tracked files before `git push`

To change the limit, edit `MAX_SIZE_MB` in `scripts/git-large-file-check.sh`.

### Code Style

- Type hints throughout
- Google-style docstrings
- Follow existing code patterns
- Add tests for new features

---

## License

GNU General Public License v3 or later (GPLv3+)

See [LICENSE](LICENSE) for details.

---

## Links

- **Repository**: https://github.com/mapmanager/nicewidgets
- **Issues**: https://github.com/mapmanager/nicewidgets/issues
- **Documentation**: _(Coming soon)_
