# Developer notes

NiceWidgets is extracted from the CloudScope monorepo as a standalone package.
This repository does not preserve the original Git history.

## Boundaries

- Runtime code under `src/nicewidgets/` must not import `cloudscope`,
  `acqstore`, or `acqstore_server`.
- Demos and manual scripts belong in `examples/` and `scripts/`, not in the
  installed package.
- Tests live under `tests/nicewidgets/`.

## Local commands

```bash
uv sync --group dev --group docs
uv run pytest
DISABLE_MKDOCS_2_WARNING=true uv run mkdocs build --strict
uv build
```
