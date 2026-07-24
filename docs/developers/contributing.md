# Contributing

Keep NiceWidgets free of CloudScope and AcqStore imports in runtime package
code. Prefer small, focused changes that preserve existing public import paths.

Before proposing a change:

```bash
uv sync --group dev
uv run pytest
```

When editing documentation under `docs/`, keep pages understandable without the
CloudScope monorepo.
