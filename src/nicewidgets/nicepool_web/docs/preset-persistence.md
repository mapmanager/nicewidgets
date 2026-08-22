# Saved plot persistence

NicePool plot presets are JSON objects containing a name and one serializable
`PlotState`. The reusable engine and Custom Element expose preset get/set APIs;
the host application decides where those objects are stored.

## Pure web development demo

The standalone Vite demo stores presets in browser `localStorage` under this
key:

```text
nicepool-web-demo-presets-v3
```

The key is supplied by `src/app/App.vue`. The load, validation, save, and delete
behavior is implemented by `src/vue/NicePoolWidget.vue`.

A normal reload or a hard reload does not clear `localStorage`. During early
development, the key is advanced when the state contract changes so obsolete
presets are ignored instead of migrated. Delete a current preset
with the Saved plot controls, clear the site's browser storage, or remove the
key explicitly when a clean demo state is required.

Browser storage is isolated by origin. For example, the Vite development
server at `localhost:5173` and a NiceGUI server at `localhost:8080` cannot see
each other's saved presets even if they use the same storage key.

## Embedded clients

The Custom Element does not choose persistent storage. An embedding client can
load presets with `setPlotPresets()`, read them with `getPlotPresets()`, and
observe `nicepool-presets-change`. This keeps NicePool reusable:

- a browser SPA may use `localStorage` or a server API;
- a Python/NiceGUI host may use a JSON file, application settings, or a server;
- another thin client can use the same versioned preset JSON contract.

`setData()` remains a full dataset-dependent reset. A host should validate and
reload only the presets that are compatible with the replacement dataset.
