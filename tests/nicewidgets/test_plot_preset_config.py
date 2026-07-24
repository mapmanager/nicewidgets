from __future__ import annotations

import json

from nicewidgets.nicepool.plot_preset_config import PlotPresetStore


def test_plot_preset_store_load_missing_file(tmp_path):
    path = tmp_path / "nicepoolplots.json"

    store = PlotPresetStore.load(path=path)

    assert store.names() == []
    assert store.path == path


def test_plot_preset_store_upsert_save_and_reload(tmp_path):
    path = tmp_path / "nicepoolplots.json"
    store = PlotPresetStore(path=path)

    normalized = store.upsert("  Velocity plot  ", {"layout": "1x1", "plot_states": [{"xcol": "a"}]})
    store.save()
    loaded = PlotPresetStore.load(path=path)

    assert normalized == "Velocity plot"
    assert loaded.names() == ["Velocity plot"]
    assert loaded.get("Velocity plot") == {"layout": "1x1", "plot_states": [{"xcol": "a"}]}


def test_plot_preset_store_skips_malformed_presets(tmp_path):
    path = tmp_path / "nicepoolplots.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "unknown": True,
                "presets": {
                    "good": {"layout": "1x1", "plot_states": []},
                    "bad": ["not", "a", "dict"],
                    " ": {"layout": "1x2"},
                },
            }
        ),
        encoding="utf-8",
    )

    store = PlotPresetStore.load(path=path)

    assert store.names() == ["good"]
    assert store.get("good") == {"layout": "1x1", "plot_states": []}


def test_plot_preset_store_rejects_empty_name(tmp_path):
    store = PlotPresetStore(path=tmp_path / "nicepoolplots.json")

    try:
        store.upsert("  ", {"layout": "1x1"})
    except ValueError as exc:
        assert "cannot be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_plot_preset_store_delete_existing_preset(tmp_path):
    store = PlotPresetStore(path=tmp_path / "nicepoolplots.json")
    store.upsert("plot a", {"layout": "1x1"})
    store.upsert("plot b", {"layout": "1x2"})

    deleted = store.delete(" plot a ")

    assert deleted is True
    assert store.names() == ["plot b"]
    assert store.get("plot a") is None


def test_plot_preset_store_delete_missing_preset_is_noop(tmp_path):
    store = PlotPresetStore(path=tmp_path / "nicepoolplots.json")
    store.upsert("plot a", {"layout": "1x1"})

    deleted = store.delete("missing")

    assert deleted is False
    assert store.names() == ["plot a"]
