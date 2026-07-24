"""Named plot-preset persistence for NicePool.

This module is the backend persistence layer for reusable NicePool plot
presets. It owns JSON file I/O and keeps an in-memory dictionary of named
preset payloads. NiceGUI controls should use controller methods that delegate to
this store rather than reading or writing files directly.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

from nicewidgets.utils.logging import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION: int = 1
DEFAULT_PRESET_FILENAME: str = "nicepoolplots.json"


class PlotPresetStore:
    """In-memory NicePool plot-preset store backed by one JSON file.

    The store is intentionally tolerant when loading user JSON. Unknown root
    keys are ignored, malformed preset entries are skipped, and preset payloads
    remain raw dictionaries until the controller validates them against the
    current runtime DataFrame.

    Args:
        path: JSON file used for persistence.
        presets: Optional initial preset mapping used by tests or host apps.
    """

    def __init__(
        self,
        *,
        path: Path,
        presets: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.path = path
        self._presets: dict[str, dict[str, Any]] = {}
        if presets:
            for name, preset in presets.items():
                normalized = self.normalize_name(name)
                if normalized and isinstance(preset, dict):
                    self._presets[normalized] = deepcopy(preset)

    @staticmethod
    def default_config_path(
        *,
        app_name: str = "nicewidgets",
        filename: str = DEFAULT_PRESET_FILENAME,
        app_author: str | None = None,
    ) -> Path:
        """Return the OS-specific NicePool preset file path.

        Args:
            app_name: Platformdirs application name.
            filename: JSON filename to use inside the config directory.
            app_author: Optional platformdirs application author.

        Returns:
            Path to the preset JSON file. The parent directory is created.
        """
        directory = Path(user_config_dir(app_name, app_author))
        directory.mkdir(parents=True, exist_ok=True)
        return directory / filename

    @classmethod
    def load(
        cls,
        *,
        path: Path | None = None,
        app_name: str = "nicewidgets",
        filename: str = DEFAULT_PRESET_FILENAME,
        app_author: str | None = None,
    ) -> PlotPresetStore:
        """Load a preset store from disk.

        Args:
            path: Optional explicit JSON path. When omitted, platformdirs is
                used with ``app_name`` and ``filename``.
            app_name: Platformdirs application name.
            filename: JSON filename used when ``path`` is omitted.
            app_author: Optional platformdirs application author.

        Returns:
            A store populated with valid named preset dictionaries. Missing or
            malformed files produce an empty store.
        """
        preset_path = path or cls.default_config_path(
            app_name=app_name,
            filename=filename,
            app_author=app_author,
        )
        store = cls(path=preset_path)
        try:
            parsed = json.loads(preset_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.debug("NicePool preset file not found at %s; using empty store", preset_path)
            return store
        except json.JSONDecodeError as exc:
            logger.warning("NicePool preset file at %s is not valid JSON: %s", preset_path, exc)
            return store
        except Exception as exc:
            logger.warning("Error loading NicePool presets from %s: %s", preset_path, exc)
            return store

        if not isinstance(parsed, dict):
            logger.warning("NicePool preset file at %s does not contain a JSON object", preset_path)
            return store

        raw_presets = parsed.get("presets", {})
        if not isinstance(raw_presets, dict):
            logger.warning("NicePool preset file at %s has non-dict 'presets'; using empty store", preset_path)
            return store

        for key in parsed:
            if key not in {"schema_version", "presets"}:
                logger.warning("Unknown key %r in NicePool preset file, ignoring", key)

        for name, preset in raw_presets.items():
            normalized = cls.normalize_name(name)
            if not normalized:
                logger.warning("Skipping NicePool preset with empty name")
                continue
            if not isinstance(preset, dict):
                logger.warning("Skipping NicePool preset %r because it is not a dict", normalized)
                continue
            store._presets[normalized] = deepcopy(preset)
        return store

    @staticmethod
    def normalize_name(name: object) -> str:
        """Return a normalized preset name.

        Args:
            name: User-provided preset name.

        Returns:
            Stripped string. Empty strings are invalid preset names.
        """
        return str(name).strip()

    def save(self) -> None:
        """Write all in-memory presets to disk.

        Raises:
            OSError: If the file cannot be written.
        """
        payload = {
            "schema_version": SCHEMA_VERSION,
            "presets": deepcopy(self._presets),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved NicePool plot presets to %s", self.path)

    def names(self) -> list[str]:
        """Return preset names sorted for display in the GUI."""
        return sorted(self._presets)

    def get(self, name: str) -> dict[str, Any] | None:
        """Return a copy of a named preset payload.

        Args:
            name: Preset name.

        Returns:
            Preset dictionary copy, or ``None`` when missing.
        """
        normalized = self.normalize_name(name)
        preset = self._presets.get(normalized)
        return deepcopy(preset) if preset is not None else None

    def upsert(self, name: str, preset: dict[str, Any]) -> str:
        """Insert or replace a named preset in memory.

        Args:
            name: User-facing preset name.
            preset: JSON-friendly preset dictionary.

        Returns:
            The normalized preset name.

        Raises:
            ValueError: If ``name`` is empty or ``preset`` is not a dictionary.
        """
        normalized = self.normalize_name(name)
        if not normalized:
            raise ValueError("NicePool plot preset name cannot be empty")
        if not isinstance(preset, dict):
            raise ValueError("NicePool plot preset payload must be a dictionary")
        self._presets[normalized] = deepcopy(preset)
        return normalized

    def delete(self, name: str) -> bool:
        """Delete a named preset from memory.

        Args:
            name: Preset name.

        Returns:
            True when a preset was removed.
        """
        normalized = self.normalize_name(name)
        return self._presets.pop(normalized, None) is not None
