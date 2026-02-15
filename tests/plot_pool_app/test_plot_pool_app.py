"""Unit tests for plot_pool_app module (env helpers, defaults)."""

from __future__ import annotations

import os

import pytest

from nicewidgets.plot_pool_app import plot_pool_app


def test_env_bool_unset_returns_default():
    """_env_bool returns default when env var is unset."""
    # Use a unique var name to avoid side effects
    key = "_TEST_PLOT_POOL_APP_UNSET_XYZ"
    assert key not in os.environ
    assert plot_pool_app._env_bool(key, True) is True
    assert plot_pool_app._env_bool(key, False) is False


def test_env_bool_truthy_values():
    """_env_bool returns True for 1, true, yes, on."""
    key = "_TEST_PLOT_POOL_APP_BOOL_XYZ"
    for val in ("1", "true", "True", "yes", "on"):
        try:
            os.environ[key] = val
            assert plot_pool_app._env_bool(key, False) is True
        finally:
            os.environ.pop(key, None)


def test_env_bool_falsy_values():
    """_env_bool returns False for 0, false, no, off."""
    key = "_TEST_PLOT_POOL_APP_BOOL_XYZ"
    for val in ("0", "false", "False", "no", "off"):
        try:
            os.environ[key] = val
            assert plot_pool_app._env_bool(key, True) is False
        finally:
            os.environ.pop(key, None)


def test_env_bool_invalid_returns_default():
    """_env_bool returns default for invalid value."""
    key = "_TEST_PLOT_POOL_APP_INVALID_XYZ"
    try:
        os.environ[key] = "maybe"
        assert plot_pool_app._env_bool(key, True) is True
        assert plot_pool_app._env_bool(key, False) is False
    finally:
        os.environ.pop(key, None)


def test_env_int_unset_returns_default():
    """_env_int returns default when env var is unset."""
    key = "_TEST_PLOT_POOL_APP_INT_UNSET_XYZ"
    assert key not in os.environ
    assert plot_pool_app._env_int(key, 42) == 42
    assert plot_pool_app._env_int(key, 8080) == 8080


def test_env_int_valid_returns_parsed():
    """_env_int returns parsed int for valid value."""
    key = "_TEST_PLOT_POOL_APP_INT_XYZ"
    try:
        os.environ[key] = "123"
        assert plot_pool_app._env_int(key, 0) == 123
    finally:
        os.environ.pop(key, None)


def test_env_int_invalid_returns_default():
    """_env_int returns default for invalid value."""
    key = "_TEST_PLOT_POOL_APP_INT_INVALID_XYZ"
    try:
        os.environ[key] = "not_a_number"
        assert plot_pool_app._env_int(key, 99) == 99
    finally:
        os.environ.pop(key, None)


def test_storage_secret_defined():
    """STORAGE_SECRET is defined and non-empty."""
    assert hasattr(plot_pool_app, "STORAGE_SECRET")
    assert isinstance(plot_pool_app.STORAGE_SECRET, str)
    assert len(plot_pool_app.STORAGE_SECRET) > 0
