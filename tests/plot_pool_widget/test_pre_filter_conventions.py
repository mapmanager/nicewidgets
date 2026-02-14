"""Unit tests for pre_filter_conventions (sentinel, default_pre_filter, is_filtered, format_pre_filter_display)."""

import pytest

from nicewidgets.plot_pool_widget.pre_filter_conventions import (
    PRE_FILTER_NONE,
    default_pre_filter,
    is_filtered,
    format_pre_filter_display,
)


def test_pre_filter_none_constant():
    """PRE_FILTER_NONE is the sentinel string."""
    assert PRE_FILTER_NONE == "(none)"


def test_default_pre_filter_all_columns_none():
    """default_pre_filter returns dict mapping each column to PRE_FILTER_NONE."""
    result = default_pre_filter(["roi_id", "condition"])
    assert result == {"roi_id": PRE_FILTER_NONE, "condition": PRE_FILTER_NONE}


def test_default_pre_filter_empty_list():
    """default_pre_filter([]) returns empty dict."""
    assert default_pre_filter([]) == {}


def test_is_filtered_all_none_is_false():
    """is_filtered is False when all values are PRE_FILTER_NONE."""
    assert is_filtered({"roi_id": PRE_FILTER_NONE}) is False
    assert is_filtered({"roi_id": PRE_FILTER_NONE, "condition": PRE_FILTER_NONE}) is False


def test_is_filtered_any_value_not_none_is_true():
    """is_filtered is True when any value is not PRE_FILTER_NONE."""
    assert is_filtered({"roi_id": 1}) is True
    assert is_filtered({"roi_id": PRE_FILTER_NONE, "condition": "A"}) is True


def test_format_pre_filter_display_all_returns_all():
    """format_pre_filter_display with no active filter returns 'All'."""
    assert format_pre_filter_display({}) == "All"
    assert format_pre_filter_display({"roi_id": PRE_FILTER_NONE}) == "All"


def test_format_pre_filter_display_active_returns_key_equals_value():
    """format_pre_filter_display with active filters returns comma-separated k=v."""
    assert format_pre_filter_display({"roi_id": 1}) == "roi_id=1"
    assert format_pre_filter_display({"roi_id": 1, "condition": "A"}) == "roi_id=1, condition=A"
