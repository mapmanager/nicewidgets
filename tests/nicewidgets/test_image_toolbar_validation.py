"""Unit tests for ``image_toolbar_widget`` validation rules (no NiceGUI client)."""

from __future__ import annotations

import pytest

from nicewidgets.image_toolbar_widget.validation import (
    validate_options_update_preserves_int_selection,
    validate_options_update_preserves_selection,
    validate_scalar_int_in_options,
    validate_scalar_in_options,
    validate_set_file_ext_args,
    validate_set_roi_options_and_selection_ext,
)


class TestValidateSetFileExtArgs:
    def test_ok_empty_both(self) -> None:
        validate_set_file_ext_args(None, None, [], [])

    def test_ok_nonempty_both(self) -> None:
        validate_set_file_ext_args(0, 1, ['0', '1'], [1, 2])

    def test_roi_nonempty_requires_roi_id(self) -> None:
        with pytest.raises(ValueError, match='roi_id is required'):
            validate_set_file_ext_args(0, None, ['0'], [1])

    def test_roi_nonempty_rejects_unknown_roi(self) -> None:
        with pytest.raises(ValueError, match='not in roi_options'):
            validate_set_file_ext_args(0, 99, ['0'], [1, 2])

    def test_roi_empty_requires_none_roi(self) -> None:
        with pytest.raises(ValueError, match='roi_id must be None'):
            validate_set_file_ext_args(0, 1, ['0'], [])

    def test_channel_empty_requires_none_channel(self) -> None:
        with pytest.raises(ValueError, match='channel must be None'):
            validate_set_file_ext_args(0, None, [], [])

    def test_channel_nonempty_requires_channel(self) -> None:
        with pytest.raises(ValueError, match='channel is required'):
            validate_set_file_ext_args(None, None, ['0'], [])

    def test_channel_must_match_options(self) -> None:
        with pytest.raises(ValueError, match='not in channel_options'):
            validate_set_file_ext_args(9, None, ['0', '1'], [])


class TestValidateSetRoiOptionsAndSelectionExt:
    def test_ok_empty(self) -> None:
        validate_set_roi_options_and_selection_ext([], None)

    def test_ok_nonempty(self) -> None:
        validate_set_roi_options_and_selection_ext([1, 2], 1)

    def test_nonempty_requires_roi(self) -> None:
        with pytest.raises(ValueError, match='roi_id is required'):
            validate_set_roi_options_and_selection_ext([1], None)

    def test_empty_rejects_roi(self) -> None:
        with pytest.raises(ValueError, match='roi_id must be None'):
            validate_set_roi_options_and_selection_ext([], 1)


class TestValidateScalarInOptions:
    def test_empty_only_none(self) -> None:
        validate_scalar_in_options(None, [], field='x')
        with pytest.raises(ValueError, match='must be None'):
            validate_scalar_in_options('a', [], field='x')

    def test_nonempty_requires_member(self) -> None:
        with pytest.raises(ValueError, match='required'):
            validate_scalar_in_options(None, ['a'], field='x')
        with pytest.raises(ValueError, match='not in'):
            validate_scalar_in_options('b', ['a'], field='x')
        validate_scalar_in_options('a', ['a'], field='x')


class TestValidateScalarIntInOptions:
    def test_empty_only_none(self) -> None:
        validate_scalar_int_in_options(None, [], field='roi_id')
        with pytest.raises(ValueError, match='must be None'):
            validate_scalar_int_in_options(1, [], field='roi_id')

    def test_nonempty_requires_member(self) -> None:
        with pytest.raises(ValueError, match='required'):
            validate_scalar_int_in_options(None, [1], field='roi_id')
        with pytest.raises(ValueError, match='not in'):
            validate_scalar_int_in_options(2, [1], field='roi_id')
        validate_scalar_int_in_options(1, [1], field='roi_id')


class TestValidateOptionsUpdatePreservesSelection:
    def test_none_current_always_ok(self) -> None:
        validate_options_update_preserves_selection(field='channel', current_value_str=None, new_options=['1', '2'])

    def test_current_must_remain_in_new(self) -> None:
        validate_options_update_preserves_selection(field='channel', current_value_str='1', new_options=['1', '2'])
        with pytest.raises(ValueError, match='not in new'):
            validate_options_update_preserves_selection(field='channel', current_value_str='3', new_options=['1', '2'])


class TestValidateOptionsUpdatePreservesIntSelection:
    def test_none_current_always_ok(self) -> None:
        validate_options_update_preserves_int_selection(field='roi', current_value=None, new_options=[1, 2])

    def test_current_must_remain_in_new(self) -> None:
        validate_options_update_preserves_int_selection(field='roi', current_value=1, new_options=[1, 2])
        with pytest.raises(ValueError, match='not in new'):
            validate_options_update_preserves_int_selection(field='roi', current_value=3, new_options=[1, 2])


class TestAsIntListRejectsBool:
    def test_bool_in_roi_options_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match='int'):
            validate_set_file_ext_args(0, 1, ['0'], [True])  # type: ignore[list-item]
