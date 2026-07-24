"""Tests for TreeWidget expanded-group tracking."""

from __future__ import annotations

from nicewidgets.aggrid_common.column_def import ColumnDef
from nicewidgets.tree_widget.tree_widget import TreeWidget


def test_expand_group_tracks_expanded_ids() -> None:
    """Programmatic expand should update expanded_group_ids."""
    tree = TreeWidget(
        columns=[ColumnDef(field='name', headerName='Name')],
        row_id_field='id',
        rows=[{'id': 'file-a', 'name': 'A', 'hierarchy_path': ['file-a']}],
        path_field='hierarchy_path',
    )
    assert tree.expanded_group_ids() == frozenset()
    tree.expand_group('file-a')
    assert tree.expanded_group_ids() == frozenset({'file-a'})
    tree.collapse_all_nodes()
    assert tree.expanded_group_ids() == frozenset()
