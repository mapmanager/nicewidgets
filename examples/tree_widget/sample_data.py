"""Synthetic hierarchy rows for the TreeWidget demo.

Pure data module: no NiceGUI imports. Rows use a list ``hierarchy_path`` suitable
for :class:`~nicewidgets.tree_widget.tree_widget.TreeWidget` (``path_field``).
"""

from __future__ import annotations

from typing import Any

ROW_ID_FIELD = 'id'
PATH_FIELD = 'hierarchy_path'


def make_demo_rows() -> list[dict[str, Any]]:
    """Return a small file/ROI tree shaped like an acquisition browser.

    Returns:
        Rows with unique string ``id`` values and list ``hierarchy_path`` entries.
    """
    return [
        {
            'id': 'exp-ctrl',
            'name': 'ctrl_2025',
            'kind': 'experiment',
            'note': 'control cohort',
            'hierarchy_path': ['ctrl_2025'],
        },
        {
            'id': 'file-ctrl-0',
            'name': 'ctrl_f00.tif',
            'kind': 'file',
            'note': 'channel 0–1',
            'hierarchy_path': ['ctrl_2025', 'ctrl_f00.tif'],
        },
        {
            'id': 'roi-ctrl-0-1',
            'name': 'ROI 1',
            'kind': 'roi',
            'note': 'accepted',
            'hierarchy_path': ['ctrl_2025', 'ctrl_f00.tif', 'ROI 1'],
        },
        {
            'id': 'roi-ctrl-0-2',
            'name': 'ROI 2',
            'kind': 'roi',
            'note': 'rejected',
            'hierarchy_path': ['ctrl_2025', 'ctrl_f00.tif', 'ROI 2'],
        },
        {
            'id': 'file-ctrl-1',
            'name': 'ctrl_f01.tif',
            'kind': 'file',
            'note': 'channel 0',
            'hierarchy_path': ['ctrl_2025', 'ctrl_f01.tif'],
        },
        {
            'id': 'roi-ctrl-1-1',
            'name': 'ROI 1',
            'kind': 'roi',
            'note': 'accepted',
            'hierarchy_path': ['ctrl_2025', 'ctrl_f01.tif', 'ROI 1'],
        },
        {
            'id': 'exp-drug',
            'name': 'drugA_2025',
            'kind': 'experiment',
            'note': 'drugA cohort',
            'hierarchy_path': ['drugA_2025'],
        },
        {
            'id': 'file-drug-0',
            'name': 'drugA_f00.tif',
            'kind': 'file',
            'note': 'channel 0–1',
            'hierarchy_path': ['drugA_2025', 'drugA_f00.tif'],
        },
        {
            'id': 'roi-drug-0-1',
            'name': 'ROI 1',
            'kind': 'roi',
            'note': 'accepted',
            'hierarchy_path': ['drugA_2025', 'drugA_f00.tif', 'ROI 1'],
        },
    ]
