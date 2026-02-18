"""Gold standard aggrid for NiceGUI."""

from typing import Callable, Any
from nicegui import ui
import pandas as pd
from nicegui.events import GenericEventArguments

from nicewidgets.utils.logging import get_logger

logger = get_logger(__name__)


def gold_standard_aggrid(df: pd.DataFrame,
                        unique_row_id_col:str = None,
                        row_select_callback:Callable[[dict[str, Any]], None] = None) -> ui.aggrid:

    def _on_row_selected(e: GenericEventArguments) -> None:

        # do not use these
        # print(f"rowIndex: {e.args['rowIndex']}")
        # print(f"rowId: {e.args.get('rowId')}")

        _rowDict = e.args['data']
        _unique_id = None
        if unique_row_id_col is not None:
            _unique_id = _rowDict[unique_row_id_col]
        
        if row_select_callback is not None:
            row_select_callback(_unique_id, _rowDict)
        
    with ui.column().classes("w-full h-full min-h-0"):
        
        aggrid = ui.aggrid.from_pandas(df).classes("w-full aggrid-compact")

        
        # holy shit, this works
        # self._aggrid = ui.aggrid({}).classes("w-full aggrid-compact max-h-40")
        # row_data = self.df.to_dict("records")
        # self._aggrid.options['rowData'] = row_data

        # self._aggrid.style("height: 100%; flex: 1;")


        # keep only these columns
        keep = ['vel_mean', 'file_name']

        aggrid.options['columnDefs'] = [
            {
                'headerName': c,
                'field': c,
                # 'hide': (c not in keep),
                "checkboxSelection": False,
                "headerCheckboxSelection": False,
                "sortable": True,
                "resizable": True,
                # "flex": 1,
                # "minWidth": 100,
            }
            for c in df.columns
        ]

        aggrid.options['rowSelection'] = 'single'
        aggrid.options['suppressRowClickSelection'] = False

        # if row_id_col is not None:
        #     aggrid.options['getRowId'] = {
        #         'function': f'params.data.{row_id_col}'
        #     }

        aggrid.update()

    if row_select_callback is not None:
        # aggrid.on("rowSelected", row_select_callback)
        aggrid.on("rowSelected", _on_row_selected)


    return aggrid