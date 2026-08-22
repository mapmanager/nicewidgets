import type { DatasetInput } from '../src/core/types'

export const edgeDataset: DatasetInput = {
  rowIdColumn: 'pool_row_id',
  rows: [
    { pool_row_id: 'a', accept: 'yes', condition: 'control', cohort: 'one', x: -2, y: 1 },
    { pool_row_id: 'b', accept: 'yes', condition: 'control', cohort: 'two', x: -1, y: 3 },
    { pool_row_id: 'c', accept: 'no', condition: 'treated', cohort: 'one', x: 1, y: 5 },
    { pool_row_id: 'd', accept: 'yes', condition: 'treated', cohort: 'two', x: 2, y: null },
    { pool_row_id: 'e', accept: 'yes', condition: 'treated', cohort: 'one', x: 3, y: 9 },
  ],
}
