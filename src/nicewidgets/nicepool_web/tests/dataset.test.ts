import { describe, expect, it } from 'vitest'

import { DatasetStore, DatasetValidationError } from '../src/core'
import { edgeDataset } from './fixtures'

describe('DatasetStore', () => {
  it('builds stable identity and schema indexes', () => {
    const dataset = new DatasetStore(edgeDataset)
    expect(dataset.rowIndexById.get('c')).toBe(2)
    expect(dataset.numericColumns()).toEqual(['x', 'y'])
    expect(dataset.preFilterColumns()).toEqual(['accept'])
  })

  it('rejects duplicate IDs instead of silently choosing a row', () => {
    expect(() => new DatasetStore({
      rowIdColumn: 'id',
      rows: [{ id: 'same' }, { id: 'same' }],
    })).toThrow(DatasetValidationError)
  })

  it('retains null but rejects implicit or non-JSON missing values', () => {
    const dataset = new DatasetStore({ rowIdColumn: 'id', rows: [{ id: 'a', value: null }] })
    expect(dataset.rows[0]?.value).toBeNull()
    expect(() => new DatasetStore({ rowIdColumn: 'id', rows: [{ id: 'a', value: 1 }, { id: 'b' } as never] })).toThrow(DatasetValidationError)
    expect(() => new DatasetStore({ rowIdColumn: 'id', rows: [{ id: 'a', value: Number.NaN }] })).toThrow(DatasetValidationError)
  })
})
