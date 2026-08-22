import { describe, expect, it } from 'vitest'

import { NicePoolEngine } from '../src/core'
import { edgeDataset } from './fixtures'

describe('NicePoolEngine', () => {
  it('treats setData as a full dataset-dependent reset', () => {
    const engine = new NicePoolEngine()
    engine.setData(edgeDataset)
    engine.setPrimarySelection('b')
    engine.setPlotState({ ...engine.plotState, preFilters: { accept: 'yes' }, plotType: 'swarm', groupColumn: 'condition' })

    engine.setData({ rowIdColumn: 'id', rows: [{ id: 'new', first: 1, second: 2 }] })

    expect(engine.selection).toEqual({ primaryRowId: null, selectedRowIds: [] })
    expect(engine.plotState.plotType).toBe('scatter')
    expect(engine.plotState.preFilters).toEqual({})
    expect(engine.dataset.rowIndexById.has('b')).toBe(false)
  })

  it('rejects selection IDs outside the authoritative dataset', () => {
    const engine = new NicePoolEngine()
    engine.setData(edgeDataset)
    expect(() => engine.setPrimarySelection('missing')).toThrow(RangeError)
  })

  it('extends selection as a stable union for Shift interactions', () => {
    const engine = new NicePoolEngine()
    engine.setData(edgeDataset)
    engine.setSelection({ primaryRowId: 'a', selectedRowIds: ['a', 'b'] })

    engine.extendSelection(['b', 'c'], 'c')

    expect(engine.selection).toEqual({ primaryRowId: 'c', selectedRowIds: ['a', 'b', 'c'] })
  })
})
