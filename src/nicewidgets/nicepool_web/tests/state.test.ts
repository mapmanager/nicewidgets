import { describe, expect, it } from 'vitest'

import nicePoolStateSchema from '../schemas/nicepool-state.schema.json'
import plotPresetSchema from '../schemas/plot-preset.schema.json'
import plotStateSchema from '../schemas/plot-state.schema.json'

import {
  DatasetStore,
  NicePoolEngine,
  StateValidationError,
  defaultNicePoolState,
  validateNicePoolState,
  validatePlotPreset,
  visiblePlotCount,
} from '../src/core'
import { sampleDataset } from '../src/app/sampleData'
import { edgeDataset } from './fixtures'

describe('versioned state contracts', () => {
  it('creates four independent plot defaults from the dataset schema', () => {
    const dataset = new DatasetStore(edgeDataset)
    const state = defaultNicePoolState(dataset)
    expect(state).toMatchObject({ schemaVersion: 1, layout: '1x1', activePlotIndex: 0 })
    expect(state.plots[0]).toMatchObject({
      showLegend: true,
      legendPosition: 'bottom',
      showPlotlyToolbar: true,
      showHover: false,
      showAxes: true,
      showHorizontalGrid: true,
      showVerticalGrid: true,
      histogramBins: 50,
    })
    expect(state.plots).toHaveLength(4)
    state.plots[0].pointSize = 12
    expect(state.plots[1].pointSize).toBe(7)
  })

  it('rejects plot states that omit current required fields', () => {
    const dataset = new DatasetStore(edgeDataset)
    const state = defaultNicePoolState(dataset)
    const legacyPlot = { ...state.plots[0] } as Partial<typeof state.plots[0]>
    delete legacyPlot.histogramBins
    expect(() => validateNicePoolState(dataset, { ...state, plots: [legacyPlot, ...state.plots.slice(1)] } as never)).toThrow(StateValidationError)
  })

  it('rejects unsupported display options', () => {
    const dataset = new DatasetStore(edgeDataset)
    const state = defaultNicePoolState(dataset)
    const invalidPlot = { ...state.plots[0], legendPosition: 'center' }
    const invalidState = { ...state, plots: [invalidPlot, ...state.plots.slice(1)] }
    expect(() => validateNicePoolState(dataset, invalidState as never)).toThrow(StateValidationError)
  })

  it('validates layout visibility and rejects unknown fields transactionally', () => {
    const dataset = new DatasetStore(edgeDataset)
    const valid = defaultNicePoolState(dataset)
    expect(visiblePlotCount('2x2')).toBe(4)
    expect(() => validateNicePoolState(dataset, { ...valid, layout: '1x1', activePlotIndex: 1 })).toThrow(StateValidationError)
    expect(() => validateNicePoolState(dataset, { ...valid, extra: true } as never)).toThrow(StateValidationError)
  })

  it('leaves the current workspace unchanged when replacement state is invalid', () => {
    const engine = new NicePoolEngine()
    engine.setData(edgeDataset)
    const before = structuredClone(engine.state)
    expect(() => engine.setState({ ...before, activePlotIndex: 3 })).toThrow(StateValidationError)
    expect(engine.state).toEqual(before)
  })

  it('applies a preset only to the requested plot', () => {
    const engine = new NicePoolEngine()
    engine.setData(edgeDataset)
    engine.setLayout('1x2')
    const preset = validatePlotPreset(engine.dataset, {
      schemaVersion: 1,
      name: 'large points',
      plotState: { ...engine.plotState, pointSize: 14 },
    })
    engine.applyPlotPreset(preset, 1)
    expect(engine.state.plots[0].pointSize).toBe(7)
    expect(engine.state.plots[1].pointSize).toBe(14)
  })

  it('ships readable version-one JSON schemas', () => {
    for (const schema of [plotStateSchema, plotPresetSchema, nicePoolStateSchema]) {
      expect(schema.$schema).toBe('https://json-schema.org/draft/2020-12/schema')
      expect(schema.additionalProperties).toBe(false)
    }
  })

  it('generates negative values, outliers, and explicit missing values', () => {
    const velocities = sampleDataset(600).rows.map((row) => row.velocity)
    const numeric = velocities.filter((value): value is number => typeof value === 'number')
    expect(numeric.some((value) => value < 0)).toBe(true)
    expect(numeric.some((value) => Math.abs(value) > 20)).toBe(true)
    expect(velocities).toContain(null)
  })
})
