import { describe, expect, it } from 'vitest'

import { DatasetStore, defaultPlotState } from '../src/core'
import { prepareDistribution, prepareHistogram, prepareScatter, prepareSwarm } from '../src/plots/prepare'
import { buildPlotlySpecification, rowIdsFromPlotlyEvent } from '../src/plots/plotly'
import { summarizePlot } from '../src/plots/summary'
import { edgeDataset } from './fixtures'

describe('plot preparation and summaries', () => {
  it('summarizes the exact filtered and transformed scatter points', () => {
    const dataset = new DatasetStore(edgeDataset)
    const state = {
      ...defaultPlotState(dataset),
      xColumn: 'x',
      yColumn: 'y',
      preFilters: { accept: 'yes' },
      groupColumn: 'condition',
      useAbsoluteValue: true,
    }
    const data = prepareScatter(dataset, state)
    const summary = summarizePlot(data)

    expect(data.points.map(({ rowId }) => rowId)).toEqual(['a', 'b', 'e'])
    expect(data.points.map(({ x }) => x)).toEqual([2, 1, 3])
    expect(summary.representedRows.map(({ rowId }) => rowId)).toEqual(['a', 'b', 'e'])
    expect(summary.aggregateRows).toHaveLength(1)
  })

  it('supports categorical scatter X values with optional color grouping', () => {
    const dataset = new DatasetStore(edgeDataset)
    const baseState = {
      ...defaultPlotState(dataset),
      xColumn: 'condition',
      yColumn: 'y',
    }
    const ungrouped = prepareScatter(dataset, baseState)
    expect(ungrouped.points.map(({ x }) => x)).toEqual([
      'control', 'control', 'treated', 'treated',
    ])
    expect(summarizePlot(ungrouped).aggregateRows).toHaveLength(1)

    const colored = prepareScatter(dataset, { ...baseState, colorColumn: 'accept' })
    expect(new Set(colored.points.map(({ colorValue }) => colorValue))).toEqual(new Set(['yes', 'no']))
    expect(summarizePlot(colored).aggregateRows).toHaveLength(2)
  })

  it('retains missing rows in the dataset but omits them from required projections', () => {
    const dataset = new DatasetStore({
      rowIdColumn: 'id',
      rows: [
        { id: 'complete', x: 1, y: 2, group: 'a', color: 'red' },
        { id: 'missing-y', x: 2, y: null, group: 'a', color: 'red' },
        { id: 'missing-color', x: 3, y: 4, group: 'a', color: null },
      ],
    })
    expect(dataset.rows).toHaveLength(3)
    const state = { ...defaultPlotState(dataset), xColumn: 'x', yColumn: 'y', colorColumn: 'color' }
    expect(prepareScatter(dataset, state).points.map(({ rowId }) => rowId)).toEqual(['complete'])
  })

  it('produces stable swarm coordinates and summaries', () => {
    const dataset = new DatasetStore(edgeDataset)
    const state = {
      ...defaultPlotState(dataset),
      plotType: 'swarm' as const,
      yColumn: 'y',
      groupColumn: 'condition',
      colorColumn: 'cohort',
    }
    const first = prepareSwarm(dataset, state)
    const second = prepareSwarm(dataset, state)

    expect(first.categories).toEqual(['control', 'treated'])
    expect(first.points.map(({ x }) => x)).toEqual(second.points.map(({ x }) => x))
    expect(new Set(first.points.map(({ rowId }) => rowId)).size).toBe(first.points.length)
    expect(summarizePlot(first).representedRows).toHaveLength(4)
  })

  it('shares distribution preparation while adding box quartiles', () => {
    const dataset = new DatasetStore(edgeDataset)
    const state = {
      ...defaultPlotState(dataset), plotType: 'box' as const,
      groupColumn: 'condition', colorColumn: null, yColumn: 'y',
    }
    const data = prepareDistribution(dataset, state)
    const summary = summarizePlot(data)
    expect(data.categories).toEqual(['control', 'treated'])
    expect(summary.aggregateRows[0]?.statistics).toMatchObject({ q1: 1.5, q3: 2.5, iqr: 1 })
    expect(buildPlotlySpecification(data, { primaryRowId: null, selectedRowIds: [] }).traces[0]).toMatchObject({ type: 'box', pointpos: 0 })
    const violin = prepareDistribution(dataset, { ...state, plotType: 'violin' })
    expect(buildPlotlySpecification(violin, { primaryRowId: null, selectedRowIds: [] }).traces[0]).toMatchObject({ type: 'violin', pointpos: 0 })
  })

  it('uses shared histogram edges and deterministic cumulative proportions', () => {
    const dataset = new DatasetStore(edgeDataset)
    const histogramState = {
      ...defaultPlotState(dataset), plotType: 'histogram' as const,
      xColumn: 'x', groupColumn: 'condition', histogramBins: 2,
    }
    const histogram = prepareHistogram(dataset, histogramState)
    const grouped = new Map<string | null, typeof histogram.bins>()
    for (const group of ['control', 'treated']) grouped.set(group, histogram.bins.filter((bin) => bin.groupValue === group))
    expect(grouped.get('control')?.map(({ lower, upper }) => [lower, upper])).toEqual(grouped.get('treated')?.map(({ lower, upper }) => [lower, upper]))
    expect(histogram.bins.reduce((total, bin) => total + bin.count, 0)).toBe(histogram.points.length)
    expect(summarizePlot(histogram).bins).toHaveLength(4)

    const cumulative = prepareHistogram(dataset, { ...histogramState, plotType: 'cumulativeHistogram' })
    expect(cumulative.bins.filter((bin) => bin.groupValue === 'control').at(-1)?.cumulativeProportion).toBe(1)
    expect(buildPlotlySpecification(cumulative, { primaryRowId: null, selectedRowIds: [] }).traces[0]).toMatchObject({ type: 'scatter', mode: 'lines' })
  })

  it('decodes stable row identity from Plotly selection events', () => {
    expect(rowIdsFromPlotlyEvent({
      points: [{ customdata: ['a'] }, { customdata: ['b'] }, { customdata: ['a'] }],
    })).toEqual(['a', 'b'])
  })

  it('changes Plotly selection revision when shared selection changes', () => {
    const dataset = new DatasetStore(edgeDataset)
    const data = prepareScatter(dataset, { ...defaultPlotState(dataset), xColumn: 'x', yColumn: 'y' })
    const multiple = buildPlotlySpecification(data, { primaryRowId: 'a', selectedRowIds: ['a', 'b'] })
    const single = buildPlotlySpecification(data, { primaryRowId: 'c', selectedRowIds: ['c'] })
    const cleared = buildPlotlySpecification(data, { primaryRowId: null, selectedRowIds: [] })
    expect(multiple.layout.selectionrevision).toBe('["a","b"]')
    expect(single.layout.selectionrevision).toBe('["c"]')
    expect(cleared.layout.selectionrevision).toBe('[]')
  })

  it('maps display options and theme only at the Plotly boundary', () => {
    const dataset = new DatasetStore(edgeDataset)
    const state = {
      ...defaultPlotState(dataset),
      xColumn: 'x',
      yColumn: 'y',
      legendPosition: 'left' as const,
      showPlotlyToolbar: false,
      showHover: true,
      showAxes: false,
      showHorizontalGrid: false,
      showVerticalGrid: true,
    }
    const data = prepareScatter(dataset, state)
    const specification = buildPlotlySpecification(data, { primaryRowId: null, selectedRowIds: [] }, 'dark')
    expect(specification.config.displayModeBar).toBe(false)
    expect(specification.layout.paper_bgcolor).toBe('#111827')
    expect(specification.layout.hovermode).toBe('closest')
    expect(specification.layout.legend).toMatchObject({ xanchor: 'right', orientation: 'v' })
    expect(specification.layout.xaxis).toMatchObject({ showline: false, showticklabels: false, showgrid: true })
    expect(specification.layout.yaxis).toMatchObject({ showline: false, showticklabels: false, showgrid: false })
  })
})
