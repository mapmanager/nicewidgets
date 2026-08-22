import type * as Plotly from 'plotly.js'

import type { LegendPosition, NicePoolSelection, NicePoolTheme, RowId } from '../core/types'
import type { PreparedDistributionData, PreparedHistogramData, PreparedPlotData, PreparedPoint, PreparedSwarmData } from './types'

export interface PlotlySpecification {
  traces: Plotly.Data[]
  layout: Partial<Plotly.Layout>
  config: Partial<Plotly.Config>
}

function groupedPoints(points: readonly PreparedPoint[]): readonly [string, readonly PreparedPoint[]][] {
  const groups = new Map<string, PreparedPoint[]>()
  for (const point of points) {
    const label = point.colorValue ?? point.groupValue ?? 'data'
    const group = groups.get(label) ?? []
    group.push(point)
    groups.set(label, group)
  }
  return [...groups.entries()]
}

function selectedPointIndices(points: readonly PreparedPoint[], selection: NicePoolSelection): number[] {
  const selected = new Set(selection.selectedRowIds)
  return points.flatMap((point, index) => selected.has(point.rowId) ? [index] : [])
}

function pointTrace(
  name: string,
  points: readonly PreparedPoint[],
  selection: NicePoolSelection,
  pointSize: number,
): Plotly.Data {
  // Plotly supports selected/unselected marker styles for scatter traces, but
  // @types/plotly.js does not currently expose both properties on Data.
  return {
    type: 'scattergl',
    mode: 'markers',
    name,
    x: points.map(({ x }) => x),
    y: points.map(({ y }) => y),
    customdata: points.map(({ rowId }) => [rowId]),
    selectedpoints: selectedPointIndices(points, selection),
    marker: { size: pointSize },
    selected: { marker: { size: pointSize + 4, color: '#f97316' } },
    unselected: { marker: { opacity: selection.selectedRowIds.length ? 0.28 : 0.9 } },
    hovertemplate: 'row=%{customdata[0]}<br>x=%{x}<br>y=%{y}<extra>%{fullData.name}</extra>',
  } as unknown as Plotly.Data
}

function swarmOverlays(data: PreparedSwarmData, theme: NicePoolTheme): Plotly.Data[] {
  if (!data.state.showMean && !data.state.showErrorBars) return []
  const groups = new Map<string, { centerX: number; values: number[] }>()
  for (const point of data.points) {
    const key = JSON.stringify([point.category, point.colorValue])
    const group = groups.get(key) ?? { centerX: point.centerX, values: [] }
    group.values.push(point.y)
    groups.set(key, group)
  }
  const x: number[] = []
  const y: number[] = []
  const error: number[] = []
  for (const { centerX, values } of groups.values()) {
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length
    const variance = values.length > 1
      ? values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1)
      : 0
    const std = Math.sqrt(variance)
    x.push(centerX)
    y.push(mean)
    error.push(data.state.errorBarType === 'sem' ? std / Math.sqrt(values.length) : std)
  }
  const trace: Plotly.Data = {
    type: 'scatter',
    mode: data.state.showMean || data.state.showErrorBars ? 'markers' : 'none',
    name: `mean ± ${data.state.errorBarType}`,
    x,
    y,
    marker: data.state.showMean
      ? { symbol: 'line-ew', size: 18, color: theme === 'dark' ? '#f8fafc' : '#111827', line: { width: 3 } }
      : { size: 1, color: 'rgba(0,0,0,0)' },
    hoverinfo: 'skip',
    showlegend: false,
    ...(data.state.showErrorBars
      ? { error_y: { type: 'data' as const, array: error, visible: true } }
      : {}),
  }
  return [trace]
}

function distributionTraces(data: PreparedDistributionData): Plotly.Data[] {
  return groupedPoints(data.points).map(([name, points]) => ({
    type: data.type,
    name,
    x: points.map(({ x }) => x),
    y: points.map(({ y }) => y),
    customdata: points.map(({ rowId }) => [rowId]),
    showlegend: data.state.showLegend,
    ...(data.type === 'box'
      ? { boxpoints: data.state.showRaw ? 'all' : false, jitter: 0.3, pointpos: 0 }
      : { points: data.state.showRaw ? 'all' : false, pointpos: 0, jitter: 0.3, box: { visible: true }, meanline: { visible: true } }),
    hovertemplate: 'row=%{customdata[0]}<br>group=%{x}<br>y=%{y}<extra>%{fullData.name}</extra>',
  } as unknown as Plotly.Data))
}

function histogramTraces(data: PreparedHistogramData): Plotly.Data[] {
  const grouped = new Map<string, { name: string; bins: typeof data.bins }>()
  for (const bin of data.bins) {
    const key = JSON.stringify([bin.groupValue, bin.colorValue])
    const name = [bin.groupValue, bin.colorValue].filter((value) => value !== null).join(' / ') || 'data'
    const entry = grouped.get(key) ?? { name, bins: [] }
    grouped.set(key, { name, bins: [...entry.bins, bin] })
  }
  return [...grouped.values()].map(({ name, bins }) => data.type === 'histogram'
    ? ({
        type: 'bar', name,
        x: bins.map(({ center }) => center),
        y: bins.map(({ count }) => count),
        width: bins.map(({ lower, upper }) => upper - lower),
        opacity: grouped.size > 1 ? 0.65 : 0.9,
        hovertemplate: 'bin=%{x}<br>count=%{y}<extra>%{fullData.name}</extra>',
      } as Plotly.Data)
    : ({
        type: 'scatter', mode: 'lines', name,
        x: [bins[0]!.lower, ...bins.map(({ upper }) => upper)],
        y: [0, ...bins.map(({ cumulativeProportion }) => cumulativeProportion)],
        line: { shape: 'hv' },
        hovertemplate: 'x=%{x}<br>cumulative=%{y:.3f}<extra>%{fullData.name}</extra>',
      } as Plotly.Data))
}

function legendLayout(position: LegendPosition): Partial<Plotly.Legend> {
  if (position === 'bottom') return { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.2, yanchor: 'top' }
  if (position === 'top') return { orientation: 'h', x: 0.5, xanchor: 'center', y: 1.08, yanchor: 'bottom' }
  if (position === 'left') return { orientation: 'v', x: -0.03, xanchor: 'right', y: 1, yanchor: 'top' }
  return { orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top' }
}

/** Convert prepared plot data into Plotly-only trace and layout objects. */
export function buildPlotlySpecification(
  data: PreparedPlotData,
  selection: NicePoolSelection,
  theme: NicePoolTheme = 'dark',
): PlotlySpecification {
  let traces: Plotly.Data[]
  if (data.type === 'box' || data.type === 'violin') traces = distributionTraces(data)
  else if (data.type === 'histogram' || data.type === 'cumulativeHistogram') traces = histogramTraces(data)
  else {
    traces = data.state.showRaw
      ? groupedPoints(data.points).map(([name, points]) => pointTrace(name, points, selection, data.state.pointSize))
      : []
    if (data.type === 'swarm') traces.push(...swarmOverlays(data, theme))
  }
  const dark = theme === 'dark'
  const axisStyle: Partial<Plotly.LayoutAxis> = {
    color: dark ? '#d1d5db' : '#374151',
    showline: data.state.showAxes,
    ticks: data.state.showAxes ? 'outside' : '',
    showticklabels: data.state.showAxes,
    zeroline: data.state.showAxes,
    linecolor: dark ? '#9ca3af' : '#6b7280',
    gridcolor: dark ? '#374151' : '#e5e7eb',
    zerolinecolor: dark ? '#6b7280' : '#9ca3af',
  }
  const layout: Partial<Plotly.Layout> = {
    autosize: true,
    margin: {
      l: data.state.legendPosition === 'left' && data.state.showLegend ? 110 : 58,
      r: data.state.legendPosition === 'right' && data.state.showLegend ? 110 : 20,
      t: data.state.legendPosition === 'top' && data.state.showLegend ? 58 : 24,
      b: data.state.legendPosition === 'bottom' && data.state.showLegend ? 90 : 64,
    },
    paper_bgcolor: dark ? '#111827' : '#ffffff',
    plot_bgcolor: dark ? '#111827' : '#ffffff',
    font: { color: dark ? '#e5e7eb' : '#172033' },
    dragmode: data.type === 'scatter' || data.type === 'swarm' ? 'lasso' : 'zoom',
    selectionrevision: JSON.stringify(selection.selectedRowIds),
    showlegend: data.state.showLegend,
    hovermode: data.state.showHover ? 'closest' : false,
    legend: legendLayout(data.state.legendPosition),
    uirevision: 'nicepool',
    xaxis: data.type === 'swarm'
      ? {
          ...axisStyle,
          title: { text: data.state.showAxes ? data.state.groupColumn ?? '' : '' },
          showgrid: data.state.showVerticalGrid,
          tickmode: 'array',
          tickvals: data.categories.map((_, index) => index),
          ticktext: [...data.categories],
          tickangle: -25,
        }
      : {
          ...axisStyle,
          title: { text: data.state.showAxes ? (data.type === 'box' || data.type === 'violin' ? data.state.groupColumn ?? '' : data.state.xColumn) : '' },
          showgrid: data.state.showVerticalGrid,
        },
    yaxis: {
      ...axisStyle,
      title: { text: data.state.showAxes ? (data.type === 'histogram' ? 'Count' : data.type === 'cumulativeHistogram' ? 'Cumulative proportion' : data.state.yColumn) : '' },
      showgrid: data.state.showHorizontalGrid,
    },
    ...(data.type === 'histogram' ? { barmode: 'overlay' as const } : {}),
  }
  return {
    traces,
    layout,
    config: { responsive: true, displaylogo: false, displayModeBar: data.state.showPlotlyToolbar },
  }
}

interface PlotlySelectedPoint {
  customdata?: unknown
}

interface PlotlySelectionEvent {
  points?: PlotlySelectedPoint[]
}

function rowIdFromCustomData(value: unknown): RowId | null {
  if (Array.isArray(value) && value[0] != null) return String(value[0])
  return value == null ? null : String(value)
}

/** Decode stable row IDs from Plotly click or area-selection event data. */
export function rowIdsFromPlotlyEvent(event: PlotlySelectionEvent | null | undefined): RowId[] {
  const ids = (event?.points ?? []).flatMap(({ customdata }) => {
    const rowId = rowIdFromCustomData(customdata)
    return rowId === null ? [] : [rowId]
  })
  return [...new Set(ids)]
}
