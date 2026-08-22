import type { DescriptiveStatistics, QuartileStatistics } from '../core/statistics'
import type { NicePoolValue, PlotState, RowId } from '../core/types'

export interface PreparedPoint {
  rowId: RowId
  sourceIndex: number
  x: number | string
  y: number
  groupValue: string | null
  colorValue: string | null
}

export interface PreparedScatterData {
  type: 'scatter'
  state: PlotState
  points: readonly PreparedPoint[]
}

export interface PreparedSwarmPoint extends PreparedPoint {
  x: number
  centerX: number
  category: string
}

export interface PreparedSwarmData {
  type: 'swarm'
  state: PlotState
  categories: readonly string[]
  points: readonly PreparedSwarmPoint[]
}

export interface PreparedDistributionData {
  type: 'box' | 'violin'
  state: PlotState
  categories: readonly string[]
  points: readonly PreparedPoint[]
}

export interface PreparedHistogramBin {
  groupValue: string | null
  colorValue: string | null
  lower: number
  upper: number
  center: number
  count: number
  cumulativeCount: number
  cumulativeProportion: number
}

export interface PreparedHistogramData {
  type: 'histogram' | 'cumulativeHistogram'
  state: PlotState
  points: readonly PreparedPoint[]
  bins: readonly PreparedHistogramBin[]
}

export type PreparedPlotData = PreparedScatterData | PreparedSwarmData | PreparedDistributionData | PreparedHistogramData

export interface SummaryRow {
  groupValue: string | null
  colorValue: string | null
  statistics: DescriptiveStatistics & Partial<QuartileStatistics>
}

export interface RepresentedDataRow {
  rowId: RowId
  x: NicePoolValue
  y: number
  groupValue: string | null
  colorValue: string | null
}

/** Structured report derived from the exact prepared data represented by a plot. */
export interface PlotSummary {
  plotType: PreparedPlotData['type']
  parameters: PlotState
  aggregateRows: readonly SummaryRow[]
  representedRows: readonly RepresentedDataRow[]
  bins?: readonly PreparedHistogramBin[]
}

/** Plotly-compatible output plus the prepared data and its report. */
export interface PreparedPlot {
  data: PreparedPlotData
  summary: PlotSummary
}
