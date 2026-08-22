/** Scalar values accepted at the NicePool dataset boundary. */
export type NicePoolValue = string | number | boolean | null

/** One row in the caller-provided authoritative dataset. */
export type NicePoolRow = Readonly<Record<string, NicePoolValue>>

export type ColumnType = 'number' | 'string' | 'boolean' | 'categorical'

/** Optional caller-owned declaration for one dataset column. */
export interface ColumnSchema {
  name: string
  type: ColumnType
  label?: string
}

/** Complete replacement payload accepted by {@link NicePoolEngine.setData}. */
export interface DatasetInput {
  rows: readonly NicePoolRow[]
  rowIdColumn: string
  schema?: readonly ColumnSchema[]
}

/** Stable external row identity. */
export type RowId = string

/** Primary and multi-row selection owned by the NicePool engine. */
export interface NicePoolSelection {
  primaryRowId: RowId | null
  selectedRowIds: readonly RowId[]
}

export type PlotType = 'scatter' | 'swarm' | 'box' | 'violin' | 'histogram' | 'cumulativeHistogram'
export type ErrorBarType = 'std' | 'sem'
export type PlotLayout = '1x1' | '1x2' | '2x1' | '2x2'
export type LegendPosition = 'bottom' | 'right' | 'top' | 'left'
export type NicePoolTheme = 'dark' | 'light'

/** Serializable configuration for one Slice 1 plot. */
export interface PlotState {
  plotType: PlotType
  preFilters: Readonly<Record<string, NicePoolValue>>
  xColumn: string
  yColumn: string
  groupColumn: string | null
  colorColumn: string | null
  useAbsoluteValue: boolean
  removeValuesThreshold: number | null
  swarmJitterAmount: number
  swarmGroupOffset: number
  jitterSeed: number
  histogramBins: number
  showRaw: boolean
  showMean: boolean
  showErrorBars: boolean
  errorBarType: ErrorBarType
  pointSize: number
  showLegend: boolean
  legendPosition: LegendPosition
  showPlotlyToolbar: boolean
  showHover: boolean
  showAxes: boolean
  showHorizontalGrid: boolean
  showVerticalGrid: boolean
  cvEpsilon: number
}

/** Serializable state for the complete four-slot plotting workspace. */
export interface NicePoolState {
  schemaVersion: 1
  layout: PlotLayout
  activePlotIndex: number
  plots: readonly [PlotState, PlotState, PlotState, PlotState]
}

/** Named single-plot state applied only to the active plot. */
export interface PlotPreset {
  schemaVersion: 1
  name: string
  plotState: PlotState
}

/** Error raised when a complete dataset cannot satisfy the public contract. */
export class DatasetValidationError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'DatasetValidationError'
  }
}

/** Error raised when plot state is incompatible with the current dataset. */
export class PlotConfigurationError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'PlotConfigurationError'
  }
}

/** Error raised when caller-supplied serialized state violates its contract. */
export class StateValidationError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'StateValidationError'
  }
}
