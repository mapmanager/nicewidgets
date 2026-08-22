import { categoryKey, filteredRowIndices, finiteNumber, type DatasetStore } from '../core/dataset'
import { PlotConfigurationError, type NicePoolValue, type PlotState } from '../core/types'
import type {
  PreparedPlotData,
  PreparedDistributionData,
  PreparedHistogramBin,
  PreparedHistogramData,
  PreparedPoint,
  PreparedScatterData,
  PreparedSwarmData,
  PreparedSwarmPoint,
} from './types'

function transformedNumber(value: NicePoolValue | undefined, state: PlotState): number | null {
  const numeric = finiteNumber(value)
  if (numeric === null) return null
  const transformed = state.useAbsoluteValue ? Math.abs(numeric) : numeric
  const threshold = state.removeValuesThreshold
  return threshold !== null && Math.abs(transformed) > threshold ? null : transformed
}

interface GroupedCandidate {
  sourceIndex: number
  rowId: string
  groupValue: string
  colorValue: string | null
  y: number
}

function groupedCandidates(dataset: DatasetStore, state: PlotState): GroupedCandidate[] {
  if (!state.groupColumn) throw new PlotConfigurationError(`${state.plotType} plot requires a group column`)
  const columns = new Set(dataset.schema.map(({ name }) => name))
  if (!columns.has(state.groupColumn)) throw new PlotConfigurationError(`Unknown group column ${JSON.stringify(state.groupColumn)}`)
  if (!columns.has(state.yColumn)) throw new PlotConfigurationError(`Unknown Y column ${JSON.stringify(state.yColumn)}`)
  return filteredRowIndices(dataset, state.preFilters).flatMap((sourceIndex) => {
    const { row, rowId, groupValue, colorValue } = pointMetadata(dataset, sourceIndex, state.groupColumn, state.colorColumn)
    const y = transformedNumber(row[state.yColumn], state)
    return y === null || groupValue === null || (state.colorColumn !== null && colorValue === null)
      ? []
      : [{ sourceIndex, rowId, groupValue, colorValue, y }]
  })
}

function validateColumns(dataset: DatasetStore, state: PlotState): void {
  const columns = new Set(dataset.schema.map(({ name }) => name))
  if (!columns.has(state.xColumn)) throw new PlotConfigurationError(`Unknown X column ${JSON.stringify(state.xColumn)}`)
  if (!columns.has(state.yColumn)) throw new PlotConfigurationError(`Unknown Y column ${JSON.stringify(state.yColumn)}`)
}

function pointMetadata(
  dataset: DatasetStore,
  sourceIndex: number,
  groupColumn: string | null,
  colorColumn: string | null,
) {
  const row = dataset.rows[sourceIndex]!
  return {
    row,
    rowId: String(row[dataset.rowIdColumn]),
    groupValue: groupColumn && row[groupColumn] != null ? categoryKey(row[groupColumn]!) : null,
    colorValue:
      colorColumn && row[colorColumn] != null
        ? categoryKey(row[colorColumn]!)
        : null,
  }
}

/** Prepare scatter values once for both Plotly rendering and plot summaries. */
export function prepareScatter(dataset: DatasetStore, state: PlotState): PreparedScatterData {
  validateColumns(dataset, state)
  const xIsNumeric = dataset.schema.find(({ name }) => name === state.xColumn)?.type === 'number'
  const points: PreparedPoint[] = []
  for (const sourceIndex of filteredRowIndices(dataset, state.preFilters)) {
    const { row, rowId, groupValue, colorValue } = pointMetadata(
      dataset,
      sourceIndex,
      null,
      state.colorColumn,
    )
    const y = transformedNumber(row[state.yColumn], state)
    if (y === null) continue
    if (state.colorColumn !== null && colorValue === null) continue
    const rawX = row[state.xColumn]
    const x = xIsNumeric ? transformedNumber(rawX, state) : rawX == null ? null : categoryKey(rawX)
    if (x === null || typeof x === 'boolean') continue
    points.push({ rowId, sourceIndex, x, y, groupValue, colorValue })
  }
  return Object.freeze({ type: 'scatter', state, points: Object.freeze(points) })
}

function hashString(value: string, seed: number): number {
  let hash = 2166136261 ^ seed
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

function unitRandom(value: string, seed: number): number {
  let state = hashString(value, seed)
  state += 0x6d2b79f5
  let mixed = state
  mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1)
  mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61)
  return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296
}

/** Prepare deterministic categorical swarm coordinates and row identity. */
export function prepareSwarm(dataset: DatasetStore, state: PlotState): PreparedSwarmData {
  const candidates = groupedCandidates(dataset, state)
  const categories = [...new Set(candidates.map(({ groupValue }) => groupValue))].sort()
  const categoryPositions = new Map(categories.map((category, index) => [category, index]))
  const colors = [...new Set(candidates.map(({ colorValue }) => colorValue).filter((value) => value !== null))].sort()
  const colorPositions = new Map(colors.map((color, index) => [color, index]))
  const points: PreparedSwarmPoint[] = candidates.map(({ sourceIndex, rowId, groupValue, colorValue, y }) => {
    const categoryPosition = categoryPositions.get(groupValue)!
    const jitter = (unitRandom(`${rowId}\u0000${groupValue}\u0000${colorValue ?? ''}`, state.jitterSeed) - 0.5) * state.swarmJitterAmount
    const colorPosition = colorValue === null ? 0 : colorPositions.get(colorValue) ?? 0
    const offset = colors.length > 1 ? (colorPosition - (colors.length - 1) / 2) * state.swarmGroupOffset : 0
    const centerX = categoryPosition + offset
    return {
      rowId,
      sourceIndex,
      category: groupValue,
      x: centerX + jitter,
      centerX,
      y,
      groupValue,
      colorValue,
    }
  })
  return Object.freeze({ type: 'swarm', state, categories: Object.freeze(categories), points: Object.freeze(points) })
}

/** Prepare categorical observations shared by box and violin rendering. */
export function prepareDistribution(dataset: DatasetStore, state: PlotState): PreparedDistributionData {
  if (state.plotType !== 'box' && state.plotType !== 'violin') throw new PlotConfigurationError('Distribution preparation requires box or violin state')
  const candidates = groupedCandidates(dataset, state)
  const categories = [...new Set(candidates.map(({ groupValue }) => groupValue))].sort()
  const points: PreparedPoint[] = candidates.map(({ sourceIndex, rowId, groupValue, colorValue, y }) => ({
    sourceIndex, rowId, x: groupValue, y, groupValue, colorValue,
  }))
  return Object.freeze({ type: state.plotType, state, categories: Object.freeze(categories), points: Object.freeze(points) })
}

/** Prepare globally aligned bins shared by histogram rendering and summaries. */
export function prepareHistogram(dataset: DatasetStore, state: PlotState): PreparedHistogramData {
  if (state.plotType !== 'histogram' && state.plotType !== 'cumulativeHistogram') throw new PlotConfigurationError('Histogram preparation requires histogram state')
  const numeric = new Set(dataset.numericColumns())
  if (!numeric.has(state.xColumn)) throw new PlotConfigurationError('Histogram X column must be numeric')
  const points: PreparedPoint[] = []
  for (const sourceIndex of filteredRowIndices(dataset, state.preFilters)) {
    const { row, rowId, groupValue, colorValue } = pointMetadata(dataset, sourceIndex, state.groupColumn, state.colorColumn)
    const value = transformedNumber(row[state.xColumn], state)
    if (value === null || (state.groupColumn !== null && groupValue === null) || (state.colorColumn !== null && colorValue === null)) continue
    points.push({ sourceIndex, rowId, x: value, y: value, groupValue, colorValue })
  }
  if (!points.length) return Object.freeze({ type: state.plotType, state, points: Object.freeze([]), bins: Object.freeze([]) })

  const values = points.map(({ x }) => x as number)
  let minimum = Math.min(...values)
  let maximum = Math.max(...values)
  if (minimum === maximum) {
    const padding = Math.abs(minimum) * 0.05 || 0.5
    minimum -= padding
    maximum += padding
  }
  const width = (maximum - minimum) / state.histogramBins
  const groups = new Map<string, PreparedPoint[]>()
  for (const point of points) {
    const key = JSON.stringify([point.groupValue, point.colorValue])
    const group = groups.get(key) ?? []
    group.push(point)
    groups.set(key, group)
  }
  const bins: PreparedHistogramBin[] = []
  for (const groupPoints of groups.values()) {
    const counts = Array.from({ length: state.histogramBins }, () => 0)
    for (const point of groupPoints) {
      const index = Math.min(state.histogramBins - 1, Math.floor(((point.x as number) - minimum) / width))
      counts[Math.max(0, index)]! += 1
    }
    let cumulativeCount = 0
    counts.forEach((count, index) => {
      const lower = minimum + index * width
      cumulativeCount += count
      bins.push({
        groupValue: groupPoints[0]!.groupValue,
        colorValue: groupPoints[0]!.colorValue,
        lower,
        upper: lower + width,
        center: lower + width / 2,
        count,
        cumulativeCount,
        cumulativeProportion: cumulativeCount / groupPoints.length,
      })
    })
  }
  return Object.freeze({ type: state.plotType, state, points: Object.freeze(points), bins: Object.freeze(bins) })
}

/** Dispatch Slice 1 plot preparation without importing Plotly or Vue. */
export function preparePlotData(dataset: DatasetStore, state: PlotState): PreparedPlotData {
  if (state.plotType === 'swarm') return prepareSwarm(dataset, state)
  if (state.plotType === 'box' || state.plotType === 'violin') return prepareDistribution(dataset, state)
  if (state.plotType === 'histogram' || state.plotType === 'cumulativeHistogram') return prepareHistogram(dataset, state)
  return prepareScatter(dataset, state)
}
