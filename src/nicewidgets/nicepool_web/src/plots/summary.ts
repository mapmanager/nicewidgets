import { descriptiveStatistics, quartileStatistics } from '../core/statistics'
import type { PlotSummary, PreparedPlotData, SummaryRow } from './types'

/** Summarize the exact immutable prepared rows consumed by Plotly rendering. */
export function summarizePlot(data: PreparedPlotData): PlotSummary {
  const grouped = new Map<string, { groupValue: string | null; colorValue: string | null; values: number[] }>()
  for (const point of data.points) {
    const key = JSON.stringify([point.groupValue, point.colorValue])
    const entry = grouped.get(key) ?? {
      groupValue: point.groupValue,
      colorValue: point.colorValue,
      values: [],
    }
    entry.values.push(point.y)
    grouped.set(key, entry)
  }
  const aggregateRows: SummaryRow[] = [...grouped.values()]
    .sort((a, b) => `${a.groupValue ?? ''}\u0000${a.colorValue ?? ''}`.localeCompare(`${b.groupValue ?? ''}\u0000${b.colorValue ?? ''}`))
    .map(({ groupValue, colorValue, values }) => ({
      groupValue,
      colorValue,
      statistics: {
        ...descriptiveStatistics(values, data.state.cvEpsilon),
        ...(data.type === 'box' || data.type === 'violin' ? quartileStatistics(values) : {}),
      },
    }))
  return Object.freeze({
    plotType: data.type,
    parameters: data.state,
    aggregateRows: Object.freeze(aggregateRows),
    representedRows: Object.freeze(data.type === 'swarm'
      ? data.points.map((point) => ({
          rowId: point.rowId,
          x: point.category,
          y: point.y,
          groupValue: point.groupValue,
          colorValue: point.colorValue,
        }))
      : data.points.map((point) => ({
          rowId: point.rowId,
          x: point.x,
          y: point.y,
          groupValue: point.groupValue,
          colorValue: point.colorValue,
        }))),
    ...('bins' in data ? { bins: Object.freeze(data.bins) } : {}),
  })
}
