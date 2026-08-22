import { describe, expect, it } from 'vitest'

import { DatasetStore, defaultPlotState } from '../src/core'
import { prepareSwarm } from '../src/plots/prepare'
import { formatPlotSummaryToTsv } from '../src/plots/summary-format'
import { summarizePlot } from '../src/plots/summary'
import { edgeDataset } from './fixtures'

describe('plot summary export', () => {
  it('includes parameters, aggregate statistics, and every represented row', () => {
    const dataset = new DatasetStore(edgeDataset)
    const state = {
      ...defaultPlotState(dataset), plotType: 'swarm' as const,
      yColumn: 'y', groupColumn: 'condition', colorColumn: 'cohort',
    }
    const summary = summarizePlot(prepareSwarm(dataset, state))
    const report = formatPlotSummaryToTsv(summary)

    expect(report).toContain('=== Plot state ===')
    expect(report).toContain('=== Summary table ===')
    expect(report).toContain('count\tmin\tmax\tmean\tmedian\tstd\tsem\tcv')
    expect(report).toContain('=== Raw data ===')
    for (const represented of summary.representedRows) expect(report).toContain(String(represented.rowId))

    const aggregateOnly = formatPlotSummaryToTsv(summary, { includeParameters: false, includeRepresentedRows: false })
    expect(aggregateOnly).not.toContain('=== Plot state ===')
    expect(aggregateOnly).toContain('=== Summary table ===')
    expect(aggregateOnly).not.toContain('=== Raw data ===')
  })
})
