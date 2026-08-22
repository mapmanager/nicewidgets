import type { DatasetInput, NicePoolRow } from '../core/types'

function deterministicNoise(index: number): number {
  const value = Math.sin(index * 12.9898) * 43758.5453
  return value - Math.floor(value) - 0.5
}

/** Build stable demonstration data without tying NicePool to a scientific domain. */
export function sampleDataset(rowCount = 600): DatasetInput {
  const conditions = ['control', 'treated', 'recovery'] as const
  const channels = ['green', 'red'] as const
  const rows: NicePoolRow[] = Array.from({ length: rowCount }, (_, index) => {
    const condition = conditions[index % conditions.length]!
    const channel = channels[Math.floor(index / conditions.length) % channels.length]!
    const conditionEffect = condition === 'control' ? 0 : condition === 'treated' ? 2.4 : 1.1
    const channelEffect = channel === 'green' ? 0 : 0.8
    const time = index / 10
    const baseVelocity = deterministicNoise(index + 71) * 6
    const velocity = index > 0 && index % 89 === 0
      ? null
      : index > 0 && index % 173 === 0
      ? (index % 346 === 0 ? -28 : 32)
      : baseVelocity + conditionEffect * 0.25
    return {
      pool_row_id: `row-${String(index + 1).padStart(4, '0')}`,
      accept: index % 11 === 0 ? 'no' : 'yes',
      channel,
      roi_id: `roi-${(index % 8) + 1}`,
      condition,
      time,
      amplitude: 8 + conditionEffect + channelEffect + deterministicNoise(index) * 3,
      velocity,
      duration: 20 + (index % 13) + deterministicNoise(index + 149) * 4,
    }
  })
  return { rows, rowIdColumn: 'pool_row_id' }
}
