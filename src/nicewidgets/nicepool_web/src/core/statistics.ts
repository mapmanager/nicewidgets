/** Descriptive statistics shared by aggregation and plot-summary layers. */
export interface DescriptiveStatistics {
  count: number
  min: number | null
  max: number | null
  mean: number | null
  median: number | null
  std: number | null
  sem: number | null
  cv: number | null
}

export interface QuartileStatistics {
  q1: number | null
  q3: number | null
  iqr: number | null
}

function quantile(sorted: readonly number[], probability: number): number | null {
  if (!sorted.length) return null
  const position = (sorted.length - 1) * probability
  const lower = Math.floor(position)
  const fraction = position - lower
  return sorted[lower]! + ((sorted[lower + 1] ?? sorted[lower]!) - sorted[lower]!) * fraction
}

/** Calculate linear-interpolated Q1, Q3, and IQR over finite values. */
export function quartileStatistics(values: readonly number[]): QuartileStatistics {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b)
  const q1 = quantile(finite, 0.25)
  const q3 = quantile(finite, 0.75)
  return { q1, q3, iqr: q1 === null || q3 === null ? null : q3 - q1 }
}

/** Calculate pandas-compatible sample statistics over finite values. */
export function descriptiveStatistics(
  values: readonly number[],
  cvEpsilon = 1e-10,
): DescriptiveStatistics {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b)
  const count = finite.length
  if (count === 0) {
    return { count, min: null, max: null, mean: null, median: null, std: null, sem: null, cv: null }
  }
  const sum = finite.reduce((total, value) => total + value, 0)
  const mean = sum / count
  const middle = Math.floor(count / 2)
  const median =
    count % 2 === 0 ? (finite[middle - 1]! + finite[middle]!) / 2 : finite[middle]!
  let std: number | null = null
  let sem: number | null = null
  let cv: number | null = null
  if (count > 1) {
    const variance = finite.reduce((total, value) => total + (value - mean) ** 2, 0) / (count - 1)
    std = Math.sqrt(variance)
    sem = std / Math.sqrt(count)
    if (Math.abs(mean) >= cvEpsilon) cv = std / mean
  }
  return { count, min: finite[0]!, max: finite[count - 1]!, mean, median, std, sem, cv }
}
