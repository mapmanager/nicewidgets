import { describe, expect, it } from 'vitest'

import { descriptiveStatistics, quartileStatistics } from '../src/core'

describe('descriptiveStatistics', () => {
  it('uses sample standard deviation and standard error', () => {
    const result = descriptiveStatistics([1, 3, 5])
    expect(result).toMatchObject({ count: 3, min: 1, max: 5, mean: 3, median: 3, std: 2 })
    expect(result.sem).toBeCloseTo(2 / Math.sqrt(3))
    expect(result.cv).toBeCloseTo(2 / 3)
  })

  it('returns undefined statistics as null', () => {
    expect(descriptiveStatistics([])).toEqual({
      count: 0, min: null, max: null, mean: null, median: null, std: null, sem: null, cv: null,
    })
    expect(descriptiveStatistics([4]).std).toBeNull()
  })

  it('guards coefficient of variation near zero', () => {
    expect(descriptiveStatistics([-1, 1], 0.01).cv).toBeNull()
  })

  it('calculates interpolated quartiles and IQR', () => {
    expect(quartileStatistics([1, 3, 5, 9])).toEqual({ q1: 2.5, q3: 6, iqr: 3.5 })
  })
})
