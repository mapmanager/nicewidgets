# Group plot — methods

This document describes the **Group** plot type in the plot pool: the algorithm, axis meaning, and interpretation.

## Algorithm

1. **Group** the filtered dataframe by the Group/Color column (e.g. `grandparent_folder`, `roi_id`).
2. For each group, take the **Y column** values, with optional absolute value and remove-values filtering applied.
3. **Aggregate** those values with the chosen **Y stat (grouped)**:
   - **mean**, **median**, **sum**, **count**, **std**, **min**, **max**
   - **sem** — standard error of the mean: \(\sigma / \sqrt{n}\) (sample std, ddof=1).
   - **cv** — coefficient of variation: \(\sigma / \mu\). Undefined when \(\mu = 0\); we set the result to missing (NaN) when \(|\mu| < \varepsilon\) to avoid division by zero and unstable values. The threshold \(\varepsilon\) is configurable in the UI (**CV ε (|μ| < this → NaN)**; default \(10^{-10}\)).
4. Plot **one point per group**: x = group label, y = aggregated value. Points are drawn as markers and connected with lines in x-order.

So: *group by category → one summary number per group (the chosen stat of the Y column) → plot category vs that number.*

## X-axis

The **group column** (Group/Color). Each tick is one distinct category (e.g. folder name or roi_id). Order is from the groupby (sorted group labels). Axis title is the group column name.

## Y-axis

The **aggregated statistic** of the **Y column** for that group. Axis title is `{stat}({ycol})`, e.g. `mean(vel_mean)`, `sem(score_peak)`, or `cv(vel_mean)`.

## What each point represents

**One point = one group.**

- **x**: That group’s label.
- **y**: The single number from applying the chosen stat to the Y column over all rows in that group.

So the plot shows **summary per category** (e.g. mean velocity per folder, or count of events per roi_id). The line connects these per-group summaries in category order.

## Notes on CV and SEM

- **CV (coefficient of variation)** is a dimensionless measure of relative spread. We return missing (NaN) when \(|\mu| < \varepsilon\) (configurable; default \(10^{-10}\)) so that division by zero and near-zero means do not produce infinite or misleading values; those groups will show a gap in the plot. The standard deviation in the numerator uses **ddof=1** (sample std, divisor \(n-1\)).
- **SEM (standard error of the mean)** quantifies uncertainty in the sample mean; it is computed as the sample standard deviation divided by \(\sqrt{n}\) for each group. We use **ddof=1** so the standard deviation is the sample (Bessel-corrected) estimate.
