# Spec 006 Part A — Table 2 in the paper's statistics

## Definitions

Sample, data and units are spec 001's main sample: `^GSPC` with `data/vix_1990_data.csv`, 1990-01 through 2007-12 (216 months), VRP in monthly percent squared, TB3MS lagged one month. The dependent variable is the h-month average excess return times 1200 (annualized percent). N = 216 − h.

The Hodrick (1992) 1B covariance is the spec 003 construction: u = e − mean(e), z_t = (1, VRP_t)', w_t = u_{t+1} × (z_t + … + z_{t−h+1}), on the dates t with t−h+1 and t+1 inside the sample. V = Z⁻¹ S Z⁻¹ / (T h²). The constant's t uses V[0, 0] and the slope's t uses V[1, 1], from the same w and Z. `src/inference.py` returns V[1, 1] only and is not modified: `hodrick_cov` in `src/table2_full.py` builds the full matrix, and its [1, 1] element is asserted equal to `inference.hodrick_t` at every horizon (1e-12).

Newey–West t uses `statsmodels` HAC, Bartlett kernel, maxlags = max(1, h−1), `use_correction=False`. R² and adjusted R² = 1 − (1 − R²)(N − 1)/(N − 2) are in percent.

## Table 2

Hodrick 1B t-statistics in parentheses; BTZ values below each row of mine.

| h | constant | VRP coef | t NW (coef) | R² | adj. R² | N |
|---|---|---|---|---|---|---|
| 1 | -0.56 (-0.14) | 0.388 (1.44) | 1.63 | 1.36 | 0.90 | 215 |
| BTZ | -0.55 (-0.13) | 0.39 (1.76) | — | — | 1.07 | — |
| 3 | -0.87 (-0.24) | 0.420 (2.22) | 3.46 | 4.98 | 4.53 | 213 |
| BTZ | -2.08 (-0.56) | 0.47 (2.86) | — | — | 6.82 | — |
| 6 | 0.74 (0.21) | 0.288 (1.86) | 2.98 | 4.92 | 4.46 | 210 |
| BTZ | 1.12 (0.33) | 0.30 (2.15) | — | — | 5.42 | — |
| 9 | 2.52 (0.79) | 0.165 (1.24) | 1.60 | 2.35 | 1.87 | 207 |
| BTZ | 3.63 (1.15) | 0.17 (1.36) | — | — | 2.30 | — |
| 12 | 3.06 (1.01) | 0.134 (1.10) | 1.42 | 1.87 | 1.39 | 204 |
| BTZ | 4.62 (1.50) | 0.12 (1.00) | — | — | 1.23 | — |
| 15 | 3.34 (1.13) | 0.110 (0.97) | 1.18 | 1.45 | 0.96 | 201 |
| BTZ | 4.84 (1.59) | 0.11 (0.94) | — | — | 1.00 | — |
| 18 | 3.49 (1.15) | 0.093 (0.86) | 1.12 | 1.18 | 0.67 | 198 |
| BTZ | 5.61 (1.81) | 0.06 (0.56) | — | — | 0.05 | — |
| 24 | 3.89 (1.31) | 0.056 (0.56) | 1.01 | 0.50 | -0.02 | 192 |
| BTZ | 6.48 (2.07) | 0.01 (0.11) | — | — | -0.50 | — |

Ratios to BTZ at h = 1, 3, 6:

| ratio | h = 1 | h = 3 | h = 6 |
|---|---|---|---|
| t1B(mine) / t1B(BTZ) | 0.820 | 0.776 | 0.863 |
| √(adj R²(mine) / adj R²(BTZ)) | 0.915 | 0.815 | 0.908 |

The coefficient peaks at h = 3 and the Hodrick t at h = 3. The largest |coef − BTZ| for h ≤ 15 is 0.0504, at h = 3. Horizons with a Hodrick |t| above 1.96: h = 3. The largest Hodrick t is 2.22, at h = 3.

## Checks

1. Pass. At h = 1, 3, 6, 9, 12, 24 the coefficient, Newey–West t and R² reproduce `results/001-baseline-1990/table2_main.csv`; largest gap 2.2e-16 (tolerance 1e-8).
2. Pass. At h = 1, 3, 6 the slope's Hodrick t reproduces R-base in `results/003-inference/table_inference.csv`; largest gap 2.4e-15 (tolerance 1e-8).
3. Pass. The constant's Hodrick t uses element [0, 0] of the matrix whose [1, 1] element gives the slope's t. At h = 1, with the OLS residual in place of u, both standard errors equal White HC0 to 1e-10, on the data and on a simulated sample.
4. Pass. Every quick-check value is reproduced (±0.005 coefficient, ±0.02 t and adjusted R², N exact). Largest gaps: coef 0.0004, t1B 0.005, t NW 0.005, adj. R² 0.005.

## Deviations from this spec

1. The spec says to generalize `src/inference.py` `hodrick_t` for the constant. The task also said not to change the 001–005 modules, so the generalization lives in `src/table2_full.py` (`hodrick_cov`), reusing `inference._rolling_sum`, and is asserted to reproduce `hodrick_t` at every horizon. `hodrick_t`'s output is unchanged because the function is unchanged.

2. The ratio "row" is stored as two columns of `table2_full.csv`, `ratio_t1B` and `ratio_sqrt_adjR2`, filled at h = 1, 3, 6 only, and shown above as two rows.

