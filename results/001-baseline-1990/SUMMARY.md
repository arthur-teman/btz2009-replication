# Spec 001 — Baseline 1990–2007 in BTZ units

## Data vintages

- `data/gspc_data.csv`: Yahoo `^GSPC` daily Close, request window 1989-12-01 to 2026-01-01 (end exclusive). Downloaded 2026-10-15. Rows in the file: 9,087, from 1989-12-01 to 2025-12-31.
- `data/vix_1990_data.csv`: Yahoo `^VIX` daily Close, same request window. Downloaded 2026-10-15. Rows in the file: 9,067, from 1990-01-02 to 2025-12-31. Yahoo has no `^VIX` print before 1990-01-02, so the file starts on the first available session inside the window.
- `data/tb3ms.csv`: existing FRED TB3MS file, not re-downloaded. Monthly average, percent per year, from 1934-01-01 to 2026-08-01.
- `data/spy_data.csv`: existing file, used only for R1 returns and RV. From 1993-01-29 to 2025-12-31. The adjusted series is stored in `Close` (the file has no separate adjusted-close column). January 1993 closes near 24, the dividend-adjusted level.

IV for every sample, including R1, uses `data/vix_1990_data.csv`. R1 changes the price series used for returns and RV, and the sample window.

## Definitions used

Month t is the calendar month, labelled at month-end (`ME`). RV is the sum of squared daily log returns inside the month, times 10⁴, with `min_count=1`. IV is the last VIX print of the month, squared, divided by 12. VRP = IV − RV, in monthly percent squared. The excess return is the monthly index log return minus TB3MS of the previous month divided by 1200. Specification (A) regresses the h-month average of future excess returns on VRP. Specification (B) multiplies that average by 1200 before the regression. Newey–West standard errors use the Bartlett kernel as implemented by `statsmodels` OLS `cov_type='HAC'`, with `use_correction=False`. The two lag choices are `maxlags = max(1, h−1)` and `maxlags = h`. At h = 1 the two choices are the same lag. AR(1) in Table 1 is the OLS slope of the series on its own lag and a constant. Standard deviations use the sample divisor (n − 1). R² is reported in percent.

## Checks

### 1. No look-ahead

Pass. The regression loop checked 4797 rows (every sample, every horizon, every retained month t). The last daily-return date inside RV_t and the last VIX date inside IV_t both fall in month t, and the later of those two dates is on or before the month-end label of t. The first excess return inside Y_t is the return of month t+1, and the window runs through month t+h.

The risk-free rate is lagged as specified. rf(1990-01) = TB3MS(1989-12) / 1200 = 0.00635833 (TB3MS December 1989 = 7.63). rf(1990-02) = TB3MS(1990-01) / 1200 = 0.00636667 (TB3MS January 1990 = 7.64).

### 2. Observation counts

Pass. VRP observations on the main sample, 1990-01 through 2007-12: **216** (expected 216). Regression N = 216 − h:

| h | N |
|---|---|
| 1 | 215 |
| 3 | 213 |
| 6 | 210 |
| 9 | 207 |
| 12 | 204 |
| 24 | 192 |

The same truncation, N = (months in the sample) − h, holds for the robustness samples:

| sample | window | months | N at h=1 | N at h=24 |
|---|---|---|---|---|
| main | 1990-01 → 2007-12 | 216 | 215 | 192 |
| R1 | 1993-02 → 2007-12 | 179 | 178 | 155 |
| R2 | 1990-01 → 2025-12 | 432 | 431 | 408 |

### 3. No zero RV and no missing values inside the sample

Pass. On main, R1 and R2, after the sample window is reindexed to a complete month-end calendar, IV, RV, VRP and the excess return contain no NaN, and no RV equals 0. A separate construction check confirms that a month with no prices gets RV = NaN rather than 0, and that VIX = 20 maps to IV = 400/12.

### 4. Means and standard deviations, monthly percent squared

Main sample, 1990-01 through 2007-12:

| variable | mean | sd |
|---|---|---|
| IV | 33.237 | 23.745 |
| RV | 20.844 | 21.779 |
| VRP | 12.394 | 14.167 |

Audit reference on 1993–2007, for Arthur's comparison with BTZ Table 1: IV ≈ 33.9, RV ≈ 23.9, VRP ≈ 10.0. The same window on these data:

| source | window | IV mean | RV mean | VRP mean |
|---|---|---|---|---|
| ^GSPC + new VIX | 1993-02 → 2007-12 | 33.902 | 22.032 | 11.871 |
| SPY + new VIX (R1) | 1993-02 → 2007-12 | 33.902 | 23.860 | 10.042 |

Full moments (mean, sd, min, max, AR(1), n) for IV, RV, VRP and e, on main, R1 and R2, are in `table1_summary.csv`.

### 5. Unit identity

Pass, under the identity implied by the unit definitions. For every sample and every horizon, the coefficient from regressing specification (A) on decimal VRP (VRP / 10⁴), multiplied by 1200 / 10⁴, equals coefficient (B). The largest absolute gap across the 18 regressions is 3.05e-16. Coefficient (A) × 1200 equals coefficient (B) on the same rows, and the t-statistic and R² are unchanged by the 1200 scaling of the dependent variable.

The check sentence in the spec multiplies the decimal-unit coefficient by 10⁴ / 1200. That order is the reciprocal of the scaling above. It is recorded under Deviations. The code asserts the identity that matches Y(B) = 1200 × Y(A), VRP in percent squared = 10⁴ × decimal VRP, and the audit benchmark of about 0.28.

## Comparison with the BTZ Table 2 targets

Targets are the notebook's transcription of BTZ (2009) Table 2. Coefficients below are specification (B), annualized percent on monthly percent-squared VRP. t(h−1) uses maxlags = max(1, h−1). t(h) uses maxlags = h. R² is in percent.

| h | coef(B) | BTZ coef | t(h−1) | t(h) | BTZ t | R² | BTZ R² | N |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.3879 | 0.39 | 1.63 | 1.63 | 1.76 | 1.36 | 1.07 | 215 |
| 3 | 0.4196 | 0.47 | 3.46 | 3.58 | 2.86 | 4.98 | 6.82 | 213 |
| 6 | 0.2882 | 0.30 | 2.98 | 3.09 | 2.15 | 4.92 | 5.42 | 210 |
| 9 | 0.1651 | — | 1.60 | 1.61 | — | 2.35 | — | 207 |
| 12 | 0.1340 | 0.12 | 1.42 | 1.41 | 1.00 | 1.87 | 1.23 | 204 |
| 24 | 0.0560 | — | 1.01 | 1.02 | — | 0.50 | — | 192 |

Coefficient (B) at h = 3 on the main sample is **0.4196**, outside the band 0.25–0.35. That band is the spec's reading of the 1993–2007 audit (0.28). R1 is that audit sample, now in these units and with the lagged T-bill rate: coefficient (B) at h = 3 is 0.2783 and t(h−1) is 2.60.

The hump is preserved on the main sample. Coefficient (B) peaks at h = 3, t(h−1) peaks at h = 3, t(h) peaks at h = 3, and R² peaks at h = 3.

t(h = 3) with maxlags = h−1 is 3.46 (above 1.96). t(h = 3) with maxlags = h is 3.58 (above 1.96).

R² is above the BTZ target at h = 1 and h = 12, and below it at h = 3 and h = 6. The spec expected daily RV to leave R² below the paper at every reported horizon, and said not to close that gap. RV is still the sum of squared daily log returns. `fig_hump.png` plots both t-statistics and R² for the main sample. BTZ markers are drawn at h = 1, 3, 6 and 12. Horizons 9 and 24 have no published target in the notebook.

Robustness, coefficient (B) and t(h−1):

| sample | h=3 coef(B) | h=3 t(h−1) | h=3 R² | peak of coef(B) |
|---|---|---|---|---|
| main | 0.4196 | 3.46 | 4.98 | h = 3 |
| R1 | 0.2783 | 2.60 | 2.54 | h = 3 |
| R2 | 0.0908 | 0.70 | 1.20 | h = 3 |

## Deviations from this spec

1. **Check 5 writes the scaling factors in reverse order.** The spec says the decimal-unit coefficient × 10⁴ / 1200 must equal coefficient (B). The definitions in the same spec say the opposite ratio. Decimal VRP times 10⁴ is VRP in monthly percent squared, so moving the regressor into the new units divides the coefficient by 10⁴. Specification (B) = 1200 × specification (A), so that scaling multiplies the coefficient by 1200. Combined, decimal-unit coefficient × 1200 / 10⁴ = coefficient (B). That is the identity asserted in the code. It is also the one that turns the notebook's 1993–2007 h = 3 coefficient of 2.32 into the audit value of about 0.28 (2.32 × 1200 / 10⁴ = 0.278). Applying the factors in the written order would produce a coefficient near 19, outside the expected band of 0.25–0.35.

