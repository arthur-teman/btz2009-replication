# Spec 002 — Why the full-sample result collapses

## Definitions

The sample, the variance series and specification (B) are the R2 regression of spec 001. `src/baseline_1990.py` is imported and not modified. Index `^GSPC`, implied variance from `data/vix_1990_data.csv`, risk-free rate TB3MS lagged one month. VRP is in monthly percent squared. Specification (B) multiplies the h-month average excess return by 1200. The sample is 1990-01 through 2025-12. Horizons are h = 1, 3 and 6. Newey–West standard errors use `statsmodels` HAC with `use_correction=False`. t(h−1) uses maxlags = max(1, h−1). t(h) uses maxlags = h. R² is in percent.

Row 0 is that regression with the raw VRP. B clips VRP at percentiles computed on the whole 1990–2025 sample, so those cut points use later months. B is labelled in-sample. C clips VRP_t at the 1st and 99th percentiles of VRP from 1990-01 through t. The first 60 months are a burn-in: regression rows before 1995-01 are dropped from C only. D drops regression rows whose regressor month is in 2008-09 through 2008-12 or 2020-02 through 2020-04. The forward returns of the other rows are unchanged. E re-runs the regression on 1990–2007, 2008–2025 and 2010–2025, and each h-month window stops at the end of that subsample.

F estimates the h = 3 regression on a rolling window of 120 consecutive regression rows, stepped one month at a time. Each point is dated at that window's last regressor month. The three outcome months of the last row are inside the R2 sample and after that date. N = 120 in every window.

G replaces VRP with VRP_fwd_t = IV_t − Ê_t[RV_{t+1}]. The forecast is an expanding-window monthly HAR, estimated by OLS on pairs (s, s+1) with s+1 ≤ t: RV_{s+1} on a constant, RV_s, RV^w_s and RV^d_s. RV_s is the spec 001 realized variance. RV^w_s is the sum of squared daily log returns over the last five trading days of month s, times 22/5, times 10⁴. RV^d_s is the squared daily log return of the last trading day of month s, times 22, times 10⁴. Forecasts start once 60 pairs are available, so the first forecast month is 1995-01. A negative forecast is floored at 0. That floor binds in 0 months.

DFBETA on the row-0 h = 3 OLS is coef(B) minus the coefficient estimated without that row. Leverage is the diagonal of the hat matrix. Cook's distance is the OLS Cook statistic. The hat diagonals of the 429 regression rows sum to 2.

## Checks

### 1. Row 0 reproduces spec 001 R2 at h = 3

Pass. coef(B) = 0.0908 and t(h−1) = 0.70. The unrounded estimates are 0.090806 and 0.696168, the same point as `results/001-baseline-1990/table2_main.csv` for sample R2 at h = 3. h = 1 and h = 6 of row 0 match that file as well.

### 2. Real-time cut points

Pass. For every month t, the 1st and 99th percentiles are the linear quantiles of VRP from 1990-01 through t. Adding 10⁶ to VRP in 2025-12 leaves every earlier cut point unchanged and moves the 2025-12 99th percentile. Printed cut points:

| date | p01 | p99 | months in the window |
|---|---|---|---|
| 1999-12 | -2.420 | 50.517 | 120 |
| 2007-12 | -22.881 | 55.343 | 216 |
| 2008-12 | -99.639 | 54.676 | 228 |
| 2020-12 | -110.410 | 69.100 | 372 |

In-sample B, for comparison, uses p01 = -111.603 and p99 = 68.850 on all 432 months, and p2.5 = -27.920 and p97.5 = 50.043. Those four numbers use the whole sample.

### 3. HAR

Pass. For every forecast month t, the last pair in the fit has s+1 = t, hence s+1 ≤ t, and the fit uses 60 or more pairs. Adding 50 to RV in 2020-03 leaves the coefficients and the raw forecast at every earlier month unchanged, and changes the coefficients dated 2020-03. The prediction at t is applied to RV_t, RV^w_t and RV^d_t. The last trading day inside those two components falls in month t.

| date | c | β_m | β_w | β_d | pairs | last s+1 |
|---|---|---|---|---|---|---|
| 2007-12 | 8.459 | 0.607 | -0.057 | 0.045 | 215 | 2007-12 |
| 2025-12 | 9.777 | -0.059 | 0.772 | -0.051 | 431 | 2025-12 |

Correlation between VRP_fwd and VRP, on the 372 months from 1995-01 through 2025-12 where the forecast exists: **0.461**. Forecasts below 0 are floored at 0 in **0** months.

### 4. N on every robustness row

Pass. The table below reports N at h = 1, 3 and 6 for every variant. N equals the number of regression rows after the variant's own sample rule and after dropping incomplete h-month windows.

### 5. No look-ahead in X, and Y starts at t+1

Pass on the spec 001 timing rule. The code checked 47958 regression rows: row 0, B, C, D, the three E samples, both G samples, every row of every 120-month window, and the two 2010–2025 readings used for the third case below. For each row t, the last daily-return date inside RV_t and the last VIX date inside IV_t fall in month t, and the later of those dates is on or before the month-end label. The first excess return inside Y_t is the return of month t+1, and the window runs through month t+h, inside the sample of that variant.

B is the labelled in-sample exception. Its clipped regressor uses percentiles computed through 2025-12, so a row t in B depends on later VRP values. C and G are the real-time forms of the two fixes, and checks 2 and 3 cover them. D drops a fixed list of months. It does not recompute VRP from later data.

## Robustness table

Coefficient (B) is annualized percent on monthly percent-squared VRP. t(h−1) uses maxlags = max(1, h−1). t(h) uses maxlags = h. R² is in percent. The same numbers are in `table_robust.csv`, one row per variant.

| variant | h | coef(B) | t(h−1) | t(h) | R² | N |
|---|---|---|---|---|---|---|
| 0 | 1 | 0.0665 | 0.32 | 0.32 | 0.21 | 431 |
| 0 | 3 | 0.0908 | 0.70 | 0.68 | 1.20 | 429 |
| 0 | 6 | 0.0345 | 0.49 | 0.48 | 0.33 | 426 |
| B 1/99 in-sample | 1 | 0.3707 | 1.56 | 1.56 | 2.38 | 431 |
| B 1/99 in-sample | 3 | 0.3169 | 2.31 | 2.20 | 5.36 | 429 |
| B 1/99 in-sample | 6 | 0.1575 | 1.97 | 1.96 | 2.51 | 426 |
| B 2.5/97.5 in-sample | 1 | 0.4958 | 1.98 | 1.98 | 1.97 | 431 |
| B 2.5/97.5 in-sample | 3 | 0.4622 | 3.30 | 3.11 | 5.26 | 429 |
| B 2.5/97.5 in-sample | 6 | 0.2779 | 2.96 | 2.93 | 3.60 | 426 |
| C real-time 1/99 | 1 | 0.2849 | 1.15 | 1.15 | 1.21 | 371 |
| C real-time 1/99 | 3 | 0.2703 | 1.75 | 1.67 | 3.32 | 369 |
| C real-time 1/99 | 6 | 0.1210 | 1.42 | 1.42 | 1.23 | 366 |
| D drop crisis rows | 1 | 0.4030 | 2.11 | 2.11 | 2.19 | 424 |
| D drop crisis rows | 3 | 0.2653 | 2.28 | 2.28 | 2.86 | 422 |
| D drop crisis rows | 6 | 0.1708 | 1.88 | 1.86 | 2.16 | 419 |
| E 1990-2007 | 1 | 0.3879 | 1.63 | 1.63 | 1.36 | 215 |
| E 1990-2007 | 3 | 0.4196 | 3.46 | 3.58 | 4.98 | 213 |
| E 1990-2007 | 6 | 0.2882 | 2.98 | 3.09 | 4.92 | 210 |
| E 2008-2025 | 1 | 0.0453 | 0.21 | 0.21 | 0.16 | 215 |
| E 2008-2025 | 3 | 0.0673 | 0.51 | 0.50 | 1.06 | 213 |
| E 2008-2025 | 6 | 0.0178 | 0.26 | 0.26 | 0.13 | 210 |
| E 2010-2025 | 1 | -0.1237 | -0.87 | -0.87 | 1.16 | 191 |
| E 2010-2025 | 3 | -0.0724 | -1.59 | -1.55 | 1.58 | 189 |
| E 2010-2025 | 6 | -0.0459 | -1.47 | -1.46 | 1.54 | 186 |
| G full sample | 1 | 0.4176 | 3.34 | 3.34 | 3.17 | 371 |
| G full sample | 3 | 0.3256 | 5.23 | 5.21 | 5.87 | 369 |
| G full sample | 6 | 0.1854 | 3.52 | 3.55 | 3.53 | 366 |
| G 1990-2007 | 1 | 0.1940 | 0.68 | 0.68 | 0.41 | 155 |
| G 1990-2007 | 3 | 0.1472 | 0.99 | 1.04 | 0.73 | 153 |
| G 1990-2007 | 6 | 0.0676 | 0.72 | 0.73 | 0.31 | 150 |

D drops these regressor months: 2008-09, 2008-10, 2008-11, 2008-12, 2020-02, 2020-03, 2020-04. That is 7 rows. They are inside the regression sample at every horizon, so N = 432 − h − 7.

G is not one of the three cases. On the full sample its h = 3 t(h−1) is 5.23; on 1990–2007 it is 0.99. The 1990–2007 regression starts in 1995-01 because the HAR burn-in leaves VRP_fwd undefined before then. `fig_vrp_vs_vrpfwd.png` plots the two premia.

## Which case holds

The audit reference, not a target of this spec, is a SPY sample from 1993 to 2025 with full-sample 1/99 winsorization: h = 3, t = 2.11, R² = 4.2%. B 1/99 on ^GSPC for 1990–2025 has t(h−1) = 2.31 and R² = 5.36%.

The ten rows with the largest |DFBETA| are below and in `table_influence.csv`. Y_t is specification (B) at h = 3.

| date | VRP_t | Y_t | DFBETA | Cook's D |
|---|---|---|---|---|
| 2020-03 | -508.51 | 72.58 | -0.2249 | 15.9173 |
| 2008-10 | -274.11 | -64.12 | 0.0278 | 0.2460 |
| 2008-09 | -116.50 | -102.92 | 0.0238 | 0.1906 |
| 2008-11 | -109.55 | -79.40 | 0.0171 | 0.0997 |
| 2025-04 | -138.42 | 47.59 | -0.0160 | 0.0850 |
| 2009-02 | 77.42 | 89.15 | 0.0101 | 0.0401 |
| 2008-12 | -72.86 | -49.77 | 0.0075 | 0.0210 |
| 1998-08 | 65.70 | 73.59 | 0.0068 | 0.0198 |
| 2020-10 | 83.42 | 50.87 | 0.0054 | 0.0114 |
| 2009-03 | -36.74 | 56.49 | -0.0048 | 0.0113 |

Of these ten, 6 fall in 2008 or 2020. The largest |DFBETA| is 2020-03.

Clauses, all at h = 3 and using t(h−1). A clause counts as significant when t > 1.96.

- B 1/99 in-sample significant: yes (t = 2.31).
- B 2.5/97.5 in-sample significant: yes (t = 3.30).
- C real-time 1/99 significant: no (t = 1.75).
- D crisis rows dropped, significant: yes (t = 2.28).
- 2008/2020 months top the |DFBETA| list: yes (6 of the ten rows, and the first row is in 2008 or 2020).
- E 2010–2025 insignificant: yes (t = -1.59).
- 2010–2025 with the real-time 1/99 VRP, insignificant: yes (t = -0.10, N = 189).
- 2010–2025 with 2020-02 through 2020-04 dropped, insignificant: yes (t = 0.87, N = 186).
- Rolling t stays low after 2010: no. On the 189 windows dated 2010-01 or later, the median t is 5.32, the maximum is 6.30, and the share with t > 1.96 is 59.8%. On the 120 windows dated 2010-01 through 2019-12 the median t is 5.64. On the 67 windows dated 2020-03 or later the median t is -2.28. The last window dated 2010 or later with t > 1.96 ends in 2019-06.

**Outlier story** does not hold. It fails because C has t = 1.75.

**Fragile** does not hold. It fails because D is above 1.96 (t = 2.28). The in-sample cut is not the only change that puts t above 1.96.

**Lasting change** does not hold. It fails because the rolling t does not stay low after 2010.

No threshold was changed after seeing the t-statistics.

The two 2010–2025 lines are the readings named in the third case. They are not extra rows of the robustness table, and they were not used to choose a winsorization rule.

## Deviations from this spec

1. **DFBETA is coef(B) minus the coefficient without that row.** That is the textbook leave-one-out change, not DFBETA divided by the leave-one-out standard error. The ten rows are ranked by the absolute value of this change.

2. **The 2010–2025 readings under C and under D are not rows of the robustness table.** The third case asks whether 2010–2025 stays insignificant after those two treatments. The two t-statistics are reported in the case section. The real-time percentiles in the C reading are still the expanding window from 1990-01 through t.

