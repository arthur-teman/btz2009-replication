# Spec 003 — Inference

## Definitions

Spec 001 and spec 002 are imported and not modified. The monthly series are the spec 001 panel: `^GSPC`, implied variance from `data/vix_1990_data.csv`, and TB3MS lagged one month. VRP and VRP_fwd are in monthly percent squared. The dependent variable is specification (B): the h-month average excess return multiplied by 1200, in annualized percent. Horizons are h = 1, 3 and 6. Newey–West standard errors are `statsmodels` HAC with the Bartlett kernel and `use_correction=False`. t(h−1) uses maxlags = max(1, h−1). t(h) uses maxlags = h. t(2h) uses maxlags = 2h. R² is in percent.

Part A drops regression row t when any month in t, t+1, …, t+h falls in C = {2008-09, 2008-10, 2008-11, 2008-12, 2020-02, 2020-03, 2020-04}. The monthly series themselves are not deleted. A kept row still has its own VRP.

Hodrick (1992) estimator 1B uses the monthly excess return in annualized percent, e, and the null residual u = e − mean(e) on the row's calendar window. z_t = (1, x_t)'. For the sum of h monthly returns, w_t = u_{t+1} times the sum of z_{t}, z_{t−1}, …, z_{t−h+1}. The sum is over dates t with t−h+1 inside the window and t+1 inside the window. A Part A date is left out of that sum, because it is not part of the regression being tested. On every row below, that date set has the same length T as the regression. S and Z are the averages of w_t w_t' and z_t z_t' on that date set. V_sum = Z⁻¹ S Z⁻¹ / T, and V for the average-return coefficient is V_sum / h². The t-statistic divides coef(B) from the average regression by the square root of V[1, 1].

The bootstrap imposes no predictability. μ is the mean of e on the row's calendar window, and the regressor is an AR(1) fit by OLS on that same window, including months that Part A drops from the regression. Residuals (ε_s, η_s) are drawn as pairs from the same month. x* starts at the sample's first x. e* on that first month is μ, because the AR(1) residual starts one month later; the first return is not inside any Y. Each later month is x*_t = c + ρ x*_{t−1} + η*_t and e*_t = μ + ε*_t. Y*(h) is the spec 001 forward average of e*, scaled as specification (B) because e* is already in annualized percent. One artificial sample is used for h = 1, 3 and 6. The iid scheme draws pairs independently. The block scheme is the Politis–Romano stationary bootstrap on the pairs, circular, with mean block length 6. There are 5,000 replications per row and scheme, from `numpy.random.default_rng(20261021)`, in the order R-base, R-full, R-D1, R-G, R-G07, and within each row the iid scheme then the block scheme. The null-mean of coef(B) in the table is the mean under the iid scheme. That is the Stambaugh bias for this design.

## Part A — window-overlap exclusion

D1 is VRP on 1990–2025. D2 is VRP on 2010–2025. D3 is VRP_fwd on 1995–2025. t(h−1) uses maxlags = max(1, h−1). t(h) uses maxlags = h.

| variant | h | coef(B) | t(h−1) | t(h) | R² | N | rows dropped |
|---|---|---|---|---|---|---|---|
| D1 | 1 | 0.3996 | 2.09 | 2.09 | 2.22 | 422 | 9 |
| D1 | 3 | 0.2455 | 2.25 | 2.26 | 2.89 | 416 | 13 |
| D1 | 6 | 0.1454 | 1.77 | 1.77 | 2.10 | 407 | 19 |
| D2 | 1 | 0.5184 | 1.76 | 1.76 | 5.25 | 187 | 4 |
| D2 | 3 | 0.1141 | 0.93 | 0.92 | 1.05 | 183 | 6 |
| D2 | 6 | 0.0571 | 0.60 | 0.59 | 0.59 | 177 | 9 |
| D3 | 1 | 0.4308 | 2.12 | 2.12 | 2.01 | 362 | 9 |
| D3 | 3 | 0.2946 | 2.74 | 2.70 | 3.25 | 356 | 13 |
| D3 | 6 | 0.2005 | 2.22 | 2.19 | 3.07 | 347 | 19 |

D1, h = 1, drops 9 rows (N = 422): 2008-08, 2008-09, 2008-10, 2008-11, 2008-12, 2020-01, 2020-02, 2020-03, 2020-04.

D1, h = 3, drops 13 rows (N = 416): 2008-06, 2008-07, 2008-08, 2008-09, 2008-10, 2008-11, 2008-12, 2019-11, 2019-12, 2020-01, 2020-02, 2020-03, 2020-04.

D1, h = 6, drops 19 rows (N = 407): 2008-03, 2008-04, 2008-05, 2008-06, 2008-07, 2008-08, 2008-09, 2008-10, 2008-11, 2008-12, 2019-08, 2019-09, 2019-10, 2019-11, 2019-12, 2020-01, 2020-02, 2020-03, 2020-04.

D2, h = 1, drops 4 rows (N = 187): 2020-01, 2020-02, 2020-03, 2020-04.

D2, h = 3, drops 6 rows (N = 183): 2019-11, 2019-12, 2020-01, 2020-02, 2020-03, 2020-04.

D2, h = 6, drops 9 rows (N = 177): 2019-08, 2019-09, 2019-10, 2019-11, 2019-12, 2020-01, 2020-02, 2020-03, 2020-04.

D3, h = 1, drops 9 rows (N = 362): 2008-08, 2008-09, 2008-10, 2008-11, 2008-12, 2020-01, 2020-02, 2020-03, 2020-04.

D3, h = 3, drops 13 rows (N = 356): 2008-06, 2008-07, 2008-08, 2008-09, 2008-10, 2008-11, 2008-12, 2019-11, 2019-12, 2020-01, 2020-02, 2020-03, 2020-04.

D3, h = 6, drops 19 rows (N = 347): 2008-03, 2008-04, 2008-05, 2008-06, 2008-07, 2008-08, 2008-09, 2008-10, 2008-11, 2008-12, 2019-08, 2019-09, 2019-10, 2019-11, 2019-12, 2020-01, 2020-02, 2020-03, 2020-04.

On the 1990–2025 and 1995–2025 calendars, the h = 3 list is 2008-06 through 2008-12 and 2019-11 through 2020-04. D2 starts in 2010, so only the second block is in its sample.

## Part B — variant G

G on 1995-01 through 2007-12 is the regression the HAR burn-in actually identifies. It is the same estimation sample as spec 002's row "G 1990-2007", which started the calendar in 1990 and then dropped months with no forecast. Columns match the spec 002 table.

| sample | h | coef(B) | t(h−1) | t(h) | R² | N |
|---|---|---|---|---|---|---|
| G 1995-01 → 2007-12 | 1 | 0.1940 | 0.68 | 0.68 | 0.41 | 155 |
| G 1995-01 → 2007-12 | 3 | 0.1472 | 0.99 | 1.04 | 0.73 | 153 |
| G 1995-01 → 2007-12 | 6 | 0.0676 | 0.72 | 0.73 | 0.31 | 150 |

Influence is the full-sample G regression, VRP_fwd on 1995–2025, at h = 3. DFBETA is coef(B) minus the coefficient without that row. The five largest |DFBETA| rows:

| date | VRP_fwd_t | Y_t | DFBETA | Cook's D |
|---|---|---|---|---|
| 2008-10 | -277.41 | -64.12 | -0.0669 | 0.4863 |
| 1998-08 | -53.57 | 73.59 | -0.0317 | 0.1214 |
| 2008-09 | -36.28 | -102.92 | 0.0243 | 0.0782 |
| 2009-02 | 72.34 | 89.15 | 0.0211 | 0.0548 |
| 2008-11 | -31.53 | -79.40 | 0.0167 | 0.0385 |

VRP and VRP_fwd in monthly percent squared:

| date | VRP | VRP_fwd |
|---|---|---|
| 2008-10 | -274.11 | -277.41 |
| 2020-03 | -508.51 | 28.29 |
| 2025-04 | -138.42 | 35.96 |

AR(1) slope on 1995-01 through 2025-12: VRP 0.126, VRP_fwd 0.341.

## Inference

t_NW(h−1) uses maxlags = max(1, h−1). t_NW(2h) uses maxlags = 2h. t_1B is the Hodrick statistic. Bootstrap p-values are two-sided shares of replications at least as large as the observed statistic in absolute value. The null-mean of the coefficient is the iid scheme. R² is in percent.

| id | h | coef(B) | null-mean coef | t_NW(h−1) | t_NW(2h) | t_1B | p_NW iid | p_NW block | p_1B iid | p_1B block | R² | N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R-base | 1 | 0.3879 | 0.0061 | 1.63 | 1.89 | 1.44 | 0.1268 | 0.3432 | 0.1530 | 0.3520 | 1.36 | 215 |
| R-base | 3 | 0.4196 | 0.0062 | 3.46 | 3.95 | 2.22 | 0.0020 | 0.1668 | 0.0230 | 0.2468 | 4.98 | 213 |
| R-base | 6 | 0.2882 | 0.0059 | 2.98 | 3.56 | 1.86 | 0.0120 | 0.1698 | 0.0638 | 0.2786 | 4.92 | 210 |
| R-full | 1 | 0.0665 | 0.0002 | 0.32 | 0.31 | 0.40 | 0.7816 | 0.8218 | 0.7202 | 0.7932 | 0.21 | 431 |
| R-full | 3 | 0.0908 | 0.0004 | 0.70 | 0.66 | 1.30 | 0.5672 | 0.6570 | 0.1962 | 0.5192 | 1.20 | 429 |
| R-full | 6 | 0.0345 | 0.0003 | 0.49 | 0.50 | 0.64 | 0.6914 | 0.7230 | 0.5368 | 0.6890 | 0.33 | 426 |
| R-D1 | 1 | 0.3996 | 0.0008 | 2.09 | 2.12 | 2.05 | 0.0890 | 0.3342 | 0.0264 | 0.2486 | 2.22 | 422 |
| R-D1 | 3 | 0.2455 | -0.0003 | 2.25 | 2.30 | 2.14 | 0.0892 | 0.2856 | 0.0254 | 0.2738 | 2.89 | 416 |
| R-D1 | 6 | 0.1454 | -0.0001 | 1.77 | 1.78 | 0.87 | 0.1658 | 0.3112 | 0.3978 | 0.5834 | 2.10 | 407 |
| R-G | 1 | 0.4176 | -0.0039 | 3.34 | 3.57 | 2.03 | 0.0060 | 0.2790 | 0.0370 | 0.3574 | 3.17 | 371 |
| R-G | 3 | 0.3256 | -0.0015 | 5.23 | 5.37 | 2.78 | 0.0004 | 0.1110 | 0.0040 | 0.1868 | 5.87 | 369 |
| R-G | 6 | 0.1854 | -0.0007 | 3.52 | 3.49 | 2.13 | 0.0074 | 0.1702 | 0.0364 | 0.2644 | 3.53 | 366 |
| R-G07 | 1 | 0.1940 | 0.0057 | 0.68 | 0.75 | 0.63 | 0.5220 | 0.5140 | 0.5442 | 0.5200 | 0.41 | 155 |
| R-G07 | 3 | 0.1472 | 0.0057 | 0.99 | 1.17 | 0.87 | 0.4026 | 0.4660 | 0.4024 | 0.4766 | 0.73 | 153 |
| R-G07 | 6 | 0.0676 | 0.0056 | 0.72 | 0.76 | 0.49 | 0.5420 | 0.5716 | 0.6482 | 0.6240 | 0.31 | 150 |

Empirical size of the nominal 5% test: the share of replications with an absolute t-statistic above 1.96.

| id | h | NW iid | 1B iid | NW block | 1B block |
|---|---|---|---|---|---|
| R-base | 1 | 6.84% | 4.80% | 21.60% | 13.34% |
| R-base | 3 | 7.88% | 4.86% | 61.92% | 38.00% |
| R-base | 6 | 8.42% | 4.78% | 44.98% | 23.26% |
| R-full | 1 | 11.70% | 3.94% | 35.56% | 26.82% |
| R-full | 3 | 12.94% | 4.68% | 32.38% | 33.22% |
| R-full | 6 | 13.04% | 4.74% | 27.44% | 15.66% |
| R-D1 | 1 | 11.18% | 3.64% | 35.62% | 27.30% |
| R-D1 | 3 | 13.08% | 4.48% | 32.80% | 32.80% |
| R-D1 | 6 | 13.00% | 4.74% | 27.82% | 15.58% |
| R-G | 1 | 8.36% | 4.60% | 63.32% | 39.82% |
| R-G | 3 | 10.92% | 4.50% | 67.20% | 52.34% |
| R-G | 6 | 11.78% | 5.00% | 53.00% | 33.38% |
| R-G07 | 1 | 7.26% | 4.84% | 7.74% | 2.14% |
| R-G07 | 3 | 9.62% | 5.16% | 15.34% | 8.08% |
| R-G07 | 6 | 10.30% | 5.14% | 11.60% | 5.50% |

Hodrick standard error at h = 1 divided by the White HC0 standard error:

| id | ratio |
|---|---|
| R-base | 1.038 |
| R-full | 0.868 |
| R-D1 | 0.972 |
| R-G | 1.568 |
| R-G07 | 1.031 |

## Checks

### 1. Earlier coefficients and Newey–West t-statistics

Pass, to 1e-6, against the saved spec 001 and spec 002 tables. R-base matches the spec 001 main sample at h = 1, 3 and 6. R-full matches spec 001 sample R2, which is spec 002 row 0. R-G matches spec 002 "G full sample". R-G07 matches spec 002 "G 1990-2007". The reference values at h = 3 are R-base 0.4196 / 3.46, R-full 0.0908 / 0.70, and R-G 0.3256 / 5.23. R-D1 is a new sample. Its coef(B) and t(h−1) are the Part A D1 estimates, from the same `fit_hac` call.

### 2. Hodrick at h = 1 against White HC0

The null-residual ratio is inside [0.9, 1.1] for R-base, R-D1, R-G07. It is outside that band for R-full 0.868, R-G 1.568. All five ratios: R-base 1.038, R-full 0.868, R-D1 0.972, R-G 1.568, R-G07 1.031. The estimator was not replaced. u = e − mean(e) and the OLS residual differ when a month has a regressor far from its mean and a return that the fitted line absorbs. On R-G the largest such month is 2008-10: VRP_fwd is far below its mean and the next return is largely what that level predicts, so the OLS residual is small and the null residual is not. On every row, the same Hodrick sandwich with the OLS residual in place of the null residual reproduces the HC0 standard error to 1e-8. A simulated-data check does the same.

### 3. Bootstrap centering and the AR(1) slopes

The iid scheme passes. For every row and horizon, the mean of t*_NW under iid draws is inside ±0.15. The block scheme does not: R-base h=1 (1.28), R-base h=3 (2.32), R-base h=6 (1.83), R-full h=1 (0.81), R-full h=3 (1.14), R-full h=6 (0.76), R-D1 h=1 (0.80), R-D1 h=3 (1.14), R-D1 h=6 (0.75), R-G h=1 (2.52), R-G h=3 (2.89), R-G h=6 (2.12), R-G07 h=1 (0.68), R-G07 h=3 (0.90), R-G07 h=6 (0.58).

The block bootstrap draws consecutive pairs (ε_s, η_s). Inside a block, η at one month stays next to ε at the following month. That lead-lag correlation is the sample predictability, because ε is e − μ and not a residual from a regression on lagged x. With mean block length 6, about five sixths of the month-to-month steps keep the historical successor, so the block distribution sits near the sample coefficient rather than near 0. The iid scheme breaks that link and keeps only the contemporaneous Stambaugh correlation. The block p-values and the block rejection shares are computed exactly as specified. They are not null p-values or null sizes.

| id | ρ | mean t*_NW, iid, h=1 / 3 / 6 | mean t*_NW, block, h=1 / 3 / 6 |
|---|---|---|---|
| R-base | 0.174 | 0.028 / 0.060 / 0.089 | 1.283 / 2.317 / 1.828 |
| R-full | 0.129 | -0.065 / -0.033 / -0.018 | 0.809 / 1.145 / 0.759 |
| R-D1 | 0.129 | -0.053 / -0.046 / -0.035 | 0.802 / 1.145 / 0.753 |
| R-G | 0.341 | -0.057 / -0.037 / -0.027 | 2.517 / 2.888 / 2.121 |
| R-G07 | 0.304 | 0.042 / 0.060 / 0.084 | 0.681 / 0.899 / 0.581 |

R-full and R-D1 share the 1990–2025 VRP series, so they share ρ. Part A changes the regression rows, not the AR(1).

### 4. Part A drops, and nothing that remains touches C

Pass. The dropped rows are listed in Part A. For every kept regression row, at every horizon of D1, D2 and D3, none of t, t+1, …, t+h is in C. The h = 3 list on D1 and on D3 is the list named in the spec.

### 5. No look-ahead

Pass, on the spec 001 rule, for 6345 regression rows: R-base, R-full, R-D1, R-G, R-G07, D2 and D3, at h = 1, 3 and 6. D1 is R-D1 and is not counted twice. For each row t, the last daily-return date inside RV_t and the last VIX date inside IV_t fall in month t, and the later of those dates is on or before the month-end label. The first excess return inside Y_t is the return of month t+1, and the window runs through month t+h, inside that row's sample. VRP_fwd is the spec 002 real-time HAR forecast. This check does not reopen that construction.

## Flags at h = 3

The flags are independent. F1, F2 and F3 are properties of the named row. F4 is a property of D2, so it is the same fact in both columns. A flag uses the inequalities written in the spec: Hodrick |t| above 1.96, bootstrap p-values below 0.05, NW size above 8%, |DFBETA| above half of |coef(B)|, and the D2 t-statistic above 1.96. F1 and F2 use the block numbers as computed. Check 3 says the block distribution is not centered at 0, so those two flags' block legs are not null tests.

| flag | R-base | R-G |
|---|---|---|
| F1 survives conservative inference | no. t_1B = 2.22, p_NW iid = 0.0020, p_NW block = 0.1668 | no. t_1B = 2.78, p_NW iid = 0.0004, p_NW block = 0.1110 |
| F2 NW is oversized | iid no (7.88%); block yes (61.92%) | iid yes (10.92%); block yes (67.20%) |
| F3 single-row influence | no. Largest |DFBETA| / |coef(B)| = 0.151 at 1998-08 | no. Largest |DFBETA| / |coef(B)| = 0.206 at 2008-10 |
| F4 2020 explains the post-2010 failure | no. D2 t(h−1) = 0.93 | no. D2 t(h−1) = 0.93 |

F4 does not hold. On 2010–2025, after the overlap exclusion, the h = 3 Newey–West t-statistic is 0.93.

On the iid scheme, the one centered at 0, R-base and R-G at h = 3 have a Hodrick absolute t-statistic above 1.96 and a Newey–West bootstrap p-value below 0.05. F1 is no because the block p-value is above 0.05, and check 3 says that p-value is not a null p-value.

## Comparison with BTZ

The Hodrick t-statistic for R-base at h = 3 is 2.22. BTZ Table 2 reports 2.86. The gap is -0.64.

The gap is larger than 0.15, the distance fixed before the run as the reading of "close". The Hodrick t-statistic is not close to 2.86. No other standard-error estimator was computed.

## Deviations from this spec

1. **Check 2's 10% band does not hold for every row.** The Hodrick formula uses the null residual. That standard error is outside 10% of HC0 for R-full, R-G. The code asserts a different identity: the same sandwich with the OLS residual equals the HC0 standard error. That identity holds. The null residual was kept.

2. **Check 3 does not hold for the block scheme.** The iid mean of t*_NW is inside ±0.15 for every row and horizon. The stationary bootstrap on the pairs keeps the sample lead-lag between η and the next ε, so its mean tracks the sample t-statistic. Outside the band: R-base h=1, R-base h=3, R-base h=6, R-full h=1, R-full h=3, R-full h=6, R-D1 h=1, R-D1 h=3, R-D1 h=6, R-G h=1, R-G h=3, R-G h=6, R-G07 h=1, R-G07 h=3, R-G07 h=6. The resampling rule was not changed.

The spec left four choices implicit. They are fixed in the definitions above, and they are not further deviations: Hodrick's Z and S use the dates on which w_t is defined, and a Part A date is removed from that sum as well as from the regression; the AR(1) and μ use every month of the calendar window; the first bootstrap return is μ; and each replication builds one monthly sample, from which all three horizons are formed. The null-mean of coef(B) stored in `table_inference.csv` is the iid mean.

