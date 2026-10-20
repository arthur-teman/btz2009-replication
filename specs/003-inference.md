# Spec 003 — Inference: do the VRP results survive conservative standard errors?

## Goal
Spec 002 left three open points:

1. The forecast-based VRP (variant G) gives h=3 t = 5.23 over 1995–2025. This is above every other result, including the BTZ baseline. Newey–West with h−1 Bartlett lags is known to over-reject in overlapping regressions with a persistent regressor (Ang & Bekaert, 2007). Before this number goes into the report, it must be checked with conservative inference.
2. Spec 001 found t-statistics above BTZ (h=3: 3.46 vs 2.86). A more conservative standard error may explain the gap.
3. Variant D (crisis exclusion) only dropped rows whose **regressor** month was in a crisis. The rows dated 2019-12 and 2020-01 still contain the March 2020 crash in their dependent variable. D must be redone.

This spec does not change the spec 001 or spec 002 pipelines. It reuses them.

## Inputs
- `src/baseline_1990.py` (spec 001) and `src/influence.py` (spec 002), unchanged.
- Units: BTZ units. VRP and VRP_fwd in monthly %², dependent variable in annualized % (specification B).
- Horizons: h ∈ {1, 3, 6}.

## Part A — Fix of variant D (window-overlap exclusion)
Crisis months: C = {2008-09 … 2008-12, 2020-02 … 2020-04}.

Drop regression row t if **any** of the months t, t+1, …, t+h is in C. For h=3, this removes the rows dated 2008-06 … 2008-12 and 2019-11 … 2020-04.

Run on:
- D1: 1990–2025, regressor VRP.
- D2: 2010–2025, regressor VRP.
- D3: 1995–2025, regressor VRP_fwd.

List the dropped rows and report N.

## Part B — Complete variant G
1. Report G on 1995-01 → 2007-12. The HAR burn-in starts the sample in 1995. Report it with the same columns as the spec 002 table.
2. Influence on G (full sample, h=3): list the top 5 rows by |DFBETA| (date, VRP_fwd_t, Y_t, DFBETA, Cook's D). DFBETA = coef(B) − coef(B) with row t deleted, as in spec 002.
3. Report VRP and VRP_fwd at 2008-10, 2020-03 and 2025-04, side by side.
4. Report the AR(1) coefficient of VRP and of VRP_fwd over 1995–2025.

## Part C — Inference
Rows to test (all at h ∈ {1, 3, 6}):

| id | regressor | sample |
|---|---|---|
| R-base | VRP | 1990–2007 (spec 001 main) |
| R-full | VRP | 1990–2025 (spec 002 row 0) |
| R-D1 | VRP | 1990–2025, Part A exclusion |
| R-G | VRP_fwd | 1995–2025 |
| R-G07 | VRP_fwd | 1995–2007 |

Report every one of these statistics for each row.

### C1. Newey–West
- maxlags = max(1, h−1): already computed. Report it again here.
- maxlags = 2h: new.

### C2. Hodrick (1992) 1B standard errors
Notation:
- e_{t+1} is the monthly excess return in annualized % (1200 × decimal);
- u_{t+1} = e_{t+1} − mean(e) over the sample (the residual under the null of no predictability);
- z_t = (1, x_t)', where x_t is the regressor.

For the regression on the h-month **sum** of returns:
- w_t = u_{t+1} · Σ_{j=0}^{h−1} z_{t−j}, for every t with t−h+1 ≥ the first sample month and t+1 ≤ the last return month;
- S = (1/T) Σ_t w_t w_t';
- Z = (1/T) Σ_t z_t z_t';
- V_sum = Z⁻¹ S Z⁻¹ / T.

Our dependent variable is the h-month **average**, so V_avg = V_sum / h². The t-statistic is coef(B) / sqrt(V_avg[1,1]).

Unit test: at h=1, the 1B standard error must be within 10% of the White (HC0) standard error from statsmodels. Report the ratio.

### C3. Bootstrap under the null of no predictability
For each row:
- Fit under the null: μ = mean(e); ε_t = e_t − μ. Fit an AR(1) for the regressor: x_t = c + ρ x_{t−1} + η_t (OLS on the row's sample).
- Draw residuals in **pairs** (ε_s, η_s) with the same date s, to keep their contemporaneous correlation (the Stambaugh bias channel).
- Build the artificial sample: x*_1 = x_1, then x*_t = c + ρ x*_{t−1} + η*_t, and e*_t = μ + ε*_t. Then form Y*(h) exactly as in spec 001 and run the same regression.
- Two resampling schemes:
  - (i) iid draws of the pairs;
  - (ii) stationary block bootstrap (Politis–Romano) on the pairs, with mean block length 6.
- 5,000 replications per scheme, `numpy.random.default_rng(20261021)`.
- For each replication, store coef(B), the NW t-stat (maxlags h−1) and the Hodrick 1B t-stat.

Outputs per row:
- the two-sided bootstrap p-value of the observed NW t: share of |t*_NW| ≥ |t_NW, obs|;
- the same p-value for Hodrick;
- the null-mean of coef(B), i.e. the Stambaugh bias;
- the **empirical size** of the nominal 5% tests: share of |t*_NW| > 1.96 and share of |t*_1B| > 1.96.

## Checks (assert in code, report in SUMMARY.md)
1. Every row reproduces its earlier coef(B) and NW t(h−1) to 1e-6. Reference values: R-base h=3 = 0.4196 / 3.46; R-full h=3 = 0.0908 / 0.70; R-G h=3 = 0.3256 / 5.23.
2. Hodrick at h=1: ratio to HC0 within [0.9, 1.1].
3. Bootstrap: the null-mean of t*_NW is within ±0.15 of 0. Report the AR(1) ρ used for each row.
4. Part A: the dropped rows are listed, and none of the remaining rows has a month of C in t…t+h.
5. Same no-look-ahead check as spec 001, check 1, on every row.

## Reading rules, set before seeing the results
These are **independent flags**, not exclusive cases. Report each flag for R-base and R-G at h=3.

- **F1 Survives conservative inference**: Hodrick |t| > 1.96, and the bootstrap p-value of the NW t is < 0.05 under both (i) and (ii).
- **F2 NW is oversized**: the empirical size of the nominal 5% NW test is > 8%.
- **F3 Single-row influence**: one row moves coef(B) by more than 50% of its full-sample value (|DFBETA| > 0.5·|coef|).
- **F4 2020 explains the post-2010 failure**: R-D2 at h=3 has t(h−1) > 1.96. If it does not, report its t.

Hypothesis to check, not a target: Hodrick t for R-base at h=3 is close to the BTZ value (2.86). If it is, report it. If it is not, do not try other SE types to reach it.

## Expected outputs
In `results/003-inference/`:
- `table_inference.csv`: one row per (row id, h), with columns coef(B), null-mean coef, t_NW(h−1), t_NW(2h), t_1B, p_boot_NW (iid), p_boot_NW (block), p_boot_1B (iid), p_boot_1B (block), R², N.
- `table_size.csv`: empirical size of the NW and 1B tests per row and h.
- `table_D_fixed.csv` and `table_G_influence.csv`.
- `fig_boot_h3.png`: histograms of t*_NW under the null for R-base and R-G at h=3, with the observed t marked.
- `SUMMARY.md`, containing:
  - checks 1–5;
  - flags F1–F4 for R-base and R-G;
  - the comparison with BTZ (h=3 t = 2.86);
  - deviations from this spec.

Code: a new module `src/inference.py`, called from a new notebook section 2.5.
Commit message: `003 conservative inference and fixed crisis exclusion`.
