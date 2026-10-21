# Fix 003a — Null-imposing bootstrap, matched-sample rows, horse race

Context: spec 003 is pushed (commits bf864fd and 708a8f9). Two problems remain.

1. The stationary block bootstrap did not impose the null of no predictability. Consecutive pairs keep the lead-lag between the VRP shock at s and the return at s+1, so the block distribution is centred near the sample t-statistic, not at 0.
2. A quick check run with `src/inference.py` functions shows that VRP and VRP_fwd work in **different sub-periods**. The spec 003 rows do not show this, because they compare the two measures on different samples (R-base is 1990–2007, R-G is 1995–2025).

Do not change the spec 001, 002 or 003 pipelines. Add to them.

## 1. Replace the block scheme with a wild bootstrap
For each replication, draw w_s iid Rademacher (±1 with probability 1/2) for every date s.

Set ε*_s = w_s·ε_s and η*_s = w_s·η_s. Both members of the pair get the same weight, and the original date order is kept. Build x* and e* exactly as in spec 003, C3.

This keeps the heteroskedasticity and the contemporaneous ε–η correlation, and removes every lead-lag between the two.

Settings: 5,000 replications, `numpy.random.default_rng(20261021)`.

Remove the block columns from the tables and replace them with wild columns: p_boot_NW_wild, p_boot_1B_wild, size_NW_wild, size_1B_wild.

Re-evaluate F1 with (i) iid and (ii) wild.

## 2. Replace check 2 of spec 003
New unit test: the 1B sandwich computed with the **OLS** residual equals White HC0 to 1e-6.

Report the null-residual 1B/HC0 ratio for information only. A ratio different from 1 is expected, because 1B uses the null residual by construction.

## 3. Matched-sample rows
Run these rows through C1–C3 (NW h−1, NW 2h, Hodrick 1B, iid and wild bootstrap) at h ∈ {1, 3, 6}.

| id | regressor | sample |
|---|---|---|
| V-95-07 | VRP | 1995-01 → 2007-12 |
| G-95-07 | VRP_fwd | 1995-01 → 2007-12 (= R-G07) |
| V-08-25 | VRP | 2008-01 → 2025-12 |
| G-08-25 | VRP_fwd | 2008-01 → 2025-12 |
| V-10-25 | VRP | 2010-01 → 2025-12 |
| G-10-25 | VRP_fwd | 2010-01 → 2025-12 |
| V-10-19 | VRP | 2010-01 → 2019-12 |
| G-10-19 | VRP_fwd | 2010-01 → 2019-12 |
| R-D3 | VRP_fwd | 1995–2025, spec 003 Part A exclusion |

Targets at h=3 from the quick check. Reproduce them to ±0.01 on the coefficient and ±0.02 on t:

| id | coef(B) | t NW(h−1) | t 1B | R² % | N |
|---|---|---|---|---|---|
| V-95-07 | 0.373 | 2.82 | 1.81 | 4.24 | 153 |
| G-95-07 | 0.147 | 0.99 | 0.87 | 0.73 | 153 |
| V-08-25 | 0.067 | 0.51 | 0.92 | 1.06 | 213 |
| G-08-25 | 0.393 | 5.02 | 2.75 | 10.63 | 213 |
| V-10-25 | −0.072 | −1.59 | −1.12 | 1.58 | 189 |
| G-10-25 | 0.401 | 3.01 | 1.91 | 5.52 | 189 |
| V-10-19 | 0.166 | 1.67 | 1.07 | 1.65 | 117 |
| G-10-19 | 0.218 | 1.51 | 0.86 | 2.01 | 117 |

The quick check computed the Hodrick t with `hodrick_t` on positions t = h−1 … n−2 of each subsample. If spec 003's `hod_pos` rule gives different positions, keep spec 003's rule and report the difference.

## 4. Horse race (joint regression)
Regression: Y_t(h) = a + b1·VRP_t + b2·VRP_fwd_t + u, at h ∈ {1, 3, 6}.

Samples:
- 1995–2007;
- 2008–2025;
- 1995–2025.

Report b1 and b2, their NW(h−1) and Hodrick 1B t-statistics, R², N, and the correlation between VRP and VRP_fwd on each sample.

Hodrick 1B in the multivariate case uses the same formula, with z_t = (1, VRP_t, VRP_fwd_t)'. Generalize `hodrick_t` without changing its univariate output. Unit test: with x2 = 0·x1 removed, the univariate result is unchanged.

No bootstrap for the joint regression.

## 5. Decomposition of the difference
VRP_fwd − VRP = RV_t − Ê_t[RV_{t+1}]. Call it the mean-reversion term MR_t.

Report:
- the mean, standard deviation and 1st/99th percentiles of MR_t on 1995–2007 and on 2008–2025;
- the 10 months with the largest |MR_t|.

## Reading rules, fixed before the run
These are independent flags. Report each one, with its numbers, at h=3.

- **H1 Sub-period split**: VRP has Hodrick |t| > 1.96 or iid p < 0.05 on 1995–2007, AND VRP_fwd has both on 2008–2025, AND the reverse pairs (VRP_fwd on 1995–2007, VRP on 2008–2025) have neither.
- **H2 G holds post-2010**: G-10-25 has Hodrick |t| > 1.96, iid p < 0.05 and wild p < 0.05.
- **H3 Calm decade**: neither V-10-19 nor G-10-19 has Hodrick |t| > 1.96.
- **H4 Horse race**: on 1995–2025, the Hodrick |t| of b2 is > 1.96 and that of b1 is ≤ 1.96.

If a flag fails, report the numbers. Do not add variants to rescue it.

## Outputs
Add to `results/003-inference/`:
- the new rows in `table_inference.csv` and `table_size.csv`;
- `table_horse_race.csv`;
- `table_mr.csv`;
- `fig_subperiod_h3.png`: Hodrick t at h=3 for VRP and VRP_fwd, side by side, for each sub-period, with a 1.96 line;
- a "003a" section in SUMMARY.md, containing:
  - the new unit test;
  - the matched-sample table compared against the targets above;
  - flags H1–H4;
  - deviations.

Commit as `003a wild bootstrap, matched samples, horse race`, then push.
