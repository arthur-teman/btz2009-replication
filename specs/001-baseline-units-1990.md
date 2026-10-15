# Spec 001 — Baseline on the true BTZ sample (1990–2007), BTZ units

## Goal
Re-run the BTZ (2009) Table 2 predictive regressions on the paper's own sample
(1990:01–2007:12) and in the paper's units, so that coefficients, t-stats and R²
can be compared one-to-one with BTZ. Keep the current SPY-based results as a robustness column.

This spec does not touch the controls (Table 4) or the European extension: those are specs 003 and 004.

## Inputs
- Existing frozen files in `data/` — **do not overwrite or modify them**.
- New downloads, frozen once as CSV (record the download date in SUMMARY.md):
  - `data/gspc_data.csv`: Yahoo `^GSPC`, daily, 1989-12-01 → 2026-01-01.
  - `data/vix_1990_data.csv`: Yahoo `^VIX`, daily, 1989-12-01 → 2026-01-01.
- `data/tb3ms.csv` (FRED TB3MS, monthly average, percent per year).

## Exact definitions
All series are monthly, indexed at month end (`'ME'`). Month t = calendar month.

- Daily log return of the index: r_d = log(P_d / P_{d-1}), with P = `^GSPC` Close.
- **RV_t** = Σ_{d ∈ month t} r_d², with `min_count=1` (an empty month must give NaN, not 0).
- **IV_t** = VIX²_{last trading day of t} / 12 (VIX in percent).
- Units: express IV, RV and VRP in **monthly percent squared**. Example: VIX = 20 gives IV = 400/12 = 33.3.
  The current notebook uses decimal units; this is a factor of 10⁴.
- **VRP_t** = IV_t − RV_t.
- Monthly index log return r_t = log(P_{last day of t} / P_{last day of t−1}).
- Risk-free rate: **rf_t = TB3MS_{t−1} / 1200**. Use the previous month's average, which is known at the start of month t.
- Excess return e_t = r_t − rf_t.
- Dependent variable, two scalings reported side by side:
  - (A) decimal monthly: (1/h) Σ_{j=1..h} e_{t+j}
  - (B) annualized percent: 1200 × (A)
- Regression: Y_t(h) = a + b · VRP_t + u. Horizons h ∈ {1, 3, 6, 9, 12, 24}.
- Standard errors: Newey–West HAC. Report two lag choices: maxlags = h−1 (min 1) and maxlags = h.
  Arthur will check which one BTZ use in the Table 2 notes.
- Main sample: VRP_t for t from 1990-01 to 2007-12. Returns may not go beyond 2007-12, so the h-month windows are truncated at the end of the sample.

## Robustness columns (same code, different inputs)
- R1: SPY adjusted close (existing `data/spy_data.csv`) for returns and RV, 1993-02 → 2007-12. This is the current notebook's sample, now in the new units and with lagged rf.
- R2: `^GSPC`, full sample 1990-01 → 2025-12.

## Checks (assert in code, report in SUMMARY.md)
1. No look-ahead: for every regression row t, the max date of the data used in X_t is ≤ month t, and the first return in Y_t is month t+1.
2. Number of VRP observations, 1990-01 → 2007-12 = 216. Regression N = 216 − h.
3. No RV equal to 0. No NaN inside the sample.
4. Means and standard deviations of IV, RV and VRP (in %²) over 1990–2007.
   For reference, the audit found on 1993–2007: IV ≈ 33.9, RV ≈ 23.9, VRP ≈ 10.0.
   Arthur will compare with BTZ Table 1.
5. Sanity check: the decimal-unit coefficient × 10⁴ / 1200 must equal the coefficient of spec (B).

## Targets (BTZ 2009, Table 2, as entered in the notebook)
| h | coef | t | R² (%) |
|---|------|------|-------|
| 1 | 0.39 | 1.76 | 1.07 |
| 3 | 0.47 | 2.86 | 6.82 |
| 6 | 0.30 | 2.15 | 5.42 |
| 12 | 0.12 | 1.00 | 1.23 |

Expected outcome:
- coefficient (B) at h=3 of the order of 0.25–0.35 (the audit found 0.28 on 1993–2007);
- hump shape preserved, with a maximum at h=3 or h=6;
- t(h=3) > 1.96.

The R² will stay below BTZ because the RV is computed from daily rather than 5-minute returns. Do not try to close this gap.

## Expected outputs
In `results/001-baseline-1990/`:
- `table1_summary.csv`: mean, sd, min, max and AR(1) of IV, RV, VRP and e, for each sample.
- `table2_main.csv`: for each h, coef(A), coef(B), t(h−1), t(h), R², N. One block per sample (main, R1, R2).
- `fig_hump.png`: t-stat and R² against h for the main sample, with the BTZ targets as markers.
- `SUMMARY.md`, containing:
  - the data vintages (download dates);
  - the results of checks 1–5;
  - the comparison table against the targets;
  - every deviation from this spec, with its reason.

Code goes in the notebook section 2 (or in a `src/` module called by the notebook). Commit message: `001 baseline 1990-2007 in BTZ units`.
