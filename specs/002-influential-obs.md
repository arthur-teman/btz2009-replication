# Spec 002 — Why the full-sample result collapses: influential observations

## Goal
Spec 001 showed that VRP predicts h=3 returns on 1990–2007 (coef(B) = 0.42, t = 3.46),
but not on 1990–2025 (R2: coef(B) = 0.09, t = 0.70).

Test the hypothesis that this collapse is driven by a few months where VRP is extremely negative
(2008-10, 2020-03), rather than by a lasting change in the relationship.

Every robustness version must also exist in a **real-time** form, so that no fix uses future information.

## Inputs
- The pipeline and data from spec 001 (`src/baseline_1990.py`), unchanged.
  - Index: `^GSPC`. Implied volatility: `data/vix_1990_data.csv`. Risk-free rate: TB3MS lagged by one month.
  - Units: BTZ units, VRP in monthly %², specification (B) in annualized %.
- Sample: 1990-01 → 2025-12 (the R2 sample of spec 001).
- Horizons: h ∈ {1, 3, 6}.
- Newey–West standard errors with two lag choices: maxlags = max(1, h−1) and maxlags = h.

## Exact definitions

### Row 0 — Baseline
The R2 regression of spec 001, unchanged. It must reproduce h=3: coef(B) 0.0908, t 0.70.

### A. Influence diagnostics (h = 3)
On the row-0 OLS, compute for each regression row t:
- the leverage h_tt;
- Cook's distance;
- the DFBETA of the VRP coefficient, i.e. the change in coef(B) when row t alone is deleted.

Report the 10 rows with the largest |DFBETA|, with columns: date, VRP_t, Y_t, DFBETA, Cook's D.

### B. Winsorization of VRP, full-sample cut points
Clip VRP at its 1st/99th percentiles, then separately at its 2.5th/97.5th percentiles, computed on 1990–2025.
This uses future information. Label it "in-sample" everywhere.

### C. Winsorization of VRP, real-time cut points
At each month t, the 1st/99th percentiles are computed on VRP_1990-01 … VRP_t only (expanding window).
- Burn-in: the first 60 months; rows before 1995-01 are dropped from this regression only.
- Report N.

### D. Exclusion of crisis rows
Drop the regression rows t whose regressor month falls in {2008-09 … 2008-12, 2020-02 … 2020-04}.
Report which rows are dropped and N.

### E. Subsamples
Same regression on:
- 1990–2007;
- 2008–2025;
- 2010–2025.

The h-month windows are truncated at the end of each subsample, as in spec 001.

### F. Rolling stability
Run the h=3 regression on a rolling window of 120 months, stepping one month at a time.
Store coef(B) and t(h−1), dated at the window's last regressor month.

### G. VRP with forecasted RV (real-time HAR)
VRP_fwd_t = IV_t − Ê_t[RV_{t+1}], in monthly %².

Ê_t[RV_{t+1}] comes from a monthly HAR regression, re-estimated by expanding-window OLS using data up to t only:

RV_{s+1} = c + β_m · RV_s + β_w · RV^w_s + β_d · RV^d_s + ε

where:
- RV_s is the month-s realized variance from spec 001;
- RV^w_s = (sum of squared daily log returns over the last 5 trading days of month s) × 22/5 × 10⁴;
- RV^d_s = (squared daily log return of the last trading day of month s) × 22 × 10⁴.

Rules:
- The fit used at month t uses only the pairs (s, s+1) with s+1 ≤ t.
- Burn-in: 60 months.
- Predictions below 0 are floored at 0, and the number of floored months is reported.

Then run the row-0 regression with VRP_fwd in place of VRP, on the full sample and on 1990–2007.

## Checks (assert in code, report in SUMMARY.md)
1. Row 0 reproduces spec 001 R2 at h=3: coef(B) 0.0908, t(h−1) 0.70.
2. Real-time cut points (C): for every t, the percentiles use no VRP dated after t.
   Print the cut points at 1999-12, 2007-12, 2008-12 and 2020-12.
3. HAR (G): for every t, the last pair used in the fit has s+1 ≤ t.
   Print the HAR coefficients at 2007-12 and 2025-12, and the correlation between VRP_fwd and VRP.
4. N is reported for every row of the robustness table.
5. Same look-ahead check as spec 001, check 1, applied to every variant.

## Targets and hypotheses
Reference value from the audit (SPY, 1993–2025, full-sample 1/99 winsorization): h=3 t = 2.11, R² = 4.2 %.

How to read the outcome at h=3:
- **Outlier story confirmed**: B, C and D all give t > 1.96, and the 2008/2020 months top the DFBETA list.
- **Fragile**: only the in-sample winsorization (B) restores significance.
- **Lasting change in the relationship**: 2010–2025 remains insignificant even after C and D, and the rolling t stays low after 2010.

Report which of these three cases holds. Do not tune choices until a significant result appears.

## Expected outputs
In `results/002-influential-obs/`:
- `table_influence.csv`: the top 10 rows by |DFBETA| (part A).
- `table_robust.csv`:
  - one row per variant: 0, B 1/99, B 2.5/97.5, C, D, E×3, G full sample, G 1990–2007;
  - for h = 1, 3, 6: coef(B), t(h−1), t(h), R², N.
- `fig_rolling_t.png`: the rolling coef(B) and t at h=3, with a 1.96 line and shaded 2008 and 2020.
- `fig_scatter_h3.png`: Y_t(3) against VRP_t, with the 10 highest-DFBETA months labelled.
- `fig_vrp_vs_vrpfwd.png`: VRP and VRP_fwd over time.
- `SUMMARY.md`:
  - checks 1–5;
  - the robustness table;
  - which of the three cases holds;
  - deviations from this spec.

Code: a new module `src/influence.py` called from a new notebook section 2.4.
Commit message: `002 influential observations and real-time robustness`.
