# Spec 004 — BTZ Table 4 with real-time controls, for VRP and VRP_fwd

## Goal
Re-run the BTZ (2009) Table 4 robustness check: does the VRP survive the standard return predictors?

Two changes compared with notebook section 4:

1. **Real-time timing.** The notebook uses the Shiller earnings and CAY as if they were known at the end of month t. In reality:
   - earnings are published with a delay;
   - CAY for a quarter is released months after the quarter ends.

   Each control is now aligned on what an investor could know at the end of month t. The notebook timing is kept as a comparison column.
2. **Both VRP measures.** Specs 003/003a showed:
   - the raw VRP predicts before 2008;
   - the forecast-based VRP_fwd predicts after 2008 and wins the horse race on 1995–2025.

   The controls are therefore run for each measure on the sub-period where it works, and on the full sample.

Do not change the 001–003a pipelines. Reuse their panel, units, VRP, VRP_fwd and the multivariate Hodrick 1B function from `src/inference_003a.py`.

## Inputs
- Spec 001 panel: `^GSPC`, `data/vix_1990_data.csv`, TB3MS lagged one month for rf. Units are BTZ units: VRP and VRP_fwd in monthly %², dependent variable in annualized %.
- VRP_fwd from spec 002 (real-time HAR, defined from 1995-01).
- Existing frozen files: `data/baa.csv`, `data/aaa.csv`, `data/gs10.csv`, `data/tb3ms.csv`, `data/cay_current.csv`.
- **New download, frozen once**: Robert Shiller's `ie_data.xls` → `data/shiller_2026.csv`, with columns date, P, E. The existing `data/shiller_data.csv` stops in 2023-06; do not modify it. Record the download date.
  - Parsing trap: the Shiller date column is a decimal year.month, and October is written `.1`, not `.10`. Parse it so that 1990.1 → 1990-10 and 1990.01 → 1990-01.
  - Rows with an empty E at the end of the file are dropped.

## Exact definitions (month t = calendar month, month-end label)
Every control has two timings: **RT** (real-time, the main specification) and **NB** (notebook timing, comparison only).

| control | RT definition | NB definition |
|---|---|---|
| log P/E | log(P_t / E_{t−3}): Shiller P (monthly average price), Shiller E (12-month trailing earnings) lagged 3 months | log(P_t / E_t) |
| DFSP | BAA_t − AAA_t (FRED monthly averages of month t) | same |
| TMSP | GS10_t − TB3MS_t | same |
| RREL | TB3MS_t − mean(TB3MS_{t−11} … TB3MS_t) | same |
| CAY | value of quarter q (q ending in month m) assigned to months m+3, m+4, m+5, until the next quarter takes over | value of quarter q assigned to months m, m+1, m+2 (forward fill from the quarter end, as in the notebook) |

DFSP, TMSP and RREL are monthly averages published in the first days of month t+1. They are treated as known at t in both timings. Write this as a limitation in SUMMARY.md.

CAY limitation (cannot be fixed with these data): the cointegrating coefficients of `cay_current.csv` are estimated on the full sample to 2019. CAY ends in 2019Q3, so with RT timing the last usable month is 2019-12.

## Regressions
Horizon h = 3 only (Table 4 is a quarterly-horizon table). Dependent variable: Y_t(3) as in spec 001, specification (B).

For each block below, and for each timing (RT, NB):
- (a) the VRP measure alone, on the block's common sample;
- (b) each control alone;
- (c) the VRP measure + one control (5 regressions);
- (d) the VRP measure + all five controls.

Every regression in a block uses the **same rows**: the intersection of the months where all variables of the block are defined. CAY limits the end of the sample, so every block that ends after 2019 is run twice: once with CAY (ending at the last RT CAY month) and once without CAY (ending 2025-12).

| block | VRP measure | sample |
|---|---|---|
| T4-base | VRP | 1990-01 → 2007-12 (BTZ sample) |
| T4-95 | VRP and VRP_fwd, separately | 1995-01 → 2007-12 |
| T4-post | VRP_fwd | 2008-01 → 2025-12 (CAY version ends 2019-12) |
| T4-full | VRP_fwd | 1995-01 → 2025-12 (CAY version ends 2019-12) |

Report for every regression: the coefficient and t-statistic of each regressor, with NW (maxlags h−1) and Hodrick 1B (multivariate version from 003a); R²; N.

## Checks (assert in code, report in SUMMARY.md)
1. Shiller parsing: no duplicate dates, 12 months per year from 1990 to 2025, and October present every year. Print the rows for 1990-09, 1990-10 and 1990-11.
2. RT timing: for every row t and every control, the latest source date used is ≤ t. For CAY RT, the quarter used ends at least 3 months before t. For E RT, the earnings month is t−3.
3. Row (a) of T4-base equals the spec 001 main regression at h=3 when run on the same rows. If CAY does not shorten T4-base, coef(B) = 0.4196 and t NW = 3.46.
4. The rows are identical across all regressions of a block (assert on the index).
5. No NaN inside any block. N is reported for every block and variant.

## Targets
**BTZ Table 4.** Arthur fills this table from the PDF before handing the spec to Cursor. Leave the cells empty if not filled.

| regression (BTZ Table 4, h=3) | coef VRP | t VRP | coef control | t control | R² |
|---|---|---|---|---|---|
| VRP + log P/E | | | | | (16.76 per the notebook) |
| VRP + DFSP | | | | | |
| VRP + TMSP | | | | | |
| VRP + RREL | | | | | |
| VRP + CAY | | (3.02 per the notebook) | | | |

**Audit reference** (SPY, 1993-12 → 2008-01, NW, old units). These cannot be compared directly, because of the different units, sample and index. They indicate the direction only:
- VRP + log P/E: R² 17.1 % with NB timing, 13.6 % with E lagged 3 months;
- VRP + CAY: R² 8.6 % with NB timing, 5.7 % with CAY lagged one quarter.

## Reading rules, fixed before the run (independent flags, at h=3, RT timing, Hodrick 1B)
- **K1 BTZ robustness**: in T4-base, VRP keeps Hodrick |t| > 1.96 in each of the five one-control regressions (c).
- **K2 Timing inflates the controls**: for log P/E and CAY in T4-base, R²(VRP + control) under NB exceeds R² under RT by more than 2 percentage points.
- **K3 Post-2008 robustness**: in T4-post, VRP_fwd keeps Hodrick |t| > 1.96 in each one-control regression (c), in the version without CAY and, for the CAY row, in the version ending 2019-12.
- **K4 Kitchen sink**: in the all-controls regression (d), Hodrick |t| > 1.96 for VRP in T4-base and for VRP_fwd in T4-post (version without CAY).

If a flag fails, report the numbers. Do not change the timing or the controls to rescue it.

## Expected outputs
In `results/004-controls/`:
- `table4_rt.csv` and `table4_nb.csv`: one row per (block, variant, regression), with columns for every regressor's coef, t NW and t 1B, plus R² and N.
- `table4_timing_compare.csv`: for T4-base, R² and the VRP t-statistics, RT against NB, regression by regression.
- `fig_r2_controls.png`: R² of control alone vs VRP + control, under RT, for T4-base and T4-post.
- `SUMMARY.md`, containing:
  - the data vintage of the new Shiller file;
  - checks 1–5;
  - the RT tables for T4-base and T4-post, compared with the BTZ targets where filled;
  - flags K1–K4;
  - the timing limitations (DFSP/TMSP/RREL publication, full-sample CAY);
  - deviations from this spec.

Code: a new module `src/controls.py`, called from a new notebook section 4.2.
Commit message: `004 Table 4 with real-time controls`. Push when done.
