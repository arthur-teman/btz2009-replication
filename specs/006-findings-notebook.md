# Spec 006 — Full Table 2 in the paper's statistics, and the findings notebook

## Goal
Two parts.

- **Part A** completes the replication of BTZ Table 2 with the paper's own statistics. Arthur read the Table 2 notes:
  - the t-statistics are **Hodrick (1992)**;
  - the fit statistic is the **adjusted R²**;
  - the horizons are h ∈ {1, 3, 6, 9, 12, 15, 18, 24}.
  
  Spec 001 used plain R², Newey–West t and no h = 15, 18.
- **Part B** builds `findings.ipynb`: one notebook that lists every finding of specs 001–006. Each finding is a claim followed by its evidence, and every number quoted in a claim is **asserted** against the results files. The LaTeX report will be written later from this notebook.

Do not change the 001–005 modules or their results. Part A is a new module that calls them.

## Part A — Table 2 in the paper's statistics

### Definitions
Main sample, data, units and dependent variable: all exactly as in spec 001.
- Data: ^GSPC with `data/vix_1990_data.csv`, 1990-01 → 2007-12.
- Units: BTZ units.
- Risk-free rate: TB3MS lagged one month.
- Dependent variable: Y_t(h) in annualized %.

Horizons: h ∈ {1, 3, 6, 9, 12, 15, 18, 24}, so N = 216 − h.

For each h, report:
- the constant and the VRP coefficient;
- Hodrick 1B t-statistics for both, using `src/inference.py` `hodrick_t` (the constant's t needs the [0,0] element; generalize without changing the existing output);
- Newey–West t with max(1, h−1) lags;
- R² and **adjusted R²** = 1 − (1 − R²)(N − 1)/(N − 2);
- N.

Add a row for the ratio t1B(mine)/t1B(BTZ), next to sqrt(adjR²(mine)/adjR²(BTZ)), at h = 1, 3, 6.

### Targets (BTZ Table 2, Hodrick t in parentheses in the paper)
| h | constant | t | VRP coef | t | adj. R² % |
|---|---|---|---|---|---|
| 1 | −0.55 | −0.13 | 0.39 | 1.76 | 1.07 |
| 3 | −2.08 | −0.56 | 0.47 | 2.86 | 6.82 |
| 6 | 1.12 | 0.33 | 0.30 | 2.15 | 5.42 |
| 9 | 3.63 | 1.15 | 0.17 | 1.36 | 2.30 |
| 12 | 4.62 | 1.50 | 0.12 | 1.00 | 1.23 |
| 15 | 4.84 | 1.59 | 0.11 | 0.94 | 1.00 |
| 18 | 5.61 | 1.81 | 0.06 | 0.56 | 0.05 |
| 24 | 6.48 | 2.07 | 0.01 | 0.11 | −0.50 |

Quick-check values Cursor must reproduce, to ±0.005 on the coefficient and ±0.02 on t and adjusted R²:

| h | coef | t1B | t NW | adj R² % | N |
|---|---|---|---|---|---|
| 1 | 0.388 | 1.44 | 1.63 | 0.90 | 215 |
| 3 | 0.420 | 2.22 | 3.46 | 4.53 | 213 |
| 6 | 0.288 | 1.86 | 2.98 | 4.46 | 210 |
| 9 | 0.165 | 1.24 | 1.60 | 1.87 | 207 |
| 12 | 0.134 | 1.10 | 1.42 | 1.39 | 204 |
| 15 | 0.110 | 0.97 | 1.18 | 0.96 | 201 |
| 18 | 0.093 | 0.86 | 1.12 | 0.67 | 198 |
| 24 | 0.056 | 0.56 | 1.01 | −0.02 | 192 |

### Checks
1. h ∈ {1, 3, 6, 9, 12, 24} reproduce `results/001-baseline-1990/table2_main.csv` (coef, NW t, R²) to 1e-8.
2. h ∈ {1, 3, 6} reproduce the R-base Hodrick t of `results/003-inference/table_inference.csv` to 1e-8.
3. The constant's Hodrick t uses the same w and Z as the slope (matrix element [0,0]). Unit test: at h = 1, with the OLS residual, it equals HC0 to 1e-10.

### Outputs
`results/006-table2-full/`:
- `table2_full.csv`;
- `fig_table2_vs_btz.png`: two panels (coefficient vs h; Hodrick t vs h), mine against BTZ, with a 1.96 line;
- `SUMMARY.md`.

Module: `src/table2_full.py`.

## Part B — `findings.ipynb` (repo root)

### Rules
- The notebook **reads** `results/*/` files. It does not re-estimate anything, except Part A's module call, if that is faster than reading its CSV.
- Restart & Run All must finish in under 2 minutes.
- Every finding has the same block:
  1. a markdown cell with the finding ID, the **claim text given below, copied verbatim**, and a "Source" line (spec, commit, results file);
  2. a code cell that loads the evidence and **asserts every number quoted in the claim**. Tolerance: the number as written, i.e. ±0.5 in its last printed digit. A failing assertion must stop the notebook;
  3. the table and/or the figure named for that finding (save figures to `results/006-findings/`).
- If a number in a claim text does not match the results, do not change the text: report it under Deviations in `SUMMARY.md`. Arthur and Claude will fix the claim.
- Formatting:
  - every table is shown with `pandas` `.style.format` to 2 decimals (3 for coefficients);
  - every figure has axis labels with units and a source line in the caption cell.
- At the end, export `results/006-findings/findings.html` with `jupyter nbconvert --to html --execute`.

### Header cell (markdown, verbatim)
```markdown
# Bollerslev, Tauchen & Zhou (2009) — replication findings

Arthur Teman, M1 QEF (HEC Paris). Code: github.com/arthur-teman/btz2009-replication, specs 001–006.

Conventions used everywhere below: S&P 500 index (^GSPC), VIX from Yahoo, realized variance from daily log returns (the paper uses 5-minute returns), variances in monthly percent squared (VIX²/12), returns in annualized percent, risk-free rate = previous month's 3-month T-bill. The main statistic is the Hodrick (1992) 1B t-statistic, as in the paper. Every number in the text is asserted against the results files when the notebook runs.
```

### Findings (claim texts, verbatim)

**F1 — Table 2 is reproduced.**
> On the paper's sample (January 1990 – December 2007), the variance risk premium coefficients are within 0.05 of BTZ Table 2 at every horizon up to 15 months (h = 3: 0.42 vs 0.47). Predictability is hump-shaped and peaks at the quarterly horizon, both for the coefficient and the Hodrick t-statistic (h = 3: 2.22 vs 2.86 in the paper). From 18 months on, neither the paper's estimates nor mine are significant.

Evidence: `table2_full.csv` (Part A), `fig_table2_vs_btz.png`. The coefficient assertion is max |coef − BTZ| ≤ 0.05 for h ≤ 15.

**F2 — The gap in t-statistics is the gap in fit, as expected from a noisier realized variance.**
> My Hodrick t-statistics are below the paper's at short horizons (h = 1, 3, 6: ratios 0.82, 0.78, 0.87), and the adjusted R² is lower in the same proportion (square-root ratios 0.92, 0.82, 0.91). The slope is almost unchanged, so the gap comes from a noisier regressor: realized variance from 22 daily returns a month instead of about 1,700 five-minute returns. Over 1990–2007 my realized variance averages 20.8 against 33.2 for the implied variance (monthly percent squared), so the average premium is 12.4.

Evidence: Part A ratio rows; `results/001-baseline-1990/table1_summary.csv`. Report the six ratios to 2 decimals before asserting. If they differ from the claim, record it under Deviations.

**F3 — Newey–West overstates significance; Hodrick does not.**
> Under the null of no predictability, simulated by bootstrap, a nominal 5% Newey–West test with h−1 lags rejects 7.9% of the time on the paper's sample and 12.9% on 1990–2025 at h = 3, while the Hodrick test rejects 4.9% and 4.7%. The paper's choice of Hodrick standard errors is the right one; my earlier Newey–West t-statistics (3.46 at h = 3) overstated the evidence.

Evidence: `results/003-inference/table_size.csv` (iid columns), R-base and R-full at h = 3.

**F4 — The paper's result survives conservative inference.**
> At h = 3 on 1990–2007, the bootstrap p-value of the Newey–West t is 0.002 (iid) and 0.007 (wild), and the Hodrick t is 2.22.

Evidence: `table_inference.csv`, row R-base, h = 3. Figure: `results/003-inference/fig_boot_h3.png` (copy it, do not redraw).

**F5 — The post-2007 collapse is one month.**
> Extending the sample to December 2025 removes the predictability (h = 3 coefficient 0.09, Newey–West t 0.70). One observation drives this: in March 2020 the variance premium is −509 (realized variance 747, implied 239), and the next quarter's return is +73% annualized; its Cook's distance is 15.9. October 2008 has a similarly negative premium (−274), but the crash went on, so it supports the slope. At the end of a crash month, past realized variance measures the crash itself, not the variance expected for the next month.

Evidence: `results/002-influential-obs/table_influence.csv` and `table_robust.csv` (row 0); `results/003-inference/table_mr.csv` for RV and the forecast in 2020-03. Figure: copy `fig_scatter_h3.png`.

**F6 — A forward-looking premium works where the backward-looking one fails, and not the other way round.**
> Replacing last month's realized variance by a real-time forecast (a monthly HAR re-estimated each month on past data only) gives a premium that predicts returns on 2008–2025 (Hodrick t 2.75, adjusted R² about 10%) but not on 1995–2007 (0.87). The original premium does the opposite (1.81 before 2008, 0.92 after). In a joint regression on 1995–2025, only the forecast-based premium is significant (Hodrick t 2.66 against −0.07).

Evidence:
- `table_inference.csv`, rows V-95-07, G-95-07, V-08-25 and G-08-25 at h = 3;
- `table_horse_race.csv`, 1995–2025, h = 3;
- copy `fig_subperiod_h3.png`.

Report the adjusted R² for G-08-25 (the claim says "about 10%": assert it is between 9.5 and 10.6).

**F7 — There is no predictability in the calm decade.**
> On 2010–2019 neither premium predicts returns (Hodrick t 1.07 and 0.86). Predictability is concentrated in stress episodes: the ten largest gaps between the two premia are all crash months (1998-08, 2002-07, 2008-09, 2008-11, 2008-12, 2009-03, 2011-08, 2020-03, 2020-04, 2025-04).

Evidence: `table_inference.csv`, rows V-10-19 and G-10-19; `table_mr.csv` (the top-10 MR months). Assert that the set of top-10 dates equals the list in the claim. Figure: copy `results/002-influential-obs/fig_rolling_t.png`.

**F8 — The premium survives the standard predictors, once they are dated in real time.**
> With earnings lagged three months and CAY used only three months after its quarter ends, the variance premium stays significant next to each control on 1990–2007 (Hodrick t from 2.06 with CAY to 2.43 with RREL) and with all five controls together (2.68). Dating the controls as in my first notebook raised the R² of the regressions with log P/E or CAY by 2.3 to 3.0 points and pushed the premium below significance next to CAY (1.92).

Evidence: `results/004-controls/table4_timing_compare.csv` and `table4_rt.csv` (T4-base). Show the T4-base RT table, plus the RT vs NB columns.

**F9 — The European extension holds only after 2008.**
> Starting European returns at the first close after the US premium is known, the US premium does not predict DAX or CAC returns before 2008 (Hodrick t 1.14 and 1.10), although the Newey–West t-statistics are above 2. After 2008 the forecast-based US premium predicts both (2.19 and 2.46). The timing correction changes t by less than 0.3, except for the CAC after 2008 (0.43). The DAX is less correlated with the US premium than the US market itself (0.14 vs 0.20).

Evidence:
- `results/005-europe/table_europe.csv` and `table_timing.csv` (h = 3, SK timing);
- the standardized coefficients for the correlation numbers;
- copy `fig_europe_h3.png`.

**F10 — What changed from my first notebook.**
A markdown cell only, no assertion. Title: "F10 — What changed from my first notebook". Then this table, verbatim:

```markdown
| v1 notebook | Problem | Corrected |
|---|---|---|
| Coefficients compared with BTZ (2.32 vs 0.47) | different units | 0.42 vs 0.47 |
| Sample 1993–2007, SPY | not the paper's sample | 1990–2007, ^GSPC |
| Risk-free rate of month t | monthly average, unknown at the start of t | rate of t−1 |
| Newey–West t only | oversized (8–13% for a 5% test) | Hodrick 1B (the paper's choice) and null bootstrap |
| "Works in growth, not in crises" | one month drives it (March 2020, Cook's D 15.9); premium mismeasured after V-shaped crashes | the forecast-based premium predicts on 2008–2025 |
| CAY and Shiller earnings used at t | publication lags: look-ahead | control R² inflated by 2.3–3.0 points; premium + CAY was not significant (1.92 instead of 2.06) |
| European return from the close of the last day | starts 4h30 before the premium is observable | skip day 1; effect below 0.3 of t except CAC post-2008 (0.43) |
| "The DAX is more sensitive" | raw slopes across markets of different volatility | the DAX correlation is lower than the US one |
| Dividend note | reversed (SPY adjusted, ^GSPC and ^FCHI price, ^GDAXI total return); a constant yield moves the intercept, not the slope | corrected |
| "US premium predicts DAX and CAC, 1994–2007" | does not survive Hodrick (about 1.1) | withdrawn; holds after 2008 with the forecast-based premium |
| Minor | "Shiller CAPE" (it is a trailing P/E), "Adjusted R²" axis label (it was plain R²), 52/395 negative months (53/396 in the output) | corrected |
```

**Open questions (markdown, verbatim)**
```markdown
## Open questions

- Out of sample: does the forecast-based premium beat the historical mean (Campbell–Thompson R²) after 2008, in real time?
- International evidence: compare with Bollerslev, Marrone, Xu & Zhou (2014), who study variance premia across countries.
- A local European premium (VSTOXX or VDAX) instead of the US VIX.
- Daily horizon: the next-day DAX effect needs the open-to-open test before it can be stated.
```

## Other changes
- **`README.md`**: replace the "Key results" table and the structure list. The new version gives:
  - the F1 table (h = 1, 3, 6, 12; Hodrick t and adjusted R², mine vs BTZ);
  - the F6 sub-period table;
  - a "Findings" link to `findings.ipynb` and `results/006-findings/findings.html`;
  - the list of specs.
  
  Keep the "How to reproduce" section, and add: run the `src/` modules in spec order, then `findings.ipynb`.
- **`btz2009_replication.ipynb`**: add only a markdown cell at the very top: "Version 1 (September 2026). Superseded by `findings.ipynb`; see the errata there."

## Expected outputs
- `results/006-table2-full/` (Part A).
- `results/006-findings/`: `findings.html`, the figures, and `SUMMARY.md` (every assertion's value, every deviation, runtime).
- `findings.ipynb` committed **with outputs**.
- Figures: force-add the `.png` files, as in specs 003 and 005.

Commit message: `006 full Table 2 in the paper's statistics and findings notebook`. Push when done.
