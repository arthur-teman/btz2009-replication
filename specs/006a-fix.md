# Fix 006a — Corrected claim texts and the wild bootstrap figure

Commit cdd755d has 5 deviations, all in the F2 claim, plus two imprecise wordings. The numbers in the results are right; the claim texts are wrong. Replace the texts below **verbatim**, update the assertions so that they test the new numbers, and rerun.

Target after the rerun: 0 deviations.

## 1. F1 claim (replace the whole quote)
> On the paper's sample (January 1990 – December 2007), the variance risk premium coefficients differ from BTZ Table 2 by at most 0.05 at every horizon up to 15 months (largest gap at h = 3: 0.420 vs 0.47). Predictability is hump-shaped and peaks at the quarterly horizon, both for the coefficient and the Hodrick t-statistic (h = 3: 2.22 vs 2.86 in the paper). From 18 months on, neither the paper's estimates nor mine are significant.

Assertion: max |coef − BTZ| over h ≤ 15 is ≤ 0.0505. Report the value.

## 2. F2 claim (replace the whole quote)
> My Hodrick t-statistics are below the paper's at short horizons (h = 1, 3, 6: ratios 0.82, 0.78, 0.86), and the adjusted R² is lower in about the same proportion (square-root ratios 0.91, 0.81, 0.91). The slope is almost unchanged, so the gap comes from a noisier regressor: realized variance from about 21 daily returns a month instead of about 1,640 five-minute returns (78 a trading day). Over 1990–2007 my realized variance averages 20.8 against 33.2 for the implied variance (monthly percent squared), so the average premium is 12.4.

Assertions:
- the six ratios, with the usual tolerance;
- the average number of trading days a month on 1990–2007 rounds to 21;
- 21 × 78 rounds to 1,640 to the nearest 10 (1,638).

## 3. F8 claim (replace the last sentence only)
> Dating the controls as in my first notebook raised the R² of the premium regressions with log P/E and with CAY by 3.0 and 2.3 points (3.3 with all five controls), and pushed the premium below significance next to CAY (1.92).

Assertions: +2.98, +2.33 and +3.33 against `table4_timing_compare.csv`.

## 4. F10 table, one cell
In the row "Newey–West t only", replace "oversized (8–13% for a 5% test)" with "oversized (7.9–12.9% for a 5% test at h = 3)".

## 5. Wild bootstrap figure for F4
The F4 claim quotes the wild bootstrap, but the copied figure shows iid and block. Add `src/fig_boot_wild.py`. It must:
- replay the spec 003a wild draws (same seed and same order of rows as `src/inference_003a.py`) for R-base and R-G at h = 3;
- assert that the replayed p-values equal `table_inference.csv` (p_boot_NW_wild) to 1e-12;
- draw the histograms of the null NW t-statistic, iid and wild side by side, with the observed t marked;
- save the figure to `results/006-findings/fig_boot_h3_iid_wild.png`.

In F4, show this figure instead of `fig_boot_h3.png`, and rewrite the caption: "Bootstrap distribution under no predictability of the Newey–West t-statistic at h = 3, iid and wild schemes (5,000 replications each), R-base (1990–2007) and R-G (1995–2025); black line = observed t."

Add one assertion: the mean of the null t is within ±0.15 of 0 in all four panels.

## 6. Rerun
- Restart & Run All, then re-export `findings.html`.
- Update `results/006-findings/SUMMARY.md`: the run log should read 0 deviations. Keep a short "Fixed in 006a" section listing what changed.

Commit as `006a corrected claims and wild bootstrap figure`, then push.
