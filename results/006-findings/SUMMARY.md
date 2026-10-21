# Spec 006 Part B — findings notebook

Written by the last cell of `findings.ipynb`. The notebook reads `results/*/` and does not re-estimate anything; the one exception is the count of trading days per month in F2, read from `data/gspc_data.csv`.

- Assertions: 74, of which 74 pass and 0 are recorded deviations.
- Runtime of Restart & Run All, inside the kernel: 0.5 s (limit 120 s).
- Figures in this folder: `fig_boot_h3_iid_wild.png`, `fig_europe_h3.png`, `fig_rolling_t.png`, `fig_scatter_h3.png`, `fig_subperiod_h3.png`, `fig_table2_vs_btz.png`.
- `findings.html` is exported with `jupyter nbconvert --to html --execute findings.ipynb`.

## Assertions

Numbers are asserted to ±0.5 in the last digit printed in the claim. "holds" rows assert a qualitative statement; the results column lists the values behind it.

| finding | assertion | claim | results | status |
|---|---|---|---|---|
| F1 | max \|coef − BTZ\| ≤ 0.0505 for h ≤ 15 | holds | max 0.0504 at h = 3 | pass |
| F1 | largest gap at h = 3 | holds | h = 3 | pass |
| F1 | coef, h = 3 | 0.420 | 0.41962 | pass |
| F1 | BTZ coef, h = 3 | 0.47 | 0.4700 | pass |
| F1 | coefficient peaks at h = 3 (mine and BTZ) | holds | argmax mine h = 3, BTZ h = 3 | pass |
| F1 | Hodrick t peaks at h = 3 (mine and BTZ) | holds | argmax mine h = 3, BTZ h = 3 | pass |
| F1 | Hodrick t, h = 3 | 2.22 | 2.2185 | pass |
| F1 | BTZ Hodrick t, h = 3 | 2.86 | 2.8600 | pass |
| F1 | no Hodrick \|t\| above 1.96 for h ≥ 18 (mine and BTZ) | holds | mine 0.86, 0.56; BTZ 0.56, 0.11 | pass |
| F2 | t1B ratio, h = 1 | 0.82 | 0.8196 | pass |
| F2 | t1B ratio, h = 3 | 0.78 | 0.7757 | pass |
| F2 | t1B ratio, h = 6 | 0.86 | 0.8631 | pass |
| F2 | sqrt adj. R² ratio, h = 1 | 0.91 | 0.9149 | pass |
| F2 | sqrt adj. R² ratio, h = 3 | 0.81 | 0.8149 | pass |
| F2 | sqrt adj. R² ratio, h = 6 | 0.91 | 0.9075 | pass |
| F2 | t-statistics below the paper's at h = 1, 3, 6 | holds | 0.820, 0.776, 0.863 | pass |
| F2 | daily returns a month | 21 | 21.01 | pass |
| F2 | five-minute returns a month, 21 × 78 (nearest 10) | 1640 | 1638.0 | pass |
| F2 | mean RV 1990–2007 | 20.8 | 20.844 | pass |
| F2 | mean IV 1990–2007 | 33.2 | 33.237 | pass |
| F2 | mean VRP 1990–2007 | 12.4 | 12.394 | pass |
| F3 | NW size %, R-base h = 3 | 7.9 | 7.880 | pass |
| F3 | NW size %, R-full h = 3 | 12.9 | 12.940 | pass |
| F3 | Hodrick size %, R-base h = 3 | 4.9 | 4.860 | pass |
| F3 | Hodrick size %, R-full h = 3 | 4.7 | 4.680 | pass |
| F3 | NW t, R-base h = 3 | 3.46 | 3.4554 | pass |
| F4 | p NW iid, R-base h = 3 | 0.002 | 0.00200 | pass |
| F4 | p NW wild, R-base h = 3 | 0.007 | 0.00680 | pass |
| F4 | Hodrick t, R-base h = 3 | 2.22 | 2.2185 | pass |
| F4 | mean of the null NW t within ±0.15 of 0 in all four panels | holds | R-base iid +0.060, R-base wild +0.054, R-G iid -0.037, R-G wild +0.052 | pass |
| F5 | coef, 1990–2025 h = 3 | 0.09 | 0.0908 | pass |
| F5 | NW t, 1990–2025 h = 3 | 0.70 | 0.6962 | pass |
| F5 | VRP 2020-03 | -509 | -508.51 | pass |
| F5 | RV 2020-03 | 747 | 747.39 | pass |
| F5 | IV 2020-03 (VRP + RV) | 239 | 238.88 | pass |
| F5 | Y(3) after 2020-03, annualized % | 73 | 72.58 | pass |
| F5 | Cook's D 2020-03 | 15.9 | 15.917 | pass |
| F5 | 2020-03 is the most influential row | holds | Cook's D 15.92, next 0.25 | pass |
| F5 | VRP 2008-10 | -274 | -274.11 | pass |
| F5 | 2008-10: return stays negative and the row raises the slope | holds | Y(3) -64.12, DFBETA +0.0278 | pass |
| F6 | Hodrick t, G 2008–2025 | 2.75 | 2.7513 | pass |
| F6 | adj. R² G 2008–2025 about 10% (9.5–10.6) | holds | 10.21% | pass |
| F6 | Hodrick t, G 1995–2007 | 0.87 | 0.8656 | pass |
| F6 | Hodrick t, V 1995–2007 | 1.81 | 1.8133 | pass |
| F6 | Hodrick t, V 2008–2025 | 0.92 | 0.9223 | pass |
| F6 | G significant after 2008 only; V stronger before 2008 | holds | G 0.87 → 2.75; V 1.81 → 0.92 | pass |
| F6 | joint Hodrick t, VRP_fwd 1995–2025 | 2.66 | 2.6607 | pass |
| F6 | joint Hodrick t, VRP 1995–2025 | -0.07 | -0.0711 | pass |
| F6 | only VRP_fwd significant in the joint regression | holds | 2.66 vs -0.07 | pass |
| F7 | Hodrick t, V 2010–2019 | 1.07 | 1.0658 | pass |
| F7 | Hodrick t, G 2010–2019 | 0.86 | 0.8573 | pass |
| F7 | neither premium significant on 2010–2019 | holds | max \|t\| 1.07 | pass |
| F7 | number of largest gaps listed | 10 | 10.00 | pass |
| F7 | top-10 \|VRP_fwd − VRP\| months equal the listed dates | holds | 1998-08, 2002-07, 2008-09, 2008-11, 2008-12, 2009-03, 2011-08, 2020-03, 2020-04, 2025-04 | pass |
| F8 | Hodrick t VRP, with CAY (RT) | 2.06 | 2.0598 | pass |
| F8 | Hodrick t VRP, with RREL (RT) | 2.43 | 2.4309 | pass |
| F8 | VRP significant next to each control; range runs from CAY to RREL | holds | log_PE 2.40, DFSP 2.23, TMSP 2.25, RREL 2.43, CAY 2.06 | pass |
| F8 | controls in the joint regression | 5 | 5.00 | pass |
| F8 | Hodrick t VRP, all controls (RT) | 2.68 | 2.6774 | pass |
| F8 | R² increase, VRP + log P/E (points) | 2.98 | 2.9806 | pass |
| F8 | R² increase, VRP + CAY (points) | 2.33 | 2.3314 | pass |
| F8 | R² increase, VRP + all five controls (points) | 3.33 | 3.3302 | pass |
| F8 | Hodrick t VRP, with CAY (first-notebook dating) | 1.92 | 1.9184 | pass |
| F8 | first-notebook dating pushes VRP below 1.96 next to CAY | holds | 1.918 | pass |
| F9 | Hodrick t, DAX pre-2008 (SK) | 1.14 | 1.1364 | pass |
| F9 | Hodrick t, CAC pre-2008 (SK) | 1.10 | 1.0959 | pass |
| F9 | NW t above 2 for both before 2008 | holds | DAX 2.53, CAC 2.09 | pass |
| F9 | Hodrick t, DAX post-2008 VRP_fwd (SK) | 2.19 | 2.1885 | pass |
| F9 | Hodrick t, CAC post-2008 VRP_fwd (SK) | 2.46 | 2.4580 | pass |
| F9 | timing shift of Hodrick t below 0.3 except CAC post-2008 | holds | largest other shift 0.253 | pass |
| F9 | timing shift, CAC post-2008 | 0.43 | 0.4304 | pass |
| F9 | correlation US VRP with DAX return (std coef) | 0.14 | 0.1424 | pass |
| F9 | correlation US VRP with US return (std coef) | 0.20 | 0.2036 | pass |
| F9 | DAX correlation below the US one | holds | 0.142 vs 0.204 | pass |

## Deviations

None: the results reproduce every number in the claims.

## Fixed in 006a

- F1: the claim says the coefficients differ from BTZ by at most 0.05 and names the largest gap (h = 3: 0.420 vs 0.47). Asserted as max |coef − BTZ| ≤ 0.0505 for h ≤ 15; the value is 0.0504.
- F2: the ratios are quoted from the unrounded statistics: t1B ratio at h = 6 0.863 (was 0.87), square-root adjusted R² ratios at h = 1 and 3 0.915 and 0.815 (were 0.92 and 0.82). "22 daily returns" is now "about 21" (21.01 on average in `data/gspc_data.csv`), and "about 1,700 five-minute returns" is now "about 1,640" (21 × 78 = 1,638). These were the five deviations of commit cdd755d.
- F8: the R² sentence names the regressions. First-notebook dating raises R² by 2.98 (VRP + log P/E), 2.33 (VRP + CAY) and 3.33 (VRP + all five controls) points, asserted against `table4_timing_compare.csv`.
- F10: the Newey–West row gives the size as 7.9–12.9% for a 5% test at h = 3.
- F4: the figure is `fig_boot_h3_iid_wild.png`, drawn by `src/fig_boot_wild.py`. It replays the spec 003 iid and the spec 003a wild draws for R-base and R-G at h = 3 (same seed and row order as `src/inference_003a.py`); the replayed p-values equal `table_inference.csv` to 1e-12, and the mean of the null t is within ±0.15 of 0 in all four panels. It replaces `fig_boot_h3.png`, which showed the iid and block schemes.

## Other notes

- Formatting: p-values are shown to 3 decimals, not 2, so that the claim's 0.002 and 0.007 are visible in the table; bootstrap sizes are shown in percent.
- Copied figures keep the labels of the spec that drew them. Where an axis has no unit (rolling coefficient in `fig_rolling_t.png`, sub-period axis in `fig_subperiod_h3.png`, t-statistics everywhere), the unit is given in the caption cell.
