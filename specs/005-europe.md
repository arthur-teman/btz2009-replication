# Spec 005 — European extension: does the US VRP predict DAX and CAC returns?

## Goal
Redo notebook section 5 on the 001–004 pipeline. Three problems with the current version:

1. **Asynchronous close (look-ahead).** Xetra and Euronext close at 17:30 CET, while the US close is at 22:00 CET. VRP_t is observable only at the US close of the last trading day T of month t. The European month-t+1 return starts at the European close of day T, about 4.5 hours **before** VRP_t is known. Part of the US afternoon news that moves VRP_t then enters the European return of day 1 of month t+1.
2. **Old units and old inference.** The section uses SPY, decimal units, an unlagged risk-free rate and NW t-statistics only.
3. **Wrong reading of the coefficients.**
   - The larger DAX coefficient reflects the DAX's larger return volatility, not a higher sensitivity.
   - The note on dividends is reversed. `^GDAXI` is a performance (total-return) index, `^FCHI` is a price index, and the US side is now `^GSPC` (price).

   A constant dividend yield shifts the intercept, not the slope.

The regressors are the US VRP and the US VRP_fwd from specs 001–002, unchanged. They are US measures by design: the question is whether US variance risk is priced in European returns.

Do not change the 001–004 pipelines. Reuse their VRP, VRP_fwd, units, `forward_average`, NW, Hodrick 1B and the iid and wild bootstrap from `src/inference.py` and `src/inference_003a.py`.

## Inputs
- `data/dax.csv` (`^GDAXI`) and `data/cac.csv` (`^FCHI`): existing frozen daily files, Close column, local currency. Do not modify them.
- `data/euribor.csv`: FRED IR3TIB01EZM156N, 3-month interbank rate, monthly, percent per year. It starts in 1994-01; before 1999 it is the OECD euro-area synthetic series.
- US side: the spec 001 panel (`^GSPC`, `data/vix_1990_data.csv`, TB3MS lagged one month) and VRP_fwd from spec 002.

## Exact definitions
All returns are log returns in local currency.

**Risk-free rate**: rf^EU_t = Euribor_{t−1} / 1200. This is the same one-month lag as the US rf in spec 001. The first usable month is 1994-02.

Two timings for the European dependent variable, with T_m = the last European trading day of month m and F_m = the first European trading day of month m:

- **NB (notebook timing)**: monthly return r_m = log(P(T_m) / P(T_{m−1})), excess return e_m = r_m − rf^EU_m, and Y^NB_t(h) = (1/h) Σ_{j=1..h} e_{t+j}, in annualized % (× 1200), as in spec 001.
- **SK (skip day 1, main specification)**: the h-month window starts at the close of F_{t+1}, the first European close after VRP_t is observable, and ends at T_{t+h}:

  Y^SK_t(h) = (1200/h) · [log(P(T_{t+h}) / P(F_{t+1})) − Σ_{j=1..h} rf^EU_{t+j}].

  The rf sum is kept over the full h months. The one missing trading day shifts the mean of Y by a negligible amount.

**Second check of the timing (VRP at T−1)**: VRP^{T−1}_t is built like spec 001 VRP, but with US daily returns and the VIX close up to the **penultimate** US trading day of month t. Pair it with Y^NB. It is a robustness row, defined for the raw VRP only. VRP_fwd keeps its spec 002 definition.

**US benchmark row**: the US index on the same sample, with the spec 001 dependent variable (no skip; the US close is synchronous with VRP).

## Regressions
Horizons h ∈ {1, 3, 6}. Univariate regressions, Y = a + b·x + u.

| market | regressor | timing | samples |
|---|---|---|---|
| US (`^GSPC`) | VRP, VRP_fwd | spec 001 | 1994-02 → 2007-12; 2008-01 → 2025-12 |
| DAX, CAC | VRP | NB, SK, T−1 | same |
| DAX, CAC | VRP_fwd | NB, SK | same |

VRP_fwd starts in 1995-01, so its pre-2008 rows run from 1995-01 → 2007-12. State this in every table.

For every regression, report:
- coef(B), t NW(h−1), t Hodrick 1B, the iid and wild bootstrap p-values of the NW t (5,000 replications, `default_rng(20261021)`), R², N;
- the **standardized coefficient**: coef(B) × sd(x) / sd(Y), with both standard deviations on the regression rows;
- coef(B) divided by the standard deviation of the market's monthly excess return (annualized %) on the same rows.

## Checks (assert in code, report in SUMMARY.md)
1. The US benchmark rows reproduce the spec 001 pipeline sliced to the same rows. The 1995–2007 and 2008–2025 rows reproduce V-95-07, G-95-07, V-08-25 and G-08-25 of 003a to 1e-6 on the coefficient and NW t.
2. Daily data quality, DAX and CAC, 1994–2025:
   - at least 15 trading days in every month;
   - list every daily |log return| > 10% with its date;
   - list every month with zero volume on all days (information only; early Yahoo index rows often have volume 0).
3. SK timing: for every row t, F_{t+1} is strictly after the last US trading day of month t. Every price in Y^SK is dated ≥ F_{t+1} and ≤ T_{t+h}.
4. T−1 timing: the latest US date inside VRP^{T−1}_t is strictly before the European close T_t. Same calendar day count: the penultimate US trading day of month t is < T_t, or equal to T_t if the European and US last days differ. Report how many months fall in each case.
5. No NaN inside any regression sample. N is reported everywhere.

## Reference values (audit, SPY-based VRP, notebook units, NW, h=3, 1994–2007)
These indicate the direction only, not targets:

| | NB | VRP at T−1 | skip day 1 |
|---|---|---|---|
| DAX t | 2.64 | 2.59 | 2.79 |
| CAC t | 2.05 | 2.02 | 2.18 |

Monthly return volatility 1994–2007: SPY 4.02 %, DAX 6.45 %, CAC 5.48 %.

## Reading rules, fixed before the run (independent flags, h=3, Hodrick 1B)
- **E1 Timing is immaterial**: for DAX and CAC, on both samples, with the raw VRP, |t1B(SK) − t1B(NB)| < 0.3 and |t1B(T−1) − t1B(NB)| < 0.3.
- **E2 Pre-2008 spillover**: with the raw VRP, SK timing, 1994-02 → 2007-12, DAX **and** CAC each have Hodrick |t| > 1.96 **and** iid p < 0.05 **and** wild p < 0.05.
- **E3 Post-2008 spillover**: same three conditions with VRP_fwd, SK timing, 2008–2025, for DAX **and** CAC.
- **E4 No extra sensitivity**: on 1994–2007 (raw VRP, SK), the standardized coefficient of the DAX is within ±25% of the US standardized coefficient.

Every condition in a flag is joined by **and**. If a flag fails, report the numbers. Do not change the timing, the index or the sample to rescue it.

## Expected outputs
In `results/005-europe/`:
- `table_europe.csv`: one row per (market, regressor, timing, sample, h), with every statistic listed above;
- `table_timing.csv`: for DAX and CAC, raw VRP, h=3, t1B and coef under NB, SK and T−1, for both samples;
- `table_data_quality.csv`: the lists from check 2;
- `fig_europe_h3.png`: Hodrick t at h=3 (SK timing) for US, DAX and CAC, raw VRP and VRP_fwd, pre-2008 and post-2008 panels, with a 1.96 line;
- `SUMMARY.md`, containing:
  - checks 1–5;
  - the h=3 table;
  - flags E1–E4;
  - the index-definition note (DAX total return, CAC and `^GSPC` price, local currency, Euribor before 1999 is synthetic);
  - deviations from this spec.

Code: a new module `src/europe.py`. The new notebook section 5.2 replaces nothing: keep the old section 5 and add a markdown cell at its top saying it is superseded by section 5.2 and why.

Commit message: `005 European extension on the BTZ pipeline`. Push when done.
