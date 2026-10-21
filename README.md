# Replication — Bollerslev, Tauchen & Zhou (2009)
**"Expected Stock Returns and Variance Risk Premia"** — *Review of Financial Studies*

**Author:** Arthur Teman — M1 QEF HEC Paris  
**Date:** September 2026

## What this replicates
The central empirical finding of BTZ (2009): the Variance Risk Premium (VRP = IV − RV)
predicts aggregate stock market returns, with predictability peaking at the quarterly horizon (h=3).

## Structure
- `findings.ipynb` — every finding of specs 001–006: each claim followed by its evidence, with every quoted number asserted against `results/`
- `src/` — one module per spec; each reads the frozen data and writes `results/<spec>/`
- `specs/` — the specifications, in order:
  - `001-baseline-units-1990.md` — Table 2 on 1990–2007 in the paper's units (`src/baseline_1990.py`)
  - `002-influential-obs.md` — influential observations and real-time robustness (`src/influence.py`)
  - `003-inference.md` — Hodrick (1992) t-statistics, null bootstrap, fixed crisis exclusion (`src/inference.py`)
  - `003a-fix.md` — wild bootstrap, matched sub-samples, horse race (`src/inference_003a.py`)
  - `004-controls-realtime.md` — Table 4 with real-time controls (`src/controls.py`)
  - `005-europe.md` — European extension, DAX and CAC (`src/europe.py`)
  - `006-findings-notebook.md` — full Table 2 in the paper's statistics (`src/table2_full.py`) and `findings.ipynb`
- `results/` — tables, figures and a `SUMMARY.md` per spec
- `btz2009_replication.ipynb` — version 1 (September 2026), superseded by `findings.ipynb`
- `data/` — all data frozen as CSV (no API calls needed to reproduce)

## Findings
The findings, with their evidence and errata for version 1, are in [`findings.ipynb`](findings.ipynb),
exported to [`results/006-findings/findings.html`](results/006-findings/findings.html).

## How to reproduce
```bash
git clone https://github.com/arthur-teman/btz2009-replication
cd btz2009-replication
pip install -r requirements.txt
jupyter notebook btz2009_replication.ipynb
# Restart & Run All
```

To rebuild the results and the findings, run the `src/` modules in spec order, then `findings.ipynb`:
```bash
python src/baseline_1990.py
python src/influence.py
python src/inference.py
python src/inference_003a.py
python src/controls.py
python src/europe.py
python src/table2_full.py
jupyter nbconvert --to notebook --execute --inplace findings.ipynb
```

## Key results
Table 2 on the paper's sample (^GSPC, January 1990 – December 2007): Hodrick (1992) 1B t-statistic of the
variance risk premium and adjusted R², this replication vs BTZ (2009).

| Horizon | Hodrick t (ours) | Hodrick t (BTZ) | adj. R² (ours) | adj. R² (BTZ) |
|---------|------------------|-----------------|----------------|---------------|
| h = 1   | 1.44             | 1.76            | 0.90%          | 1.07%         |
| h = 3   | **2.22**         | **2.86**        | 4.53%          | 6.82%         |
| h = 6   | 1.86             | **2.15**        | 4.46%          | 5.42%         |
| h = 12  | 1.10             | 1.00            | 1.39%          | 1.23%         |

By sub-period, h = 3: the original premium (implied variance minus last month's realized variance) against the
forecast-based premium (implied variance minus a real-time HAR forecast of realized variance).

| Sample    | Hodrick t, original | Hodrick t, forecast-based | adj. R², original | adj. R², forecast-based |
|-----------|---------------------|---------------------------|-------------------|-------------------------|
| 1995–2007 | 1.81                | 0.87                      | 3.61%             | 0.08%                   |
| 2008–2025 | 0.92                | **2.75**                  | 0.59%             | 10.21%                  |
| 2010–2025 | −1.12               | 1.91                      | 1.05%             | 5.01%                   |
| 2010–2019 | 1.07                | 0.86                      | 0.79%             | 1.15%                   |

## Data sources
- SPY, VIX, DAX, CAC: Yahoo Finance (frozen)
- TB3MS, BAA, AAA, GS10, Euribor: FRED (frozen)
- Shiller P/E: Robert Shiller's website (frozen)
- CAY: Lettau & Ludvigson website (frozen)
