# Replication — Bollerslev, Tauchen & Zhou (2009)
**"Expected Stock Returns and Variance Risk Premia"** — *Review of Financial Studies*

**Author:** Arthur Teman — M1 QEF HEC Paris  
**Date:** September 2026

## What this replicates
The central empirical finding of BTZ (2009): the Variance Risk Premium (VRP = IV − RV)
predicts aggregate stock market returns, with predictability peaking at the quarterly horizon (h=3).

## Structure
- `btz2009_replication.ipynb` — main notebook (sections 1–5)
  - Section 1: Data (RV, IV, VRP construction)
  - Section 2: Predictive regressions — multi-horizon (h = 1, 3, 6, 9, 12, 24 months)
  - Section 3: Conclusion and comparison with BTZ (2009)
  - Section 4: Robustness — control variables (log P/E, DFSP, TMSP, RREL, CAY)
  - Section 5: Extension — geographic spillover (DAX, CAC 40)
- `data/` — all data frozen as CSV (no API calls needed to reproduce)

## How to reproduce
```bash
git clone https://github.com/arthur-teman/btz2009-replication
cd btz2009-replication
pip install -r requirements.txt
jupyter notebook btz2009_replication.ipynb
# Restart & Run All
```

## Key results (paper sample 1993–2007)
| Horizon | t-stat (ours) | t-stat (BTZ) | R² (ours) | R² (BTZ) |
|---------|--------------|-------------|----------|---------|
| h = 1   | 0.97         | 1.76        | 0.56%    | 1.07%   |
| h = 3   | **2.59**     | **2.86**    | 2.54%    | 6.82%   |
| h = 6   | **2.39**     | 2.15        | 2.64%    | 5.42%   |
| h = 12  | 0.88         | 1.00        | 0.68%    | 1.23%   |

Hump-shaped pattern reproduced. Weaker R² explained by daily vs 5-minute RV.

## Data sources
- SPY, VIX, DAX, CAC: Yahoo Finance (frozen)
- TB3MS, BAA, AAA, GS10, Euribor: FRED (frozen)
- Shiller P/E: Robert Shiller's website (frozen, date bug fixed)
- CAY: Lettau & Ludvigson website (frozen)
