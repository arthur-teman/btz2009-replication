"""Spec 006 Part A — BTZ Table 2 in the paper's statistics.

Hodrick (1992) 1B t-statistics for the constant and the slope, Newey–West t,
R² and adjusted R², at h = 1, 3, 6, 9, 12, 15, 18, 24 on 1990–2007.
Imports the spec 001 and spec 003 modules and does not modify them.
Writes results/006-table2-full/.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import baseline_1990 as btz
from src import inference as inf

RESULTS = btz.ROOT / "results" / "006-table2-full"
HORIZONS = (1, 3, 6, 9, 12, 15, 18, 24)
RATIO_HORIZONS = (1, 3, 6)
START, END = "1990-01-31", "2007-12-31"

# BTZ (2009) Table 2: constant, its Hodrick t, VRP coefficient, its Hodrick t, adjusted R² in %.
BTZ = {
    1: (-0.55, -0.13, 0.39, 1.76, 1.07),
    3: (-2.08, -0.56, 0.47, 2.86, 6.82),
    6: (1.12, 0.33, 0.30, 2.15, 5.42),
    9: (3.63, 1.15, 0.17, 1.36, 2.30),
    12: (4.62, 1.50, 0.12, 1.00, 1.23),
    15: (4.84, 1.59, 0.11, 0.94, 1.00),
    18: (5.61, 1.81, 0.06, 0.56, 0.05),
    24: (6.48, 2.07, 0.01, 0.11, -0.50),
}

# Spec quick-check values: coef, t1B, t NW, adjusted R² in %, N.
QUICK = {
    1: (0.388, 1.44, 1.63, 0.90, 215),
    3: (0.420, 2.22, 3.46, 4.53, 213),
    6: (0.288, 1.86, 2.98, 4.46, 210),
    9: (0.165, 1.24, 1.60, 1.87, 207),
    12: (0.134, 1.10, 1.42, 1.39, 204),
    15: (0.110, 0.97, 1.18, 0.96, 201),
    18: (0.093, 0.86, 1.12, 0.67, 198),
    24: (0.056, 0.56, 1.01, -0.02, 192),
}
TOL_COEF = 0.005
TOL_STAT = 0.02


def _fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def sandwich(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """V_sum = Z⁻¹ S Z⁻¹ / T with Z = z'z / T and S = w'w / T. z, w are (T, 2)."""
    t_obs = z.shape[0]
    Z = z.T @ z / t_obs
    S = w.T @ w / t_obs
    Zinv = np.linalg.inv(Z)
    return Zinv @ S @ Zinv / t_obs


def hodrick_cov(e: np.ndarray, x: np.ndarray, h: int, hod_pos: np.ndarray) -> np.ndarray:
    """Full 2×2 Hodrick 1B covariance of (constant, slope) for the h-month average.

    Same u, w_t, z_t and date set as `inference.hodrick_t`; that function
    returns the [1, 1] element only.
    """
    e = np.asarray(e, dtype=float)
    x = np.asarray(x, dtype=float)
    u = e - e.mean()
    rolled = inf._rolling_sum(x[None, :], h)[0]
    pos = np.asarray(hod_pos, dtype=int)
    u_next = u[pos + 1]
    z = np.column_stack([np.ones(len(pos)), x[pos]])
    w = np.column_stack([u_next * h, u_next * rolled[pos]])
    return sandwich(z, w) / float(h * h)


def adjusted_r2(r2: float, n: int) -> float:
    return 1.0 - (1.0 - r2) * (n - 1) / (n - 2)


def _self_checks() -> None:
    """Check 3 on simulated data: at h = 1 with the OLS residual, both elements equal HC0."""
    rng = np.random.default_rng(6)
    n = 150
    x = rng.normal(size=n)
    y = 0.3 + 0.5 * x + rng.normal(size=n)
    hc0 = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC0")
    z = np.column_stack([np.ones(n), x])
    w = np.asarray(hc0.resid)[:, None] * z
    se = np.sqrt(np.diag(sandwich(z, w)))
    if np.max(np.abs(se - np.asarray(hc0.bse))) > 1e-10:
        raise AssertionError(f"Sandwich with OLS residuals {se} vs HC0 {np.asarray(hc0.bse)}")


def load_sample() -> pd.DataFrame:
    gspc = btz.load_close(btz.DATA / "gspc_data.csv")
    vix = btz.load_close(btz.DATA / "vix_1990_data.csv")
    rf = btz.load_rf()
    panel = btz.build_panel(gspc, vix, rf)
    sample = btz.slice_sample(panel, START, END)
    if len(sample) != 216:
        raise AssertionError(f"Main sample has {len(sample)} months, expected 216")
    if sample[["IV", "RV", "VRP", "e"]].isna().any().any():
        raise AssertionError("Main sample has missing values")
    return sample


def fit_horizon(sample: pd.DataFrame, h: int) -> dict:
    n = len(sample)
    y = 1200.0 * btz.forward_mean(sample["e"], h)
    model, idx = btz.fit_hac(y, sample["VRP"], max(1, h - 1))
    if len(idx) != n - h:
        raise AssertionError(f"h={h}: N={len(idx)}, expected {n - h}")
    hod_index = pd.DatetimeIndex(sample.index[h - 1 : n - 1])
    if len(hod_index) != len(idx):
        raise AssertionError(f"h={h}: Hodrick dates {len(hod_index)} vs N {len(idx)}")
    hod_pos = sample.index.get_indexer(hod_index)
    e = 1200.0 * sample["e"].to_numpy(dtype=float)
    x = sample["VRP"].to_numpy(dtype=float)

    const = float(model.params["const"])
    coef = float(model.params["VRP"])
    V = hodrick_cov(e, x, h, hod_pos)
    if not np.isclose(V[0, 1], V[1, 0], rtol=0, atol=1e-14):
        raise AssertionError(f"h={h}: Hodrick covariance is not symmetric")
    se_const, se_coef = np.sqrt(np.diag(V))
    t_ref, se_ref = inf.hodrick_t(e, x, h, hod_pos, coef)
    if abs(se_coef - float(se_ref[0])) > 1e-12 or abs(coef / se_coef - float(t_ref[0])) > 1e-10:
        raise AssertionError(f"h={h}: [1, 1] element does not reproduce inference.hodrick_t")

    if h == 1:
        frame = pd.concat([y.rename("y"), sample["VRP"]], axis=1).dropna()
        hc0 = sm.OLS(frame["y"], sm.add_constant(frame["VRP"])).fit(cov_type="HC0")
        z = np.column_stack([np.ones(len(frame)), frame["VRP"].to_numpy(dtype=float)])
        w = np.asarray(hc0.resid, dtype=float)[:, None] * z
        se_ols = np.sqrt(np.diag(sandwich(z, w)))
        gap = np.abs(se_ols - np.asarray(hc0.bse, dtype=float))
        if gap.max() > 1e-10:
            raise AssertionError(f"h=1: Hodrick with the OLS residual vs HC0, gap {gap}")

    r2 = float(model.rsquared)
    adj = adjusted_r2(r2, len(idx))
    if abs(adj - float(model.rsquared_adj)) > 1e-12:
        raise AssertionError(f"h={h}: adjusted R² formula disagrees with statsmodels")
    b_const, b_tc, b_coef, b_t, b_adj = BTZ[h]
    return {
        "h": h,
        "const": const,
        "t1B_const": const / se_const,
        "tNW_const": float(model.tvalues["const"]),
        "coef": coef,
        "t1B_coef": coef / se_coef,
        "tNW_coef": float(model.tvalues["VRP"]),
        "R2": 100.0 * r2,
        "adjR2": 100.0 * adj,
        "N": int(len(idx)),
        "BTZ_const": b_const,
        "BTZ_t_const": b_tc,
        "BTZ_coef": b_coef,
        "BTZ_t_coef": b_t,
        "BTZ_adjR2": b_adj,
    }


def build_table(sample: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame([fit_horizon(sample, h) for h in HORIZONS])
    table["ratio_t1B"] = np.nan
    table["ratio_sqrt_adjR2"] = np.nan
    for h in RATIO_HORIZONS:
        row = table["h"] == h
        table.loc[row, "ratio_t1B"] = table.loc[row, "t1B_coef"] / table.loc[row, "BTZ_t_coef"]
        table.loc[row, "ratio_sqrt_adjR2"] = np.sqrt(
            table.loc[row, "adjR2"] / table.loc[row, "BTZ_adjR2"]
        )
    return table


def check_references(table: pd.DataFrame) -> dict:
    by_h = table.set_index("h")
    saved = pd.read_csv(btz.RESULTS / "table2_main.csv")
    main = saved[saved["sample"] == "main"].set_index("h")
    gaps_1 = []
    for h in main.index:
        gaps_1 += [
            abs(by_h.loc[h, "coef"] - main.loc[h, "coef_B"]),
            abs(by_h.loc[h, "tNW_coef"] - main.loc[h, "t_h_minus_1"]),
            abs(by_h.loc[h, "R2"] - main.loc[h, "R2"]),
        ]
        if int(by_h.loc[h, "N"]) != int(main.loc[h, "N"]):
            raise AssertionError(f"h={h}: N differs from spec 001")
    if max(gaps_1) > 1e-8:
        raise AssertionError(f"Check 1: largest gap with spec 001 is {max(gaps_1)}")

    inference = pd.read_csv(inf.RESULTS / "table_inference.csv")
    base = inference[inference["id"] == "R-base"].set_index("h")
    gaps_2 = [abs(by_h.loc[h, "t1B_coef"] - base.loc[h, "t_1B"]) for h in RATIO_HORIZONS]
    if max(gaps_2) > 1e-8:
        raise AssertionError(f"Check 2: largest gap with spec 003 R-base is {max(gaps_2)}")

    quick_gaps = []
    for h, (coef, t1b, tnw, adj, n) in QUICK.items():
        row = by_h.loc[h]
        failures = []
        if abs(row["coef"] - coef) > TOL_COEF:
            failures.append(f"coef {row['coef']:.4f} vs {coef}")
        if abs(row["t1B_coef"] - t1b) > TOL_STAT:
            failures.append(f"t1B {row['t1B_coef']:.3f} vs {t1b}")
        if abs(row["tNW_coef"] - tnw) > TOL_STAT:
            failures.append(f"tNW {row['tNW_coef']:.3f} vs {tnw}")
        if abs(row["adjR2"] - adj) > TOL_STAT:
            failures.append(f"adjR2 {row['adjR2']:.3f} vs {adj}")
        if int(row["N"]) != n:
            failures.append(f"N {int(row['N'])} vs {n}")
        if failures:
            raise AssertionError(f"Quick check h={h}: " + "; ".join(failures))
        quick_gaps.append(
            {
                "h": h,
                "coef": abs(row["coef"] - coef),
                "t1B": abs(row["t1B_coef"] - t1b),
                "tNW": abs(row["tNW_coef"] - tnw),
                "adjR2": abs(row["adjR2"] - adj),
            }
        )
    return {
        "check1": max(gaps_1),
        "check2": max(gaps_2),
        "quick": pd.DataFrame(quick_gaps),
    }


def plot_vs_btz(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    panels = (
        (axes[0], "coef", "BTZ_coef", "VRP coefficient", "Coefficient (annualized % per monthly %²)"),
        (axes[1], "t1B_coef", "BTZ_t_coef", "Hodrick (1992) 1B t-statistic", "t-statistic (Hodrick 1B)"),
    )
    for ax, mine, theirs, title, ylabel in panels:
        ax.plot(table["h"], table[mine], marker="o", color="steelblue", label="This replication")
        ax.plot(
            table["h"], table[theirs], marker="D", color="black", ls="--", lw=0.9,
            label="BTZ (2009) Table 2",
        )
        ax.axhline(0, color="black", lw=0.4)
        ax.set_title(title)
        ax.set_xlabel("Horizon h (months)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(HORIZONS))
    axes[1].axhline(1.96, color="gray", lw=0.8, ls=":", label="1.96")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("^GSPC and VIX, 1990:01–2007:12, h-month average excess return on VRP", fontsize=11)
    fig.text(
        0.01, 0.01,
        "Source: results/006-table2-full/table2_full.csv; BTZ (2009) Table 2.",
        fontsize=7, color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(table: pd.DataFrame, checks: dict) -> None:
    by_h = table.set_index("h")
    lines: list[str] = []
    lines.append("# Spec 006 Part A — Table 2 in the paper's statistics")
    lines.append("")
    lines.append("## Definitions")
    lines.append("")
    lines.append(
        "Sample, data and units are spec 001's main sample: `^GSPC` with `data/vix_1990_data.csv`, "
        "1990-01 through 2007-12 (216 months), VRP in monthly percent squared, "
        "TB3MS lagged one month. The dependent variable is the h-month average excess return times 1200 "
        "(annualized percent). N = 216 − h."
    )
    lines.append("")
    lines.append(
        "The Hodrick (1992) 1B covariance is the spec 003 construction: u = e − mean(e), "
        "z_t = (1, VRP_t)', w_t = u_{t+1} × (z_t + … + z_{t−h+1}), on the dates t with t−h+1 and t+1 inside the sample. "
        "V = Z⁻¹ S Z⁻¹ / (T h²). The constant's t uses V[0, 0] and the slope's t uses V[1, 1], from the same w and Z. "
        "`src/inference.py` returns V[1, 1] only and is not modified: `hodrick_cov` in `src/table2_full.py` "
        "builds the full matrix, and its [1, 1] element is asserted equal to `inference.hodrick_t` at every horizon (1e-12)."
    )
    lines.append("")
    lines.append(
        "Newey–West t uses `statsmodels` HAC, Bartlett kernel, maxlags = max(1, h−1), `use_correction=False`. "
        "R² and adjusted R² = 1 − (1 − R²)(N − 1)/(N − 2) are in percent."
    )
    lines.append("")
    lines.append("## Table 2")
    lines.append("")
    lines.append("Hodrick 1B t-statistics in parentheses; BTZ values below each row of mine.")
    lines.append("")
    lines.append("| h | constant | VRP coef | t NW (coef) | R² | adj. R² | N |")
    lines.append("|---|---|---|---|---|---|---|")
    for h in HORIZONS:
        r = by_h.loc[h]
        lines.append(
            f"| {h} | {_fmt(r['const'], 2)} ({_fmt(r['t1B_const'], 2)}) | "
            f"{_fmt(r['coef'], 3)} ({_fmt(r['t1B_coef'], 2)}) | {_fmt(r['tNW_coef'], 2)} | "
            f"{_fmt(r['R2'], 2)} | {_fmt(r['adjR2'], 2)} | {int(r['N'])} |"
        )
        lines.append(
            f"| BTZ | {_fmt(r['BTZ_const'], 2)} ({_fmt(r['BTZ_t_const'], 2)}) | "
            f"{_fmt(r['BTZ_coef'], 2)} ({_fmt(r['BTZ_t_coef'], 2)}) | — | — | {_fmt(r['BTZ_adjR2'], 2)} | — |"
        )
    lines.append("")
    lines.append("Ratios to BTZ at h = 1, 3, 6:")
    lines.append("")
    lines.append("| ratio | h = 1 | h = 3 | h = 6 |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| t1B(mine) / t1B(BTZ) | "
        + " | ".join(_fmt(by_h.loc[h, "ratio_t1B"], 3) for h in RATIO_HORIZONS)
        + " |"
    )
    lines.append(
        "| √(adj R²(mine) / adj R²(BTZ)) | "
        + " | ".join(_fmt(by_h.loc[h, "ratio_sqrt_adjR2"], 3) for h in RATIO_HORIZONS)
        + " |"
    )
    lines.append("")
    peak_c = int(by_h["coef"].idxmax())
    peak_t = int(by_h["t1B_coef"].idxmax())
    gap = (by_h["coef"] - by_h["BTZ_coef"]).abs()
    gap15 = gap.loc[[h for h in HORIZONS if h <= 15]]
    significant = [h for h in HORIZONS if abs(by_h.loc[h, "t1B_coef"]) > 1.96]
    lines.append(
        f"The coefficient peaks at h = {peak_c} and the Hodrick t at h = {peak_t}. "
        f"The largest |coef − BTZ| for h ≤ 15 is {_fmt(gap15.max(), 4)}, at h = {int(gap15.idxmax())}. "
        "Horizons with a Hodrick |t| above 1.96: "
        + (", ".join(f"h = {h}" for h in significant) if significant else "none")
        + f". The largest Hodrick t is {_fmt(by_h['t1B_coef'].max(), 2)}, at h = {peak_t}."
    )
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append(
        f"1. Pass. At h = 1, 3, 6, 9, 12, 24 the coefficient, Newey–West t and R² reproduce "
        f"`results/001-baseline-1990/table2_main.csv`; largest gap {checks['check1']:.1e} (tolerance 1e-8)."
    )
    lines.append(
        f"2. Pass. At h = 1, 3, 6 the slope's Hodrick t reproduces R-base in "
        f"`results/003-inference/table_inference.csv`; largest gap {checks['check2']:.1e} (tolerance 1e-8)."
    )
    lines.append(
        "3. Pass. The constant's Hodrick t uses element [0, 0] of the matrix whose [1, 1] element gives the slope's t. "
        "At h = 1, with the OLS residual in place of u, both standard errors equal White HC0 to 1e-10, "
        "on the data and on a simulated sample."
    )
    q = checks["quick"]
    lines.append(
        "4. Pass. Every quick-check value is reproduced (±0.005 coefficient, ±0.02 t and adjusted R², N exact). "
        f"Largest gaps: coef {_fmt(q['coef'].max(), 4)}, t1B {_fmt(q['t1B'].max(), 3)}, "
        f"t NW {_fmt(q['tNW'].max(), 3)}, adj. R² {_fmt(q['adjR2'].max(), 3)}."
    )
    lines.append("")
    lines.append("## Deviations from this spec")
    lines.append("")
    lines.append(
        "1. The spec says to generalize `src/inference.py` `hodrick_t` for the constant. The task also said not to change "
        "the 001–005 modules, so the generalization lives in `src/table2_full.py` (`hodrick_cov`), reusing "
        "`inference._rolling_sum`, and is asserted to reproduce `hodrick_t` at every horizon. `hodrick_t`'s output is unchanged because the function is unchanged."
    )
    lines.append("")
    lines.append(
        "2. The ratio \"row\" is stored as two columns of `table2_full.csv`, `ratio_t1B` and `ratio_sqrt_adjR2`, "
        "filled at h = 1, 3, 6 only, and shown above as two rows."
    )
    lines.append("")
    (RESULTS / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> pd.DataFrame:
    _self_checks()
    sample = load_sample()
    table = build_table(sample)
    checks = check_references(table)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "table2_full.csv", index=False)
    plot_vs_btz(table, RESULTS / "fig_table2_vs_btz.png")
    write_summary(table, checks)
    print(f"Wrote {RESULTS}")
    print(table.round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    main()
