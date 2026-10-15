"""Spec 002 — influential observations and real-time robustness.

Imports the spec 001 pipeline and does not modify it. Writes
results/002-influential-obs/.
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
from matplotlib.patches import Patch

from src import baseline_1990 as btz

RESULTS = btz.ROOT / "results" / "002-influential-obs"
HORIZONS = (1, 3, 6)
BURN_IN_MONTHS = 60
ROLL_MONTHS = 120
REF_COEF_H3 = 0.0908
REF_T_H3 = 0.70

VARIANT_ORDER = (
    "0",
    "B 1/99 in-sample",
    "B 2.5/97.5 in-sample",
    "C real-time 1/99",
    "D drop crisis rows",
    "E 1990-2007",
    "E 2008-2025",
    "E 2010-2025",
    "G full sample",
    "G 1990-2007",
)


def _fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _month_ends(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="ME")


def crisis_months(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Regressor months named in part D."""
    blocks = (
        _month_ends("2008-09-01", "2008-12-31"),
        _month_ends("2020-02-01", "2020-04-30"),
    )
    wanted = blocks[0].append(blocks[1])
    missing = [t for t in wanted if t not in index]
    if missing:
        raise AssertionError(f"Crisis months missing from the sample index: {missing}")
    return pd.DatetimeIndex(wanted)


def assert_row_timing(sample: pd.DataFrame, t: pd.Timestamp, h: int, name: str) -> None:
    """Spec 001 check 1, for one regression row."""
    t = pd.Timestamp(t)
    rv_d = pd.Timestamp(sample.at[t, "rv_max"])
    iv_d = pd.Timestamp(sample.at[t, "iv_max"])
    if rv_d.to_period("M") != t.to_period("M"):
        raise AssertionError(f"{name} h={h}: RV at {t.date()} uses {rv_d.date()}")
    if iv_d.to_period("M") != t.to_period("M"):
        raise AssertionError(f"{name} h={h}: IV at {t.date()} uses {iv_d.date()}")
    xmax = max(rv_d, iv_d)
    if xmax.to_period("M") != t.to_period("M") or xmax > t:
        raise AssertionError(f"{name} h={h}: X at {t.date()} uses {xmax.date()}")
    loc = sample.index.get_loc(t)
    if not isinstance(loc, (int, np.integer)):
        raise AssertionError(f"{name}: {t.date()} is not a unique row")
    first = sample.index[loc + 1]
    last = sample.index[loc + h]
    if first.to_period("M") != t.to_period("M") + 1:
        raise AssertionError(
            f"{name} h={h}: first return in Y at {t.date()} is {pd.Timestamp(first).date()}"
        )
    if last.to_period("M") != t.to_period("M") + h:
        raise AssertionError(
            f"{name} h={h}: last return in Y at {t.date()} is {pd.Timestamp(last).date()}"
        )


def run_regression(
    sample: pd.DataFrame,
    vrp: pd.Series,
    name: str,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict:
    """Specification (B) on one sample. NaN regressor rows are dropped.

    Y is built on `sample`, so an h-month window that would cross the end of
    this sample is left incomplete, as in spec 001.
    """
    vrp = vrp.reindex(sample.index)
    by_h = {}
    n_checked = 0
    for h in horizons:
        y = 1200.0 * btz.forward_mean(sample["e"], h)
        model, idx = btz.fit_hac(y, vrp, max(1, h - 1))
        model_h, idx_h = btz.fit_hac(y, vrp, h)
        if not idx.equals(idx_h):
            raise AssertionError(f"{name} h={h}: the two lag choices changed the sample")
        for t in idx:
            assert_row_timing(sample, t, h, name)
            n_checked += 1
        by_h[h] = {
            "coef_B": float(model.params["VRP"]),
            "t_hm1": float(model.tvalues["VRP"]),
            "t_h": float(model_h.tvalues["VRP"]),
            "R2": float(model.rsquared * 100.0),
            "N": int(len(idx)),
            "index": pd.DatetimeIndex(idx),
        }
    return {"name": name, "by_h": by_h, "n_checked": n_checked}


def wide_row(result: dict) -> dict:
    row = {"variant": result["name"]}
    for h, res in result["by_h"].items():
        row[f"h{h}_coef_B"] = res["coef_B"]
        row[f"h{h}_t_hm1"] = res["t_hm1"]
        row[f"h{h}_t_h"] = res["t_h"]
        row[f"h{h}_R2"] = res["R2"]
        row[f"h{h}_N"] = res["N"]
    return row


def influence_h3(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leverage, Cook's distance and DFBETA on the row-0 h=3 OLS.

    DFBETA is coef(B) minus the coefficient estimated without that row.
    """
    y = 1200.0 * btz.forward_mean(sample["e"], 3)
    frame = pd.concat(
        [y.rename("y"), sample["VRP"].rename("VRP")],
        axis=1,
        sort=True,
    ).dropna()
    ols = sm.OLS(frame["y"], sm.add_constant(frame["VRP"])).fit()
    infl = ols.get_influence()
    hii = np.asarray(infl.hat_matrix_diag, dtype=float)
    cooks = np.asarray(infl.cooks_distance[0], dtype=float)
    dfb = np.asarray(infl.dfbeta, dtype=float)[:, list(ols.params.index).index("VRP")]

    resid = np.asarray(ols.resid, dtype=float)
    manual_cook = (resid ** 2) / (2.0 * float(ols.mse_resid)) * (hii / (1.0 - hii) ** 2)
    if not np.allclose(cooks, manual_cook, rtol=1e-8, atol=1e-10):
        raise AssertionError("Cook's distance does not match the OLS formula")
    if not np.isclose(hii.sum(), 2.0, rtol=1e-8, atol=1e-8):
        raise AssertionError(f"Leverages sum to {hii.sum()}, expected 2")
    if np.any(hii <= 0) or np.any(hii >= 1):
        raise AssertionError("A leverage is outside (0, 1)")

    y_np = frame["y"].to_numpy(dtype=float)
    x_np = frame["VRP"].to_numpy(dtype=float)
    beta = float(ols.params["VRP"])
    loo = np.empty(len(frame))
    for i in range(len(frame)):
        mask = np.ones(len(frame), dtype=bool)
        mask[i] = False
        fit = sm.OLS(y_np[mask], sm.add_constant(x_np[mask])).fit()
        loo[i] = beta - float(fit.params[1])
    if not np.allclose(dfb, loo, rtol=1e-8, atol=1e-10):
        raise AssertionError("DFBETA is not the leave-one-out change in coef(B)")

    full = pd.DataFrame(
        {
            "date": frame.index,
            "VRP_t": frame["VRP"].to_numpy(dtype=float),
            "Y_t": frame["y"].to_numpy(dtype=float),
            "DFBETA": dfb,
            "Cooks_D": cooks,
            "h_tt": hii,
        }
    )
    top = full.reindex(full["DFBETA"].abs().sort_values(ascending=False).index).head(10)
    top = top.drop(columns=["h_tt"]).reset_index(drop=True)
    return full, top


def winsor_insample(vrp: pd.Series, lower: float, upper: float) -> tuple[pd.Series, float, float]:
    """Full-sample percentiles. This uses later months. Labelled in-sample."""
    values = vrp.to_numpy(dtype=float)
    lo = float(np.quantile(values, lower, method="linear"))
    hi = float(np.quantile(values, upper, method="linear"))
    if not (vrp.index.max() == pd.Timestamp("2025-12-31") and len(vrp) == len(values)):
        raise AssertionError("In-sample cut points were not computed on the full VRP sample")
    clipped = vrp.clip(lower=lo, upper=hi)
    clipped.name = "VRP"
    return clipped, lo, hi


def winsor_realtime(vrp: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """1st/99th percentiles on VRP from 1990-01 through t. First 60 months dropped."""
    values = vrp.to_numpy(dtype=float)
    out = pd.Series(np.nan, index=vrp.index, name="VRP")
    rows = []
    for i, t in enumerate(vrp.index):
        window = vrp.iloc[: i + 1]
        if window.index.max() != t or (window.index > t).any():
            raise AssertionError(f"Real-time window at {pd.Timestamp(t).date()} uses a later month")
        if len(window) != i + 1:
            raise AssertionError(f"Real-time window at {pd.Timestamp(t).date()} is not 1990-01 through t")
        w = window.to_numpy(dtype=float)
        lo = float(np.quantile(w, 0.01, method="linear"))
        hi = float(np.quantile(w, 0.99, method="linear"))
        rows.append({"date": t, "p01": lo, "p99": hi, "n": int(len(window))})
        if i >= BURN_IN_MONTHS:
            out.iloc[i] = float(np.clip(values[i], lo, hi))
    cuts = pd.DataFrame(rows).set_index("date")
    return out, cuts


def assert_realtime_cuts(vrp: pd.Series, cuts: pd.DataFrame) -> None:
    """A shock to a later month leaves earlier cut points unchanged."""
    alt = vrp.copy()
    alt.iloc[-1] = float(alt.iloc[-1]) + 1.0e6
    _, cuts_alt = winsor_realtime(alt)
    earlier = cuts.index[:-1]
    if not np.allclose(cuts.loc[earlier, "p01"], cuts_alt.loc[earlier, "p01"]):
        raise AssertionError("A future VRP changed an earlier 1st percentile")
    if not np.allclose(cuts.loc[earlier, "p99"], cuts_alt.loc[earlier, "p99"]):
        raise AssertionError("A future VRP changed an earlier 99th percentile")
    last = cuts.index[-1]
    if np.isclose(cuts.loc[last, "p99"], cuts_alt.loc[last, "p99"]):
        raise AssertionError("The counterfactual did not move the last month's 99th percentile")
    for t, row in cuts.iterrows():
        window = vrp.loc[:t]
        if window.index.max() != t:
            raise AssertionError(f"Cut points at {pd.Timestamp(t).date()} see a later VRP")
        lo = float(np.quantile(window.to_numpy(dtype=float), 0.01, method="linear"))
        hi = float(np.quantile(window.to_numpy(dtype=float), 0.99, method="linear"))
        if not (np.isclose(lo, row["p01"]) and np.isclose(hi, row["p99"])):
            raise AssertionError(f"Stored cut points at {pd.Timestamp(t).date()} do not match the window")


def realized_components(close: pd.Series) -> pd.DataFrame:
    """RV^w and RV^d in monthly percent squared, from spec 001 daily log returns."""
    r2 = btz.daily_log_returns(close).dropna() ** 2
    rows = []
    for period, chunk in r2.groupby(r2.index.to_period("M")):
        if len(chunk) < 5:
            raise AssertionError(f"{period} has {len(chunk)} daily returns, need 5")
        last5 = chunk.iloc[-5:]
        last_day = chunk.index[-1]
        if last5.index.max().to_period("M") != period or last_day.to_period("M") != period:
            raise AssertionError(f"Weekly or daily RV for {period} leaves the month")
        rows.append(
            {
                "date": period.to_timestamp("M"),
                "RV_w": float(last5.sum() * (22.0 / 5.0) * 1e4),
                "RV_d": float(chunk.iloc[-1] * 22.0 * 1e4),
                "w_end": pd.Timestamp(last5.index.max()),
                "d_date": pd.Timestamp(last_day),
            }
        )
    out = pd.DataFrame(rows).set_index("date")
    out.index = pd.DatetimeIndex(out.index)
    return out


def _self_check_components() -> None:
    idx = pd.to_datetime(
        [
            "2019-12-31",
            "2020-01-02",
            "2020-01-03",
            "2020-01-06",
            "2020-01-07",
            "2020-01-08",
            "2020-01-09",
            "2020-01-10",
        ]
    )
    close = pd.Series(np.exp([0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]), index=idx)
    comp = realized_components(close)
    jan = pd.Timestamp("2020-01-31")
    if not np.isclose(comp.loc[jan, "RV_w"], 0.01 * 22.0 / 5.0 * 1e4):
        raise AssertionError("RV^w self-check failed")
    if not np.isclose(comp.loc[jan, "RV_d"], 0.01 * 22.0 * 1e4):
        raise AssertionError("RV^d self-check failed")
    if comp.loc[jan, "d_date"] != pd.Timestamp("2020-01-10"):
        raise AssertionError("RV^d is not the last trading day of the month")


def har_expanding(rv: pd.Series, rv_w: pd.Series, rv_d: pd.Series) -> pd.DataFrame:
    """Expanding-window HAR. The fit at t uses only pairs with s+1 <= t.

    Forecasts start once 60 pairs are available. Negative forecasts are floored
    at 0 and kept alongside the raw prediction.
    """
    if not rv.index.equals(rv_w.index) or not rv.index.equals(rv_d.index):
        raise AssertionError("HAR inputs are not on the same months")
    if rv.isna().any() or rv_w.isna().any() or rv_d.isna().any():
        raise AssertionError("HAR inputs have missing months")
    if not rv.index.equals(pd.DatetimeIndex(pd.date_range(rv.index.min(), rv.index.max(), freq="ME"))):
        raise AssertionError("HAR inputs are not a complete month-end calendar")

    rows = []
    y_all = rv.to_numpy(dtype=float)
    x_m = rv.to_numpy(dtype=float)
    x_w = rv_w.to_numpy(dtype=float)
    x_d = rv_d.to_numpy(dtype=float)
    for i, t in enumerate(rv.index):
        n_pairs = i  # pairs s = 0..i-1, dependent month s+1 = 1..i
        if n_pairs < BURN_IN_MONTHS:
            continue
        y = y_all[1 : i + 1]
        x = np.column_stack(
            [
                np.ones(n_pairs),
                x_m[:i],
                x_w[:i],
                x_d[:i],
            ]
        )
        last_s1 = rv.index[i]  # dependent month of the last pair
        if last_s1 > t or rv.index[i - 1].to_period("M") + 1 != t.to_period("M"):
            raise AssertionError(f"HAR fit at {pd.Timestamp(t).date()} uses a pair after t")
        fit = sm.OLS(y, x).fit()
        beta = np.asarray(fit.params, dtype=float)
        raw = float(beta @ np.array([1.0, x_m[i], x_w[i], x_d[i]]))
        rows.append(
            {
                "date": t,
                "c": float(beta[0]),
                "b_m": float(beta[1]),
                "b_w": float(beta[2]),
                "b_d": float(beta[3]),
                "n_pairs": int(n_pairs),
                "last_s1": last_s1,
                "pred_raw": raw,
                "pred": max(raw, 0.0),
                "floored": bool(raw < 0.0),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def assert_har_no_lookahead(rv: pd.Series, rv_w: pd.Series, rv_d: pd.Series, har: pd.DataFrame) -> None:
    """A shock to RV in month u leaves every fit dated before u unchanged."""
    shock_at = pd.Timestamp("2020-03-31")
    alt_rv = rv.copy()
    alt_rv.loc[shock_at] = float(alt_rv.loc[shock_at]) + 50.0
    alt = har_expanding(alt_rv, rv_w, rv_d)
    before = har.index[har.index < shock_at]
    for col in ("c", "b_m", "b_w", "b_d", "pred_raw"):
        if not np.allclose(har.loc[before, col], alt.loc[before, col]):
            raise AssertionError(f"HAR column {col} dated before 2020-03 moved when RV_2020-03 changed")
    if np.allclose(har.loc[shock_at, ["c", "b_m", "b_w", "b_d"]], alt.loc[shock_at, ["c", "b_m", "b_w", "b_d"]]):
        raise AssertionError("The 2020-03 shock did not enter the fit dated 2020-03")
    for t, row in har.iterrows():
        if row["last_s1"] > t:
            raise AssertionError(f"HAR last pair at {pd.Timestamp(t).date()} has s+1 after t")
        if int(row["n_pairs"]) < BURN_IN_MONTHS:
            raise AssertionError(f"HAR at {pd.Timestamp(t).date()} has fewer than 60 pairs")


def rolling_h3(sample: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """120 consecutive h=3 regression rows, stepped one month, dated at the last regressor."""
    y = 1200.0 * btz.forward_mean(sample["e"], 3)
    frame = pd.concat(
        [y.rename("y"), sample["VRP"].rename("VRP")],
        axis=1,
        sort=True,
    ).dropna()
    rows = []
    n_checked = 0
    for end in range(ROLL_MONTHS - 1, len(frame)):
        window = frame.iloc[end - ROLL_MONTHS + 1 : end + 1]
        model, idx = btz.fit_hac(window["y"], window["VRP"], max(1, 3 - 1))
        if len(idx) != ROLL_MONTHS:
            raise AssertionError(f"Rolling window has N={len(idx)}, expected {ROLL_MONTHS}")
        if pd.Timestamp(idx[-1]) != pd.Timestamp(window.index[-1]):
            raise AssertionError("Rolling date is not the window's last regressor month")
        for t in idx:
            assert_row_timing(sample, t, 3, "F rolling")
            n_checked += 1
        rows.append(
            {
                "date": pd.Timestamp(idx[-1]),
                "coef_B": float(model.params["VRP"]),
                "t_hm1": float(model.tvalues["VRP"]),
                "N": int(len(idx)),
            }
        )
    return pd.DataFrame(rows), n_checked


def plot_rolling(rolling: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.4), sharex=True)
    shade = [
        (pd.Timestamp("2008-01-01"), pd.Timestamp("2009-01-01")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")),
    ]
    axes[0].plot(rolling["date"], rolling["coef_B"], color="steelblue", lw=1.2)
    axes[0].axhline(0, color="black", lw=0.4)
    axes[0].set_ylabel("coef(B)")
    axes[0].set_title("Rolling h = 3, 120-month window")

    axes[1].plot(rolling["date"], rolling["t_hm1"], color="darkorange", lw=1.2)
    axes[1].axhline(1.96, color="gray", lw=0.8, ls="--", label="1.96")
    axes[1].axhline(0, color="black", lw=0.4)
    axes[1].set_ylabel("t, maxlags = h−1")
    axes[1].set_xlabel("Last regressor month of the window")
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        for i, (a, b) in enumerate(shade):
            ax.axvspan(a, b, color="0.85", zorder=0, label="2008 and 2020" if i == 0 else None)
    axes[0].legend(
        handles=[
            plt.Line2D([0], [0], color="steelblue", lw=1.2, label="coef(B)"),
            Patch(facecolor="0.85", edgecolor="none", label="2008 and 2020"),
        ],
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(full: pd.DataFrame, top: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.scatter(full["VRP_t"], full["Y_t"], s=14, color="steelblue", alpha=0.75, zorder=2)
    ax.scatter(top["VRP_t"], top["Y_t"], s=28, color="crimson", zorder=3)
    for i, row in top.iterrows():
        label = pd.Timestamp(row["date"]).strftime("%Y-%m")
        ax.annotate(
            label,
            (row["VRP_t"], row["Y_t"]),
            textcoords="offset points",
            xytext=(5, 6 if i % 2 == 0 else -9),
            fontsize=7.5,
            color="crimson",
        )
    ax.axhline(0, color="black", lw=0.4)
    ax.axvline(0, color="black", lw=0.4)
    ax.set_xlabel("VRP_t (monthly %²)")
    ax.set_ylabel("Y_t(3), annualized %")
    ax.set_title("h = 3, ten largest |DFBETA| months labelled")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_vrp(vrp: pd.Series, vrp_fwd: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.3))
    ax.plot(vrp.index, vrp.to_numpy(), color="steelblue", lw=1.0, label="VRP")
    ax.plot(vrp_fwd.index, vrp_fwd.to_numpy(), color="darkorange", lw=1.0, label="VRP_fwd")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylabel("monthly %²")
    ax.set_title("VRP and real-time HAR VRP_fwd")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _h3(result: dict) -> dict:
    return result["by_h"][3]


def _tops_crisis(top: pd.DataFrame) -> tuple[bool, int]:
    years = [pd.Timestamp(d).year for d in top["date"]]
    n_crisis = sum(year in (2008, 2020) for year in years)
    tops = years[0] in (2008, 2020) and n_crisis >= 5
    return tops, n_crisis


def _case_paragraph(name: str, holds: bool, when_true: str, when_false: str) -> str:
    if holds:
        return f"**{name}** holds. {when_true}"
    return f"**{name}** does not hold. {when_false}"


def _outlier_gap(b1: bool, b25: bool, c_sig: bool, d_sig: bool, tops: bool, h3: dict) -> str:
    missing = []
    if not b1:
        missing.append(f"B 1/99 has t = {_fmt(h3['B 1/99 in-sample']['h3_t_hm1'], 2)}")
    if not b25:
        missing.append(f"B 2.5/97.5 has t = {_fmt(h3['B 2.5/97.5 in-sample']['h3_t_hm1'], 2)}")
    if not c_sig:
        missing.append(f"C has t = {_fmt(h3['C real-time 1/99']['h3_t_hm1'], 2)}")
    if not d_sig:
        missing.append(f"D has t = {_fmt(h3['D drop crisis rows']['h3_t_hm1'], 2)}")
    if not tops:
        missing.append("the 2008/2020 months do not top the |DFBETA| list")
    return "It fails because " + "; ".join(missing) + "."


def _fragile_gap(b1: bool, b25: bool, c_sig: bool, d_sig: bool, h3: dict) -> str:
    parts = []
    if not (b1 and b25):
        parts.append("an in-sample winsorization is not above 1.96")
    if c_sig:
        parts.append(f"C is above 1.96 (t = {_fmt(h3['C real-time 1/99']['h3_t_hm1'], 2)})")
    if d_sig:
        parts.append(f"D is above 1.96 (t = {_fmt(h3['D drop crisis rows']['h3_t_hm1'], 2)})")
    if not parts:
        parts.append("the clauses do not match this reading")
    return (
        "It fails because " + ", and ".join(parts) + ". "
        "The in-sample cut is not the only change that puts t above 1.96."
    )


def _lasting_gap(e_insig: bool, c_insig: bool, d_insig: bool, rolling_low: bool) -> str:
    missing = []
    if not e_insig:
        missing.append("the raw 2010–2025 regression is above 1.96")
    if not c_insig:
        missing.append("2010–2025 under the real-time winsorization is above 1.96")
    if not d_insig:
        missing.append("2010–2025 after dropping 2020-02 through 2020-04 is above 1.96")
    if not rolling_low:
        missing.append("the rolling t does not stay low after 2010")
    return "It fails because " + "; ".join(missing) + "."


def _rolling_low(rolling: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    after = rolling[rolling["date"] >= pd.Timestamp("2010-01-31")]
    share = float((after["t_hm1"] > 1.96).mean())
    low = float(after["t_hm1"].median()) < 1.96 and share < 0.5
    return low, after


def write_summary(payload: dict) -> None:
    robust = payload["robust"].set_index("variant")
    top = payload["top"]
    cuts = payload["cuts"]
    har = payload["har"]
    rolling = payload["rolling"]
    dropped = payload["dropped"]
    h3 = {name: robust.loc[name] for name in VARIANT_ORDER}

    b1 = float(h3["B 1/99 in-sample"]["h3_t_hm1"]) > 1.96
    b25 = float(h3["B 2.5/97.5 in-sample"]["h3_t_hm1"]) > 1.96
    c_sig = float(h3["C real-time 1/99"]["h3_t_hm1"]) > 1.96
    d_sig = float(h3["D drop crisis rows"]["h3_t_hm1"]) > 1.96
    tops, n_crisis = _tops_crisis(top)
    outlier = b1 and b25 and c_sig and d_sig and tops
    # "Only B restores significance": both in-sample cuts are significant, C and D are not.
    fragile = b1 and b25 and (not c_sig) and (not d_sig)

    c2010 = _h3(payload["c_2010"])
    d2010 = _h3(payload["d_2010"])
    e2010_t = float(h3["E 2010-2025"]["h3_t_hm1"])
    e2010_insig = e2010_t <= 1.96
    c2010_insig = c2010["t_hm1"] <= 1.96
    d2010_insig = d2010["t_hm1"] <= 1.96
    rolling_low, after = _rolling_low(rolling)
    lasting = e2010_insig and c2010_insig and d2010_insig and rolling_low
    pre_2020 = rolling[
        (rolling["date"] >= pd.Timestamp("2010-01-31"))
        & (rolling["date"] < pd.Timestamp("2020-01-31"))
    ]
    from_2020 = rolling[rolling["date"] >= pd.Timestamp("2020-03-31")]
    above = after.loc[after["t_hm1"] > 1.96, "date"]
    last_above = pd.Timestamp(above.max()) if len(above) else None

    lines: list[str] = []
    lines.append("# Spec 002 — Why the full-sample result collapses")
    lines.append("")
    lines.append("## Definitions")
    lines.append("")
    lines.append(
        "The sample, the variance series and specification (B) are the R2 regression of spec 001. "
        "`src/baseline_1990.py` is imported and not modified. "
        "Index `^GSPC`, implied variance from `data/vix_1990_data.csv`, risk-free rate TB3MS lagged one month. "
        "VRP is in monthly percent squared. Specification (B) multiplies the h-month average excess return by 1200. "
        "The sample is 1990-01 through 2025-12. Horizons are h = 1, 3 and 6. "
        "Newey–West standard errors use `statsmodels` HAC with `use_correction=False`. "
        "t(h−1) uses maxlags = max(1, h−1). t(h) uses maxlags = h. R² is in percent."
    )
    lines.append("")
    lines.append(
        "Row 0 is that regression with the raw VRP. "
        "B clips VRP at percentiles computed on the whole 1990–2025 sample, so those cut points use later months. "
        "B is labelled in-sample. "
        "C clips VRP_t at the 1st and 99th percentiles of VRP from 1990-01 through t. "
        "The first 60 months are a burn-in: regression rows before 1995-01 are dropped from C only. "
        "D drops regression rows whose regressor month is in 2008-09 through 2008-12 or 2020-02 through 2020-04. "
        "The forward returns of the other rows are unchanged. "
        "E re-runs the regression on 1990–2007, 2008–2025 and 2010–2025, and each h-month window stops at the end of that subsample."
    )
    lines.append("")
    lines.append(
        "F estimates the h = 3 regression on a rolling window of 120 consecutive regression rows, "
        "stepped one month at a time. Each point is dated at that window's last regressor month. "
        "The three outcome months of the last row are inside the R2 sample and after that date. "
        "N = 120 in every window."
    )
    lines.append("")
    lines.append(
        "G replaces VRP with VRP_fwd_t = IV_t − Ê_t[RV_{t+1}]. "
        "The forecast is an expanding-window monthly HAR, estimated by OLS on pairs (s, s+1) with s+1 ≤ t: "
        "RV_{s+1} on a constant, RV_s, RV^w_s and RV^d_s. "
        "RV_s is the spec 001 realized variance. "
        "RV^w_s is the sum of squared daily log returns over the last five trading days of month s, times 22/5, times 10⁴. "
        "RV^d_s is the squared daily log return of the last trading day of month s, times 22, times 10⁴. "
        "Forecasts start once 60 pairs are available, so the first forecast month is 1995-01. "
        "A negative forecast is floored at 0. "
        f"That floor binds in {payload['n_floored']} months."
    )
    lines.append("")
    lines.append(
        "DFBETA on the row-0 h = 3 OLS is coef(B) minus the coefficient estimated without that row. "
        "Leverage is the diagonal of the hat matrix. Cook's distance is the OLS Cook statistic. "
        f"The hat diagonals of the {payload['n_influence']} regression rows sum to 2."
    )
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("### 1. Row 0 reproduces spec 001 R2 at h = 3")
    lines.append("")
    row0 = h3["0"]
    lines.append(
        f"Pass. coef(B) = {_fmt(row0['h3_coef_B'], 4)} and t(h−1) = {_fmt(row0['h3_t_hm1'], 2)}. "
        f"The unrounded estimates are {row0['h3_coef_B']:.6f} and {row0['h3_t_hm1']:.6f}, "
        "the same point as `results/001-baseline-1990/table2_main.csv` for sample R2 at h = 3. "
        "h = 1 and h = 6 of row 0 match that file as well."
    )
    lines.append("")
    lines.append("### 2. Real-time cut points")
    lines.append("")
    lines.append(
        "Pass. For every month t, the 1st and 99th percentiles are the linear quantiles of VRP from 1990-01 through t. "
        "Adding 10⁶ to VRP in 2025-12 leaves every earlier cut point unchanged and moves the 2025-12 99th percentile. "
        "Printed cut points:"
    )
    lines.append("")
    lines.append("| date | p01 | p99 | months in the window |")
    lines.append("|---|---|---|---|")
    for stamp in ("1999-12-31", "2007-12-31", "2008-12-31", "2020-12-31"):
        row = cuts.loc[pd.Timestamp(stamp)]
        lines.append(
            f"| {stamp[:7]} | {_fmt(row['p01'], 3)} | {_fmt(row['p99'], 3)} | {int(row['n'])} |"
        )
    lines.append("")
    lines.append(
        f"In-sample B, for comparison, uses p01 = {_fmt(payload['b_199'][0], 3)} and "
        f"p99 = {_fmt(payload['b_199'][1], 3)} on all {payload['n_months']} months, and "
        f"p2.5 = {_fmt(payload['b_25'][0], 3)} and p97.5 = {_fmt(payload['b_25'][1], 3)}. "
        "Those four numbers use the whole sample."
    )
    lines.append("")
    lines.append("### 3. HAR")
    lines.append("")
    lines.append(
        "Pass. For every forecast month t, the last pair in the fit has s+1 = t, hence s+1 ≤ t, "
        "and the fit uses 60 or more pairs. "
        "Adding 50 to RV in 2020-03 leaves the coefficients and the raw forecast at every earlier month unchanged, "
        "and changes the coefficients dated 2020-03. "
        "The prediction at t is applied to RV_t, RV^w_t and RV^d_t. "
        "The last trading day inside those two components falls in month t."
    )
    lines.append("")
    lines.append("| date | c | β_m | β_w | β_d | pairs | last s+1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for stamp in ("2007-12-31", "2025-12-31"):
        row = har.loc[pd.Timestamp(stamp)]
        lines.append(
            f"| {stamp[:7]} | {_fmt(row['c'], 3)} | {_fmt(row['b_m'], 3)} | {_fmt(row['b_w'], 3)} | "
            f"{_fmt(row['b_d'], 3)} | {int(row['n_pairs'])} | {pd.Timestamp(row['last_s1']).strftime('%Y-%m')} |"
        )
    lines.append("")
    lines.append(
        f"Correlation between VRP_fwd and VRP, on the {payload['corr_n']} months from 1995-01 through 2025-12 "
        f"where the forecast exists: **{_fmt(payload['corr'], 3)}**. "
        f"Forecasts below 0 are floored at 0 in **{payload['n_floored']}** months."
    )
    lines.append("")
    lines.append("### 4. N on every robustness row")
    lines.append("")
    lines.append(
        "Pass. The table below reports N at h = 1, 3 and 6 for every variant. "
        "N equals the number of regression rows after the variant's own sample rule and after dropping incomplete h-month windows."
    )
    lines.append("")
    lines.append("### 5. No look-ahead in X, and Y starts at t+1")
    lines.append("")
    lines.append(
        f"Pass on the spec 001 timing rule. The code checked {payload['n_checked']} regression rows: "
        "row 0, B, C, D, the three E samples, both G samples, every row of every 120-month window, "
        "and the two 2010–2025 readings used for the third case below. "
        "For each row t, the last daily-return date inside RV_t and the last VIX date inside IV_t fall in month t, "
        "and the later of those dates is on or before the month-end label. "
        "The first excess return inside Y_t is the return of month t+1, and the window runs through month t+h, "
        "inside the sample of that variant."
    )
    lines.append("")
    lines.append(
        "B is the labelled in-sample exception. Its clipped regressor uses percentiles computed through 2025-12, "
        "so a row t in B depends on later VRP values. C and G are the real-time forms of the two fixes, "
        "and checks 2 and 3 cover them. D drops a fixed list of months. It does not recompute VRP from later data."
    )
    lines.append("")
    lines.append("## Robustness table")
    lines.append("")
    lines.append(
        "Coefficient (B) is annualized percent on monthly percent-squared VRP. "
        "t(h−1) uses maxlags = max(1, h−1). t(h) uses maxlags = h. R² is in percent. "
        "The same numbers are in `table_robust.csv`, one row per variant."
    )
    lines.append("")
    lines.append("| variant | h | coef(B) | t(h−1) | t(h) | R² | N |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in VARIANT_ORDER:
        for h in HORIZONS:
            lines.append(
                f"| {name} | {h} | {_fmt(robust.loc[name, f'h{h}_coef_B'], 4)} | "
                f"{_fmt(robust.loc[name, f'h{h}_t_hm1'], 2)} | "
                f"{_fmt(robust.loc[name, f'h{h}_t_h'], 2)} | "
                f"{_fmt(robust.loc[name, f'h{h}_R2'], 2)} | "
                f"{int(robust.loc[name, f'h{h}_N'])} |"
            )
    lines.append("")
    lines.append(
        "D drops these regressor months: "
        + ", ".join(pd.Timestamp(t).strftime("%Y-%m") for t in dropped)
        + f". That is {len(dropped)} rows. They are inside the regression sample at every horizon, "
        "so N = 432 − h − 7."
    )
    lines.append("")
    lines.append(
        f"G is not one of the three cases. On the full sample its h = 3 t(h−1) is "
        f"{_fmt(h3['G full sample']['h3_t_hm1'], 2)}; on 1990–2007 it is "
        f"{_fmt(h3['G 1990-2007']['h3_t_hm1'], 2)}. "
        "The 1990–2007 regression starts in 1995-01 because the HAR burn-in leaves VRP_fwd undefined before then. "
        "`fig_vrp_vs_vrpfwd.png` plots the two premia."
    )
    lines.append("")
    lines.append("## Which case holds")
    lines.append("")
    lines.append(
        "The audit reference, not a target of this spec, is a SPY sample from 1993 to 2025 with full-sample "
        "1/99 winsorization: h = 3, t = 2.11, R² = 4.2%. "
        f"B 1/99 on ^GSPC for 1990–2025 has t(h−1) = {_fmt(h3['B 1/99 in-sample']['h3_t_hm1'], 2)} "
        f"and R² = {_fmt(h3['B 1/99 in-sample']['h3_R2'], 2)}%."
    )
    lines.append("")
    lines.append(
        "The ten rows with the largest |DFBETA| are below and in `table_influence.csv`. "
        "Y_t is specification (B) at h = 3."
    )
    lines.append("")
    lines.append("| date | VRP_t | Y_t | DFBETA | Cook's D |")
    lines.append("|---|---|---|---|---|")
    for _, row in top.iterrows():
        lines.append(
            f"| {pd.Timestamp(row['date']).strftime('%Y-%m')} | {_fmt(row['VRP_t'], 2)} | "
            f"{_fmt(row['Y_t'], 2)} | {_fmt(row['DFBETA'], 4)} | {_fmt(row['Cooks_D'], 4)} |"
        )
    lines.append("")
    lines.append(
        f"Of these ten, {n_crisis} fall in 2008 or 2020. "
        f"The largest |DFBETA| is {pd.Timestamp(top.iloc[0]['date']).strftime('%Y-%m')}."
    )
    lines.append("")
    lines.append(
        "Clauses, all at h = 3 and using t(h−1). "
        "A clause counts as significant when t > 1.96."
    )
    lines.append("")
    lines.append(
        f"- B 1/99 in-sample significant: {'yes' if b1 else 'no'} "
        f"(t = {_fmt(h3['B 1/99 in-sample']['h3_t_hm1'], 2)})."
    )
    lines.append(
        f"- B 2.5/97.5 in-sample significant: {'yes' if b25 else 'no'} "
        f"(t = {_fmt(h3['B 2.5/97.5 in-sample']['h3_t_hm1'], 2)})."
    )
    lines.append(
        f"- C real-time 1/99 significant: {'yes' if c_sig else 'no'} "
        f"(t = {_fmt(h3['C real-time 1/99']['h3_t_hm1'], 2)})."
    )
    lines.append(
        f"- D crisis rows dropped, significant: {'yes' if d_sig else 'no'} "
        f"(t = {_fmt(h3['D drop crisis rows']['h3_t_hm1'], 2)})."
    )
    lines.append(
        f"- 2008/2020 months top the |DFBETA| list: {'yes' if tops else 'no'} "
        f"({n_crisis} of the ten rows, and the first row is in 2008 or 2020)."
    )
    lines.append(
        f"- E 2010–2025 insignificant: {'yes' if e2010_insig else 'no'} (t = {_fmt(e2010_t, 2)})."
    )
    lines.append(
        f"- 2010–2025 with the real-time 1/99 VRP, insignificant: {'yes' if c2010_insig else 'no'} "
        f"(t = {_fmt(c2010['t_hm1'], 2)}, N = {c2010['N']})."
    )
    lines.append(
        f"- 2010–2025 with 2020-02 through 2020-04 dropped, insignificant: {'yes' if d2010_insig else 'no'} "
        f"(t = {_fmt(d2010['t_hm1'], 2)}, N = {d2010['N']})."
    )
    lines.append(
        f"- Rolling t stays low after 2010: {'yes' if rolling_low else 'no'}. "
        f"On the {len(after)} windows dated 2010-01 or later, the median t is {_fmt(float(after['t_hm1'].median()), 2)}, "
        f"the maximum is {_fmt(float(after['t_hm1'].max()), 2)}, "
        f"and the share with t > 1.96 is {_fmt(100.0 * float((after['t_hm1'] > 1.96).mean()), 1)}%. "
        f"On the {len(pre_2020)} windows dated 2010-01 through 2019-12 the median t is "
        f"{_fmt(float(pre_2020['t_hm1'].median()), 2)}. "
        f"On the {len(from_2020)} windows dated 2020-03 or later the median t is "
        f"{_fmt(float(from_2020['t_hm1'].median()), 2)}. "
        f"The last window dated 2010 or later with t > 1.96 ends in "
        f"{last_above.strftime('%Y-%m') if last_above is not None else 'no month'}."
    )
    lines.append("")
    lines.append(_case_paragraph(
        "Outlier story",
        outlier,
        "B, C and D are all above 1.96, and the 2008/2020 months top the |DFBETA| list.",
        _outlier_gap(b1, b25, c_sig, d_sig, tops, h3),
    ))
    lines.append("")
    lines.append(_case_paragraph(
        "Fragile",
        fragile,
        "Both in-sample winsorizations are above 1.96, and neither C nor D is.",
        _fragile_gap(b1, b25, c_sig, d_sig, h3),
    ))
    lines.append("")
    lines.append(_case_paragraph(
        "Lasting change",
        lasting,
        "2010–2025 stays insignificant on the raw sample, under the real-time winsorization and after the 2020 rows are dropped, and the rolling t stays low after 2010.",
        _lasting_gap(e2010_insig, c2010_insig, d2010_insig, rolling_low),
    ))
    lines.append("")
    lines.append("No threshold was changed after seeing the t-statistics.")
    lines.append("")
    lines.append(
        "The two 2010–2025 lines are the readings named in the third case. "
        "They are not extra rows of the robustness table, and they were not used to choose a winsorization rule."
    )
    lines.append("")
    lines.append("## Deviations from this spec")
    lines.append("")
    lines.append(
        "1. **DFBETA is coef(B) minus the coefficient without that row.** "
        "That is the textbook leave-one-out change, not DFBETA divided by the leave-one-out standard error. "
        "The ten rows are ranked by the absolute value of this change."
    )
    lines.append("")
    lines.append(
        "2. **The 2010–2025 readings under C and under D are not rows of the robustness table.** "
        "The third case asks whether 2010–2025 stays insignificant after those two treatments. "
        "The two t-statistics are reported in the case section. "
        "The real-time percentiles in the C reading are still the expanding window from 1990-01 through t."
    )
    lines.append("")
    (RESULTS / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_r2() -> tuple[pd.DataFrame, pd.Series]:
    gspc = btz.load_close(btz.DATA / "gspc_data.csv")
    vix = btz.load_close(btz.DATA / "vix_1990_data.csv")
    rf = btz.load_rf()
    panel = btz.build_panel(gspc, vix, rf)
    sample = btz.slice_sample(panel, "1990-01-31", "2025-12-31")
    needed = ["IV", "RV", "VRP", "e", "rv_max", "iv_max"]
    if sample[needed].isna().any().any():
        raise AssertionError("R2 sample has missing values")
    if len(sample) != 432:
        raise AssertionError(f"R2 sample has {len(sample)} months, expected 432")
    return sample, gspc


def _assert_row0(result: dict) -> None:
    saved = pd.read_csv(btz.RESULTS / "table2_main.csv")
    ref = saved[saved["sample"] == "R2"].set_index("h")
    for h in HORIZONS:
        got = result["by_h"][h]
        row = ref.loc[h]
        for key, col in (
            ("coef_B", "coef_B"),
            ("t_hm1", "t_h_minus_1"),
            ("t_h", "t_h"),
            ("R2", "R2"),
            ("N", "N"),
        ):
            if not np.isclose(got[key], row[col], rtol=0, atol=1e-10):
                raise AssertionError(
                    f"Row 0 h={h} {key} = {got[key]}, spec 001 R2 has {row[col]}"
                )
    h3 = result["by_h"][3]
    if round(h3["coef_B"], 4) != REF_COEF_H3:
        raise AssertionError(f"coef(B) rounds to {round(h3['coef_B'], 4)}, expected {REF_COEF_H3}")
    if round(h3["t_hm1"], 2) != REF_T_H3:
        raise AssertionError(f"t(h−1) rounds to {round(h3['t_hm1'], 2)}, expected {REF_T_H3}")


def _assert_components_match_rv(sample: pd.DataFrame, gspc: pd.Series) -> pd.DataFrame:
    comp = realized_components(gspc).reindex(sample.index)
    if comp[["RV_w", "RV_d", "w_end", "d_date"]].isna().any().any():
        raise AssertionError("A sample month has no weekly or daily RV")
    r2 = btz.daily_log_returns(gspc).dropna() ** 2
    rebuilt = r2.resample("ME").sum(min_count=1) * 1e4
    rebuilt.index = pd.DatetimeIndex(rebuilt.index).to_period("M").to_timestamp("M")
    gap = (rebuilt.reindex(sample.index) - sample["RV"]).abs().max()
    if not np.isclose(gap, 0.0, atol=1e-8):
        raise AssertionError(f"Rebuilt RV differs from spec 001 RV by {gap}")
    return comp


def main() -> dict:
    _self_check_components()
    sample, gspc = _load_r2()
    comp = _assert_components_match_rv(sample, gspc)
    for t, row in comp.iterrows():
        if pd.Timestamp(row["w_end"]).to_period("M") != pd.Timestamp(t).to_period("M"):
            raise AssertionError(f"RV^w at {pd.Timestamp(t).date()} leaves the month")
        if pd.Timestamp(row["d_date"]).to_period("M") != pd.Timestamp(t).to_period("M"):
            raise AssertionError(f"RV^d at {pd.Timestamp(t).date()} leaves the month")

    vrp = sample["VRP"]
    row0 = run_regression(sample, vrp, "0")
    _assert_row0(row0)

    full_infl, top = influence_h3(sample)
    if not np.isclose(_h3(row0)["coef_B"], sm.OLS(
        full_infl["Y_t"], sm.add_constant(full_infl["VRP_t"])
    ).fit().params.iloc[1]):
        raise AssertionError("Influence OLS coefficient is not row-0 coef(B)")

    vrp_b1, lo1, hi1 = winsor_insample(vrp, 0.01, 0.99)
    vrp_b25, lo25, hi25 = winsor_insample(vrp, 0.025, 0.975)
    b1 = run_regression(sample, vrp_b1, "B 1/99 in-sample")
    b25 = run_regression(sample, vrp_b25, "B 2.5/97.5 in-sample")

    vrp_c, cuts = winsor_realtime(vrp)
    assert_realtime_cuts(vrp, cuts)
    first_c = vrp.index[BURN_IN_MONTHS]
    if first_c != pd.Timestamp("1995-01-31"):
        raise AssertionError(f"C burn-in ends at {pd.Timestamp(first_c).date()}, expected 1995-01")
    if vrp_c.loc[: pd.Timestamp("1994-12-31")].notna().any():
        raise AssertionError("C keeps a row before 1995-01")
    c_reg = run_regression(sample, vrp_c, "C real-time 1/99")
    if _h3(c_reg)["index"].min() != pd.Timestamp("1995-01-31"):
        raise AssertionError("C regression starts before 1995-01")

    dropped = crisis_months(sample.index)
    vrp_d = vrp.copy()
    vrp_d.loc[dropped] = np.nan
    d_reg = run_regression(sample, vrp_d, "D drop crisis rows")
    for h in HORIZONS:
        present = d_reg["by_h"][h]["index"].intersection(dropped)
        if len(present):
            raise AssertionError(f"D still contains crisis rows at h={h}")
        if d_reg["by_h"][h]["N"] != 432 - h - len(dropped):
            raise AssertionError(f"D N at h={h} is {d_reg['by_h'][h]['N']}")

    e_1990 = run_regression(
        btz.slice_sample(sample, "1990-01-31", "2007-12-31"), vrp, "E 1990-2007"
    )
    e_2008 = run_regression(
        btz.slice_sample(sample, "2008-01-31", "2025-12-31"), vrp, "E 2008-2025"
    )
    e_2010 = run_regression(
        btz.slice_sample(sample, "2010-01-31", "2025-12-31"), vrp, "E 2010-2025"
    )
    saved = pd.read_csv(btz.RESULTS / "table2_main.csv")
    main_h3 = saved[(saved["sample"] == "main") & (saved["h"] == 3)].iloc[0]
    if not np.isclose(_h3(e_1990)["coef_B"], main_h3["coef_B"], rtol=0, atol=1e-10):
        raise AssertionError("E 1990-2007 does not reproduce the spec 001 main sample")

    har = har_expanding(sample["RV"], comp["RV_w"], comp["RV_d"])
    assert_har_no_lookahead(sample["RV"], comp["RV_w"], comp["RV_d"], har)
    if har.index.min() != pd.Timestamp("1995-01-31"):
        raise AssertionError(f"First HAR forecast is {har.index.min().date()}, expected 1995-01")
    if int(har.iloc[0]["n_pairs"]) != BURN_IN_MONTHS:
        raise AssertionError("The first HAR forecast does not use exactly 60 pairs")
    for t, row in har.iterrows():
        if pd.Timestamp(comp.at[t, "d_date"]).to_period("M") != pd.Timestamp(t).to_period("M"):
            raise AssertionError(f"VRP_fwd at {pd.Timestamp(t).date()} uses a daily return outside t")
        if pd.Timestamp(comp.at[t, "w_end"]).to_period("M") != pd.Timestamp(t).to_period("M"):
            raise AssertionError(f"VRP_fwd at {pd.Timestamp(t).date()} uses a weekly return outside t")

    vrp_fwd = (sample["IV"] - har["pred"]).rename("VRP")
    n_floored = int(har["floored"].sum())
    both = pd.concat([vrp.rename("VRP"), vrp_fwd.rename("VRP_fwd")], axis=1).dropna()
    corr = float(both["VRP"].corr(both["VRP_fwd"]))
    g_full = run_regression(sample, vrp_fwd, "G full sample")
    g_early = run_regression(
        btz.slice_sample(sample, "1990-01-31", "2007-12-31"),
        vrp_fwd,
        "G 1990-2007",
    )
    if _h3(g_full)["index"].min() != pd.Timestamp("1995-01-31"):
        raise AssertionError("G full sample starts before the HAR burn-in")
    if _h3(g_early)["index"].min() != pd.Timestamp("1995-01-31"):
        raise AssertionError("G 1990-2007 starts before the HAR burn-in")
    if _h3(g_early)["index"].max() > pd.Timestamp("2007-12-31"):
        raise AssertionError("G 1990-2007 uses a regressor after 2007-12")

    c_2010 = run_regression(
        btz.slice_sample(sample, "2010-01-31", "2025-12-31"),
        vrp_c,
        "C on 2010-2025",
    )
    vrp_d_2010 = vrp.copy()
    drop_2010 = dropped[dropped >= pd.Timestamp("2010-01-31")]
    vrp_d_2010.loc[drop_2010] = np.nan
    d_2010 = run_regression(
        btz.slice_sample(sample, "2010-01-31", "2025-12-31"),
        vrp_d_2010,
        "D on 2010-2025",
    )

    rolling, n_roll = rolling_h3(sample)
    if len(rolling) != _h3(row0)["N"] - ROLL_MONTHS + 1:
        raise AssertionError("Unexpected number of rolling windows")
    if (rolling["N"] != ROLL_MONTHS).any():
        raise AssertionError("A rolling window does not have 120 rows")

    results = [row0, b1, b25, c_reg, d_reg, e_1990, e_2008, e_2010, g_full, g_early]
    if [r["name"] for r in results] != list(VARIANT_ORDER):
        raise AssertionError("Robustness rows are not in the spec order")
    robust = pd.DataFrame([wide_row(r) for r in results])

    n_checked = sum(r["n_checked"] for r in results) + n_roll + c_2010["n_checked"] + d_2010["n_checked"]

    print("Check 2 cut points")
    for stamp in ("1999-12-31", "2007-12-31", "2008-12-31", "2020-12-31"):
        row = cuts.loc[pd.Timestamp(stamp)]
        print(f"  {stamp[:7]}  p01={row['p01']:.4f}  p99={row['p99']:.4f}  n={int(row['n'])}")
    print("Check 3 HAR coefficients")
    for stamp in ("2007-12-31", "2025-12-31"):
        row = har.loc[pd.Timestamp(stamp)]
        print(
            f"  {stamp[:7]}  c={row['c']:.4f}  b_m={row['b_m']:.4f}  "
            f"b_w={row['b_w']:.4f}  b_d={row['b_d']:.4f}  "
            f"pairs={int(row['n_pairs'])}  last_s1={pd.Timestamp(row['last_s1']).strftime('%Y-%m')}"
        )
    print(f"corr(VRP_fwd, VRP) = {corr:.4f}  on {len(both)} months")
    print(f"floored forecasts = {n_floored}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    top_out = top.copy()
    top_out["date"] = pd.to_datetime(top_out["date"]).dt.strftime("%Y-%m-%d")
    top_out.to_csv(RESULTS / "table_influence.csv", index=False)
    robust.to_csv(RESULTS / "table_robust.csv", index=False)
    plot_rolling(rolling, RESULTS / "fig_rolling_t.png")
    plot_scatter(full_infl, top, RESULTS / "fig_scatter_h3.png")
    plot_vrp(vrp, vrp_fwd, RESULTS / "fig_vrp_vs_vrpfwd.png")

    payload = {
        "robust": robust,
        "top": top,
        "cuts": cuts,
        "har": har,
        "rolling": rolling,
        "dropped": dropped,
        "b_199": (lo1, hi1),
        "b_25": (lo25, hi25),
        "n_months": int(len(sample)),
        "n_floored": n_floored,
        "corr": corr,
        "corr_n": int(len(both)),
        "n_checked": n_checked,
        "n_influence": int(len(full_infl)),
        "c_2010": c_2010,
        "d_2010": d_2010,
    }
    write_summary(payload)
    print(f"Wrote {RESULTS}")
    show = robust.copy()
    for col in show.columns:
        if col != "variant" and not col.endswith("_N"):
            show[col] = show[col].map(lambda v: round(float(v), 4))
    print(show.to_string(index=False))
    return payload


if __name__ == "__main__":
    main()
