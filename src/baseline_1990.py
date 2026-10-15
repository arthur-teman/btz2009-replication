"""Spec 001 — BTZ (2009) Table 2 on 1990–2007, in the paper's units.

Reads frozen CSVs, builds monthly IV, RV and VRP, and runs the predictive
regressions. Writes results/001-baseline-1990/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "001-baseline-1990"

# Yahoo window requested by the spec. Files are frozen on first download.
YAHOO_START = "1989-12-01"
YAHOO_END = "2026-01-01"
YAHOO_DOWNLOAD_DATE = "2026-10-15"

HORIZONS = [1, 3, 6, 9, 12, 24]
BTZ_TARGETS = {
    1: {"coef": 0.39, "t": 1.76, "R2": 1.07},
    3: {"coef": 0.47, "t": 2.86, "R2": 6.82},
    6: {"coef": 0.30, "t": 2.15, "R2": 5.42},
    12: {"coef": 0.12, "t": 1.00, "R2": 1.23},
}

SAMPLES = (
    # name, start month-end, end month-end, price source
    ("main", "1990-01-31", "2007-12-31", "gspc"),
    ("R1", "1993-02-28", "2007-12-31", "spy"),
    ("R2", "1990-01-31", "2025-12-31", "gspc"),
)


def _flatten_yahoo(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        elif "Close" in df.columns.get_level_values(-1):
            df.columns = df.columns.get_level_values(-1)
        else:
            raise RuntimeError(f"Unrecognised Yahoo columns: {list(df.columns)}")
    return df


def download_yahoo(ticker: str, path: Path) -> None:
    """Download daily bars once. Never overwrites an existing freeze."""
    if path.exists():
        return
    import yfinance as yf

    df = yf.download(
        ticker,
        start=YAHOO_START,
        end=YAHOO_END,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned no rows for {ticker}")
    df = _flatten_yahoo(df)
    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    keep = ["Date", "Close", "High", "Low", "Open", "Volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ticker} is missing {missing}; got {list(df.columns)}")
    df = df[keep].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if getattr(df["Date"].dt, "tz", None) is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)
    df = df.sort_values("Date")
    df = df[(df["Date"] >= YAHOO_START) & (df["Date"] < YAHOO_END)]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_close(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    if getattr(df["Date"].dt, "tz", None) is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)
    df = df.dropna(subset=["Close"]).sort_values("Date")
    df = df.drop_duplicates("Date", keep="last")
    close = df.set_index("Date")["Close"].astype(float)
    close.index.name = "Date"
    return close


def load_rf() -> pd.Series:
    """rf_t = TB3MS_{t-1} / 1200, indexed at month end t."""
    tb = pd.read_csv(DATA / "tb3ms.csv", parse_dates=["DATE"])
    s = tb.set_index("DATE")["TB3MS"].astype(float).sort_index()
    s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp("M")
    s = s[~s.index.duplicated(keep="last")]
    rf = s.shift(1) / 1200.0
    rf.name = "rf"
    return rf


def daily_log_returns(close: pd.Series) -> pd.Series:
    close = close.sort_index().astype(float)
    return np.log(close).diff()


def monthly_rv(close: pd.Series) -> pd.Series:
    """RV in monthly percent squared. Empty months are NaN, not 0."""
    r2 = daily_log_returns(close) ** 2
    rv = r2.resample("ME").sum(min_count=1) * 1e4
    rv.index = pd.DatetimeIndex(rv.index).to_period("M").to_timestamp("M")
    rv.name = "RV"
    return rv


def monthly_iv(vix: pd.Series) -> pd.Series:
    """IV_t = VIX^2 on the last print of month t, divided by 12."""
    eom = vix.sort_index().astype(float).resample("ME").last()
    iv = (eom ** 2) / 12.0
    iv.index = pd.DatetimeIndex(iv.index).to_period("M").to_timestamp("M")
    iv.name = "IV"
    return iv


def monthly_log_return(close: pd.Series) -> pd.Series:
    eom = close.sort_index().astype(float).resample("ME").last()
    r = np.log(eom).diff()
    r.index = pd.DatetimeIndex(r.index).to_period("M").to_timestamp("M")
    r.name = "r"
    return r


def x_max_date(close: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """Last return date in RV, last VIX date in IV, and the later of the two."""
    ret_dates = daily_log_returns(close).dropna().index
    rv_max = pd.Series(ret_dates, index=ret_dates).resample("ME").max()
    vix_ok = vix.dropna()
    iv_max = pd.Series(vix_ok.index, index=vix_ok.index).resample("ME").max()
    both = pd.concat(
        [rv_max.rename("rv_max"), iv_max.rename("iv_max")],
        axis=1,
        sort=True,
    )
    both.index = pd.DatetimeIndex(both.index).to_period("M").to_timestamp("M")
    both["x_max"] = both[["rv_max", "iv_max"]].max(axis=1)
    return both


def build_panel(close: pd.Series, vix: pd.Series, rf: pd.Series) -> pd.DataFrame:
    df = pd.concat(
        [
            monthly_iv(vix),
            monthly_rv(close),
            monthly_log_return(close),
            rf,
            x_max_date(close, vix),
        ],
        axis=1,
        sort=True,
    )
    df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp("M")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["VRP"] = df["IV"] - df["RV"]
    df["e"] = df["r"] - df["rf"]
    return df


def slice_sample(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="ME")
    return panel.reindex(idx)


def forward_mean(e: pd.Series, h: int) -> pd.Series:
    """(1/h) * sum_{j=1..h} e_{t+j}. Incomplete windows are NaN."""
    window = pd.concat([e.shift(-j) for j in range(1, h + 1)], axis=1, sort=True)
    return window.mean(axis=1, skipna=False)


def fit_hac(y: pd.Series, x: pd.Series, maxlags: int):
    frame = pd.concat([y.rename("y"), x.rename("VRP")], axis=1, sort=True).dropna()
    model = sm.OLS(frame["y"], sm.add_constant(frame["VRP"])).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(maxlags), "use_correction": False},
    )
    return model, frame.index


def ar1(series: pd.Series) -> float:
    """OLS slope of x_t on a constant and x_{t-1}."""
    s = series.dropna()
    y = s.iloc[1:].to_numpy(dtype=float)
    x = sm.add_constant(s.iloc[:-1].to_numpy(dtype=float))
    return float(sm.OLS(y, x).fit().params[1])


def summarize(sample: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for variable in ["IV", "RV", "VRP", "e"]:
        s = sample[variable]
        rows.append(
            {
                "sample": name,
                "variable": variable,
                "mean": float(s.mean()),
                "sd": float(s.std(ddof=1)),
                "min": float(s.min()),
                "max": float(s.max()),
                "ar1": ar1(s),
                "n": int(s.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def _self_checks() -> None:
    """Spec examples that do not depend on the market data."""
    assert np.isclose((20.0 ** 2) / 12.0, 400.0 / 12.0)

    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-03-02", "2020-03-03"])
    close = pd.Series([100.0, 110.0, 100.0, 100.0], index=idx)
    rv = monthly_rv(close)
    feb = pd.Timestamp("2020-02-29")
    assert feb in rv.index
    assert pd.isna(rv.loc[feb])
    jan_ret = np.log(110.0 / 100.0)
    assert np.isclose(rv.loc[pd.Timestamp("2020-01-31")], (jan_ret ** 2) * 1e4)

    vix = pd.Series([20.0], index=pd.to_datetime(["2020-01-15"]))
    iv = monthly_iv(vix)
    assert np.isclose(iv.loc[pd.Timestamp("2020-01-31")], 400.0 / 12.0)

    e = pd.Series(
        [0.1, 0.2, 0.3, 0.4],
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )
    y = forward_mean(e, 2)
    assert np.isclose(y.iloc[0], (0.2 + 0.3) / 2)
    assert np.isclose(y.iloc[1], (0.3 + 0.4) / 2)
    assert pd.isna(y.iloc[2]) and pd.isna(y.iloc[3])


def run_sample(panel: pd.DataFrame, name: str, start: str, end: str) -> dict:
    sample = slice_sample(panel, start, end)
    needed = ["IV", "RV", "VRP", "e", "rv_max", "iv_max", "x_max"]
    if sample[needed].isna().any().any():
        bad = sample[needed].isna().sum()
        raise AssertionError(f"{name} has NaNs inside the sample:\n{bad}")
    if (sample["RV"] == 0).any():
        raise AssertionError(f"{name} has a month with RV equal to 0")

    n_vrp = len(sample)
    if name == "main":
        if n_vrp != 216:
            raise AssertionError(f"Main sample has {n_vrp} VRP rows, expected 216")

    rows = []
    n_lookahead = 0
    identity_errors = []
    for h in HORIZONS:
        y_a = forward_mean(sample["e"], h)
        y_b = 1200.0 * y_a
        x_pct = sample["VRP"]
        x_dec = sample["VRP"] / 1e4

        model_a, idx_a = fit_hac(y_a, x_pct, max(1, h - 1))
        model_b, idx_b = fit_hac(y_b, x_pct, max(1, h - 1))
        model_b_h, idx_h = fit_hac(y_b, x_pct, h)
        model_dec, idx_dec = fit_hac(y_a, x_dec, max(1, h - 1))

        if not (idx_a.equals(idx_b) and idx_a.equals(idx_h) and idx_a.equals(idx_dec)):
            raise AssertionError(f"{name} h={h}: scalings changed the estimation sample")
        if len(idx_a) != n_vrp - h:
            raise AssertionError(
                f"{name} h={h}: N={len(idx_a)}, expected {n_vrp - h}"
            )

        for t in idx_a:
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
            first = sample.index[loc + 1]
            last = sample.index[loc + h]
            if first.to_period("M") != t.to_period("M") + 1:
                raise AssertionError(f"{name} h={h}: first return in Y at {t.date()} is {first.date()}")
            if last.to_period("M") != t.to_period("M") + h:
                raise AssertionError(f"{name} h={h}: last return in Y at {t.date()} is {last.date()}")
            n_lookahead += 1

        coef_a = float(model_a.params["VRP"])
        coef_b = float(model_b.params["VRP"])
        coef_dec = float(model_dec.params["VRP"])
        # Y(B) = 1200 * Y(A) and VRP_percent^2 = 1e4 * VRP_decimal.
        scaled = coef_dec * 1200.0 / 1e4
        identity_errors.append(abs(scaled - coef_b))
        identity_errors.append(abs(coef_a * 1200.0 - coef_b))
        if not np.isclose(scaled, coef_b, rtol=1e-8, atol=1e-8):
            raise AssertionError(
                f"{name} h={h}: decimal coef {coef_dec} * 1200/1e4 = {scaled}, coef(B) = {coef_b}"
            )
        if not np.isclose(coef_a * 1200.0, coef_b, rtol=1e-8, atol=1e-8):
            raise AssertionError(f"{name} h={h}: coef(A)*1200 != coef(B)")
        if not np.isclose(model_a.tvalues["VRP"], model_b.tvalues["VRP"], rtol=1e-6, atol=1e-6):
            raise AssertionError(f"{name} h={h}: t-stat changed with the 1200 scaling")
        if not np.isclose(model_a.rsquared, model_b.rsquared, rtol=1e-8, atol=1e-10):
            raise AssertionError(f"{name} h={h}: R^2 changed with the 1200 scaling")

        rows.append(
            {
                "sample": name,
                "h": h,
                "coef_A": coef_a,
                "coef_B": coef_b,
                "t_h_minus_1": float(model_b.tvalues["VRP"]),
                "t_h": float(model_b_h.tvalues["VRP"]),
                "R2": float(model_b.rsquared * 100.0),
                "N": int(len(idx_a)),
            }
        )

    return {
        "sample": sample,
        "table2": pd.DataFrame(rows),
        "n_lookahead": n_lookahead,
        "max_identity_error": float(max(identity_errors)),
    }


def plot_hump(table2: pd.DataFrame, path: Path) -> None:
    main = table2[table2["sample"] == "main"].sort_values("h")
    btz_h = list(BTZ_TARGETS)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    axes[0].plot(
        main["h"], main["t_h_minus_1"], marker="o", color="steelblue",
        label="t, maxlags = h−1 (min 1)",
    )
    axes[0].plot(
        main["h"], main["t_h"], marker="s", color="darkorange",
        label="t, maxlags = h",
    )
    axes[0].scatter(
        btz_h,
        [BTZ_TARGETS[h]["t"] for h in btz_h],
        marker="D",
        s=46,
        color="black",
        zorder=5,
        label="BTZ Table 2",
    )
    axes[0].axhline(1.96, color="gray", lw=0.8, ls="--", label="1.96")
    axes[0].set_title("t-statistic")
    axes[0].set_xlabel("Horizon h (months)")
    axes[0].set_ylabel("t")

    axes[1].plot(
        main["h"], main["R2"], marker="o", color="steelblue", label="This replication"
    )
    axes[1].scatter(
        btz_h,
        [BTZ_TARGETS[h]["R2"] for h in btz_h],
        marker="D",
        s=46,
        color="black",
        zorder=5,
        label="BTZ Table 2",
    )
    axes[1].set_title("R² (%)")
    axes[1].set_xlabel("Horizon h (months)")
    axes[1].set_ylabel("R² (%)")

    for ax in axes:
        ax.set_xticks(HORIZONS)
        ax.legend(frameon=False, fontsize=8)
        ax.axhline(0, color="black", lw=0.4)

    fig.suptitle("Main sample, ^GSPC, 1990:01–2007:12", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _peak(table: pd.DataFrame, column: str) -> int:
    return int(table.loc[table[column].idxmax(), "h"])


def write_summary(
    meta: dict,
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    results: dict,
) -> None:
    main = table2[table2["sample"] == "main"].set_index("h")
    lines: list[str] = []
    lines.append("# Spec 001 — Baseline 1990–2007 in BTZ units")
    lines.append("")
    lines.append("## Data vintages")
    lines.append("")
    lines.append(
        f"- `data/gspc_data.csv`: Yahoo `^GSPC` daily Close, request window "
        f"{YAHOO_START} to {YAHOO_END} (end exclusive). Downloaded {YAHOO_DOWNLOAD_DATE}. "
        f"Rows in the file: {meta['gspc_rows']:,}, from {meta['gspc_start']} to {meta['gspc_end']}."
    )
    lines.append(
        f"- `data/vix_1990_data.csv`: Yahoo `^VIX` daily Close, same request window. "
        f"Downloaded {YAHOO_DOWNLOAD_DATE}. Rows in the file: {meta['vix_rows']:,}, "
        f"from {meta['vix_start']} to {meta['vix_end']}. Yahoo has no `^VIX` print before "
        f"{meta['vix_start']}, so the file starts on the first available session inside the window."
    )
    lines.append(
        f"- `data/tb3ms.csv`: existing FRED TB3MS file, not re-downloaded. "
        f"Monthly average, percent per year, from {meta['tb_start']} to {meta['tb_end']}."
    )
    lines.append(
        f"- `data/spy_data.csv`: existing file, used only for R1 returns and RV. "
        f"From {meta['spy_start']} to {meta['spy_end']}. The adjusted series is stored in `Close` "
        f"(the file has no separate adjusted-close column). January 1993 closes near 24, "
        f"the dividend-adjusted level."
    )
    lines.append("")
    lines.append(
        "IV for every sample, including R1, uses `data/vix_1990_data.csv`. "
        "R1 changes the price series used for returns and RV, and the sample window."
    )
    lines.append("")
    lines.append("## Definitions used")
    lines.append("")
    lines.append(
        "Month t is the calendar month, labelled at month-end (`ME`). "
        "RV is the sum of squared daily log returns inside the month, times 10⁴, "
        "with `min_count=1`. IV is the last VIX print of the month, squared, divided by 12. "
        "VRP = IV − RV, in monthly percent squared. "
        "The excess return is the monthly index log return minus TB3MS of the previous month divided by 1200. "
        "Specification (A) regresses the h-month average of future excess returns on VRP. "
        "Specification (B) multiplies that average by 1200 before the regression. "
        "Newey–West standard errors use the Bartlett kernel as implemented by "
        "`statsmodels` OLS `cov_type='HAC'`, with `use_correction=False`. "
        "The two lag choices are `maxlags = max(1, h−1)` and `maxlags = h`. "
        "At h = 1 the two choices are the same lag. "
        "AR(1) in Table 1 is the OLS slope of the series on its own lag and a constant. "
        "Standard deviations use the sample divisor (n − 1). R² is reported in percent."
    )
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("### 1. No look-ahead")
    lines.append("")
    lines.append(
        f"Pass. The regression loop checked {meta['n_lookahead']} rows "
        f"(every sample, every horizon, every retained month t). "
        f"The last daily-return date inside RV_t and the last VIX date inside IV_t both fall in month t, "
        f"and the later of those two dates is on or before the month-end label of t. "
        f"The first excess return inside Y_t is the return of month t+1, and the window runs through month t+h."
    )
    lines.append("")
    lines.append(
        "The risk-free rate is lagged as specified. "
        f"rf(1990-01) = TB3MS(1989-12) / 1200 = {_fmt(meta['rf_1990_01'], 8)} "
        f"(TB3MS December 1989 = 7.63). "
        f"rf(1990-02) = TB3MS(1990-01) / 1200 = {_fmt(meta['rf_1990_02'], 8)} "
        f"(TB3MS January 1990 = 7.64)."
    )
    lines.append("")
    lines.append("### 2. Observation counts")
    lines.append("")
    lines.append(
        "Pass. VRP observations on the main sample, 1990-01 through 2007-12: "
        f"**{meta['n_main']}** (expected 216). Regression N = 216 − h:"
    )
    lines.append("")
    lines.append("| h | N |")
    lines.append("|---|---|")
    for h, n in meta["n_by_h"]["main"]:
        lines.append(f"| {h} | {n} |")
    lines.append("")
    lines.append("The same truncation, N = (months in the sample) − h, holds for the robustness samples:")
    lines.append("")
    lines.append("| sample | window | months | N at h=1 | N at h=24 |")
    lines.append("|---|---|---|---|---|")
    for name, start, end, _price in SAMPLES:
        months = results[name]["sample"].shape[0]
        n1 = int(results[name]["table2"].set_index("h").loc[1, "N"])
        n24 = int(results[name]["table2"].set_index("h").loc[24, "N"])
        lines.append(f"| {name} | {start[:7]} → {end[:7]} | {months} | {n1} | {n24} |")
    lines.append("")
    lines.append("### 3. No zero RV and no missing values inside the sample")
    lines.append("")
    lines.append(
        "Pass. On main, R1 and R2, after the sample window is reindexed to a complete "
        "month-end calendar, IV, RV, VRP and the excess return contain no NaN, and no RV equals 0. "
        "A separate construction check confirms that a month with no prices gets RV = NaN rather than 0, "
        "and that VIX = 20 maps to IV = 400/12."
    )
    lines.append("")
    lines.append("### 4. Means and standard deviations, monthly percent squared")
    lines.append("")
    lines.append("Main sample, 1990-01 through 2007-12:")
    lines.append("")
    lines.append("| variable | mean | sd |")
    lines.append("|---|---|---|")
    t1_main = table1[table1["sample"] == "main"]
    for _, row in t1_main.iterrows():
        if row["variable"] == "e":
            continue
        lines.append(
            f"| {row['variable']} | {_fmt(row['mean'], 3)} | {_fmt(row['sd'], 3)} |"
        )
    lines.append("")
    lines.append(
        "Audit reference on 1993–2007, for Arthur's comparison with BTZ Table 1: "
        "IV ≈ 33.9, RV ≈ 23.9, VRP ≈ 10.0. "
        "The same window on these data:"
    )
    lines.append("")
    lines.append("| source | window | IV mean | RV mean | VRP mean |")
    lines.append("|---|---|---|---|---|")
    for label, key in (
        ("^GSPC + new VIX", "gspc_1993"),
        ("SPY + new VIX (R1)", "spy_1993"),
    ):
        mom = meta[key]
        lines.append(
            f"| {label} | 1993-02 → 2007-12 | {_fmt(mom['IV'], 3)} | "
            f"{_fmt(mom['RV'], 3)} | {_fmt(mom['VRP'], 3)} |"
        )
    lines.append("")
    lines.append(
        "Full moments (mean, sd, min, max, AR(1), n) for IV, RV, VRP and e, "
        "on main, R1 and R2, are in `table1_summary.csv`."
    )
    lines.append("")
    lines.append("### 5. Unit identity")
    lines.append("")
    lines.append(
        "Pass, under the identity implied by the unit definitions. "
        "For every sample and every horizon, the coefficient from regressing specification (A) "
        "on decimal VRP (VRP / 10⁴), multiplied by 1200 / 10⁴, equals coefficient (B). "
        f"The largest absolute gap across the 18 regressions is {meta['max_identity_error']:.2e}. "
        "Coefficient (A) × 1200 equals coefficient (B) on the same rows, and the t-statistic and R² "
        "are unchanged by the 1200 scaling of the dependent variable."
    )
    lines.append("")
    lines.append(
        "The check sentence in the spec multiplies the decimal-unit coefficient by 10⁴ / 1200. "
        "That order is the reciprocal of the scaling above. "
        "It is recorded under Deviations. The code asserts the identity that matches "
        "Y(B) = 1200 × Y(A), VRP in percent squared = 10⁴ × decimal VRP, and the audit "
        "benchmark of about 0.28."
    )
    lines.append("")
    lines.append("## Comparison with the BTZ Table 2 targets")
    lines.append("")
    lines.append(
        "Targets are the notebook's transcription of BTZ (2009) Table 2. "
        "Coefficients below are specification (B), annualized percent on monthly percent-squared VRP. "
        "t(h−1) uses maxlags = max(1, h−1). t(h) uses maxlags = h. R² is in percent."
    )
    lines.append("")
    lines.append("| h | coef(B) | BTZ coef | t(h−1) | t(h) | BTZ t | R² | BTZ R² | N |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for h in HORIZONS:
        row = main.loc[h]
        tgt = BTZ_TARGETS.get(h)
        if tgt is None:
            b_coef, b_t, b_r2 = "—", "—", "—"
        else:
            b_coef = _fmt(tgt["coef"], 2)
            b_t = _fmt(tgt["t"], 2)
            b_r2 = _fmt(tgt["R2"], 2)
        lines.append(
            f"| {h} | {_fmt(row['coef_B'], 4)} | {b_coef} | "
            f"{_fmt(row['t_h_minus_1'], 2)} | {_fmt(row['t_h'], 2)} | {b_t} | "
            f"{_fmt(row['R2'], 2)} | {b_r2} | {int(row['N'])} |"
        )
    lines.append("")
    h3 = main.loc[3]
    r1_h3 = table2.set_index(["sample", "h"]).loc[("R1", 3)]
    in_band = 0.25 <= float(h3["coef_B"]) <= 0.35
    band_text = "inside" if in_band else "outside"
    lines.append(
        f"Coefficient (B) at h = 3 on the main sample is **{_fmt(h3['coef_B'], 4)}**, "
        f"{band_text} the band 0.25–0.35. That band is the spec's reading of the 1993–2007 audit (0.28). "
        f"R1 is that audit sample, now in these units and with the lagged T-bill rate: "
        f"coefficient (B) at h = 3 is {_fmt(float(r1_h3['coef_B']), 4)} "
        f"and t(h−1) is {_fmt(float(r1_h3['t_h_minus_1']), 2)}."
    )
    lines.append("")
    peak_c = _peak(main.reset_index(), "coef_B")
    peak_t1 = _peak(main.reset_index(), "t_h_minus_1")
    peak_th = _peak(main.reset_index(), "t_h")
    peak_r = _peak(main.reset_index(), "R2")
    hump_ok = peak_c in (3, 6) and peak_t1 in (3, 6)
    hump_text = "preserved" if hump_ok else "not preserved"
    lines.append(
        f"The hump is {hump_text} on the main sample. "
        f"Coefficient (B) peaks at h = {peak_c}, t(h−1) peaks at h = {peak_t1}, "
        f"t(h) peaks at h = {peak_th}, and R² peaks at h = {peak_r}."
    )
    lines.append("")
    t_pass_1 = h3["t_h_minus_1"] > 1.96
    t_pass_h = h3["t_h"] > 1.96
    lines.append(
        f"t(h = 3) with maxlags = h−1 is {_fmt(h3['t_h_minus_1'], 2)} "
        f"({'above' if t_pass_1 else 'not above'} 1.96). "
        f"t(h = 3) with maxlags = h is {_fmt(h3['t_h'], 2)} "
        f"({'above' if t_pass_h else 'not above'} 1.96)."
    )
    lines.append("")
    def _horizons(hs: list[int]) -> str:
        labels = [f"h = {h}" for h in hs]
        if len(labels) <= 1:
            return labels[0] if labels else "none of the reported horizons"
        return ", ".join(labels[:-1]) + " and " + labels[-1]

    higher = [h for h, tgt in BTZ_TARGETS.items() if main.loc[h, "R2"] > tgt["R2"]]
    lower = [h for h, tgt in BTZ_TARGETS.items() if main.loc[h, "R2"] <= tgt["R2"]]
    if higher and lower:
        r2_vs_btz = (
            f"R² is above the BTZ target at {_horizons(higher)}, "
            f"and below it at {_horizons(lower)}. "
        )
    elif higher:
        r2_vs_btz = f"R² is above the BTZ target at {_horizons(higher)}. "
    else:
        r2_vs_btz = f"R² is below the BTZ target at {_horizons(lower)}. "
    lines.append(
        r2_vs_btz
        + "The spec expected daily RV to leave R² below the paper at every reported horizon, "
        "and said not to close that gap. RV is still the sum of squared daily log returns. "
        "`fig_hump.png` plots both t-statistics and R² for the main sample. "
        "BTZ markers are drawn at h = 1, 3, 6 and 12. Horizons 9 and 24 have no published target in the notebook."
    )
    lines.append("")
    lines.append("Robustness, coefficient (B) and t(h−1):")
    lines.append("")
    lines.append("| sample | h=3 coef(B) | h=3 t(h−1) | h=3 R² | peak of coef(B) |")
    lines.append("|---|---|---|---|---|")
    for name, _, _, _ in SAMPLES:
        block = table2[table2["sample"] == name]
        r3 = block[block["h"] == 3].iloc[0]
        lines.append(
            f"| {name} | {_fmt(r3['coef_B'], 4)} | {_fmt(r3['t_h_minus_1'], 2)} | "
            f"{_fmt(r3['R2'], 2)} | h = {_peak(block, 'coef_B')} |"
        )
    lines.append("")
    lines.append("## Deviations from this spec")
    lines.append("")
    lines.append(
        "1. **Check 5 writes the scaling factors in reverse order.** "
        "The spec says the decimal-unit coefficient × 10⁴ / 1200 must equal coefficient (B). "
        "The definitions in the same spec say the opposite ratio. "
        "Decimal VRP times 10⁴ is VRP in monthly percent squared, so moving the regressor "
        "into the new units divides the coefficient by 10⁴. "
        "Specification (B) = 1200 × specification (A), so that scaling multiplies the coefficient by 1200. "
        "Combined, decimal-unit coefficient × 1200 / 10⁴ = coefficient (B). "
        "That is the identity asserted in the code. "
        "It is also the one that turns the notebook's 1993–2007 h = 3 coefficient of 2.32 "
        "into the audit value of about 0.28 (2.32 × 1200 / 10⁴ = 0.278). "
        "Applying the factors in the written order would produce a coefficient near 19, "
        "outside the expected band of 0.25–0.35."
    )
    lines.append("")
    (RESULTS / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _span(close: pd.Series) -> tuple[str, str, int]:
    return str(close.index.min().date()), str(close.index.max().date()), int(close.shape[0])


def main() -> dict:
    _self_checks()
    download_yahoo("^GSPC", DATA / "gspc_data.csv")
    download_yahoo("^VIX", DATA / "vix_1990_data.csv")

    gspc = load_close(DATA / "gspc_data.csv")
    vix = load_close(DATA / "vix_1990_data.csv")
    spy = load_close(DATA / "spy_data.csv")
    rf = load_rf()

    rf_jan = float(rf.loc[pd.Timestamp("1990-01-31")])
    rf_feb = float(rf.loc[pd.Timestamp("1990-02-28")])
    if not np.isclose(rf_jan, 7.63 / 1200.0):
        raise AssertionError(f"rf(1990-01) = {rf_jan}, expected 7.63/1200")
    if not np.isclose(rf_feb, 7.64 / 1200.0):
        raise AssertionError(f"rf(1990-02) = {rf_feb}, expected 7.64/1200")

    panel_gspc = build_panel(gspc, vix, rf)
    panel_spy = build_panel(spy, vix, rf)
    panels = {"gspc": panel_gspc, "spy": panel_spy}

    results = {}
    table1_parts = []
    table2_parts = []
    for name, start, end, price in SAMPLES:
        out = run_sample(panels[price], name, start, end)
        results[name] = out
        table1_parts.append(summarize(out["sample"], name))
        table2_parts.append(out["table2"])

    table1 = pd.concat(table1_parts, ignore_index=True)
    table2 = pd.concat(table2_parts, ignore_index=True)

    gspc_1993 = slice_sample(panel_gspc, "1993-02-28", "2007-12-31")
    spy_1993 = results["R1"]["sample"]

    tb = pd.read_csv(DATA / "tb3ms.csv", parse_dates=["DATE"])
    meta = {
        "gspc_start": _span(gspc)[0],
        "gspc_end": _span(gspc)[1],
        "gspc_rows": _span(gspc)[2],
        "vix_start": _span(vix)[0],
        "vix_end": _span(vix)[1],
        "vix_rows": _span(vix)[2],
        "spy_start": _span(spy)[0],
        "spy_end": _span(spy)[1],
        "tb_start": str(tb["DATE"].min().date()),
        "tb_end": str(tb["DATE"].max().date()),
        "n_lookahead": int(sum(results[n]["n_lookahead"] for n, *_ in SAMPLES)),
        "n_main": int(results["main"]["sample"].shape[0]),
        "n_by_h": {
            name: [(int(r.h), int(r.N)) for r in results[name]["table2"].itertuples()]
            for name, *_ in SAMPLES
        },
        "max_identity_error": float(
            max(results[n]["max_identity_error"] for n, *_ in SAMPLES)
        ),
        "rf_1990_01": rf_jan,
        "rf_1990_02": rf_feb,
        "gspc_1993": {
            "IV": float(gspc_1993["IV"].mean()),
            "RV": float(gspc_1993["RV"].mean()),
            "VRP": float(gspc_1993["VRP"].mean()),
        },
        "spy_1993": {
            "IV": float(spy_1993["IV"].mean()),
            "RV": float(spy_1993["RV"].mean()),
            "VRP": float(spy_1993["VRP"].mean()),
        },
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    table1.to_csv(RESULTS / "table1_summary.csv", index=False)
    table2.to_csv(RESULTS / "table2_main.csv", index=False)
    plot_hump(table2, RESULTS / "fig_hump.png")
    write_summary(meta, table1, table2, results)

    print(f"Wrote {RESULTS}")
    print(table2.round(4).to_string(index=False))
    return {"table1": table1, "table2": table2, "meta": meta}


if __name__ == "__main__":
    main()
