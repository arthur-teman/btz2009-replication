"""Spec 005 — European extension: does the US VRP predict DAX and CAC returns?

Imports the spec 001, 002, 003 and 003a pipelines and does not modify them.
Writes results/005-europe/.
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

from src import baseline_1990 as btz
from src import inference as inf3
from src import inference_003a as inf3a
from src import influence as inf

RESULTS = btz.ROOT / "results" / "005-europe"
DATA = btz.DATA
HORIZONS = inf3.HORIZONS
N_REP = inf3.N_REP
SEED = inf3.SEED
T_CRIT = 1.96
P_CRIT = 0.05
E1_GAP = 0.3
E4_BAND = 0.25
MIN_DAYS = 15
BIG_MOVE = 0.10
QUALITY_START = "1994-01-01"
QUALITY_END = "2025-12-31"

# market, ticker, file
EU_MARKETS = (("DAX", "^GDAXI", "dax.csv"), ("CAC", "^FCHI", "cac.csv"))
TICKERS = {"US": "^GSPC", "DAX": "^GDAXI", "CAC": "^FCHI"}
MARKET_ORDER = ("US", "DAX", "CAC")
# sample id, first VRP month, first VRP_fwd month, last month
SAMPLES = (
    ("pre-2008", "1994-02-28", "1995-01-31", "2007-12-31"),
    ("post-2008", "2008-01-31", "2008-01-31", "2025-12-31"),
)
SAMPLE_IDS = tuple(s[0] for s in SAMPLES)
US_TIMING = "spec001"
TIMING_LABEL = {"spec001": "spec 001", "NB": "NB", "SK": "SK", "T-1": "T−1"}
FWD_NOTE = "VRP_fwd starts in 1995-01, so its pre-2008 rows run from 1995-01 to 2007-12."

# 003a rows reproduced by check 1: id, regressor, first month, last month.
REF_003A = (
    ("V-95-07", "VRP", "1995-01-31", "2007-12-31"),
    ("G-95-07", "VRP_fwd", "1995-01-31", "2007-12-31"),
    ("V-08-25", "VRP", "2008-01-31", "2025-12-31"),
    ("G-08-25", "VRP_fwd", "2008-01-31", "2025-12-31"),
)
REF_TOL = 1e-6

# Audit reference, direction only: SPY-based VRP, notebook units, NW, h = 3, 1994–2007.
AUDIT_T = {"DAX": {"NB": 2.64, "T-1": 2.59, "SK": 2.79}, "CAC": {"NB": 2.05, "T-1": 2.02, "SK": 2.18}}
AUDIT_VOL = {"US": 4.02, "DAX": 6.45, "CAC": 5.48}

_fmt = inf3._fmt
_ym = inf3._ym
_p_value = inf3._p_value


def _yn(flag: bool) -> str:
    return "yes" if flag else "no"


def _month_end(index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index).to_period("M").to_timestamp("M")


# ── Data ─────────────────────────────────────────────────────────────────────


def load_euribor() -> pd.Series:
    """rf^EU_t = Euribor_{t−1} / 1200, indexed at month end t."""
    df = pd.read_csv(DATA / "euribor.csv", parse_dates=["DATE"])
    s = df.set_index("DATE")["IR3TIB01EZM156N"].astype(float).sort_index()
    s.index = _month_end(s.index)
    if s.index.duplicated().any():
        raise AssertionError("Euribor has two values in one month")
    full = pd.date_range(s.index.min(), s.index.max(), freq="ME")
    if not s.index.equals(full):
        raise AssertionError("Euribor is not a complete monthly calendar")
    rf = s.shift(1) / 1200.0
    rf.name = "rf"
    return rf


def european_monthly(close: pd.Series, rf: pd.Series) -> pd.DataFrame:
    """Monthly frame: first and last trading day, r, the day-1 return d, e = r − rf.

    r_m = log(P(T_m) / P(T_{m−1})), d_m = log(P(F_m) / P(T_{m−1})), both in decimal.
    """
    close = close.sort_index().astype(float)
    month = _month_end(close.index)
    grp = pd.Series(close.index, index=close.index).groupby(month)
    out = pd.DataFrame({"F": grp.min(), "T": grp.max(), "n_days": grp.count()})
    out["P_F"] = close.reindex(out["F"]).to_numpy()
    out["P_T"] = close.reindex(out["T"]).to_numpy()
    full = pd.date_range(out.index.min(), out.index.max(), freq="ME")
    if not out.index.equals(full):
        raise AssertionError("A calendar month has no trading day")
    out["T_prev"] = out["T"].shift(1)
    out["r"] = np.log(out["P_T"]).diff()
    out["d"] = np.log(out["P_F"]) - np.log(out["P_T"].shift(1))
    out["rf"] = rf.reindex(out.index)
    out["e"] = out["r"] - out["rf"]
    ref = btz.monthly_log_return(close).reindex(out.index)
    if not np.allclose(out["r"].dropna(), ref.dropna(), rtol=0, atol=1e-12) or not out["r"].isna().equals(ref.isna()):
        raise AssertionError("Monthly return differs from the spec 001 definition")
    return out


def vrp_cut(gspc: pd.Series, vix: pd.Series, drop_last: int) -> pd.DataFrame:
    """Spec 001 RV, IV and VRP on the US trading days of month t, minus the last `drop_last`.

    The US calendar is the ^GSPC calendar. Returns RV, IV, VRP, the cut date, and
    the last US trading day of the month.
    """
    gspc = gspc.sort_index().astype(float)
    r = btz.daily_log_returns(gspc).dropna()
    days = pd.Series(gspc.index, index=gspc.index)
    month = _month_end(days.index)
    rank_from_end = days.groupby(month).cumcount(ascending=False)
    cut = days[rank_from_end == drop_last]
    cut.index = _month_end(cut.index)
    last = days.groupby(month).max()
    keep_days = days[rank_from_end >= drop_last].index
    r_keep = r[r.index.isin(keep_days)]
    rv = (r_keep ** 2).groupby(_month_end(r_keep.index)).sum(min_count=1) * 1e4
    rv_max = pd.Series(r_keep.index, index=r_keep.index).groupby(_month_end(r_keep.index)).max()
    vix = vix.sort_index().astype(float)
    cut = cut[cut.index >= _month_end([vix.index.min()])[0]]
    if not cut.isin(vix.index).all():
        missing = cut[~cut.isin(vix.index)]
        raise AssertionError(f"No VIX close on the cut date for {len(missing)} months")
    iv = (vix.reindex(cut.to_numpy()).to_numpy() ** 2) / 12.0
    out = pd.DataFrame({"cut": cut, "last_US": last.reindex(cut.index), "rv_max": rv_max.reindex(cut.index)})
    out["IV"] = iv
    out["RV"] = rv.reindex(cut.index)
    out["VRP"] = out["IV"] - out["RV"]
    return out


def load_us_penultimate(gspc: pd.Series, vix: pd.Series, full: pd.DataFrame) -> pd.DataFrame:
    """VRP^{T−1}: spec 001 VRP with data up to the penultimate US trading day of month t."""
    same = vrp_cut(gspc, vix, 0).reindex(full.index)
    for col in ("IV", "RV", "VRP"):
        if not np.allclose(same[col].to_numpy(), full[col].to_numpy(), rtol=0, atol=1e-9):
            raise AssertionError(f"vrp_cut with no day dropped does not reproduce spec 001 {col}")
    pen = vrp_cut(gspc, vix, 1).reindex(full.index)
    if pen[["IV", "RV", "VRP"]].isna().any().any():
        raise AssertionError("VRP^{T−1} is missing in 1990–2025")
    if not (pen["rv_max"] == pen["cut"]).all() or not (pen["cut"] < pen["last_US"]).all():
        raise AssertionError("VRP^{T−1} uses a US date after the penultimate trading day")
    return pen


def load_markets(loaded: dict) -> dict:
    gspc = btz.load_close(DATA / "gspc_data.csv")
    vix = btz.load_close(DATA / "vix_1990_data.csv")
    rf = load_euribor()
    if rf.first_valid_index() != pd.Timestamp("1994-02-28"):
        raise AssertionError(f"First usable Euribor month is {_ym(rf.first_valid_index())}, expected 1994-02")
    pen = load_us_penultimate(gspc, vix, loaded["full"])
    closes = {name: btz.load_close(DATA / fname) for name, _tic, fname in EU_MARKETS}
    eu = {name: european_monthly(closes[name], rf) for name in closes}
    us_days = pd.Series(gspc.index, index=gspc.index)
    us_last = us_days.groupby(_month_end(us_days.index)).max()
    return {"gspc": gspc, "vix": vix, "rf": rf, "pen": pen, "closes": closes, "eu": eu, "us_last": us_last}


# ── Check 2: daily data quality ──────────────────────────────────────────────


def data_quality(closes: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    info = {}
    for name, _tic, fname in EU_MARKETS:
        raw = pd.read_csv(DATA / fname, parse_dates=["Date"]).set_index("Date").sort_index()
        close = closes[name]
        r = btz.daily_log_returns(close)
        inside = (close.index >= QUALITY_START) & (close.index <= QUALITY_END)
        c = close[inside]
        counts = c.groupby(_month_end(c.index)).count()
        months = pd.date_range(QUALITY_START, QUALITY_END, freq="ME")
        counts = counts.reindex(months).fillna(0).astype(int)
        short = counts[counts < MIN_DAYS]
        if len(short):
            raise AssertionError(f"{name}: months with fewer than {MIN_DAYS} trading days: " + ", ".join(_ym(t) for t in short.index))
        rows.append({"market": name, "check": "min_trading_days", "date": _ym(counts.idxmin()), "value": int(counts.min()),
                     "note": f"fewest trading days in a month, {len(months)} months 1994-01 to 2025-12; every month has at least {MIN_DAYS}"})
        r_in = r[inside].dropna()
        big = r_in[r_in.abs() > BIG_MOVE]
        for t, v in big.items():
            rows.append({"market": name, "check": "abs_log_return_gt_10pct", "date": t.strftime("%Y-%m-%d"), "value": float(v), "note": "daily log return"})
        vol = raw["Volume"][(raw.index >= QUALITY_START) & (raw.index <= QUALITY_END)]
        zero = vol.groupby(_month_end(vol.index)).apply(lambda s: bool((s == 0).all()))
        zero_months = zero[zero].index
        for t in zero_months:
            rows.append({"market": name, "check": "zero_volume_month", "date": _ym(t), "value": int(counts.loc[t]),
                         "note": "volume 0 on all trading days (information only); value = trading days"})
        nz = zero[~zero].index
        info[name] = {
            "min_days": int(counts.min()),
            "min_month": _ym(counts.idxmin()),
            "big": big,
            "n_zero": len(zero_months),
            "zero_first": _ym(zero_months.min()) if len(zero_months) else "",
            "zero_last": _ym(zero_months.max()) if len(zero_months) else "",
            "zero_contiguous": bool(len(zero_months) == 0 or zero_months.max() < nz.min()),
            "n_days": int(len(c)),
        }
    return pd.DataFrame(rows), info


# ── SK timing: Hodrick 1B and null bootstrap with the day-1 term ─────────────


def hodrick_t_skip(
    e: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
    h: int,
    hod_pos: np.ndarray,
    coef: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hodrick 1B t for Y^SK_t(h) = (1/h)(Σ_{j=1..h} e_{t+j} − d_{t+1}).

    Under the null, h·(Y_t − Ȳ) = Σ_j u_{t+j} − δ_{t+1}, so the 1B score at
    position p is u_{p+1}·Σ_{i=0..h−1} z_{p−i} − δ_{p+1}·z_p, with u = e − ē and
    δ = d − d̄. With d = 0 it is `inference.hodrick_t`.
    """
    e = np.atleast_2d(np.asarray(e, dtype=float))
    d = np.atleast_2d(np.asarray(d, dtype=float))
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if not (e.shape == d.shape == x.shape):
        raise AssertionError("e, d and x shapes do not match")
    coef = np.asarray(coef, dtype=float)
    if coef.ndim == 0:
        coef = np.full(e.shape[0], float(coef))
    u = e - e.mean(axis=1, keepdims=True)
    delta = d - d.mean(axis=1, keepdims=True)
    pos = np.asarray(hod_pos, dtype=int)
    rolled = inf3._rolling_sum(x, h)
    ones = np.ones((e.shape[0], len(pos)))
    z = np.stack([ones, x[:, pos]], axis=-1)
    zsum = np.stack([h * ones, rolled[:, pos]], axis=-1)
    w = u[:, pos + 1, None] * zsum - delta[:, pos + 1, None] * z
    se = np.sqrt(inf3._hodrick_variance(z, w, h))
    if np.any(~np.isfinite(se)) or np.any(se <= 0):
        raise AssertionError("Hodrick standard error is not finite and positive")
    return coef / se, se


def simulate_skip(
    rng: np.random.Generator,
    e: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
    scheme: str,
    n_rep: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spec 003 C3 null samples, with δ_s = d_s − d̄ drawn with (ε_s, η_s).

    iid: one index per date shared by (ε, δ, η), as `inference.simulate_null`.
    wild: one Rademacher weight per date shared by (ε, δ, η), as
    `inference_003a.simulate_wild`. The generator calls are the same, so e* and
    x* equal those functions' output from the same generator state.
    """
    e = np.asarray(e, dtype=float)
    d = np.asarray(d, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(e)
    mu_e = float(e.mean())
    mu_d = float(d.mean())
    eps = e - mu_e
    dlt = d - mu_d
    c, rho, eta = inf3.fit_ar1(x)
    m = n - 1
    if len(eta) != m:
        raise AssertionError("Residual triples are not aligned")
    if scheme == "iid":
        draw = rng.integers(0, m, size=(n_rep, m))
        eta_s = eta[draw]
        eps_s = eps[1:][draw]
        dlt_s = dlt[1:][draw]
    elif scheme == "wild":
        weight = rng.integers(0, 2, size=(n_rep, m)) * 2.0 - 1.0
        eta_s = weight * eta[None, :]
        eps_s = weight * eps[1:][None, :]
        dlt_s = weight * dlt[1:][None, :]
    else:
        raise AssertionError(f"Unknown scheme {scheme}")
    x_star = np.empty((n_rep, n))
    e_star = np.empty((n_rep, n))
    d_star = np.empty((n_rep, n))
    x_star[:, 0] = x[0]
    e_star[:, 0] = mu_e
    d_star[:, 0] = mu_d
    level = np.full(n_rep, x[0], dtype=float)
    for t in range(m):
        level = c + rho * level + eta_s[:, t]
        x_star[:, t + 1] = level
        e_star[:, t + 1] = mu_e + eps_s[:, t]
        d_star[:, t + 1] = mu_d + dlt_s[:, t]
    return e_star, d_star, x_star


def _boot_stats_skip(row: dict, e_star: np.ndarray, d_star: np.ndarray, x_star: np.ndarray, label: str) -> dict:
    by_h = {}
    for h in HORIZONS:
        reg = row["by_h"][h]["reg_pos"]
        hod = row["by_h"][h]["hod_pos"]
        y_star = inf3.forward_average(e_star, h, reg) - d_star[:, reg + 1] / float(h)
        t_nw, coef = inf3.newey_west_t(y_star, x_star[:, reg], max(1, h - 1))
        t_1b, _se = hodrick_t_skip(e_star, d_star, x_star, h, hod, coef)
        if not (np.isfinite(t_nw).all() and np.isfinite(t_1b).all() and np.isfinite(coef).all()):
            raise AssertionError(f"{row['name']} {label} h={h}: a replication is not finite")
        by_h[h] = {"coef": coef, "t_nw": t_nw, "t_1b": t_1b}
    return by_h


def _self_checks() -> None:
    rng = np.random.default_rng(5)
    n = 160
    e = rng.normal(size=n)
    d = 0.3 * e + rng.normal(scale=0.4, size=n)
    x = rng.normal(size=n)
    for h in HORIZONS:
        pos = np.arange(h - 1, n - 1)
        ref_t, ref_se = inf3.hodrick_t(e, x, h, pos, 0.2)
        got_t, got_se = hodrick_t_skip(e, np.zeros(n), x, h, pos, 0.2)
        if not (np.allclose(got_t, ref_t, rtol=0, atol=1e-12) and np.allclose(got_se, ref_se, rtol=0, atol=1e-12)):
            raise AssertionError(f"h={h}: hodrick_t_skip with d = 0 differs from inference.hodrick_t")
    pos1 = np.arange(0, n - 1)
    ref_t, _ = inf3.hodrick_t(e - d, x, 1, pos1, 0.2)
    got_t, _ = hodrick_t_skip(e, d, x, 1, pos1, 0.2)
    if not np.allclose(got_t, ref_t, rtol=0, atol=1e-12):
        raise AssertionError("At h = 1, hodrick_t_skip is not inference.hodrick_t on e − d")
    for scheme in ("iid", "wild"):
        a = np.random.default_rng(99)
        b = np.random.default_rng(99)
        e_s, _d_s, x_s = simulate_skip(a, e, d, x, scheme, 40)
        if scheme == "iid":
            e_r, x_r = inf3.simulate_null(b, e, x, "iid", 40)
        else:
            e_r, x_r = inf3a.simulate_wild(b, e, x, 40)
        if not (np.array_equal(e_s, e_r) and np.array_equal(x_s, x_r)):
            raise AssertionError(f"simulate_skip {scheme} does not reproduce the spec 003/003a draws")
        if a.integers(0, 1 << 30) != b.integers(0, 1 << 30):
            raise AssertionError(f"simulate_skip {scheme} consumes the generator differently")


# ── Fitting ──────────────────────────────────────────────────────────────────


def _fit_skip_horizon(sample: pd.DataFrame, x: pd.Series, h: int, name: str) -> dict:
    n = len(sample)
    x_s = x.reindex(sample.index)
    y = 1200.0 * btz.forward_mean(sample["e"], h) - 1200.0 * sample["d"].shift(-1) / float(h)
    model, idx = btz.fit_hac(y, x_s, max(1, h - 1))
    if not pd.DatetimeIndex(idx).equals(pd.DatetimeIndex(sample.index[: n - h])):
        raise AssertionError(f"{name} h={h}: SK rows are not the complete windows")
    for t in idx:
        inf.assert_row_timing(sample, t, h, name)
    reg_pos = np.arange(n - h)
    hod_pos = np.arange(h - 1, n - 1)
    e = 1200.0 * sample["e"].to_numpy(dtype=float)
    d = 1200.0 * sample["d"].to_numpy(dtype=float)
    x_np = x_s.to_numpy(dtype=float)
    if not (np.isfinite(e).all() and np.isfinite(d).all() and np.isfinite(x_np).all()):
        raise AssertionError(f"{name}: e, d or the regressor is missing inside the sample")
    y_np = inf3.forward_average(e, h, reg_pos)[0] - d[reg_pos + 1] / float(h)
    if not np.allclose(y_np, y.loc[idx].to_numpy(dtype=float), rtol=0, atol=1e-9):
        raise AssertionError(f"{name} h={h}: numpy Y^SK differs from the pandas Y^SK")
    t_np, b_np = inf3.newey_west_t(y_np, x_np[reg_pos], max(1, h - 1))
    coef = float(model.params["VRP"])
    t_hm1 = float(model.tvalues["VRP"])
    if abs(float(b_np[0]) - coef) > 1e-8 or abs(float(t_np[0]) - t_hm1) > 1e-8:
        raise AssertionError(f"{name} h={h}: numpy NW differs from statsmodels")
    t_1b, se_1b = hodrick_t_skip(e, d, x_np, h, hod_pos, coef)
    return {
        "h": h,
        "coef_B": coef,
        "t_hm1": t_hm1,
        "t_1b": float(t_1b[0]),
        "se_1b": float(se_1b[0]),
        "R2": float(model.rsquared * 100.0),
        "N": int(len(idx)),
        "index": pd.DatetimeIndex(idx),
        "reg_pos": reg_pos,
        "hod_pos": hod_pos,
        "e": e,
        "d": d,
        "x": x_np,
        "y": y_np,
    }


def run_row_skip(sample: pd.DataFrame, x: pd.Series, name: str) -> dict:
    by_h = {h: _fit_skip_horizon(sample, x, h, name) for h in HORIZONS}
    return {"name": name, "by_h": by_h}


def _row_samples(loaded: dict, mk: dict) -> list[dict]:
    """The regression rows, in the order the bootstrap draws them."""
    full = loaded["full"]
    regressors = {"VRP": loaded["vrp"], "VRP_fwd": loaded["vrp_fwd"]}
    specs = []
    for sid, start_v, start_g, end in SAMPLES:
        for reg, start in (("VRP", start_v), ("VRP_fwd", start_g)):
            specs.append({"market": "US", "regressor": reg, "timing": US_TIMING, "sample": sid, "start": start, "end": end})
        for market, _tic, _f in EU_MARKETS:
            for timing in ("NB", "SK", "T-1"):
                specs.append({"market": market, "regressor": "VRP", "timing": timing, "sample": sid, "start": start_v, "end": end})
            for timing in ("NB", "SK"):
                specs.append({"market": market, "regressor": "VRP_fwd", "timing": timing, "sample": sid, "start": start_g, "end": end})
    for s in specs:
        base = btz.slice_sample(full, s["start"], s["end"])
        sample = base.copy()
        x = regressors[s["regressor"]]
        if s["market"] != "US":
            eu = mk["eu"][s["market"]].reindex(base.index)
            sample["e"] = eu["e"]
            sample["d"] = eu["d"]
            sample["F"] = eu["F"]
            sample["T"] = eu["T"]
        if s["timing"] == "T-1":
            pen = mk["pen"].reindex(base.index)
            x = pen["VRP"].rename("VRP")
            sample["rv_max"] = pen["rv_max"]
            sample["iv_max"] = pen["cut"]
        need = ["e", "rv_max", "iv_max"] + (["d"] if s["timing"] == "SK" else [])
        if sample[need].isna().any().any() or x.reindex(base.index).isna().any():
            raise AssertionError(f"{s}: NaN inside the regression sample")
        s["frame"] = sample
        s["x"] = x
        s["name"] = f"{s['market']}|{s['regressor']}|{s['timing']}|{s['sample']}"
    return specs


def fit_all(specs: list[dict]) -> None:
    crisis = inf3.crisis_months()
    for s in specs:
        if s["timing"] == "SK":
            s["row"] = run_row_skip(s["frame"], s["x"], s["name"])
        else:
            s["row"] = inf3.run_row(s["frame"], s["x"], crisis, s["name"])
            s["row"]["name"] = s["name"]
            for h in HORIZONS:
                r = s["row"]["by_h"][h]
                r["y"] = inf3.forward_average(r["e"], h, r["reg_pos"])[0]
        n = len(s["frame"])
        for h in HORIZONS:
            r = s["row"]["by_h"][h]
            if r["N"] != n - h:
                raise AssertionError(f"{s['name']} h={h}: N = {r['N']}, expected {n - h}")
            if not (np.isfinite(r["y"]).all() and np.isfinite(r["x"]).all() and np.isfinite(r["e"]).all()):
                raise AssertionError(f"{s['name']} h={h}: NaN inside the regression")


def run_bootstraps(specs: list[dict]) -> None:
    """iid then wild for each row, in table order, from one default_rng(SEED)."""
    rng = np.random.default_rng(SEED)
    for s in specs:
        row = s["row"]
        r1 = row["by_h"][1]
        if s["timing"] == "SK":
            boots = {}
            for scheme in ("iid", "wild"):
                e_s, d_s, x_s = simulate_skip(rng, r1["e"], r1["d"], r1["x"], scheme, N_REP)
                boots[scheme] = _boot_stats_skip(row, e_s, d_s, x_s, scheme)
        else:
            e_s, x_s = inf3.simulate_null(rng, r1["e"], r1["x"], "iid", N_REP)
            boots = {"iid": inf3a._boot_stats(row, e_s, x_s, "iid")}
            e_s, x_s = inf3a.simulate_wild(rng, r1["e"], r1["x"], N_REP)
            boots["wild"] = inf3a._boot_stats(row, e_s, x_s, "wild")
        s["boots"] = boots
        print(
            f"  {s['name']:28s} h3 t_NW={row['by_h'][3]['t_hm1']:6.2f} t_1B={row['by_h'][3]['t_1b']:6.2f} "
            f"p_iid={_p_value(boots['iid'][3]['t_nw'], row['by_h'][3]['t_hm1']):.4f} "
            f"p_wild={_p_value(boots['wild'][3]['t_nw'], row['by_h'][3]['t_hm1']):.4f}"
        )


def records(specs: list[dict]) -> pd.DataFrame:
    out = []
    for s in specs:
        for h in HORIZONS:
            r = s["row"]["by_h"][h]
            x_reg = r["x"][r["reg_pos"]]
            e_rows = 1200.0 * s["frame"]["e"].loc[r["index"]].to_numpy(dtype=float)
            sd_x = float(np.std(x_reg, ddof=1))
            sd_y = float(np.std(r["y"], ddof=1))
            sd_e = float(np.std(e_rows, ddof=1))
            std_coef = r["coef_B"] * sd_x / sd_y
            corr = float(np.corrcoef(x_reg, r["y"])[0, 1])
            if abs(std_coef - corr) > 1e-9 or abs(np.sign(r["coef_B"]) * np.sqrt(r["R2"] / 100.0) - corr) > 1e-9:
                raise AssertionError(f"{s['name']} h={h}: standardized coefficient is not the correlation")
            out.append(
                {
                    "market": s["market"],
                    "index": TICKERS[s["market"]],
                    "regressor": s["regressor"],
                    "timing": s["timing"],
                    "sample": s["sample"],
                    "start": s["start"][:7],
                    "end": s["end"][:7],
                    "h": h,
                    "coef_B": r["coef_B"],
                    "t_NW_hm1": r["t_hm1"],
                    "t_1B": r["t_1b"],
                    "p_boot_NW_iid": _p_value(s["boots"]["iid"][h]["t_nw"], r["t_hm1"]),
                    "p_boot_NW_wild": _p_value(s["boots"]["wild"][h]["t_nw"], r["t_hm1"]),
                    "R2": r["R2"],
                    "N": r["N"],
                    "std_coef": std_coef,
                    "coef_over_sd_e": r["coef_B"] / sd_e,
                    "sd_x": sd_x,
                    "sd_Y": sd_y,
                    "sd_e_ann": sd_e,
                    "first_row": _ym(r["index"][0]),
                    "last_row": _ym(r["index"][-1]),
                    "note": FWD_NOTE if (s["regressor"] == "VRP_fwd" and s["sample"] == "pre-2008") else "",
                }
            )
    return pd.DataFrame(out)


def _pick(tab: pd.DataFrame, market: str, regressor: str, timing: str, sample: str, h: int = 3) -> pd.Series:
    sel = tab[(tab["market"] == market) & (tab["regressor"] == regressor) & (tab["timing"] == timing)
              & (tab["sample"] == sample) & (tab["h"] == h)]
    if len(sel) != 1:
        raise AssertionError(f"{market} {regressor} {timing} {sample} h={h}: {len(sel)} rows")
    return sel.iloc[0]


# ── Checks 1, 3, 4 ───────────────────────────────────────────────────────────


def check_us_benchmark(loaded: dict, specs: list[dict]) -> list[dict]:
    """Check 1. US rows against the spec 001/002 regressions and the 003a rows."""
    full = loaded["full"]
    for s in [s for s in specs if s["market"] == "US"]:
        if s["regressor"] == "VRP":
            ref = btz.run_sample(full, s["name"], s["start"], s["end"])["table2"].set_index("h")
            pairs = {h: (ref.loc[h, "coef_B"], ref.loc[h, "t_h_minus_1"], ref.loc[h, "R2"], ref.loc[h, "N"]) for h in HORIZONS}
        else:
            sample = btz.slice_sample(full, s["start"], s["end"])
            ref = inf.run_regression(sample, loaded["vrp_fwd"], s["name"])
            pairs = {h: (ref["by_h"][h]["coef_B"], ref["by_h"][h]["t_hm1"], ref["by_h"][h]["R2"], ref["by_h"][h]["N"]) for h in HORIZONS}
        for h in HORIZONS:
            r = s["row"]["by_h"][h]
            c, t, r2, n = pairs[h]
            if abs(r["coef_B"] - c) > 1e-8 or abs(r["t_hm1"] - t) > 1e-8 or abs(r["R2"] - r2) > 1e-8 or r["N"] != int(n):
                raise AssertionError(f"{s['name']} h={h} does not reproduce the spec 001/002 regression")
    table = pd.read_csv(inf3.RESULTS / "table_inference.csv").set_index(["id", "h"])
    crisis = inf3.crisis_months()
    regressors = {"VRP": loaded["vrp"], "VRP_fwd": loaded["vrp_fwd"]}
    out = []
    for rid, reg, start, end in REF_003A:
        sample = btz.slice_sample(full, start, end)
        own = [s for s in specs if s["market"] == "US" and s["regressor"] == reg and s["start"] == start and s["end"] == end]
        row = own[0]["row"] if own else inf3.run_row(sample, regressors[reg], crisis, f"check {rid}")
        for h in HORIZONS:
            r = row["by_h"][h]
            ref = table.loc[(rid, h)]
            gaps = (abs(r["coef_B"] - float(ref["coef_B"])), abs(r["t_hm1"] - float(ref["t_NW_hm1"])), abs(r["t_1b"] - float(ref["t_1B"])))
            if max(gaps) > REF_TOL or r["N"] != int(ref["N"]):
                raise AssertionError(f"{rid} h={h}: gaps {gaps}, N {r['N']} vs {int(ref['N'])}")
            out.append({"id": rid, "h": h, "table_row": bool(own), "coef_B": r["coef_B"], "t_NW_hm1": r["t_hm1"],
                        "t_1B": r["t_1b"], "N": r["N"], "max_gap": max(gaps)})
    return out


def check_skip_timing(specs: list[dict], mk: dict) -> int:
    """Check 3. Y^SK from the two prices, and F_{t+1} after the last US day of month t."""
    n_checked = 0
    for s in [s for s in specs if s["timing"] == "SK"]:
        close = mk["closes"][s["market"]]
        frame = s["frame"]
        eu = mk["eu"][s["market"]]
        for h in HORIZONS:
            r = s["row"]["by_h"][h]
            for k, t in enumerate(r["index"]):
                loc = frame.index.get_loc(t)
                m1 = frame.index[loc + 1]
                mh = frame.index[loc + h]
                f_date = pd.Timestamp(eu.at[m1, "F"])
                t_date = pd.Timestamp(eu.at[mh, "T"])
                us_last = pd.Timestamp(mk["us_last"].loc[t])
                if not f_date > us_last:
                    raise AssertionError(f"{s['name']} {_ym(t)}: F_t+1 {f_date.date()} is not after the US close {us_last.date()}")
                if not (f_date.to_period("M") == m1.to_period("M") and t_date.to_period("M") == mh.to_period("M") and f_date <= t_date):
                    raise AssertionError(f"{s['name']} {_ym(t)} h={h}: price dates outside F_t+1 … T_t+h")
                rf_sum = float(eu["rf"].loc[frame.index[loc + 1 : loc + h + 1]].sum())
                y_px = 1200.0 / h * (np.log(close.loc[t_date]) - np.log(close.loc[f_date]) - rf_sum)
                if abs(y_px - r["y"][k]) > 1e-9:
                    raise AssertionError(f"{s['name']} {_ym(t)} h={h}: Y^SK from prices {y_px} vs {r['y'][k]}")
                n_checked += 1
    return n_checked


def check_penultimate_timing(specs: list[dict], mk: dict) -> pd.DataFrame:
    """Check 4. The latest US date in VRP^{T−1}_t against the European close T_t."""
    rows = []
    pen = mk["pen"]
    for s in [s for s in specs if s["timing"] == "T-1"]:
        eu = mk["eu"][s["market"]]
        months = s["row"]["by_h"][1]["index"]
        before = equal = 0
        equal_months = []
        for t in months:
            cut = pd.Timestamp(pen.at[t, "cut"])
            latest = max(pd.Timestamp(pen.at[t, "rv_max"]), cut)
            if latest != cut:
                raise AssertionError(f"{s['name']} {_ym(t)}: VRP^T−1 uses {latest.date()} after {cut.date()}")
            t_eu = pd.Timestamp(eu.at[t, "T"])
            last_us = pd.Timestamp(pen.at[t, "last_US"])
            if cut < t_eu:
                before += 1
            elif cut == t_eu and last_us != t_eu:
                equal += 1
                equal_months.append(_ym(t))
            else:
                raise AssertionError(f"{s['name']} {_ym(t)}: penultimate US day {cut.date()} vs European close {t_eu.date()}")
        rows.append({"market": s["market"], "sample": s["sample"], "rows": len(months), "pen_before_T": before,
                     "pen_equal_T": equal, "equal_months": ", ".join(equal_months)})
    return pd.DataFrame(rows)


# ── Flags, tables, figure ────────────────────────────────────────────────────


def flags(tab: pd.DataFrame) -> dict:
    e1_parts = []
    for market in ("DAX", "CAC"):
        for sid in SAMPLE_IDS:
            nb = _pick(tab, market, "VRP", "NB", sid)["t_1B"]
            sk = _pick(tab, market, "VRP", "SK", sid)["t_1B"]
            t1 = _pick(tab, market, "VRP", "T-1", sid)["t_1B"]
            e1_parts.append((market, sid, sk - nb, t1 - nb, abs(sk - nb) < E1_GAP and abs(t1 - nb) < E1_GAP))

    def spill(regressor: str, sid: str) -> tuple[bool, dict]:
        parts = {}
        for market in ("DAX", "CAC"):
            r = _pick(tab, market, regressor, "SK", sid)
            ok = abs(r["t_1B"]) > T_CRIT and r["p_boot_NW_iid"] < P_CRIT and r["p_boot_NW_wild"] < P_CRIT
            parts[market] = (ok, r)
        return all(v[0] for v in parts.values()), parts

    e2, e2p = spill("VRP", "pre-2008")
    e3, e3p = spill("VRP_fwd", "post-2008")
    us = _pick(tab, "US", "VRP", US_TIMING, "pre-2008")["std_coef"]
    dax = _pick(tab, "DAX", "VRP", "SK", "pre-2008")["std_coef"]
    e4 = abs(dax - us) <= E4_BAND * abs(us)
    return {
        "E1": (all(p[4] for p in e1_parts), e1_parts),
        "E2": (e2, e2p),
        "E3": (e3, e3p),
        "E4": (e4, (dax, us, dax / us - 1.0)),
    }


def timing_table(tab: pd.DataFrame) -> pd.DataFrame:
    out = []
    for market in ("DAX", "CAC"):
        for sid in SAMPLE_IDS:
            nb, sk, t1 = (_pick(tab, market, "VRP", tm, sid) for tm in ("NB", "SK", "T-1"))
            if not (nb["N"] == sk["N"] == t1["N"]):
                raise AssertionError(f"{market} {sid}: the timings have different N")
            out.append({
                "market": market, "regressor": "VRP", "sample": sid, "start": nb["start"], "end": nb["end"], "h": 3, "N": int(nb["N"]),
                "coef_NB": nb["coef_B"], "coef_SK": sk["coef_B"], "coef_T1": t1["coef_B"],
                "t1B_NB": nb["t_1B"], "t1B_SK": sk["t_1B"], "t1B_T1": t1["t_1B"],
                "t1B_SK_minus_NB": sk["t_1B"] - nb["t_1B"], "t1B_T1_minus_NB": t1["t_1B"] - nb["t_1B"],
                "tNW_NB": nb["t_NW_hm1"], "tNW_SK": sk["t_NW_hm1"], "tNW_T1": t1["t_NW_hm1"],
            })
    return pd.DataFrame(out)


def plot_h3(tab: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    titles = {"pre-2008": "Pre-2008 (VRP 1994-02, VRP_fwd 1995-01, to 2007-12)", "post-2008": "Post-2008 (2008-01 to 2025-12)"}
    width = 0.36
    pos = np.arange(len(MARKET_ORDER))
    vals_all = []
    for ax, sid in zip(axes, SAMPLE_IDS):
        v, g = [], []
        for market in MARKET_ORDER:
            timing = US_TIMING if market == "US" else "SK"
            v.append(_pick(tab, market, "VRP", timing, sid)["t_1B"])
            g.append(_pick(tab, market, "VRP_fwd", timing, sid)["t_1B"])
        vals_all += v + g
        ax.bar(pos - width / 2, v, width, color="steelblue", label="VRP")
        ax.bar(pos + width / 2, g, width, color="darkorange", label="VRP_fwd")
        for xpos, val in zip(np.concatenate([pos - width / 2, pos + width / 2]), v + g):
            ax.text(xpos, val + (0.08 if val >= 0 else -0.08), _fmt(val, 2), ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
        ax.axhline(T_CRIT, color="black", lw=0.9, ls="--", label="±1.96")
        ax.axhline(-T_CRIT, color="black", lw=0.9, ls="--")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xticks(pos)
        ax.set_xticklabels(["US (^GSPC)", "DAX, SK", "CAC, SK"])
        ax.set_title(titles[sid], fontsize=10)
    span = vals_all + [T_CRIT, -T_CRIT]
    axes[0].set_ylim(min(span) - 0.5, max(span) + 0.5)
    axes[0].set_ylabel("Hodrick 1B t-statistic, h = 3")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("US VRP and VRP_fwd predicting US, DAX and CAC excess returns, h = 3", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── SUMMARY.md ───────────────────────────────────────────────────────────────


def _vs_us(value: float, us: float) -> str:
    rel = 100.0 * (value / us - 1.0)
    return f"{_fmt(abs(rel), 0)}% {'above' if rel >= 0 else 'below'}"


def _row_label(r: pd.Series) -> str:
    return f"| {r['market']} | {r['regressor']} | {TIMING_LABEL[r['timing']]} | {r['start']} → {r['end']} |"


def write_summary(p: dict) -> None:
    tab, tim, fl = p["tab"], p["timing"], p["flags"]
    dq, ref, pen_tab = p["dq_info"], p["ref_rows"], p["pen_check"]
    L: list[str] = []
    L.append("# Spec 005 — European extension: does the US VRP predict DAX and CAC returns?")
    L.append("")
    L.append(
        "Code: `src/europe.py`, called from notebook section 5.2. It imports the spec 001, 002, 003 and 003a modules and does not modify them. "
        "VRP and VRP_fwd are the spec 001 and spec 002 series (`inference._load`). The US rows and the NB and T−1 rows go through `inference.run_row` "
        "(spec 001 Y, statsmodels NW, `inference.hodrick_t`), and their bootstrap is `inference.simulate_null` (iid) and `inference_003a.simulate_wild` (wild) "
        "with `inference_003a._boot_stats`. The SK rows need a day-1 term in the 1B score and in the null draws; see Deviations."
    )
    L.append("")
    L.append("## Indices and data")
    L.append("")
    L.append(
        "- **Index definitions.** `^GDAXI` (DAX) is a performance index: dividends are reinvested, so it is a total-return index. "
        "`^FCHI` (CAC 40) and `^GSPC` (S&P 500) are price indices. All returns are log returns in local currency (EUR for DAX and CAC, USD for the US), with no currency conversion. "
        "A constant dividend yield shifts the intercept of the regression, not the slope, so the index type does not explain a difference in coefficients."
    )
    L.append(
        "- **Risk-free rate.** `data/euribor.csv` is FRED IR3TIB01EZM156N, the 3-month interbank rate, monthly, percent per year. "
        "Before 1999 it is the OECD synthetic euro-area series, not a traded Euribor. rf^EU_t = Euribor_{t−1} / 1200, the one-month lag of the US rf; the first usable month is 1994-02."
    )
    L.append(
        f"- **Daily files.** `data/dax.csv` ({len(p['closes']['DAX']):,} days, {p['closes']['DAX'].index.min().date()} → {p['closes']['DAX'].index.max().date()}) and "
        f"`data/cac.csv` ({len(p['closes']['CAC']):,} days, {p['closes']['CAC'].index.min().date()} → {p['closes']['CAC'].index.max().date()}), Close column, unchanged. "
        "US side: the spec 001 panel (`^GSPC`, `data/vix_1990_data.csv`, TB3MS lagged one month) and the spec 002 VRP_fwd."
    )
    L.append("")
    L.append("## Definitions")
    L.append("")
    L.append(
        "T_m and F_m are the last and first European trading days of month m, from the market's own daily file. "
        "**NB**: e_m = log(P(T_m)/P(T_{m−1})) − rf^EU_m and Y^NB_t(h) = (1200/h) Σ_{j=1..h} e_{t+j}, as spec 001. "
        "**SK**: Y^SK_t(h) = (1200/h)[log(P(T_{t+h})/P(F_{t+1})) − Σ_{j=1..h} rf^EU_{t+j}]. "
        "Writing d_m = log(P(F_m)/P(T_{m−1})), Y^SK_t(h) = Y^NB_t(h) − 1200·d_{t+1}/h. "
        "**T−1**: VRP^{T−1}_t is the spec 001 VRP with the US daily returns and the VIX close up to the penultimate `^GSPC` trading day of month t, paired with Y^NB; raw VRP only. "
        "**US benchmark**: `^GSPC` with the spec 001 Y on the same months."
    )
    L.append("")
    L.append(
        "Samples, regressor months t: pre-2008 is 1994-02 → 2007-12 for VRP and 1995-01 → 2007-12 for VRP_fwd; post-2008 is 2008-01 → 2025-12. "
        "As in specs 001–003a, Y_t(h) only uses returns inside the sample, so N = months − h. "
        f"{FWD_NOTE}"
    )
    L.append("")
    L.append(
        "Statistics: coef(B) in BTZ units (annualized % per monthly %²); t NW with maxlags = max(1, h−1), Bartlett, no small-sample correction; "
        "t Hodrick 1B with the null residual on positions h−1 … n−2; bootstrap p-values of the NW t, spec 003 C3 null (AR(1) regressor, constant mean return), "
        f"{N_REP:,} replications each for iid and wild, all from one `numpy.random.default_rng({SEED})` in the row order of `table_europe.csv`, iid then wild within a row. "
        "Standardized coefficient = coef(B) × sd(x) / sd(Y) on the regression rows. coef / sd(e) divides coef(B) by the standard deviation of the market's monthly excess return "
        "(NB definition, × 1200) over the regression-row months t."
    )
    L.append("")
    L.append(
        "For a univariate OLS regression, coef × sd(x) / sd(Y) is the sample correlation of x and Y, that is sign(coef) × √R². "
        "The code asserts this on every row (1e-9). E4 therefore compares the correlations of the US VRP with US and DAX returns."
    )
    L.append("")

    L.append("## Checks")
    L.append("")
    L.append("### 1. US benchmark")
    L.append("")
    L.append(
        "Pass. On both samples and at h = 1, 3, 6, the US VRP rows equal `baseline_1990.run_sample` on the same months, and the US VRP_fwd rows equal "
        "`influence.run_regression` (coef(B), t NW(h−1), R² and N, to 1e-8). Against `results/003-inference/table_inference.csv` (003a), to "
        f"{REF_TOL:g} on coef(B), t NW(h−1) and t 1B, with the same N:"
    )
    L.append("")
    L.append("| 003a row | h | coef(B) | t NW(h−1) | t 1B | N | largest gap | source |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in ref:
        src = "table row" if r["table_row"] else "check-only fit, 1995-01 → 2007-12"
        L.append(f"| {r['id']} | {r['h']} | {_fmt(r['coef_B'], 4)} | {_fmt(r['t_NW_hm1'], 2)} | {_fmt(r['t_1B'], 2)} | {r['N']} | {r['max_gap']:.1e} | {src} |")
    L.append("")
    L.append(
        "G-95-07, V-08-25 and G-08-25 are rows of `table_europe.csv`. The pre-2008 VRP row starts in 1994-02, so V-95-07 is checked on a separate fit over 1995-01 → 2007-12 through the same code path."
    )
    L.append("")
    L.append("### 2. Daily data quality, 1994–2025")
    L.append("")
    for market in ("DAX", "CAC"):
        q = dq[market]
        big = q["big"]
        big_txt = ", ".join(f"{t.date()} ({100 * v:+.2f}%)" for t, v in big.items()) if len(big) else "none"
        zero_txt = (
            f"{q['n_zero']} months, {q['zero_first']} → {q['zero_last']}" + (", one unbroken run at the start of the file" if q["zero_contiguous"] else ", not contiguous")
            if q["n_zero"] else "none"
        )
        L.append(
            f"- **{market}**: every month has at least {MIN_DAYS} trading days (pass; fewest {q['min_days']}, in {q['min_month']}). "
            f"Daily |log return| > 10%: {big_txt}. Months with zero volume on every day (information only): {zero_txt}."
        )
    L.append("")
    big_months = sorted({t.strftime("%Y-%m") for m in ("DAX", "CAC") for t in dq[m]["big"].index})
    L.append(
        "The full lists are in `table_data_quality.csv`. The large daily moves fall in " + ", ".join(big_months)
        + ("; the same days appear in both files." if set(dq["CAC"]["big"].index) <= set(dq["DAX"]["big"].index) else ".")
    )
    L.append("")
    L.append("### 3. SK timing")
    L.append("")
    L.append(
        f"Pass, on all {p['n_sk_checked']:,} SK regression rows (DAX and CAC, VRP and VRP_fwd, both samples, h = 1, 3, 6). "
        "For every row t, F_{t+1} is strictly after the last `^GSPC` trading day of month t. "
        "Y^SK is recomputed from the two prices P(F_{t+1}) and P(T_{t+h}) and the h Euribor terms, and equals the regression Y to 1e-9; "
        "F_{t+1} is in month t+1, T_{t+h} in month t+h, and F_{t+1} ≤ T_{t+h}. So every price in Y^SK is dated in [F_{t+1}, T_{t+h}]."
    )
    L.append("")
    L.append("### 4. T−1 timing")
    L.append("")
    L.append(
        "Pass as written. The latest US date inside VRP^{T−1}_t is the penultimate `^GSPC` day of month t (asserted for RV and IV). "
        "In every row it is either before T_t, or equal to T_t in a month where the European and US last days differ. No month has the penultimate US day after T_t."
    )
    L.append("")
    L.append("| market | sample | rows (h = 1) | penultimate US day < T_t | = T_t (last days differ) | months with equality |")
    L.append("|---|---|---|---|---|---|")
    for _, r in pen_tab.iterrows():
        L.append(f"| {r['market']} | {r['sample']} | {r['rows']} | {r['pen_before_T']} | {r['pen_equal_T']} | {r['equal_months'] or '—'} |")
    L.append("")
    L.append(
        "When the dates are equal, the US close on that day (22:00 CET) is still after the European close (17:30 CET). "
        "In those months VRP^{T−1}_t is not known before T_t, so T−1 does not remove the overlap there. They are mostly Decembers, when Xetra or Euronext close before 31 December."
    )
    L.append("")
    L.append("### 5. No NaN, N")
    L.append("")
    L.append(
        "Pass. e, d (SK), the regressor, Y and the US date columns are finite on every regression sample (asserted per row and horizon), and N = months − h everywhere. "
        "The spec 001 look-ahead rule (`influence.assert_row_timing`) also passes on every row."
    )
    L.append("")
    L.append("| sample | regressor | months | N, h = 1 | h = 3 | h = 6 |")
    L.append("|---|---|---|---|---|---|")
    for sid in SAMPLE_IDS:
        for reg in ("VRP", "VRP_fwd"):
            sel = tab[(tab["sample"] == sid) & (tab["regressor"] == reg)]
            ns = [sorted(sel[sel["h"] == h]["N"].unique()) for h in HORIZONS]
            if any(len(n) != 1 for n in ns):
                raise AssertionError(f"{sid} {reg}: N differs across markets or timings")
            first = sel.iloc[0]
            L.append(f"| {sid} | {reg} | {first['start']} → {first['end']} ({ns[0][0] + 1}) | {ns[0][0]} | {ns[1][0]} | {ns[2][0]} |")
    L.append("")
    L.append(f"{FWD_NOTE} N is the same for every market and timing within a sample and regressor.")
    L.append("")

    L.append("## Results at h = 3")
    L.append("")
    L.append(
        f"coef(B) in BTZ units. p iid and p wild are bootstrap p-values of the NW t. Std coef = coef × sd(x)/sd(Y). R² in percent. {FWD_NOTE} "
        "US rows use the spec 001 timing. h = 1 and 6 are in `table_europe.csv`."
    )
    L.append("")
    L.append("| market | regressor | timing | sample | coef(B) | t NW | t 1B | p iid | p wild | R² | N | std coef | coef / sd(e) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    h3 = tab[tab["h"] == 3]
    for _, r in h3.iterrows():
        L.append(
            f"{_row_label(r)} {_fmt(r['coef_B'], 4)} | {_fmt(r['t_NW_hm1'], 2)} | {_fmt(r['t_1B'], 2)} | {_fmt(r['p_boot_NW_iid'], 4)} | "
            f"{_fmt(r['p_boot_NW_wild'], 4)} | {_fmt(r['R2'], 2)} | {int(r['N'])} | {_fmt(r['std_coef'], 3)} | {_fmt(r['coef_over_sd_e'], 5)} |"
        )
    L.append("")
    L.append("![Hodrick t at h = 3](fig_europe_h3.png)")
    L.append("")
    L.append("### Timing, raw VRP, h = 3")
    L.append("")
    L.append("Samples 1994-02 → 2007-12 and 2008-01 → 2025-12 (raw VRP only, so no VRP_fwd start applies).")
    L.append("")
    L.append("| market | sample | N | coef NB | coef SK | coef T−1 | t 1B NB | t 1B SK | t 1B T−1 | SK − NB | T−1 − NB |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in tim.iterrows():
        L.append(
            f"| {r['market']} | {r['sample']} | {r['N']} | {_fmt(r['coef_NB'], 4)} | {_fmt(r['coef_SK'], 4)} | {_fmt(r['coef_T1'], 4)} | "
            f"{_fmt(r['t1B_NB'], 2)} | {_fmt(r['t1B_SK'], 2)} | {_fmt(r['t1B_T1'], 2)} | {_fmt(r['t1B_SK_minus_NB'], 2)} | {_fmt(r['t1B_T1_minus_NB'], 2)} |"
        )
    L.append("")
    L.append(
        "Audit reference (direction only, not targets): SPY-based VRP, notebook units, NW, h = 3, 1994–2007. "
        "The comparable figures here are the NW t on the pre-2008 rows, with `^GSPC`, BTZ units and the lagged rf:"
    )
    L.append("")
    L.append("| market | NB audit | NB here | T−1 audit | T−1 here | SK audit | SK here |")
    L.append("|---|---|---|---|---|---|---|")
    for market in ("DAX", "CAC"):
        cells = []
        for timing in ("NB", "T-1", "SK"):
            cells.append(_fmt(AUDIT_T[market][timing], 2))
            cells.append(_fmt(_pick(tab, market, "VRP", timing, "pre-2008")["t_NW_hm1"], 2))
        L.append(f"| {market} | " + " | ".join(cells) + " |")
    L.append("")
    vols = {m: _pick(tab, m, "VRP", US_TIMING if m == "US" else "NB", "pre-2008")["sd_e_ann"] / 12.0 for m in MARKET_ORDER}
    L.append(
        "Monthly excess-return volatility on the pre-2008 regression months (1994-02 → 2007-09 at h = 3), in % per month: "
        + ", ".join(f"{m} {_fmt(vols[m], 2)} (audit {_fmt(AUDIT_VOL[m], 2)})" for m in MARKET_ORDER)
        + ". The audit figures are monthly return volatilities on 1994–2007, with SPY for the US."
    )
    L.append("")

    L.append("## Reading the coefficients")
    L.append("")
    us, dax, cac = (_pick(tab, m, "VRP", US_TIMING if m == "US" else "SK", "pre-2008") for m in MARKET_ORDER)
    L.append(
        f"Pre-2008, raw VRP, h = 3 (SK for DAX and CAC): coef(B) is {_fmt(us['coef_B'], 3)} for the US, {_fmt(dax['coef_B'], 3)} for the DAX and {_fmt(cac['coef_B'], 3)} for the CAC. "
        f"The monthly excess-return standard deviations on the same rows are {_fmt(us['sd_e_ann'] / 12, 2)}%, {_fmt(dax['sd_e_ann'] / 12, 2)}% and {_fmt(cac['sd_e_ann'] / 12, 2)}%. "
        f"Scaled by them, coef / sd(e) is {_fmt(us['coef_over_sd_e'], 5)}, {_fmt(dax['coef_over_sd_e'], 5)} and {_fmt(cac['coef_over_sd_e'], 5)}, "
        f"and the standardized coefficients are {_fmt(us['std_coef'], 3)}, {_fmt(dax['std_coef'], 3)} and {_fmt(cac['std_coef'], 3)}. "
        f"The DAX raw coefficient is {_vs_us(dax['coef_B'], us['coef_B'])} the US one, "
        f"its volatility-scaled coefficient {_vs_us(dax['coef_over_sd_e'], us['coef_over_sd_e'])} "
        f"and its standardized coefficient {_vs_us(dax['std_coef'], us['std_coef'])}. "
        "The raw coefficient is in annualized percent of the index's own return, so it grows with the index's volatility; the scaled coefficients and E4 compare sensitivities. "
        "The dividend treatment cannot explain a slope difference: `^GDAXI` reinvests dividends and `^FCHI` and `^GSPC` do not, but a constant dividend yield only moves the intercept."
    )
    L.append("")

    L.append("## Flags (h = 3, Hodrick 1B)")
    L.append("")
    L.append("Fixed before the run. Every condition within a flag is joined by **and**. p iid and p wild are the bootstrap p-values of the NW t.")
    L.append("")
    L.append("| flag | holds | numbers |")
    L.append("|---|---|---|")
    e1, e1p = fl["E1"]
    L.append(
        f"| E1 timing is immaterial | {_yn(e1)} | raw VRP, \\|Δt 1B\\| < {E1_GAP}: "
        + "; ".join(f"{m} {sid}: SK − NB {_fmt(a, 2)}, T−1 − NB {_fmt(b, 2)} ({_yn(ok)})" for m, sid, a, b, ok in e1p)
        + " |"
    )
    for key, title, reg, sid in (("E2", "E2 pre-2008 spillover", "VRP", "1994-02 → 2007-12"), ("E3", "E3 post-2008 spillover", "VRP_fwd", "2008-01 → 2025-12")):
        ok, parts = fl[key]
        L.append(
            f"| {title} | {_yn(ok)} | {reg}, SK, {sid}: "
            + "; ".join(
                f"{m} t 1B {_fmt(r['t_1B'], 2)}, p iid {_fmt(r['p_boot_NW_iid'], 4)}, p wild {_fmt(r['p_boot_NW_wild'], 4)} ({_yn(o)})"
                for m, (o, r) in parts.items()
            )
            + " |"
        )
    e4, (sd_dax, sd_us, rel) = fl["E4"]
    L.append(
        f"| E4 no extra sensitivity | {_yn(e4)} | 1994-02 → 2007-12, raw VRP: std coef DAX (SK) {_fmt(sd_dax, 3)}, US {_fmt(sd_us, 3)}, "
        f"DAX / US − 1 = {_fmt(100 * rel, 1)}% (band ±{_fmt(100 * E4_BAND, 0)}%) |"
    )
    L.append("")
    L.append("A flag that fails is reported with its numbers. The timing, the indices and the samples were not changed after the run.")
    L.append("")

    L.append("## Deviations from this spec")
    L.append("")
    devs = [
        "**Hodrick 1B and the bootstrap for SK.** Y^SK is not a forward average of one monthly series, so `inference.hodrick_t` and the spec 003 null draws do not apply to it directly. "
        "`hodrick_t_skip` adds the day-1 term: the 1B score at position p is u_{p+1}·Σ_{i<h} z_{p−i} − δ_{p+1}·z_p, with δ the demeaned day-1 return d. "
        "With d = 0 it equals `inference.hodrick_t` to 1e-12, and at h = 1 it equals `inference.hodrick_t` on e − d (both asserted on simulated data). "
        "`simulate_skip` draws δ_s with the same index (iid) or the same Rademacher weight (wild) as (ε_s, η_s). From the same generator state, its e* and x* are identical to "
        "`inference.simulate_null` and `inference_003a.simulate_wild`, and it consumes the generator identically (asserted).",
        "**Check 4.** The spec's first sentence asks for the latest US date to be strictly before T_t, and its second admits equality when the last days differ. "
        "The code asserts the second, more precise reading and reports the months with equality. In those months the US close on the penultimate day is after the European close on the same day.",
        "**coef / sd(e).** \"The same rows\" is read as the regression-row months t, with the market's NB monthly excess return in annualized percent. "
        "For the SK rows the same NB monthly series is used, because it is the market's monthly excess return.",
        "**Check 1, V-95-07.** The table's pre-2008 VRP rows start in 1994-02, as the spec requires, so V-95-07 (1995-01 → 2007-12) is reproduced by a separate fit through the same code. It is not a table row.",
        "**Bootstrap draws.** The US rows G-95-07, V-08-25 and G-08-25 are redrawn from the spec 005 generator, so their p-values can differ in the last digits from 003a. Check 1 covers coef, t NW, t 1B and N only.",
        "**Notebook numbering.** The old section 5 already has a \"5.2 Results\" heading. It is left unchanged. The new section is titled \"5.2 European extension on the BTZ pipeline (spec 005)\" and placed after the old section, "
        "and the note at the top of section 5 points to it by that title.",
        "`table_europe.csv` adds sd_x, sd_Y, sd_e_ann, first_row, last_row and a note column (the VRP_fwd start) to the statistics the spec lists. "
        "`table_data_quality.csv` holds the three lists of check 2 in one file, with a `check` column.",
    ]
    for i, text in enumerate(devs, start=1):
        L.append(f"{i}. {text}")
    L.append("")
    (RESULTS / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> dict:
    inf3._self_checks()
    inf3a._self_checks()
    _self_checks()
    loaded = inf3._load()
    mk = load_markets(loaded)
    dq_table, dq_info = data_quality(mk["closes"])

    specs = _row_samples(loaded, mk)
    print(f"Fitting {len(specs)} rows")
    fit_all(specs)
    ref_rows = check_us_benchmark(loaded, specs)
    n_sk = check_skip_timing(specs, mk)
    pen_check = check_penultimate_timing(specs, mk)

    print(f"Bootstrap, {N_REP} replications, seed {SEED}")
    run_bootstraps(specs)
    tab = records(specs)
    if tab[["coef_B", "t_NW_hm1", "t_1B", "p_boot_NW_iid", "p_boot_NW_wild", "R2", "std_coef", "coef_over_sd_e"]].isna().any().any():
        raise AssertionError("A statistic is NaN")
    tim = timing_table(tab)
    fl = flags(tab)

    RESULTS.mkdir(parents=True, exist_ok=True)
    tab.to_csv(RESULTS / "table_europe.csv", index=False)
    tim.to_csv(RESULTS / "table_timing.csv", index=False)
    dq_table.to_csv(RESULTS / "table_data_quality.csv", index=False)
    plot_h3(tab, RESULTS / "fig_europe_h3.png")
    payload = {
        "tab": tab,
        "timing": tim,
        "flags": fl,
        "dq_info": dq_info,
        "ref_rows": ref_rows,
        "pen_check": pen_check,
        "n_sk_checked": n_sk,
        "closes": mk["closes"],
    }
    write_summary(payload)
    for k, (ok, _v) in fl.items():
        print(f"  {k}: {_yn(ok)}")
    print(f"Wrote {RESULTS}")
    return payload


if __name__ == "__main__":
    main()
