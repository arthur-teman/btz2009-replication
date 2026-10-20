"""Spec 003 — conservative inference and the fixed crisis exclusion.

Imports the spec 001 and spec 002 pipelines and does not modify them.
Writes results/003-inference/.
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
from src import influence as inf

RESULTS = btz.ROOT / "results" / "003-inference"
HORIZONS = (1, 3, 6)
N_REP = 5000
SEED = 20261021
BLOCK_LENGTH = 6
# Fixed before reading the estimates. A Hodrick t within this distance of the
# published BTZ h=3 t-statistic (2.86) is reported as close.
BTZ_H3_T = 2.86
CLOSE_TO_BTZ = 0.15
T_MEAN_BAND = 0.15

ROW_ORDER = ("R-base", "R-full", "R-D1", "R-G", "R-G07")
PART_A_ORDER = ("D1", "D2", "D3")

REF_H3 = {
    "R-base": (0.4196, 3.46),
    "R-full": (0.0908, 0.70),
    "R-G": (0.3256, 5.23),
}


def _fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _ym(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m")


def crisis_months() -> pd.DatetimeIndex:
    """C = {2008-09 … 2008-12, 2020-02 … 2020-04}."""
    blocks = (
        pd.date_range("2008-09-01", "2008-12-31", freq="ME"),
        pd.date_range("2020-02-01", "2020-04-30", freq="ME"),
    )
    out = blocks[0].append(blocks[1])
    if len(out) != 7:
        raise AssertionError(f"Crisis set has {len(out)} months, expected 7")
    return pd.DatetimeIndex(out)


def _window_months(t, n_months: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(t).to_period("M")
    return pd.DatetimeIndex([(start + j).to_timestamp("M") for j in range(n_months)])


def overlapping_drop(index: pd.DatetimeIndex, h: int, crisis: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Regression rows t whose window t, t+1, …, t+h meets the crisis set.

    Only dates that would otherwise enter the regression (a complete h-month
    forward window) are included.
    """
    crisis_set = set(pd.DatetimeIndex(crisis))
    index = pd.DatetimeIndex(index)
    n = len(index)
    eligible = index[: n - h]
    dropped = []
    for t in eligible:
        window = _window_months(t, h + 1)
        if any(m in crisis_set for m in window):
            dropped.append(pd.Timestamp(t))
    return pd.DatetimeIndex(dropped)


def _assert_h3_drop_matches_spec(index: pd.DatetimeIndex, dropped: pd.DatetimeIndex) -> None:
    """The spec names the h=3 rows removed on a sample that contains both crises."""
    expected = pd.date_range("2008-06-01", "2008-12-31", freq="ME").append(
        pd.date_range("2019-11-01", "2020-04-30", freq="ME")
    )
    if not pd.DatetimeIndex(dropped).equals(pd.DatetimeIndex(expected)):
        raise AssertionError(
            "h=3 overlap exclusion is not 2008-06…2008-12 and 2019-11…2020-04: "
            + ", ".join(_ym(t) for t in dropped)
        )
    if not expected.isin(index).all():
        raise AssertionError("A spec h=3 drop date is outside the sample index")


def _no_crisis_left(index: pd.DatetimeIndex, h: int, crisis: pd.DatetimeIndex) -> None:
    crisis_set = set(pd.DatetimeIndex(crisis))
    for t in index:
        window = _window_months(t, h + 1)
        hit = [m for m in window if m in crisis_set]
        if hit:
            raise AssertionError(
                f"{_ym(t)} at h={h} still touches " + ", ".join(_ym(m) for m in hit)
            )


def newey_west_t(y: np.ndarray, x: np.ndarray, maxlags: int) -> tuple[np.ndarray, np.ndarray]:
    """Slope t-statistic, Bartlett HAC, use_correction False.

    y and x have shape (replications, T). Matches statsmodels OLS cov_type='HAC'.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim == 1:
        y = y[None, :]
        x = x[None, :]
    x_mean = x.mean(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    xd = x - x_mean
    yd = y - y_mean
    ssx = np.sum(xd * xd, axis=1)
    b = np.sum(xd * yd, axis=1) / ssx
    a = y_mean[:, 0] - b * x_mean[:, 0]
    resid = y - a[:, None] - b[:, None] * x
    exog = np.stack([np.ones(y.shape), x], axis=-1)
    score = resid[:, :, None] * exog
    meat = np.einsum("bti,btj->bij", score, score)
    for lag in range(1, int(maxlags) + 1):
        weight = 1.0 - lag / (int(maxlags) + 1.0)
        cross = np.einsum("bti,btj->bij", score[:, lag:, :], score[:, :-lag, :])
        meat = meat + weight * (cross + np.transpose(cross, (0, 2, 1)))
    xtx = np.einsum("bti,btj->bij", exog, exog)
    cov = np.matmul(np.matmul(np.linalg.inv(xtx), meat), np.linalg.inv(xtx))
    se = np.sqrt(cov[:, 1, 1])
    return b / se, b


def _hodrick_variance(z: np.ndarray, w: np.ndarray, h: int) -> np.ndarray:
    """V_avg[1, 1] from Z = (1/T) z'z and S = (1/T) w'w, shape (replications,)."""
    t_obs = z.shape[1]
    Z = np.einsum("bti,btj->bij", z, z) / t_obs
    S = np.einsum("bti,btj->bij", w, w) / t_obs
    Zinv = np.linalg.inv(Z)
    v_sum = np.matmul(np.matmul(Zinv, S), Zinv) / t_obs
    return v_sum[:, 1, 1] / float(h * h)


def _rolling_sum(x: np.ndarray, h: int) -> np.ndarray:
    """Sum of h terms ending at each date. Leading h-1 columns are unused."""
    csum = np.cumsum(x, axis=1)
    out = np.empty_like(x)
    out[:, : h - 1] = np.nan
    out[:, h - 1] = csum[:, h - 1]
    if x.shape[1] > h:
        out[:, h:] = csum[:, h:] - csum[:, :-h]
    return out


def hodrick_t(
    e: np.ndarray,
    x: np.ndarray,
    h: int,
    hod_pos: np.ndarray,
    coef: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hodrick (1992) 1B t-statistic for the h-month average regression.

    e and x are shaped (replications, N), in the regression's units (e is the
    monthly excess return in annualized percent). u is e minus its sample mean.
    w_t and z_t are both taken on `hod_pos`: dates t with t-h+1 inside the
    sample and t+1 inside the return sample. coef is the OLS slope on the
    h-month average, one value per replication, or a scalar broadcast to all.
    """
    e = np.asarray(e, dtype=float)
    x = np.asarray(x, dtype=float)
    if e.ndim == 1:
        e = e[None, :]
        x = x[None, :]
    coef = np.asarray(coef, dtype=float)
    if coef.ndim == 0:
        coef = np.full(e.shape[0], float(coef))
    u = e - e.mean(axis=1, keepdims=True)
    rolled = _rolling_sum(x, h)
    pos = np.asarray(hod_pos, dtype=int)
    u_next = u[:, pos + 1]
    z = np.stack([np.ones((e.shape[0], len(pos))), x[:, pos]], axis=-1)
    w = np.stack([u_next * h, u_next * rolled[:, pos]], axis=-1)
    v11 = _hodrick_variance(z, w, h)
    se = np.sqrt(v11)
    if np.any(~np.isfinite(se)) or np.any(se <= 0):
        raise AssertionError("Hodrick standard error is not finite and positive")
    return coef / se, se


def fit_ar1(x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """x_t = c + rho x_{t-1} + eta_t. eta is aligned with x[1:]."""
    y = np.asarray(x[1:], dtype=float)
    lag = np.asarray(x[:-1], dtype=float)
    model = sm.OLS(y, sm.add_constant(lag)).fit()
    c = float(np.asarray(model.params)[0])
    rho = float(np.asarray(model.params)[1])
    eta = np.asarray(model.resid, dtype=float)
    if len(eta) != len(x) - 1:
        raise AssertionError("AR(1) residual length does not match the sample")
    return c, rho, eta


def _stationary_indices(rng: np.random.Generator, n_rep: int, m: int, mean_block: int) -> np.ndarray:
    """Politis–Romano indices. Block length is geometric with mean `mean_block`."""
    p = 1.0 / float(mean_block)
    new_block = rng.random((n_rep, m)) < p
    new_block[:, 0] = True
    draws = rng.integers(0, m, size=(n_rep, m))
    idx = np.empty((n_rep, m), dtype=np.int64)
    idx[:, 0] = draws[:, 0]
    for t in range(1, m):
        nxt = idx[:, t - 1] + 1
        nxt[nxt == m] = 0
        idx[:, t] = np.where(new_block[:, t], draws[:, t], nxt)
    return idx


def simulate_null(
    rng: np.random.Generator,
    e: np.ndarray,
    x: np.ndarray,
    scheme: str,
    n_rep: int,
) -> np.ndarray:
    """Artificial samples under no predictability. Returns e_star, x_star.

    Pairs (epsilon_s, eta_s) share the date s of the AR(1) innovation.
    x*_1 = x_1. The first return has no innovation pair, so e*_1 = mu.
    That return does not enter Y.
    """
    e = np.asarray(e, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(e)
    mu = float(e.mean())
    eps = e - mu
    c, rho, eta = fit_ar1(x)
    m = n - 1
    if len(eta) != m or len(eps[1:]) != m:
        raise AssertionError("Residual pairs are not aligned")
    if scheme == "iid":
        draw = rng.integers(0, m, size=(n_rep, m))
    elif scheme == "block":
        draw = _stationary_indices(rng, n_rep, m, BLOCK_LENGTH)
    else:
        raise AssertionError(f"Unknown scheme {scheme}")
    eta_s = eta[draw]
    eps_s = eps[1:][draw]
    x_star = np.empty((n_rep, n))
    e_star = np.empty((n_rep, n))
    x_star[:, 0] = x[0]
    e_star[:, 0] = mu
    level = np.full(n_rep, x[0], dtype=float)
    for t in range(m):
        level = c + rho * level + eta_s[:, t]
        x_star[:, t + 1] = level
        e_star[:, t + 1] = mu + eps_s[:, t]
    return e_star, x_star


def forward_average(e: np.ndarray, h: int, positions: np.ndarray) -> np.ndarray:
    """(1/h) sum of e[t+1] … e[t+h] at each position. e is (replications, N)."""
    if e.ndim == 1:
        e = e[None, :]
    cume = np.cumsum(e, axis=1)
    pos = np.asarray(positions, dtype=int)
    if np.any(pos + h >= e.shape[1]):
        raise AssertionError("A forward window runs off the end of the sample")
    return (cume[:, pos + h] - cume[:, pos]) / float(h)


def _self_checks() -> None:
    """Algebra that does not depend on the market data."""
    rng = np.random.default_rng(7)
    n = 120
    x = rng.normal(size=n)
    y = 0.4 * x + rng.normal(size=n)
    for lags in (1, 2, 6, 12):
        model = sm.OLS(y, sm.add_constant(x)).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": lags, "use_correction": False},
        )
        t_np, b_np = newey_west_t(y, x, lags)
        if abs(float(b_np[0]) - float(np.asarray(model.params)[1])) > 1e-10:
            raise AssertionError("Numpy slope does not match statsmodels")
        if abs(float(t_np[0]) - float(np.asarray(model.tvalues)[1])) > 1e-8:
            raise AssertionError(
                f"NW t at maxlags={lags}: numpy {float(t_np[0])} vs statsmodels "
                f"{float(np.asarray(model.tvalues)[1])}"
            )

    # h=1 Hodrick with the OLS residual is White HC0. The production function uses
    # the null residual; this check builds the same sandwich on the OLS residual.
    e = rng.normal(size=n)
    xx = rng.normal(size=n)
    y1 = e[1:]
    x1 = xx[:-1]
    hc0 = sm.OLS(y1, sm.add_constant(x1)).fit(cov_type="HC0")
    resid = np.asarray(hc0.resid, dtype=float)
    z = np.stack([np.ones((1, n - 1)), x1[None, :]], axis=-1)
    w = resid[None, :, None] * z
    se = float(np.sqrt(_hodrick_variance(z, w, 1)[0]))
    se_hc0 = float(np.asarray(hc0.bse)[1])
    if abs(se / se_hc0 - 1.0) > 1e-8:
        raise AssertionError(f"Hodrick h=1 with OLS residuals / HC0 = {se / se_hc0}")

    # Forward average matches spec 001 on a short series.
    series = pd.Series(np.arange(6, dtype=float), index=pd.date_range("2020-01-31", periods=6, freq="ME"))
    got = forward_average(series.to_numpy(), 2, np.array([0, 1, 2]))[0]
    ref = btz.forward_mean(series, 2).iloc[:3].to_numpy()
    if not np.allclose(got, ref):
        raise AssertionError("Forward average does not match spec 001")

    crisis = crisis_months()
    names = [_ym(t) for t in crisis]
    if names != ["2008-09", "2008-10", "2008-11", "2008-12", "2020-02", "2020-03", "2020-04"]:
        raise AssertionError(f"Crisis months are {names}")


def _load() -> dict:
    gspc = btz.load_close(btz.DATA / "gspc_data.csv")
    vix = btz.load_close(btz.DATA / "vix_1990_data.csv")
    rf = btz.load_rf()
    panel = btz.build_panel(gspc, vix, rf)
    full = btz.slice_sample(panel, "1990-01-31", "2025-12-31")
    needed = ["IV", "RV", "VRP", "e", "rv_max", "iv_max"]
    if full[needed].isna().any().any():
        raise AssertionError("The 1990–2025 sample has missing values")
    if len(full) != 432:
        raise AssertionError(f"1990–2025 has {len(full)} months, expected 432")
    comp = inf.realized_components(gspc).reindex(full.index)
    if comp[["RV_w", "RV_d"]].isna().any().any():
        raise AssertionError("A sample month has no weekly or daily RV")
    har = inf.har_expanding(full["RV"], comp["RV_w"], comp["RV_d"])
    if har.index.min() != pd.Timestamp("1995-01-31"):
        raise AssertionError("HAR burn-in does not start in 1995-01")
    vrp_fwd = (full["IV"] - har["pred"]).rename("VRP")
    if vrp_fwd.loc[pd.Timestamp("1995-01-31"):].isna().any():
        raise AssertionError("VRP_fwd is missing on or after 1995-01")
    if vrp_fwd.loc[: pd.Timestamp("1994-12-31")].notna().any():
        raise AssertionError("VRP_fwd is defined before the HAR burn-in")
    return {"full": full, "vrp": full["VRP"], "vrp_fwd": vrp_fwd, "har": har}


def _fit_horizon(sample: pd.DataFrame, x: pd.Series, h: int, dropped: pd.DatetimeIndex) -> dict:
    x_h = x.reindex(sample.index).copy()
    if len(dropped):
        x_h.loc[pd.DatetimeIndex(dropped)] = np.nan
    y = 1200.0 * btz.forward_mean(sample["e"], h)
    model, idx = btz.fit_hac(y, x_h, max(1, h - 1))
    model_h, idx_h = btz.fit_hac(y, x_h, h)
    model_2h, idx_2h = btz.fit_hac(y, x_h, 2 * h)
    if not (idx.equals(idx_h) and idx.equals(idx_2h)):
        raise AssertionError(f"h={h}: lag choices changed the estimation sample")
    for t in idx:
        inf.assert_row_timing(sample, t, h, "spec 003")
    n = len(sample)
    potential = pd.DatetimeIndex(sample.index[: n - h])
    if not pd.DatetimeIndex(dropped).isin(potential).all():
        raise AssertionError("A dropped row was not an otherwise valid regression row")
    expected = potential.difference(pd.DatetimeIndex(dropped))
    if not expected.equals(pd.DatetimeIndex(idx)):
        raise AssertionError("fit_hac sample is not the overlap-excluded window")
    hod_index = pd.DatetimeIndex(sample.index[h - 1 : n - 1]).difference(pd.DatetimeIndex(dropped))
    if len(hod_index) != len(idx):
        raise AssertionError(
            f"h={h}: Hodrick dates {len(hod_index)} vs regression rows {len(idx)}"
        )
    reg_pos = sample.index.get_indexer(idx)
    hod_pos = sample.index.get_indexer(hod_index)
    if np.any(reg_pos < 0) or np.any(hod_pos < 0):
        raise AssertionError("A regression or Hodrick date is missing from the sample")

    e = 1200.0 * sample["e"].to_numpy(dtype=float)
    x_np = x.reindex(sample.index).to_numpy(dtype=float)
    if not (np.isfinite(e).all() and np.isfinite(x_np).all()):
        raise AssertionError("e or the regressor is missing inside the row sample")
    y_np = forward_average(e, h, reg_pos)[0]
    x_reg = x_np[reg_pos]
    t_np, b_np = newey_west_t(y_np, x_reg, max(1, h - 1))
    coef = float(model.params["VRP"])
    t_hm1 = float(model.tvalues["VRP"])
    if abs(float(b_np[0]) - coef) > 1e-8 or abs(float(t_np[0]) - t_hm1) > 1e-8:
        raise AssertionError(
            f"h={h}: numpy NW ({float(b_np[0])}, {float(t_np[0])}) "
            f"vs statsmodels ({coef}, {t_hm1})"
        )
    t_1b, se_1b = hodrick_t(e, x_np, h, hod_pos, coef)
    frame = pd.concat([y.rename("y"), x_h.rename("VRP")], axis=1).dropna()
    hc0 = sm.OLS(frame["y"], sm.add_constant(frame["VRP"])).fit(cov_type="HC0")
    se_hc0 = float(hc0.bse["VRP"])
    ratio = float(se_1b[0] / se_hc0) if h == 1 else np.nan
    if h == 1:
        # The spec's 10% band compares the null residual with the OLS residual.
        # The sandwich itself is checked by putting the OLS residual into it.
        if not frame.index.equals(pd.DatetimeIndex(idx)):
            raise AssertionError("HC0 sample is not the regression sample")
        resid = np.asarray(hc0.resid, dtype=float)
        z = np.stack([np.ones((1, len(resid))), frame["VRP"].to_numpy(dtype=float)[None, :]], axis=-1)
        w = resid[None, :, None] * z
        se_ols = float(np.sqrt(_hodrick_variance(z, w, 1)[0]))
        if abs(se_ols / se_hc0 - 1.0) > 1e-8:
            raise AssertionError(f"Hodrick h=1 with the OLS residual / HC0 = {se_ols / se_hc0}")
    return {
        "h": h,
        "coef_B": coef,
        "t_hm1": t_hm1,
        "t_h": float(model_h.tvalues["VRP"]),
        "t_2h": float(model_2h.tvalues["VRP"]),
        "R2": float(model.rsquared * 100.0),
        "N": int(len(idx)),
        "index": pd.DatetimeIndex(idx),
        "reg_pos": reg_pos,
        "hod_pos": hod_pos,
        "t_1b": float(t_1b[0]),
        "se_1b": float(se_1b[0]),
        "se_hc0": se_hc0,
        "hc0_ratio": float(ratio) if h == 1 else np.nan,
        "e": e,
        "x": x_np,
        "dropped": pd.DatetimeIndex(dropped),
    }


def run_row(sample: pd.DataFrame, x: pd.Series, crisis: pd.DatetimeIndex, name: str) -> dict:
    by_h = {}
    n_checked = 0
    for h in HORIZONS:
        dropped = overlapping_drop(sample.index, h, crisis) if name in {"D1", "D2", "D3", "R-D1"} else pd.DatetimeIndex([])
        # R-D1 is the Part A sample. D1, D2, D3 pass the flag via `name`.
        if name in {"D1", "D2", "D3", "R-D1"}:
            _no_crisis_left(
                pd.DatetimeIndex(sample.index[: len(sample) - h]).difference(dropped),
                h,
                crisis,
            )
        fitted = _fit_horizon(sample, x, h, dropped)
        n_checked += fitted["N"]
        by_h[h] = fitted
    e0 = by_h[1]["e"]
    x0 = by_h[1]["x"]
    for h in HORIZONS:
        if not np.array_equal(by_h[h]["e"], e0) or not np.array_equal(by_h[h]["x"], x0):
            raise AssertionError(f"{name}: horizons are not on the same monthly series")
    c, rho, _eta = fit_ar1(x0)
    if not np.isclose(rho, btz.ar1(pd.Series(x0)), rtol=0, atol=1e-10):
        raise AssertionError(f"{name}: AR(1) slope does not match spec 001's ar1")
    return {"name": name, "by_h": by_h, "n_checked": n_checked, "rho": rho, "ar_c": c}


def _bootstrap_row(rng: np.random.Generator, row: dict) -> dict:
    e = row["by_h"][1]["e"]
    x = row["by_h"][1]["x"]
    out = {}
    for scheme in ("iid", "block"):
        e_star, x_star = simulate_null(rng, e, x, scheme, N_REP)
        by_h = {}
        for h in HORIZONS:
            reg = row["by_h"][h]["reg_pos"]
            hod = row["by_h"][h]["hod_pos"]
            y_star = forward_average(e_star, h, reg)
            x_reg = x_star[:, reg]
            t_nw, coef = newey_west_t(y_star, x_reg, max(1, h - 1))
            t_1b, _se = hodrick_t(e_star, x_star, h, hod, coef)
            if not (np.isfinite(t_nw).all() and np.isfinite(t_1b).all() and np.isfinite(coef).all()):
                raise AssertionError(f"{row['name']} {scheme} h={h}: a replication is not finite")
            by_h[h] = {"coef": coef, "t_nw": t_nw, "t_1b": t_1b}
        out[scheme] = by_h
    return out


def _p_value(draws: np.ndarray, observed: float) -> float:
    return float(np.mean(np.abs(draws) >= abs(observed)))


def _size(draws: np.ndarray) -> float:
    return float(np.mean(np.abs(draws) > 1.96))


def _influence(y: np.ndarray, x: np.ndarray, dates: pd.DatetimeIndex, x_name: str) -> pd.DataFrame:
    """DFBETA = coef(B) minus the coefficient without that row. Same check as spec 002."""
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    beta = float(np.asarray(ols.params)[1])
    infl = ols.get_influence()
    hii = np.asarray(infl.hat_matrix_diag, dtype=float)
    cooks = np.asarray(infl.cooks_distance[0], dtype=float)
    manual = (np.asarray(ols.resid) ** 2) / (2.0 * float(ols.mse_resid)) * (hii / (1.0 - hii) ** 2)
    if not np.allclose(cooks, manual, rtol=1e-8, atol=1e-10):
        raise AssertionError("Cook's distance does not match the OLS formula")
    loo = np.empty(len(y))
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        fit = sm.OLS(y[mask], sm.add_constant(x[mask])).fit()
        loo[i] = beta - float(np.asarray(fit.params)[1])
    dfb = np.asarray(infl.dfbeta, dtype=float)[:, 1]
    if not np.allclose(dfb, loo, rtol=1e-8, atol=1e-10):
        raise AssertionError("DFBETA is not the leave-one-out change in coef(B)")
    frame = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates),
            x_name: np.asarray(x, dtype=float),
            "Y_t": np.asarray(y, dtype=float),
            "DFBETA": loo,
            "Cooks_D": cooks,
        }
    )
    frame = frame.reindex(frame["DFBETA"].abs().sort_values(ascending=False).index)
    return frame.reset_index(drop=True)


def _assert_references(rows: dict) -> None:
    saved_main = pd.read_csv(btz.RESULTS / "table2_main.csv")
    saved_inf = pd.read_csv(btz.ROOT / "results" / "002-influential-obs" / "table_robust.csv")
    main = saved_main[saved_main["sample"] == "main"].set_index("h")
    r2 = saved_main[saved_main["sample"] == "R2"].set_index("h")
    robust = saved_inf.set_index("variant")
    checks = {
        "R-base": main,
        "R-full": r2,
    }
    for name, ref in checks.items():
        for h in HORIZONS:
            got = rows[name]["by_h"][h]
            if abs(got["coef_B"] - float(ref.loc[h, "coef_B"])) > 1e-6:
                raise AssertionError(f"{name} h={h} coef does not reproduce the earlier estimate")
            if abs(got["t_hm1"] - float(ref.loc[h, "t_h_minus_1"])) > 1e-6:
                raise AssertionError(f"{name} h={h} NW t does not reproduce the earlier estimate")
    for name, variant, prefix in (
        ("R-G", "G full sample", "h"),
        ("R-G07", "G 1990-2007", "h"),
    ):
        for h in HORIZONS:
            got = rows[name]["by_h"][h]
            coef_col = f"{prefix}{h}_coef_B"
            t_col = f"{prefix}{h}_t_hm1"
            if abs(got["coef_B"] - float(robust.loc[variant, coef_col])) > 1e-6:
                raise AssertionError(f"{name} h={h} coef does not reproduce {variant}")
            if abs(got["t_hm1"] - float(robust.loc[variant, t_col])) > 1e-6:
                raise AssertionError(f"{name} h={h} NW t does not reproduce {variant}")
    for name, (coef_ref, t_ref) in REF_H3.items():
        got = rows[name]["by_h"][3]
        if round(got["coef_B"], 4) != coef_ref:
            raise AssertionError(f"{name} h=3 coef rounds to {round(got['coef_B'], 4)}, expected {coef_ref}")
        if round(got["t_hm1"], 2) != t_ref:
            raise AssertionError(f"{name} h=3 t rounds to {round(got['t_hm1'], 2)}, expected {t_ref}")


def _assert_bootstrap_centering(rows: dict, boots: dict) -> None:
    """The iid scheme imposes the null. Its mean t must sit near 0.

    The block scheme resamples consecutive pairs, so it keeps the sample
    correlation between η_t and ε_{t+1}. That correlation is the sample
    predictability. The block mean is reported and is not forced into the band.
    """
    for name in ROW_ORDER:
        for h in HORIZONS:
            mean_t = float(boots[name]["iid"][h]["t_nw"].mean())
            if abs(mean_t) > T_MEAN_BAND:
                raise AssertionError(
                    f"{name} h={h} iid: mean of t*_NW is {mean_t}, outside ±{T_MEAN_BAND}"
                )


def plot_bootstrap(boots: dict, rows: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    specs = (("R-base", axes[0]), ("R-G", axes[1]))
    colors = {"iid": "steelblue", "block": "darkorange"}
    for name, ax in specs:
        observed = rows[name]["by_h"][3]["t_hm1"]
        draws = []
        for scheme in ("iid", "block"):
            t = boots[name][scheme][3]["t_nw"]
            draws.append(t)
            ax.hist(
                t,
                bins=40,
                color=colors[scheme],
                alpha=0.55,
                label=f"{scheme}, mean {_fmt(float(t.mean()), 2)}",
            )
        span = np.concatenate(draws)
        xmax = max(4.0, abs(observed) + 0.6, float(np.quantile(np.abs(span), 0.999)) + 0.4)
        ax.set_xlim(-xmax, xmax)
        ax.axvline(observed, color="black", lw=1.2, label=f"observed t = {_fmt(observed, 2)}")
        ax.axvline(0, color="gray", lw=0.6)
        ax.set_title(f"{name}, h = 3")
        ax.set_xlabel("t*_NW, maxlags = h−1")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Replications")
    fig.suptitle("Bootstrap draws of the Newey–West t, 5,000 replications", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _flag_f1(t_1b: float, p_iid: float, p_block: float) -> bool:
    return abs(t_1b) > 1.96 and p_iid < 0.05 and p_block < 0.05


def _flag_f3(influence: pd.DataFrame, coef: float) -> bool:
    return bool((influence["DFBETA"].abs() > 0.5 * abs(coef)).any())


def write_summary(payload: dict) -> None:
    rows = payload["rows"]
    part_a = payload["part_a"]
    boots = payload["boots"]
    inf_g = payload["influence_g"]
    inf_base = payload["influence_base"]
    spots = payload["spots"]
    ar_vrp_1995 = payload["ar_vrp_1995"]
    lines: list[str] = []
    lines.append("# Spec 003 — Inference")
    lines.append("")
    lines.append("## Definitions")
    lines.append("")
    lines.append(
        "Spec 001 and spec 002 are imported and not modified. "
        "The monthly series are the spec 001 panel: `^GSPC`, implied variance from `data/vix_1990_data.csv`, "
        "and TB3MS lagged one month. VRP and VRP_fwd are in monthly percent squared. "
        "The dependent variable is specification (B): the h-month average excess return multiplied by 1200, in annualized percent. "
        "Horizons are h = 1, 3 and 6. "
        "Newey–West standard errors are `statsmodels` HAC with the Bartlett kernel and `use_correction=False`. "
        "t(h−1) uses maxlags = max(1, h−1). t(h) uses maxlags = h. t(2h) uses maxlags = 2h. R² is in percent."
    )
    lines.append("")
    lines.append(
        "Part A drops regression row t when any month in t, t+1, …, t+h falls in "
        "C = {2008-09, 2008-10, 2008-11, 2008-12, 2020-02, 2020-03, 2020-04}. "
        "The monthly series themselves are not deleted. A kept row still has its own VRP."
    )
    lines.append("")
    lines.append(
        "Hodrick (1992) estimator 1B uses the monthly excess return in annualized percent, "
        "e, and the null residual u = e − mean(e) on the row's calendar window. "
        "z_t = (1, x_t)'. For the sum of h monthly returns, "
        "w_t = u_{t+1} times the sum of z_{t}, z_{t−1}, …, z_{t−h+1}. "
        "The sum is over dates t with t−h+1 inside the window and t+1 inside the window. "
        "A Part A date is left out of that sum, because it is not part of the regression being tested. "
        "On every row below, that date set has the same length T as the regression. "
        "S and Z are the averages of w_t w_t' and z_t z_t' on that date set. "
        "V_sum = Z⁻¹ S Z⁻¹ / T, and V for the average-return coefficient is V_sum / h². "
        "The t-statistic divides coef(B) from the average regression by the square root of V[1, 1]."
    )
    lines.append("")
    lines.append(
        "The bootstrap imposes no predictability. μ is the mean of e on the row's calendar window, "
        "and the regressor is an AR(1) fit by OLS on that same window, including months that Part A drops from the regression. "
        "Residuals (ε_s, η_s) are drawn as pairs from the same month. "
        "x* starts at the sample's first x. e* on that first month is μ, because the AR(1) residual starts one month later; "
        "the first return is not inside any Y. "
        "Each later month is x*_t = c + ρ x*_{t−1} + η*_t and e*_t = μ + ε*_t. "
        "Y*(h) is the spec 001 forward average of e*, scaled as specification (B) because e* is already in annualized percent. "
        "One artificial sample is used for h = 1, 3 and 6. "
        "The iid scheme draws pairs independently. "
        "The block scheme is the Politis–Romano stationary bootstrap on the pairs, circular, with mean block length 6. "
        f"There are {N_REP:,} replications per row and scheme, from `numpy.random.default_rng({SEED})`, "
        "in the order R-base, R-full, R-D1, R-G, R-G07, and within each row the iid scheme then the block scheme. "
        "The null-mean of coef(B) in the table is the mean under the iid scheme. That is the Stambaugh bias for this design."
    )
    lines.append("")
    lines.append("## Part A — window-overlap exclusion")
    lines.append("")
    lines.append(
        "D1 is VRP on 1990–2025. D2 is VRP on 2010–2025. D3 is VRP_fwd on 1995–2025. "
        "t(h−1) uses maxlags = max(1, h−1). t(h) uses maxlags = h."
    )
    lines.append("")
    lines.append("| variant | h | coef(B) | t(h−1) | t(h) | R² | N | rows dropped |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name in PART_A_ORDER:
        for h in HORIZONS:
            res = part_a[name]["by_h"][h]
            lines.append(
                f"| {name} | {h} | {_fmt(res['coef_B'], 4)} | {_fmt(res['t_hm1'], 2)} | "
                f"{_fmt(res['t_h'], 2)} | {_fmt(res['R2'], 2)} | {res['N']} | {len(res['dropped'])} |"
            )
    lines.append("")
    for name in PART_A_ORDER:
        for h in HORIZONS:
            dropped = part_a[name]["by_h"][h]["dropped"]
            listed = ", ".join(_ym(t) for t in dropped) if len(dropped) else "none"
            lines.append(
                f"{name}, h = {h}, drops {len(dropped)} rows (N = {part_a[name]['by_h'][h]['N']}): {listed}."
            )
            lines.append("")
    lines.append(
        "On the 1990–2025 and 1995–2025 calendars, the h = 3 list is 2008-06 through 2008-12 and 2019-11 through 2020-04. "
        "D2 starts in 2010, so only the second block is in its sample."
    )
    lines.append("")
    lines.append("## Part B — variant G")
    lines.append("")
    lines.append(
        "G on 1995-01 through 2007-12 is the regression the HAR burn-in actually identifies. "
        "It is the same estimation sample as spec 002's row \"G 1990-2007\", which started the calendar in 1990 and then dropped months with no forecast. "
        "Columns match the spec 002 table."
    )
    lines.append("")
    lines.append("| sample | h | coef(B) | t(h−1) | t(h) | R² | N |")
    lines.append("|---|---|---|---|---|---|---|")
    for h in HORIZONS:
        res = rows["R-G07"]["by_h"][h]
        lines.append(
            f"| G 1995-01 → 2007-12 | {h} | {_fmt(res['coef_B'], 4)} | {_fmt(res['t_hm1'], 2)} | "
            f"{_fmt(res['t_h'], 2)} | {_fmt(res['R2'], 2)} | {res['N']} |"
        )
    lines.append("")
    lines.append(
        "Influence is the full-sample G regression, VRP_fwd on 1995–2025, at h = 3. "
        "DFBETA is coef(B) minus the coefficient without that row. The five largest |DFBETA| rows:"
    )
    lines.append("")
    lines.append("| date | VRP_fwd_t | Y_t | DFBETA | Cook's D |")
    lines.append("|---|---|---|---|---|")
    for _, rec in inf_g.head(5).iterrows():
        lines.append(
            f"| {_ym(rec['date'])} | {_fmt(rec['VRP_fwd_t'], 2)} | {_fmt(rec['Y_t'], 2)} | "
            f"{_fmt(rec['DFBETA'], 4)} | {_fmt(rec['Cooks_D'], 4)} |"
        )
    lines.append("")
    lines.append("VRP and VRP_fwd in monthly percent squared:")
    lines.append("")
    lines.append("| date | VRP | VRP_fwd |")
    lines.append("|---|---|---|")
    for stamp, rec in spots.iterrows():
        lines.append(f"| {_ym(stamp)} | {_fmt(rec['VRP'], 2)} | {_fmt(rec['VRP_fwd'], 2)} |")
    lines.append("")
    lines.append(
        f"AR(1) slope on 1995-01 through 2025-12: VRP {_fmt(ar_vrp_1995, 3)}, "
        f"VRP_fwd {_fmt(rows['R-G']['rho'], 3)}."
    )
    lines.append("")
    lines.append("## Inference")
    lines.append("")
    lines.append(
        "t_NW(h−1) uses maxlags = max(1, h−1). t_NW(2h) uses maxlags = 2h. "
        "t_1B is the Hodrick statistic. Bootstrap p-values are two-sided shares of replications at least as large as the observed statistic in absolute value. "
        "The null-mean of the coefficient is the iid scheme. R² is in percent."
    )
    lines.append("")
    lines.append(
        "| id | h | coef(B) | null-mean coef | t_NW(h−1) | t_NW(2h) | t_1B | "
        "p_NW iid | p_NW block | p_1B iid | p_1B block | R² | N |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ROW_ORDER:
        for h in HORIZONS:
            res = rows[name]["by_h"][h]
            iid = boots[name]["iid"][h]
            blk = boots[name]["block"][h]
            lines.append(
                f"| {name} | {h} | {_fmt(res['coef_B'], 4)} | {_fmt(float(iid['coef'].mean()), 4)} | "
                f"{_fmt(res['t_hm1'], 2)} | {_fmt(res['t_2h'], 2)} | {_fmt(res['t_1b'], 2)} | "
                f"{_fmt(_p_value(iid['t_nw'], res['t_hm1']), 4)} | "
                f"{_fmt(_p_value(blk['t_nw'], res['t_hm1']), 4)} | "
                f"{_fmt(_p_value(iid['t_1b'], res['t_1b']), 4)} | "
                f"{_fmt(_p_value(blk['t_1b'], res['t_1b']), 4)} | "
                f"{_fmt(res['R2'], 2)} | {res['N']} |"
            )
    lines.append("")
    lines.append(
        "Empirical size of the nominal 5% test: the share of replications with an absolute t-statistic above 1.96."
    )
    lines.append("")
    lines.append("| id | h | NW iid | 1B iid | NW block | 1B block |")
    lines.append("|---|---|---|---|---|---|")
    for name in ROW_ORDER:
        for h in HORIZONS:
            iid = boots[name]["iid"][h]
            blk = boots[name]["block"][h]
            lines.append(
                f"| {name} | {h} | {_fmt(100 * _size(iid['t_nw']), 2)}% | "
                f"{_fmt(100 * _size(iid['t_1b']), 2)}% | "
                f"{_fmt(100 * _size(blk['t_nw']), 2)}% | "
                f"{_fmt(100 * _size(blk['t_1b']), 2)}% |"
            )
    lines.append("")
    lines.append("Hodrick standard error at h = 1 divided by the White HC0 standard error:")
    lines.append("")
    lines.append("| id | ratio |")
    lines.append("|---|---|")
    for name in ROW_ORDER:
        lines.append(f"| {name} | {_fmt(rows[name]['by_h'][1]['hc0_ratio'], 3)} |")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("### 1. Earlier coefficients and Newey–West t-statistics")
    lines.append("")
    lines.append(
        "Pass, to 1e-6, against the saved spec 001 and spec 002 tables. "
        "R-base matches the spec 001 main sample at h = 1, 3 and 6. "
        "R-full matches spec 001 sample R2, which is spec 002 row 0. "
        "R-G matches spec 002 \"G full sample\". "
        "R-G07 matches spec 002 \"G 1990-2007\". "
        "The reference values at h = 3 are R-base 0.4196 / 3.46, R-full 0.0908 / 0.70, and R-G 0.3256 / 5.23. "
        "R-D1 is a new sample. Its coef(B) and t(h−1) are the Part A D1 estimates, from the same `fit_hac` call."
    )
    lines.append("")
    lines.append("### 2. Hodrick at h = 1 against White HC0")
    lines.append("")
    ratios = [(name, rows[name]["by_h"][1]["hc0_ratio"]) for name in ROW_ORDER]
    inside = [name for name, ratio in ratios if 0.9 <= ratio <= 1.1]
    outside = [(name, ratio) for name, ratio in ratios if not (0.9 <= ratio <= 1.1)]
    ratio_text = ", ".join(f"{name} {_fmt(ratio, 3)}" for name, ratio in ratios)
    if not outside:
        band = f"Pass. Every null-residual ratio is inside [0.9, 1.1]: {ratio_text}."
    else:
        outside_text = ", ".join(f"{name} {_fmt(ratio, 3)}" for name, ratio in outside)
        driver = ""
        if any(name == "R-G" for name, _ratio in outside):
            driver = " On R-G the largest such month is 2008-10: VRP_fwd is far below its mean and the next return is largely what that level predicts, so the OLS residual is small and the null residual is not."
        band = (
            f"The null-residual ratio is inside [0.9, 1.1] for {', '.join(inside)}. "
            f"It is outside that band for {outside_text}. All five ratios: {ratio_text}. "
            "The estimator was not replaced. "
            "u = e − mean(e) and the OLS residual differ when a month has a regressor far from its mean and a return that the fitted line absorbs."
            + driver
        )
    lines.append(
        band + " "
        "On every row, the same Hodrick sandwich with the OLS residual in place of the null residual reproduces the HC0 standard error to 1e-8. "
        "A simulated-data check does the same."
    )
    lines.append("")
    lines.append("### 3. Bootstrap centering and the AR(1) slopes")
    lines.append("")
    block_outside = []
    for name in ROW_ORDER:
        for h in HORIZONS:
            mean_t = float(boots[name]["block"][h]["t_nw"].mean())
            if abs(mean_t) > T_MEAN_BAND:
                block_outside.append((name, h, mean_t))
    if not block_outside:
        lines.append(
            f"Pass. For every row, every horizon and both schemes, the mean of t*_NW is inside ±{T_MEAN_BAND}."
        )
    else:
        listed = ", ".join(
            f"{name} h={h} ({_fmt(mean_t, 2)})" for name, h, mean_t in block_outside
        )
        lines.append(
            f"The iid scheme passes. For every row and horizon, the mean of t*_NW under iid draws is inside ±{T_MEAN_BAND}. "
            f"The block scheme does not: {listed}."
        )
        lines.append("")
        lines.append(
            "The block bootstrap draws consecutive pairs (ε_s, η_s). "
            "Inside a block, η at one month stays next to ε at the following month. "
            "That lead-lag correlation is the sample predictability, because ε is e − μ and not a residual from a regression on lagged x. "
            "With mean block length 6, about five sixths of the month-to-month steps keep the historical successor, so the block distribution sits near the sample coefficient rather than near 0. "
            "The iid scheme breaks that link and keeps only the contemporaneous Stambaugh correlation. "
            "The block p-values and the block rejection shares are computed exactly as specified. They are not null p-values or null sizes."
        )
    lines.append("")
    lines.append("| id | ρ | mean t*_NW, iid, h=1 / 3 / 6 | mean t*_NW, block, h=1 / 3 / 6 |")
    lines.append("|---|---|---|---|")
    for name in ROW_ORDER:
        iid_means = " / ".join(_fmt(float(boots[name]["iid"][h]["t_nw"].mean()), 3) for h in HORIZONS)
        blk_means = " / ".join(_fmt(float(boots[name]["block"][h]["t_nw"].mean()), 3) for h in HORIZONS)
        lines.append(f"| {name} | {_fmt(rows[name]['rho'], 3)} | {iid_means} | {blk_means} |")
    lines.append("")
    lines.append(
        "R-full and R-D1 share the 1990–2025 VRP series, so they share ρ. "
        "Part A changes the regression rows, not the AR(1)."
    )
    lines.append("")
    lines.append("### 4. Part A drops, and nothing that remains touches C")
    lines.append("")
    lines.append(
        "Pass. The dropped rows are listed in Part A. "
        "For every kept regression row, at every horizon of D1, D2 and D3, none of t, t+1, …, t+h is in C. "
        "The h = 3 list on D1 and on D3 is the list named in the spec."
    )
    lines.append("")
    lines.append("### 5. No look-ahead")
    lines.append("")
    lines.append(
        f"Pass, on the spec 001 rule, for {payload['n_checked']} regression rows: "
        "R-base, R-full, R-D1, R-G, R-G07, D2 and D3, at h = 1, 3 and 6. "
        "D1 is R-D1 and is not counted twice. "
        "For each row t, the last daily-return date inside RV_t and the last VIX date inside IV_t fall in month t, "
        "and the later of those dates is on or before the month-end label. "
        "The first excess return inside Y_t is the return of month t+1, and the window runs through month t+h, inside that row's sample. "
        "VRP_fwd is the spec 002 real-time HAR forecast. This check does not reopen that construction."
    )
    lines.append("")
    lines.append("## Flags at h = 3")
    lines.append("")
    lines.append(
        "The flags are independent. F1, F2 and F3 are properties of the named row. "
        "F4 is a property of D2, so it is the same fact in both columns. "
        "A flag uses the inequalities written in the spec: Hodrick |t| above 1.96, bootstrap p-values below 0.05, "
        "NW size above 8%, |DFBETA| above half of |coef(B)|, and the D2 t-statistic above 1.96. "
        "F1 and F2 use the block numbers as computed. Check 3 says the block distribution is not centered at 0, so those two flags' block legs are not null tests."
    )
    lines.append("")
    lines.append("| flag | R-base | R-G |")
    lines.append("|---|---|---|")
    flag_text = {}
    for name, influence in (("R-base", inf_base), ("R-G", inf_g)):
        res = rows[name]["by_h"][3]
        p_iid = _p_value(boots[name]["iid"][3]["t_nw"], res["t_hm1"])
        p_blk = _p_value(boots[name]["block"][3]["t_nw"], res["t_hm1"])
        size_iid = _size(boots[name]["iid"][3]["t_nw"])
        size_blk = _size(boots[name]["block"][3]["t_nw"])
        ratio = float(influence["DFBETA"].abs().max() / abs(res["coef_B"]))
        top_date = _ym(influence.iloc[0]["date"])
        f1 = _flag_f1(res["t_1b"], p_iid, p_blk)
        f3 = _flag_f3(influence, res["coef_B"])
        flag_text[name] = {
            "F1": (
                f"{'yes' if f1 else 'no'}. t_1B = {_fmt(res['t_1b'], 2)}, "
                f"p_NW iid = {_fmt(p_iid, 4)}, p_NW block = {_fmt(p_blk, 4)}"
            ),
            "F2": (
                f"iid {'yes' if size_iid > 0.08 else 'no'} ({_fmt(100 * size_iid, 2)}%); "
                f"block {'yes' if size_blk > 0.08 else 'no'} ({_fmt(100 * size_blk, 2)}%)"
            ),
            "F3": (
                f"{'yes' if f3 else 'no'}. Largest |DFBETA| / |coef(B)| = {_fmt(ratio, 3)} "
                f"at {top_date}"
            ),
        }
    d2 = part_a["D2"]["by_h"][3]
    f4 = d2["t_hm1"] > 1.96
    f4_text = f"{'yes' if f4 else 'no'}. D2 t(h−1) = {_fmt(d2['t_hm1'], 2)}"
    lines.append(f"| F1 survives conservative inference | {flag_text['R-base']['F1']} | {flag_text['R-G']['F1']} |")
    lines.append(f"| F2 NW is oversized | {flag_text['R-base']['F2']} | {flag_text['R-G']['F2']} |")
    lines.append(f"| F3 single-row influence | {flag_text['R-base']['F3']} | {flag_text['R-G']['F3']} |")
    lines.append(f"| F4 2020 explains the post-2010 failure | {f4_text} | {f4_text} |")
    lines.append("")
    if not f4:
        lines.append(
            f"F4 does not hold. On 2010–2025, after the overlap exclusion, the h = 3 Newey–West t-statistic is {_fmt(d2['t_hm1'], 2)}."
        )
        lines.append("")
    iid_only = []
    for name in ("R-base", "R-G"):
        res = rows[name]["by_h"][3]
        p_iid = _p_value(boots[name]["iid"][3]["t_nw"], res["t_hm1"])
        if abs(res["t_1b"]) > 1.96 and p_iid < 0.05:
            iid_only.append(name)
    if iid_only:
        lines.append(
            "On the iid scheme, the one centered at 0, "
            + " and ".join(iid_only)
            + " at h = 3 "
            + ("has" if len(iid_only) == 1 else "have")
            + " a Hodrick absolute t-statistic above 1.96 and a Newey–West bootstrap p-value below 0.05. "
            "F1 is no because the block p-value is above 0.05, and check 3 says that p-value is not a null p-value."
        )
        lines.append("")
    lines.append("## Comparison with BTZ")
    lines.append("")
    t_1b = rows["R-base"]["by_h"][3]["t_1b"]
    gap = t_1b - BTZ_H3_T
    lines.append(
        f"The Hodrick t-statistic for R-base at h = 3 is {_fmt(t_1b, 2)}. "
        f"BTZ Table 2 reports {BTZ_H3_T:.2f}. The gap is {gap:+.2f}."
    )
    lines.append("")
    if abs(gap) <= CLOSE_TO_BTZ:
        lines.append(
            f"That gap is within {CLOSE_TO_BTZ}, the distance fixed before the run as the reading of \"close\"."
        )
    else:
        lines.append(
            f"The gap is larger than {CLOSE_TO_BTZ}, the distance fixed before the run as the reading of \"close\". "
            "The Hodrick t-statistic is not close to 2.86. No other standard-error estimator was computed."
        )
    lines.append("")
    lines.append("## Deviations from this spec")
    lines.append("")
    deviations = []
    outside = [
        name for name in ROW_ORDER
        if not (0.9 <= rows[name]["by_h"][1]["hc0_ratio"] <= 1.1)
    ]
    if outside:
        deviations.append(
            "**Check 2's 10% band does not hold for every row.** "
            "The Hodrick formula uses the null residual. "
            f"That standard error is outside 10% of HC0 for {', '.join(outside)}. "
            "The code asserts a different identity: the same sandwich with the OLS residual equals the HC0 standard error. "
            "That identity holds. The null residual was kept."
        )
    block_outside = [
        f"{name} h={h}"
        for name in ROW_ORDER
        for h in HORIZONS
        if abs(float(boots[name]["block"][h]["t_nw"].mean())) > T_MEAN_BAND
    ]
    if block_outside:
        deviations.append(
            "**Check 3 does not hold for the block scheme.** "
            "The iid mean of t*_NW is inside ±0.15 for every row and horizon. "
            "The stationary bootstrap on the pairs keeps the sample lead-lag between η and the next ε, "
            "so its mean tracks the sample t-statistic. "
            f"Outside the band: {', '.join(block_outside)}. "
            "The resampling rule was not changed."
        )
    if not deviations:
        lines.append("None.")
        lines.append("")
    for i, text in enumerate(deviations, start=1):
        lines.append(f"{i}. {text}")
        lines.append("")
    lines.append(
        "The spec left four choices implicit. They are fixed in the definitions above, and they are not further deviations: "
        "Hodrick's Z and S use the dates on which w_t is defined, and a Part A date is removed from that sum as well as from the regression; "
        "the AR(1) and μ use every month of the calendar window; "
        "the first bootstrap return is μ; "
        "and each replication builds one monthly sample, from which all three horizons are formed. "
        "The null-mean of coef(B) stored in `table_inference.csv` is the iid mean."
    )
    lines.append("")
    (RESULTS / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _spots(vrp: pd.Series, vrp_fwd: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(["2008-10-31", "2020-03-31", "2025-04-30"])
    frame = pd.concat([vrp.rename("VRP"), vrp_fwd.rename("VRP_fwd")], axis=1)
    out = frame.loc[dates]
    if out.isna().any().any():
        raise AssertionError("A requested VRP month is missing")
    return out


def _write_tables(rows: dict, part_a: dict, boots: dict, influence_g: pd.DataFrame) -> None:
    inf_rows = []
    size_rows = []
    for name in ROW_ORDER:
        for h in HORIZONS:
            res = rows[name]["by_h"][h]
            iid = boots[name]["iid"][h]
            blk = boots[name]["block"][h]
            inf_rows.append(
                {
                    "id": name,
                    "h": h,
                    "coef_B": res["coef_B"],
                    "null_mean_coef": float(iid["coef"].mean()),
                    "t_NW_hm1": res["t_hm1"],
                    "t_NW_2h": res["t_2h"],
                    "t_1B": res["t_1b"],
                    "p_boot_NW_iid": _p_value(iid["t_nw"], res["t_hm1"]),
                    "p_boot_NW_block": _p_value(blk["t_nw"], res["t_hm1"]),
                    "p_boot_1B_iid": _p_value(iid["t_1b"], res["t_1b"]),
                    "p_boot_1B_block": _p_value(blk["t_1b"], res["t_1b"]),
                    "R2": res["R2"],
                    "N": res["N"],
                }
            )
            size_rows.append(
                {
                    "id": name,
                    "h": h,
                    "size_NW_iid": _size(iid["t_nw"]),
                    "size_1B_iid": _size(iid["t_1b"]),
                    "size_NW_block": _size(blk["t_nw"]),
                    "size_1B_block": _size(blk["t_1b"]),
                }
            )
    pd.DataFrame(inf_rows).to_csv(RESULTS / "table_inference.csv", index=False)
    pd.DataFrame(size_rows).to_csv(RESULTS / "table_size.csv", index=False)

    d_rows = []
    for name in PART_A_ORDER:
        for h in HORIZONS:
            res = part_a[name]["by_h"][h]
            d_rows.append(
                {
                    "variant": name,
                    "h": h,
                    "coef_B": res["coef_B"],
                    "t_hm1": res["t_hm1"],
                    "t_h": res["t_h"],
                    "R2": res["R2"],
                    "N": res["N"],
                    "n_dropped": int(len(res["dropped"])),
                    "dropped": ";".join(_ym(t) for t in res["dropped"]),
                }
            )
    pd.DataFrame(d_rows).to_csv(RESULTS / "table_D_fixed.csv", index=False)

    top = influence_g.head(5).copy()
    top["date"] = pd.to_datetime(top["date"]).dt.strftime("%Y-%m-%d")
    top.to_csv(RESULTS / "table_G_influence.csv", index=False)


def main() -> dict:
    _self_checks()
    loaded = _load()
    full = loaded["full"]
    vrp = loaded["vrp"]
    vrp_fwd = loaded["vrp_fwd"]
    crisis = crisis_months()
    if not set(crisis).issubset(set(pd.DatetimeIndex(full.index))):
        raise AssertionError("A crisis month is outside 1990–2025")

    samples = {
        "R-base": btz.slice_sample(full, "1990-01-31", "2007-12-31"),
        "R-full": full,
        "R-D1": full,
        "R-G": btz.slice_sample(full, "1995-01-31", "2025-12-31"),
        "R-G07": btz.slice_sample(full, "1995-01-31", "2007-12-31"),
        "D2": btz.slice_sample(full, "2010-01-31", "2025-12-31"),
        "D3": btz.slice_sample(full, "1995-01-31", "2025-12-31"),
    }
    regressors = {
        "R-base": vrp,
        "R-full": vrp,
        "R-D1": vrp,
        "R-G": vrp_fwd,
        "R-G07": vrp_fwd,
        "D2": vrp,
        "D3": vrp_fwd,
    }
    expected_n = {
        "R-base": 216,
        "R-full": 432,
        "R-D1": 432,
        "R-G": 372,
        "R-G07": 156,
        "D2": 192,
        "D3": 372,
    }
    for name, sample in samples.items():
        if len(sample) != expected_n[name]:
            raise AssertionError(f"{name} has {len(sample)} months, expected {expected_n[name]}")

    print("Fitting rows")
    fitted = {}
    for name in ("R-base", "R-full", "R-D1", "R-G", "R-G07", "D2", "D3"):
        fitted[name] = run_row(samples[name], regressors[name], crisis, name)
        print(f"  {name}  rho={fitted[name]['rho']:.3f}  h3 coef={fitted[name]['by_h'][3]['coef_B']:.4f}  "
              f"t={fitted[name]['by_h'][3]['t_hm1']:.2f}  t_1B={fitted[name]['by_h'][3]['t_1b']:.2f}")

    _assert_h3_drop_matches_spec(samples["R-D1"].index, fitted["R-D1"]["by_h"][3]["dropped"])
    _assert_h3_drop_matches_spec(samples["D3"].index, fitted["D3"]["by_h"][3]["dropped"])
    d2_h3 = [_ym(t) for t in fitted["D2"]["by_h"][3]["dropped"]]
    if d2_h3 != ["2019-11", "2019-12", "2020-01", "2020-02", "2020-03", "2020-04"]:
        raise AssertionError(f"D2 h=3 drops {d2_h3}")
    _assert_references(fitted)

    rows = {name: fitted[name] for name in ROW_ORDER}
    part_a = {"D1": fitted["R-D1"], "D2": fitted["D2"], "D3": fitted["D3"]}

    print(f"Bootstrap, {N_REP} replications, seed {SEED}")
    rng = np.random.default_rng(SEED)
    boots = {}
    for name in ROW_ORDER:
        boots[name] = _bootstrap_row(rng, rows[name])
        mean_t = float(boots[name]["iid"][3]["t_nw"].mean())
        size = _size(boots[name]["iid"][3]["t_nw"])
        print(f"  {name}  h3 iid mean t={mean_t:.3f}  size={size:.3f}")
    _assert_bootstrap_centering(rows, boots)

    g3 = rows["R-G"]["by_h"][3]
    base3 = rows["R-base"]["by_h"][3]
    y_g = forward_average(g3["e"], 3, g3["reg_pos"])[0]
    x_g = g3["x"][g3["reg_pos"]]
    influence_g = _influence(y_g, x_g, g3["index"], "VRP_fwd_t")
    y_b = forward_average(base3["e"], 3, base3["reg_pos"])[0]
    influence_base = _influence(y_b, base3["x"][base3["reg_pos"]], base3["index"], "VRP_t")
    g_slope = float(np.asarray(sm.OLS(y_g, sm.add_constant(x_g)).fit().params)[1])
    if abs(g_slope - g3["coef_B"]) > 1e-8:
        raise AssertionError("G influence sample is not the h=3 regression")

    n_checked = sum(fitted[name]["n_checked"] for name in fitted)
    spots = _spots(vrp, vrp_fwd)
    ar_vrp_1995 = btz.ar1(vrp.reindex(samples["R-G"].index))

    RESULTS.mkdir(parents=True, exist_ok=True)
    _write_tables(rows, part_a, boots, influence_g)
    plot_bootstrap(boots, rows, RESULTS / "fig_boot_h3.png")
    payload = {
        "rows": rows,
        "part_a": part_a,
        "boots": boots,
        "influence_g": influence_g,
        "influence_base": influence_base,
        "spots": spots,
        "ar_vrp_1995": ar_vrp_1995,
        "n_checked": n_checked,
    }
    write_summary(payload)
    print(f"Wrote {RESULTS}")
    return payload


if __name__ == "__main__":
    main()
