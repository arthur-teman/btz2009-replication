"""Spec 003a — wild bootstrap, matched-sample rows, horse race.

Imports the spec 001, 002 and 003 pipelines and does not modify them.
Adds to results/003-inference/.
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
from src import inference as inf3
from src import influence as inf

RESULTS = inf3.RESULTS
HORIZONS = inf3.HORIZONS
N_REP = inf3.N_REP
SEED = inf3.SEED
T_CRIT = 1.96
P_CRIT = 0.05
SECTION_MARKER = "## 003a — Wild bootstrap, matched samples, horse race"

ORIGINAL_ROWS = inf3.ROW_ORDER
# id, regressor, first month, last month
MATCHED = (
    ("V-95-07", "VRP", "1995-01-31", "2007-12-31"),
    ("G-95-07", "VRP_fwd", "1995-01-31", "2007-12-31"),
    ("V-08-25", "VRP", "2008-01-31", "2025-12-31"),
    ("G-08-25", "VRP_fwd", "2008-01-31", "2025-12-31"),
    ("V-10-25", "VRP", "2010-01-31", "2025-12-31"),
    ("G-10-25", "VRP_fwd", "2010-01-31", "2025-12-31"),
    ("V-10-19", "VRP", "2010-01-31", "2019-12-31"),
    ("G-10-19", "VRP_fwd", "2010-01-31", "2019-12-31"),
    ("R-D3", "VRP_fwd", "1995-01-31", "2025-12-31"),
)
NEW_ROWS = tuple(m[0] for m in MATCHED)
ALL_ROWS = ORIGINAL_ROWS + NEW_ROWS
MATCHED_MONTHS = {
    "V-95-07": 156, "G-95-07": 156, "V-08-25": 216, "G-08-25": 216,
    "V-10-25": 192, "G-10-25": 192, "V-10-19": 120, "G-10-19": 120, "R-D3": 372,
}

# Quick-check targets at h=3: coef(B), t NW(h−1), t 1B, R² %, N.
TARGETS_H3 = {
    "V-95-07": (0.373, 2.82, 1.81, 4.24, 153),
    "G-95-07": (0.147, 0.99, 0.87, 0.73, 153),
    "V-08-25": (0.067, 0.51, 0.92, 1.06, 213),
    "G-08-25": (0.393, 5.02, 2.75, 10.63, 213),
    "V-10-25": (-0.072, -1.59, -1.12, 1.58, 189),
    "G-10-25": (0.401, 3.01, 1.91, 5.52, 189),
    "V-10-19": (0.166, 1.67, 1.07, 1.65, 117),
    "G-10-19": (0.218, 1.51, 0.86, 2.01, 117),
}
COEF_TOL = 0.01
T_TOL = 0.02

# Spec 003 SUMMARY.md, iid scheme, h=3: p_NW, p_1B, size NW, size 1B.
REF_003_IID_H3 = {
    "R-base": (0.0020, 0.0230, 0.0788, 0.0486),
    "R-G": (0.0004, 0.0040, 0.1092, 0.0450),
}

HORSE_SAMPLES = (
    ("1995–2007", "1995-01-31", "2007-12-31"),
    ("2008–2025", "2008-01-31", "2025-12-31"),
    ("1995–2025", "1995-01-31", "2025-12-31"),
)
MR_SAMPLES = (
    ("1995–2007", "1995-01-31", "2007-12-31"),
    ("2008–2025", "2008-01-31", "2025-12-31"),
)
FIG_PERIODS = (
    ("1995–2007", "V-95-07", "G-95-07"),
    ("2008–2025", "V-08-25", "G-08-25"),
    ("2010–2025", "V-10-25", "G-10-25"),
    ("2010–2019", "V-10-19", "G-10-19"),
)

_fmt = inf3._fmt
_ym = inf3._ym
_p_value = inf3._p_value
_size = inf3._size


def hodrick_t(
    e: np.ndarray,
    x: np.ndarray,
    h: int,
    hod_pos: np.ndarray,
    coef: np.ndarray | float,
    u_next: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Hodrick (1992) 1B t-statistics with k regressors, z_t = (1, x_t')'.

    Univariate call, as in spec 003: e and x both (N,) or both (replications, N).
    Returns t and se of shape (replications,), equal to `inference.hodrick_t`.
    Multivariate call: x is (N, k) with e (N,), or (replications, N, k) with e
    (replications, N). coef is (k,) or (replications, k). Returns (replications, k).
    u_next replaces u_{t+1} on `hod_pos`; by default u = e − mean(e).
    """
    e = np.asarray(e, dtype=float)
    x = np.asarray(x, dtype=float)
    univariate = x.ndim == e.ndim
    if e.ndim == 1:
        e = e[None, :]
        x = x[None, ...]
    if univariate:
        x = x[:, :, None]
    if x.ndim != 3 or x.shape[:2] != e.shape:
        raise AssertionError("e and x shapes do not match")
    n_rep, _n, k = x.shape
    coef = np.asarray(coef, dtype=float)
    if coef.ndim == 0:
        coef = np.full((n_rep, k), float(coef))
    elif univariate and coef.ndim == 1:
        coef = coef[:, None]
    elif coef.ndim == 1:
        coef = np.broadcast_to(coef[None, :], (n_rep, k))
    if coef.shape != (n_rep, k):
        raise AssertionError(f"coef has shape {coef.shape}, expected {(n_rep, k)}")
    pos = np.asarray(hod_pos, dtype=int)
    if u_next is None:
        u = e - e.mean(axis=1, keepdims=True)
        u_next = u[:, pos + 1]
    else:
        u_next = np.atleast_2d(np.asarray(u_next, dtype=float))
        if u_next.shape != (n_rep, len(pos)):
            raise AssertionError("u_next is not aligned with hod_pos")
    rolled = inf3._rolling_sum(x, h)
    ones = np.ones((n_rep, len(pos), 1))
    z = np.concatenate([ones, x[:, pos, :]], axis=2)
    zsum = np.concatenate([h * ones, rolled[:, pos, :]], axis=2)
    w = u_next[:, :, None] * zsum
    t_obs = len(pos)
    Z = np.einsum("bti,btj->bij", z, z) / t_obs
    S = np.einsum("bti,btj->bij", w, w) / t_obs
    Zinv = np.linalg.inv(Z)
    v = np.matmul(np.matmul(Zinv, S), Zinv) / t_obs / float(h * h)
    se = np.sqrt(np.diagonal(v, axis1=1, axis2=2)[:, 1:])
    if np.any(~np.isfinite(se)) or np.any(se <= 0):
        raise AssertionError("Hodrick standard error is not finite and positive")
    t = coef / se
    if univariate:
        return t[:, 0], se[:, 0]
    return t, se


def ols_residual_hc0_gap(e: np.ndarray, x: np.ndarray, hod_pos: np.ndarray) -> float:
    """Largest |1B se / HC0 se − 1| at h=1 when the 1B sandwich uses the OLS residual.

    x is (N,) or (N, k). The regression is e_{t+1} on (1, x_t) over hod_pos.
    """
    e = np.asarray(e, dtype=float)
    x = np.asarray(x, dtype=float)
    pos = np.asarray(hod_pos, dtype=int)
    xm = x[:, None] if x.ndim == 1 else x
    ols = sm.OLS(e[pos + 1], sm.add_constant(xm[pos], has_constant="add")).fit(cov_type="HC0")
    resid = np.asarray(ols.resid, dtype=float)
    slopes = np.asarray(ols.params, dtype=float)[1:]
    _t, se = hodrick_t(e, xm, 1, pos, slopes, u_next=resid)
    ratio = se[0] / np.asarray(ols.bse, dtype=float)[1:]
    return float(np.max(np.abs(ratio - 1.0)))


def _self_checks() -> None:
    rng = np.random.default_rng(11)
    n = 150
    e = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = 0.5 * x1 + rng.normal(size=n)
    for h in HORIZONS:
        pos = np.arange(h - 1, n - 1)
        ref_t, ref_se = inf3.hodrick_t(e, x1, h, pos, 0.3)
        got_t, got_se = hodrick_t(e, x1, h, pos, 0.3)
        col_t, col_se = hodrick_t(e, x1[:, None], h, pos, np.array([0.3]))
        if not (np.allclose(got_t, ref_t, rtol=0, atol=1e-12) and np.allclose(got_se, ref_se, rtol=0, atol=1e-12)):
            raise AssertionError(f"h={h}: generalized hodrick_t changes the univariate output")
        if not (np.allclose(col_t[:, 0], ref_t, rtol=0, atol=1e-12) and np.allclose(col_se[:, 0], ref_se, rtol=0, atol=1e-12)):
            raise AssertionError(f"h={h}: one-column multivariate call differs from the univariate call")
        joint_t, _ = hodrick_t(e, np.column_stack([x1, x2]), h, pos, np.array([0.3, -0.1]))
        if joint_t.shape != (1, 2) or not np.isfinite(joint_t).all():
            raise AssertionError("Bivariate Hodrick t is not finite")
    pos1 = np.arange(0, n - 1)
    for xm in (x1, np.column_stack([x1, x2])):
        gap = ols_residual_hc0_gap(e, xm, pos1)
        if gap > 1e-6:
            raise AssertionError(f"1B with the OLS residual differs from HC0 by {gap}")
    w = rng.integers(0, 2, size=100_000) * 2 - 1
    if set(np.unique(w)) != {-1, 1} or abs(w.mean()) > 0.02:
        raise AssertionError("Rademacher draws are not ±1 with probability 1/2")


def simulate_wild(rng: np.random.Generator, e: np.ndarray, x: np.ndarray, n_rep: int) -> tuple[np.ndarray, np.ndarray]:
    """Spec 003 C3 artificial samples with (ε*_s, η*_s) = w_s (ε_s, η_s), w_s Rademacher."""
    e = np.asarray(e, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(e)
    mu = float(e.mean())
    eps = e - mu
    c, rho, eta = inf3.fit_ar1(x)
    m = n - 1
    if len(eta) != m:
        raise AssertionError("Residual pairs are not aligned")
    weight = rng.integers(0, 2, size=(n_rep, m)) * 2.0 - 1.0
    eta_s = weight * eta[None, :]
    eps_s = weight * eps[1:][None, :]
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


def _boot_stats(row: dict, e_star: np.ndarray, x_star: np.ndarray, label: str) -> dict:
    by_h = {}
    for h in HORIZONS:
        reg = row["by_h"][h]["reg_pos"]
        hod = row["by_h"][h]["hod_pos"]
        y_star = inf3.forward_average(e_star, h, reg)
        t_nw, coef = inf3.newey_west_t(y_star, x_star[:, reg], max(1, h - 1))
        t_1b, _se = hodrick_t(e_star, x_star, h, hod, coef)
        if not (np.isfinite(t_nw).all() and np.isfinite(t_1b).all() and np.isfinite(coef).all()):
            raise AssertionError(f"{row['name']} {label} h={h}: a replication is not finite")
        by_h[h] = {"coef": coef, "t_nw": t_nw, "t_1b": t_1b}
    return by_h


def replay_003_iid(rows: dict) -> dict:
    """The spec 003 iid draws for its five rows, from the spec 003 stream.

    Spec 003 drew iid then block per row from one generator. The block draws are
    taken from the generator and discarded so that each iid sample is the same.
    """
    rng = np.random.default_rng(SEED)
    out = {}
    for name in ORIGINAL_ROWS:
        e = rows[name]["by_h"][1]["e"]
        x = rows[name]["by_h"][1]["x"]
        e_star, x_star = inf3.simulate_null(rng, e, x, "iid", N_REP)
        out[name] = _boot_stats(rows[name], e_star, x_star, "iid")
        m = len(e) - 1
        rng.random((N_REP, m))
        rng.integers(0, m, size=(N_REP, m))
    for name, (p_nw, p_1b, s_nw, s_1b) in REF_003_IID_H3.items():
        got = out[name][3]
        res = rows[name]["by_h"][3]
        replay = (
            round(_p_value(got["t_nw"], res["t_hm1"]), 4),
            round(_p_value(got["t_1b"], res["t_1b"]), 4),
            round(_size(got["t_nw"]), 4),
            round(_size(got["t_1b"]), 4),
        )
        if replay != (p_nw, p_1b, s_nw, s_1b):
            raise AssertionError(f"{name}: replayed iid draws {replay} are not spec 003's {(p_nw, p_1b, s_nw, s_1b)}")
    return out


def run_bootstraps(rows: dict) -> dict:
    """iid for the new rows and wild for every row, from one default_rng(SEED)."""
    boots = {name: {} for name in ALL_ROWS}
    replay = replay_003_iid(rows)
    for name in ORIGINAL_ROWS:
        boots[name]["iid"] = replay[name]
    rng = np.random.default_rng(SEED)
    for name in ALL_ROWS:
        e = rows[name]["by_h"][1]["e"]
        x = rows[name]["by_h"][1]["x"]
        if name in NEW_ROWS:
            e_star, x_star = inf3.simulate_null(rng, e, x, "iid", N_REP)
            boots[name]["iid"] = _boot_stats(rows[name], e_star, x_star, "iid")
        e_star, x_star = simulate_wild(rng, e, x, N_REP)
        boots[name]["wild"] = _boot_stats(rows[name], e_star, x_star, "wild")
        print(
            f"  {name:8s} h3 mean t*_NW iid={boots[name]['iid'][3]['t_nw'].mean():.3f} "
            f"wild={boots[name]['wild'][3]['t_nw'].mean():.3f}"
        )
    for name in ALL_ROWS:
        for h in HORIZONS:
            mean_t = float(boots[name]["iid"][h]["t_nw"].mean())
            if abs(mean_t) > inf3.T_MEAN_BAND:
                raise AssertionError(f"{name} h={h} iid: mean of t*_NW is {mean_t}")
    return boots


def record(rows: dict, boots: dict, name: str, h: int) -> dict:
    res = rows[name]["by_h"][h]
    iid = boots[name]["iid"][h]
    wild = boots[name]["wild"][h]
    return {
        "id": name,
        "h": h,
        "coef_B": res["coef_B"],
        "null_mean_coef": float(iid["coef"].mean()),
        "t_NW_hm1": res["t_hm1"],
        "t_NW_2h": res["t_2h"],
        "t_1B": res["t_1b"],
        "p_boot_NW_iid": _p_value(iid["t_nw"], res["t_hm1"]),
        "p_boot_NW_wild": _p_value(wild["t_nw"], res["t_hm1"]),
        "p_boot_1B_iid": _p_value(iid["t_1b"], res["t_1b"]),
        "p_boot_1B_wild": _p_value(wild["t_1b"], res["t_1b"]),
        "R2": res["R2"],
        "N": res["N"],
        "size_NW_iid": _size(iid["t_nw"]),
        "size_1B_iid": _size(iid["t_1b"]),
        "size_NW_wild": _size(wild["t_nw"]),
        "size_1B_wild": _size(wild["t_1b"]),
        "mean_t_NW_iid": float(iid["t_nw"].mean()),
        "mean_t_NW_wild": float(wild["t_nw"].mean()),
        "wild_null_mean_coef": float(wild["coef"].mean()),
    }


def fit_rows(loaded: dict) -> tuple[dict, int]:
    full = loaded["full"]
    regressors = {"VRP": loaded["vrp"], "VRP_fwd": loaded["vrp_fwd"]}
    crisis = inf3.crisis_months()
    samples = {
        "R-base": (btz.slice_sample(full, "1990-01-31", "2007-12-31"), "VRP"),
        "R-full": (full, "VRP"),
        "R-D1": (full, "VRP"),
        "R-G": (btz.slice_sample(full, "1995-01-31", "2025-12-31"), "VRP_fwd"),
        "R-G07": (btz.slice_sample(full, "1995-01-31", "2007-12-31"), "VRP_fwd"),
    }
    for name, reg, start, end in MATCHED:
        samples[name] = (btz.slice_sample(full, start, end), reg)
        if len(samples[name][0]) != MATCHED_MONTHS[name]:
            raise AssertionError(f"{name} has {len(samples[name][0])} months")
    rows = {}
    n_checked = 0
    for name in ALL_ROWS:
        sample, reg = samples[name]
        # run_row applies the Part A exclusion to the names it knows; R-D3 is its D3.
        row = inf3.run_row(sample, regressors[reg], crisis, "D3" if name == "R-D3" else name)
        row["name"] = name
        rows[name] = row
        if name in NEW_ROWS:
            n_checked += sum(row["by_h"][h]["N"] for h in HORIZONS)
    inf3._assert_references(rows)
    d_fixed = pd.read_csv(RESULTS / "table_D_fixed.csv").set_index(["variant", "h"])
    for h in HORIZONS:
        got = rows["R-D3"]["by_h"][h]
        ref = d_fixed.loc[("D3", h)]
        if abs(got["coef_B"] - float(ref["coef_B"])) > 1e-6 or abs(got["t_hm1"] - float(ref["t_hm1"])) > 1e-6:
            raise AssertionError(f"R-D3 h={h} does not reproduce spec 003 D3")
        a = rows["G-95-07"]["by_h"][h]
        b = rows["R-G07"]["by_h"][h]
        if a["coef_B"] != b["coef_B"] or a["t_hm1"] != b["t_hm1"] or a["t_1b"] != b["t_1b"]:
            raise AssertionError(f"G-95-07 h={h} is not R-G07")
    for name in NEW_ROWS:
        for h in HORIZONS:
            r = rows[name]["by_h"][h]
            n = len(r["e"])
            dropped = r["dropped"]
            rule = pd.DatetimeIndex(samples[name][0].index[h - 1 : n - 1]).difference(dropped)
            if not np.array_equal(r["hod_pos"], samples[name][0].index.get_indexer(rule)):
                raise AssertionError(f"{name} h={h}: Hodrick positions differ from the spec 003 rule")
            # Spec 003a quick check: positions h−1 … n−2 of the subsample.
            if len(dropped) == 0 and not np.array_equal(r["hod_pos"], np.arange(h - 1, n - 1)):
                raise AssertionError(f"{name} h={h}: Hodrick positions are not h−1 … n−2")
    return rows, n_checked


def horse_race(loaded: dict) -> tuple[pd.DataFrame, float, int]:
    full = loaded["full"]
    out = []
    worst_gap = 0.0
    n_checked = 0
    for label, start, end in HORSE_SAMPLES:
        sample = btz.slice_sample(full, start, end)
        xs = pd.concat(
            [loaded["vrp"].reindex(sample.index).rename("VRP"), loaded["vrp_fwd"].reindex(sample.index).rename("VRP_fwd")],
            axis=1,
        )
        if xs.isna().any().any():
            raise AssertionError(f"Horse race {label}: a regressor is missing")
        corr = float(xs["VRP"].corr(xs["VRP_fwd"]))
        e = 1200.0 * sample["e"].to_numpy(dtype=float)
        X = xs.to_numpy(dtype=float)
        n = len(sample)
        for h in HORIZONS:
            y = 1200.0 * btz.forward_mean(sample["e"], h)
            frame = pd.concat([y.rename("y"), xs], axis=1).dropna()
            if not frame.index.equals(pd.DatetimeIndex(sample.index[: n - h])):
                raise AssertionError(f"Horse race {label} h={h}: rows are not the complete windows")
            for t in frame.index:
                inf.assert_row_timing(sample, t, h, "spec 003a horse race")
            n_checked += len(frame)
            model = sm.OLS(frame["y"], sm.add_constant(frame[["VRP", "VRP_fwd"]])).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": max(1, h - 1), "use_correction": False},
            )
            coef = model.params[["VRP", "VRP_fwd"]].to_numpy(dtype=float)
            hod_pos = np.arange(h - 1, n - 1)
            if len(hod_pos) != len(frame):
                raise AssertionError("Hodrick dates and regression rows differ")
            t_1b, _se = hodrick_t(e, X, h, hod_pos, coef)
            if h == 1:
                worst_gap = max(worst_gap, ols_residual_hc0_gap(e, X, hod_pos))
            out.append(
                {
                    "sample": label,
                    "h": h,
                    "b1_VRP": coef[0],
                    "t_NW_hm1_b1": float(model.tvalues["VRP"]),
                    "t_1B_b1": float(t_1b[0, 0]),
                    "b2_VRP_fwd": coef[1],
                    "t_NW_hm1_b2": float(model.tvalues["VRP_fwd"]),
                    "t_1B_b2": float(t_1b[0, 1]),
                    "R2": float(model.rsquared * 100.0),
                    "N": int(model.nobs),
                    "corr_VRP_VRP_fwd": corr,
                }
            )
    return pd.DataFrame(out), worst_gap, n_checked


def mean_reversion(loaded: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = loaded["full"]
    idx = pd.date_range("1995-01-31", "2025-12-31", freq="ME")
    mr = (loaded["vrp_fwd"] - loaded["vrp"]).reindex(idx)
    rv_hat = loaded["har"]["pred"].reindex(idx)
    direct = full["RV"].reindex(idx) - rv_hat
    if mr.isna().any() or not np.allclose(mr.to_numpy(), direct.to_numpy(), rtol=0, atol=1e-9):
        raise AssertionError("VRP_fwd − VRP is not RV_t − Ê_t[RV_{t+1}]")
    stats = []
    for label, start, end in MR_SAMPLES:
        s = mr.loc[pd.Timestamp(start) : pd.Timestamp(end)].to_numpy(dtype=float)
        stats.append(
            {
                "part": "stats",
                "sample": label,
                "N": len(s),
                "mean": float(s.mean()),
                "sd": float(s.std(ddof=1)),
                "p01": float(np.percentile(s, 1)),
                "p99": float(np.percentile(s, 99)),
            }
        )
    top_idx = mr.abs().sort_values(ascending=False).index[:10]
    top = pd.DataFrame(
        {
            "part": "top10",
            "rank": np.arange(1, 11),
            "date": [d.strftime("%Y-%m") for d in top_idx],
            "MR": mr.loc[top_idx].to_numpy(),
            "VRP": loaded["vrp"].loc[top_idx].to_numpy(),
            "VRP_fwd": loaded["vrp_fwd"].loc[top_idx].to_numpy(),
            "RV": full["RV"].loc[top_idx].to_numpy(),
            "RV_hat": rv_hat.loc[top_idx].to_numpy(),
        }
    )
    return pd.DataFrame(stats), top


def plot_subperiods(recs: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    pos = np.arange(len(FIG_PERIODS))
    width = 0.36
    v = [recs[(vid, 3)]["t_1B"] for _label, vid, _gid in FIG_PERIODS]
    g = [recs[(gid, 3)]["t_1B"] for _label, _vid, gid in FIG_PERIODS]
    ax.bar(pos - width / 2, v, width, color="steelblue", label="VRP")
    ax.bar(pos + width / 2, g, width, color="darkorange", label="VRP_fwd")
    for xpos, val in zip(np.concatenate([pos - width / 2, pos + width / 2]), v + g):
        ax.text(xpos, val + (0.08 if val >= 0 else -0.08), _fmt(val, 2), ha="center",
                va="bottom" if val >= 0 else "top", fontsize=8)
    ax.axhline(T_CRIT, color="black", lw=0.9, ls="--", label="±1.96")
    ax.axhline(-T_CRIT, color="black", lw=0.9, ls="--")
    ax.axhline(0, color="gray", lw=0.6)
    span = v + g + [T_CRIT, -T_CRIT]
    ax.set_ylim(min(span) - 0.4, max(span) + 0.4)
    ax.set_xticks(pos)
    ax.set_xticklabels([label for label, _v, _g in FIG_PERIODS])
    ax.set_ylabel("Hodrick 1B t-statistic, h = 3")
    ax.set_title("VRP and VRP_fwd by sub-period, h = 3")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_tables(recs: dict, horse: pd.DataFrame, mr_stats: pd.DataFrame, mr_top: pd.DataFrame) -> None:
    inf_cols = [
        "id", "h", "coef_B", "null_mean_coef", "t_NW_hm1", "t_NW_2h", "t_1B",
        "p_boot_NW_iid", "p_boot_NW_wild", "p_boot_1B_iid", "p_boot_1B_wild", "R2", "N",
    ]
    size_cols = ["id", "h", "size_NW_iid", "size_1B_iid", "size_NW_wild", "size_1B_wild"]
    ordered = [recs[(name, h)] for name in ALL_ROWS for h in HORIZONS]
    pd.DataFrame(ordered)[inf_cols].to_csv(RESULTS / "table_inference.csv", index=False)
    pd.DataFrame(ordered)[size_cols].to_csv(RESULTS / "table_size.csv", index=False)
    horse.to_csv(RESULTS / "table_horse_race.csv", index=False)
    pd.concat([mr_stats, mr_top], ignore_index=True, sort=False).to_csv(RESULTS / "table_mr.csv", index=False)


def _sig_t(rec: dict) -> bool:
    return abs(rec["t_1B"]) > T_CRIT


def _sig_p(rec: dict) -> bool:
    return rec["p_boot_NW_iid"] < P_CRIT


def _yn(flag: bool) -> str:
    return "yes" if flag else "no"


def _brief(rec: dict) -> str:
    return (
        f"t_1B = {_fmt(rec['t_1B'], 2)}, p_NW iid = {_fmt(rec['p_boot_NW_iid'], 4)}, "
        f"p_NW wild = {_fmt(rec['p_boot_NW_wild'], 4)}"
    )


def summary_section(payload: dict) -> list[str]:
    recs = payload["recs"]
    rows = payload["rows"]
    horse = payload["horse"]
    mr_stats = payload["mr_stats"]
    mr_top = payload["mr_top"]
    L: list[str] = []
    L.append(SECTION_MARKER)
    L.append("")
    L.append(
        "This section is written by `src/inference_003a.py`. It imports `src/inference.py` and the spec 001 and spec 002 modules and does not modify them. "
        "Everything above this section is the spec 003 run as committed. "
        "`table_inference.csv` and `table_size.csv` now come from this section: the block columns are removed, wild columns replace them, and the new rows are appended."
    )
    L.append("")
    L.append("### Wild bootstrap")
    L.append("")
    L.append(
        "Each replication draws w_s, ±1 with probability 1/2, for every date s of the AR(1) innovations. "
        "ε*_s = w_s ε_s and η*_s = w_s η_s, so both members of a pair get the same sign, and the dates keep their order. "
        "x* and e* are then built exactly as in spec 003 C3. "
        "The weight keeps each month's variance and the contemporaneous ε–η product, and its independence across months removes every lead-lag between η_s and ε_{s+1}."
    )
    L.append("")
    L.append(
        f"There are {N_REP:,} replications per row and scheme. "
        "For the five spec 003 rows, the iid draws are the spec 003 draws: the spec 003 generator is replayed, and its block draws are taken and discarded. "
        "The code checks the replay against the spec 003 h = 3 values for R-base and R-G. "
        f"All wild draws, and the iid draws for the new rows, come from a second `numpy.random.default_rng({SEED})`, "
        "in the order " + ", ".join(ALL_ROWS) + "; within a new row, iid then wild."
    )
    L.append("")
    L.append("### Unit tests")
    L.append("")
    L.append(
        "**Check 2, replaced.** At h = 1, the 1B sandwich with the OLS residual in place of u_{t+1} equals the White HC0 standard error. "
        f"Pass, to 1e-6. The largest relative gap is {payload['hc0_gap_rows']:.1e} over the {len(ALL_ROWS)} rows, "
        f"{payload['hc0_gap_horse']:.1e} over the three horse-race samples (both slopes), and below 1e-6 on simulated data with one and with two regressors."
    )
    L.append("")
    L.append(
        "**Generalized Hodrick.** `hodrick_t` in `src/inference_003a.py` takes k regressors, with z_t = (1, x_t')'. "
        "With the second regressor removed, its output equals `inference.hodrick_t` to 1e-12: on simulated data at h = 1, 3 and 6, "
        "and on every row of this section, where the matched-sample and bootstrap t-statistics are computed with the new function."
    )
    L.append("")
    L.append(
        "For information, the null-residual 1B / HC0 ratio at h = 1. It is not expected to be 1, because 1B uses the null residual by construction."
    )
    L.append("")
    L.append("| id | ratio |")
    L.append("|---|---|")
    for name in ALL_ROWS:
        L.append(f"| {name} | {_fmt(rows[name]['by_h'][1]['hc0_ratio'], 3)} |")
    L.append("")
    L.append("### Inference, all rows")
    L.append("")
    L.append(
        "Columns as in spec 003, with wild in place of block. The null-mean of coef(B) is the iid mean. R² is in percent."
    )
    L.append("")
    L.append("| id | h | coef(B) | null-mean coef | t_NW(h−1) | t_NW(2h) | t_1B | p_NW iid | p_NW wild | p_1B iid | p_1B wild | R² | N |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ALL_ROWS:
        for h in HORIZONS:
            r = recs[(name, h)]
            L.append(
                f"| {name} | {h} | {_fmt(r['coef_B'], 4)} | {_fmt(r['null_mean_coef'], 4)} | {_fmt(r['t_NW_hm1'], 2)} | "
                f"{_fmt(r['t_NW_2h'], 2)} | {_fmt(r['t_1B'], 2)} | {_fmt(r['p_boot_NW_iid'], 4)} | {_fmt(r['p_boot_NW_wild'], 4)} | "
                f"{_fmt(r['p_boot_1B_iid'], 4)} | {_fmt(r['p_boot_1B_wild'], 4)} | {_fmt(r['R2'], 2)} | {r['N']} |"
            )
    L.append("")
    L.append("Empirical size of the nominal 5% tests, and the mean of t*_NW under each scheme:")
    L.append("")
    L.append("| id | h | NW iid | 1B iid | NW wild | 1B wild | mean t*_NW iid | mean t*_NW wild |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name in ALL_ROWS:
        for h in HORIZONS:
            r = recs[(name, h)]
            L.append(
                f"| {name} | {h} | {_fmt(100 * r['size_NW_iid'], 2)}% | {_fmt(100 * r['size_1B_iid'], 2)}% | "
                f"{_fmt(100 * r['size_NW_wild'], 2)}% | {_fmt(100 * r['size_1B_wild'], 2)}% | "
                f"{_fmt(r['mean_t_NW_iid'], 3)} | {_fmt(r['mean_t_NW_wild'], 3)} |"
            )
    L.append("")
    wild_out = [
        f"{name} h={h} ({_fmt(recs[(name, h)]['mean_t_NW_wild'], 2)})"
        for name in ALL_ROWS for h in HORIZONS
        if abs(recs[(name, h)]["mean_t_NW_wild"]) > inf3.T_MEAN_BAND
    ]
    if wild_out:
        L.append(
            f"Spec 003 check 3, applied to the wild scheme: the mean of t*_NW is outside ±{inf3.T_MEAN_BAND} for " + ", ".join(wild_out) + ". "
            f"The iid mean is inside ±{inf3.T_MEAN_BAND} for every row and horizon."
        )
    else:
        L.append(
            f"Spec 003 check 3 now passes for both schemes: for every row and horizon, the mean of t*_NW is inside ±{inf3.T_MEAN_BAND} under iid and under wild draws."
        )
    L.append("")
    L.append("### F1 re-evaluated, h = 3")
    L.append("")
    L.append("F1: Hodrick |t| > 1.96, and the bootstrap p-value of the NW t below 0.05 under iid and under wild draws.")
    L.append("")
    L.append("| row | F1 | t_1B | p_NW iid | p_NW wild |")
    L.append("|---|---|---|---|---|")
    for name in ("R-base", "R-G"):
        r = recs[(name, 3)]
        f1 = _sig_t(r) and r["p_boot_NW_iid"] < P_CRIT and r["p_boot_NW_wild"] < P_CRIT
        L.append(
            f"| {name} | {_yn(f1)} | {_fmt(r['t_1B'], 2)} | {_fmt(r['p_boot_NW_iid'], 4)} | {_fmt(r['p_boot_NW_wild'], 4)} |"
        )
    L.append("")
    L.append("### Matched samples against the quick-check targets, h = 3")
    L.append("")
    L.append(
        f"Tolerance: ±{COEF_TOL} on coef(B), ±{T_TOL} on each t. The Hodrick positions are spec 003's `hod_pos` rule, "
        "dates t with t−h+1 and t+1 inside the subsample. Without a Part A exclusion this is positions h−1 … n−2, the positions the quick check used, so there is no difference to report; the code asserts it."
    )
    L.append("")
    L.append("| id | coef(B) | target | t NW(h−1) | target | t 1B | target | R² % | target | N | target | within tolerance |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    misses = []
    for name, (tc, tt, tb, tr, tn) in TARGETS_H3.items():
        r = recs[(name, 3)]
        ok = (
            abs(r["coef_B"] - tc) <= COEF_TOL
            and abs(r["t_NW_hm1"] - tt) <= T_TOL
            and abs(r["t_1B"] - tb) <= T_TOL
            and r["N"] == tn
        )
        if not ok:
            misses.append(name)
        L.append(
            f"| {name} | {_fmt(r['coef_B'], 3)} | {_fmt(tc, 3)} | {_fmt(r['t_NW_hm1'], 2)} | {_fmt(tt, 2)} | "
            f"{_fmt(r['t_1B'], 2)} | {_fmt(tb, 2)} | {_fmt(r['R2'], 2)} | {_fmt(tr, 2)} | {r['N']} | {tn} | {_yn(ok)} |"
        )
    L.append("")
    L.append(
        "All eight rows reproduce the targets." if not misses
        else "Rows outside tolerance: " + ", ".join(misses) + "."
    )
    L.append("")
    L.append("G-95-07 is R-G07, and R-D3 reproduces spec 003 D3 to 1e-6. Both are asserted.")
    L.append("")
    L.append("![Hodrick t at h = 3 by sub-period](fig_subperiod_h3.png)")
    L.append("")
    L.append("### Horse race")
    L.append("")
    L.append(
        "Y_t(h) = a + b1·VRP_t + b2·VRP_fwd_t + u. NW uses maxlags = max(1, h−1). "
        "Hodrick 1B uses the same formula with z_t = (1, VRP_t, VRP_fwd_t)' and the null residual. "
        "The correlation is between the monthly VRP and VRP_fwd series over the sample's calendar months. No bootstrap."
    )
    L.append("")
    L.append("| sample | h | b1 (VRP) | t NW | t 1B | b2 (VRP_fwd) | t NW | t 1B | R² | N | corr |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in horse.iterrows():
        L.append(
            f"| {r['sample']} | {r['h']} | {_fmt(r['b1_VRP'], 4)} | {_fmt(r['t_NW_hm1_b1'], 2)} | {_fmt(r['t_1B_b1'], 2)} | "
            f"{_fmt(r['b2_VRP_fwd'], 4)} | {_fmt(r['t_NW_hm1_b2'], 2)} | {_fmt(r['t_1B_b2'], 2)} | "
            f"{_fmt(r['R2'], 2)} | {int(r['N'])} | {_fmt(r['corr_VRP_VRP_fwd'], 3)} |"
        )
    L.append("")
    L.append("### Decomposition of the difference")
    L.append("")
    L.append(
        "MR_t = VRP_fwd_t − VRP_t = RV_t − Ê_t[RV_{t+1}], where Ê is the spec 002 real-time HAR forecast. "
        "The identity is asserted to 1e-9. Monthly percent squared. Percentiles are linear interpolation."
    )
    L.append("")
    L.append("| sample | N | mean | sd | 1st pct | 99th pct |")
    L.append("|---|---|---|---|---|---|")
    for _, r in mr_stats.iterrows():
        L.append(
            f"| {r['sample']} | {int(r['N'])} | {_fmt(r['mean'], 2)} | {_fmt(r['sd'], 2)} | {_fmt(r['p01'], 2)} | {_fmt(r['p99'], 2)} |"
        )
    L.append("")
    L.append("The 10 months with the largest |MR_t|, 1995–2025:")
    L.append("")
    L.append("| rank | month | MR | VRP | VRP_fwd | RV | Ê[RV] |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in mr_top.iterrows():
        L.append(
            f"| {int(r['rank'])} | {r['date']} | {_fmt(r['MR'], 2)} | {_fmt(r['VRP'], 2)} | {_fmt(r['VRP_fwd'], 2)} | "
            f"{_fmt(r['RV'], 2)} | {_fmt(r['RV_hat'], 2)} |"
        )
    L.append("")
    L.append("### Flags at h = 3")
    L.append("")
    L.append(
        "Independent flags, with the inequalities written in the spec. \"iid p\" and \"wild p\" are the bootstrap p-values of the NW t(h−1), as in F1. "
        "H1 is read literally: the VRP leg on 1995–2007 needs either condition, the VRP_fwd leg on 2008–2025 needs both, and each reverse pair needs neither."
    )
    L.append("")
    v95, g08 = recs[("V-95-07", 3)], recs[("G-08-25", 3)]
    g95, v08 = recs[("G-95-07", 3)], recs[("V-08-25", 3)]
    h1_parts = (
        _sig_t(v95) or _sig_p(v95),
        _sig_t(g08) and _sig_p(g08),
        not (_sig_t(g95) or _sig_p(g95)),
        not (_sig_t(v08) or _sig_p(v08)),
    )
    h1 = all(h1_parts)
    g1025 = recs[("G-10-25", 3)]
    h2 = _sig_t(g1025) and g1025["p_boot_NW_iid"] < P_CRIT and g1025["p_boot_NW_wild"] < P_CRIT
    v1019, g1019 = recs[("V-10-19", 3)], recs[("G-10-19", 3)]
    h3 = not _sig_t(v1019) and not _sig_t(g1019)
    hr = horse[(horse["sample"] == "1995–2025") & (horse["h"] == 3)].iloc[0]
    h4 = abs(hr["t_1B_b2"]) > T_CRIT and abs(hr["t_1B_b1"]) <= T_CRIT
    L.append("| flag | holds | numbers |")
    L.append("|---|---|---|")
    L.append(
        f"| H1 sub-period split | {_yn(h1)} | "
        f"V-95-07 {_brief(v95)} (leg {_yn(h1_parts[0])}); G-08-25 {_brief(g08)} (leg {_yn(h1_parts[1])}); "
        f"G-95-07 {_brief(g95)} (neither: {_yn(h1_parts[2])}); V-08-25 {_brief(v08)} (neither: {_yn(h1_parts[3])}) |"
    )
    L.append(f"| H2 G holds post-2010 | {_yn(h2)} | G-10-25 {_brief(g1025)} |")
    L.append(f"| H3 calm decade | {_yn(h3)} | V-10-19 t_1B = {_fmt(v1019['t_1B'], 2)}; G-10-19 t_1B = {_fmt(g1019['t_1B'], 2)} |")
    L.append(
        f"| H4 horse race | {_yn(h4)} | 1995–2025: b2 t_1B = {_fmt(hr['t_1B_b2'], 2)}, b1 t_1B = {_fmt(hr['t_1B_b1'], 2)} |"
    )
    L.append("")
    L.append(
        f"No look-ahead: the spec 001 rule passes on the {payload['n_checked']} new regression rows (matched samples and horse race, h = 1, 3 and 6)."
    )
    L.append("")
    L.append("### Deviations")
    L.append("")
    devs = [
        "`hodrick_t` is generalized in a new function in `src/inference_003a.py`, not edited inside `src/inference.py`, because the spec 003 pipeline is not to be changed. "
        "The univariate output is identical, which is asserted.",
        "The iid columns of the five spec 003 rows are the spec 003 draws, replayed. They are not new draws. Only the wild draws, and the iid draws of the new rows, come from the 003a generator.",
        "`table_mr.csv` holds both parts of the decomposition in one file: the `part` column is `stats` for the sample moments and `top10` for the largest months.",
        "The spec 003 section above still shows the block columns and the old check 2 and check 3 text. It is the record of that run. The CSV tables and this section supersede it.",
    ]
    if misses:
        devs.append(
            "Matched-sample targets not reproduced within tolerance for " + ", ".join(misses) + ". The numbers are reported as computed."
        )
    for i, text in enumerate(devs, start=1):
        L.append(f"{i}. {text}")
    L.append("")
    return L


def write_summary(lines: list[str]) -> None:
    path = RESULTS / "SUMMARY.md"
    text = path.read_text(encoding="utf-8")
    cut = text.find(SECTION_MARKER)
    if cut >= 0:
        text = text[:cut]
    text = text.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> dict:
    _self_checks()
    inf3._self_checks()
    loaded = inf3._load()

    print("Fitting rows")
    rows, n_checked = fit_rows(loaded)
    for name in ALL_ROWS:
        for h in HORIZONS:
            r = rows[name]["by_h"][h]
            t_new, _ = hodrick_t(r["e"], r["x"], h, r["hod_pos"], r["coef_B"])
            if abs(float(t_new[0]) - r["t_1b"]) > 1e-12:
                raise AssertionError(f"{name} h={h}: generalized Hodrick differs from spec 003")
        r3 = rows[name]["by_h"][3]
        print(f"  {name:8s} h3 coef={r3['coef_B']:.4f} t={r3['t_hm1']:.2f} t_1B={r3['t_1b']:.2f} N={r3['N']}")
    hc0_gap_rows = 0.0
    for name in ALL_ROWS:
        r = rows[name]["by_h"][1]
        hc0_gap_rows = max(hc0_gap_rows, ols_residual_hc0_gap(r["e"], r["x"], r["hod_pos"]))
    if hc0_gap_rows > 1e-6:
        raise AssertionError(f"1B with the OLS residual differs from HC0 by {hc0_gap_rows}")

    print(f"Bootstrap, {N_REP} replications, seed {SEED}")
    boots = run_bootstraps(rows)
    recs = {(name, h): record(rows, boots, name, h) for name in ALL_ROWS for h in HORIZONS}

    horse, hc0_gap_horse, n_horse = horse_race(loaded)
    if hc0_gap_horse > 1e-6:
        raise AssertionError(f"Bivariate 1B with the OLS residual differs from HC0 by {hc0_gap_horse}")
    mr_stats, mr_top = mean_reversion(loaded)

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_tables(recs, horse, mr_stats, mr_top)
    plot_subperiods(recs, RESULTS / "fig_subperiod_h3.png")
    payload = {
        "rows": rows,
        "boots": boots,
        "recs": recs,
        "horse": horse,
        "mr_stats": mr_stats,
        "mr_top": mr_top,
        "hc0_gap_rows": hc0_gap_rows,
        "hc0_gap_horse": hc0_gap_horse,
        "n_checked": n_checked + n_horse,
    }
    write_summary(summary_section(payload))
    print(f"Wrote {RESULTS}")
    return payload


if __name__ == "__main__":
    main()
