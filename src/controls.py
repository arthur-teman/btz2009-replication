"""Spec 004 — BTZ Table 4 with real-time controls, for VRP and VRP_fwd.

Imports the spec 001, 002, 003 and 003a pipelines and does not modify them.
Writes results/004-controls/.
"""

from __future__ import annotations

import io
import sys
import urllib.request
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
from src import inference_003a as inf3a
from src import influence as inf

RESULTS = btz.ROOT / "results" / "004-controls"
DATA = btz.DATA
SHILLER_PATH = DATA / "shiller_2026.csv"
# The Yale URL (econ.yale.edu/~shiller/data/ie_data.xls) still serves a file
# that ends in 2023-09. The maintained file is linked from shillerdata.com.
SHILLER_URL = (
    "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/"
    "1449043e-a1ed-4e2e-8e9c-7b9a2a064493/ie_data.xls?ver=1791297298731"
)
SHILLER_DOWNLOAD_DATE = "2026-10-21"

H = 3
T_CRIT = 1.96
K2_GAP = 2.0
CONTROLS = ("log_PE", "DFSP", "TMSP", "RREL", "CAY")
TIMINGS = ("RT", "NB")
REGRESSORS = ("VRP", "VRP_fwd") + CONTROLS
CAY_COL = "cay=c-0.218 a-0.801y+0.441"

# block, VRP measure, first month, last month
BLOCKS = (
    ("T4-base", "VRP", "1990-01-31", "2007-12-31"),
    ("T4-95", "VRP", "1995-01-31", "2007-12-31"),
    ("T4-95", "VRP_fwd", "1995-01-31", "2007-12-31"),
    ("T4-post", "VRP_fwd", "2008-01-31", "2025-12-31"),
    ("T4-full", "VRP_fwd", "1995-01-31", "2025-12-31"),
)
WITH_CAY = "with CAY"
NO_CAY = "without CAY"
# The spec fixes the end of every CAY version at 2019-12.
CAY_WINDOW_END = pd.Timestamp("2019-12-31")

# Spec 003/003a rows whose sample is the block's sample, for row (a), h = 3.
REF_003 = {
    ("T4-base", "VRP", WITH_CAY): "R-base",
    ("T4-95", "VRP", WITH_CAY): "V-95-07",
    ("T4-95", "VRP_fwd", WITH_CAY): "G-95-07",
    ("T4-post", "VRP_fwd", NO_CAY): "G-08-25",
    ("T4-full", "VRP_fwd", NO_CAY): "R-G",
}
REF_001_H3 = (0.4196, 3.46)

# BTZ Table 4 targets, h = 3: coef VRP, t VRP, coef control, t control, R².
# Arthur fills these from the PDF. None means not filled.
BTZ_TARGETS = {c: (None, None, None, None, None) for c in CONTROLS}

_fmt = inf3._fmt
_ym = inf3._ym


def _yn(flag: bool) -> str:
    return "yes" if flag else "no"


def _f(value, digits: int) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return _fmt(float(value), digits)


def _mcode(index: pd.DatetimeIndex) -> np.ndarray:
    """Month number year*12 + month − 1, so that t − 3 is mcode − 3."""
    index = pd.DatetimeIndex(index)
    return (index.year * 12 + index.month - 1).to_numpy(dtype=np.int64)


def _month_end(index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index).to_period("M").to_timestamp("M")


# ── Shiller ──────────────────────────────────────────────────────────────────


def parse_shiller_date(value: float) -> pd.Timestamp:
    """Decimal year.month. October is .1, so the month is read on two decimals."""
    year, month = f"{float(value):.2f}".split(".")
    return pd.Timestamp(year=int(year), month=int(month), day=1) + pd.offsets.MonthEnd(0)


def download_shiller(path: Path = SHILLER_PATH) -> None:
    """Download ie_data.xls once and freeze date, P, E. Never overwrites."""
    if path.exists():
        return
    req = urllib.request.Request(SHILLER_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    sheet = pd.read_excel(io.BytesIO(raw), sheet_name="Data", header=None)
    header = sheet.index[sheet[0].astype(str).str.strip() == "Date"]
    if len(header) != 1:
        raise RuntimeError("ie_data.xls: the Date header row is not unique")
    body = sheet.loc[header[0] + 1 :, [0, 1, 3]]
    body.columns = ["date", "P", "E"]
    body = body[pd.to_numeric(body["date"], errors="coerce").notna()].copy()
    body["date"] = [parse_shiller_date(v) for v in body["date"]]
    body["P"] = pd.to_numeric(body["P"], errors="coerce")
    body["E"] = pd.to_numeric(body["E"], errors="coerce")
    last_e = body["E"].last_valid_index()
    body = body.loc[:last_e]
    if body[["P", "E"]].isna().any().any():
        raise RuntimeError("ie_data.xls has an empty P or E before the last earnings row")
    path.parent.mkdir(parents=True, exist_ok=True)
    body.to_csv(path, index=False, date_format="%Y-%m-%d")


def load_shiller(path: Path = SHILLER_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index = _month_end(df.index)
    return df[["P", "E"]].astype(float)


def check_shiller(shiller: pd.DataFrame) -> dict:
    """Check 1."""
    idx = shiller.index
    if idx.has_duplicates:
        raise AssertionError("Shiller file has duplicate dates")
    for year in range(1990, 2026):
        months = sorted(idx[idx.year == year].month)
        if months != list(range(1, 13)):
            raise AssertionError(f"Shiller {year} has months {months}")
    complete_years = [y for y in range(idx.year.min(), 2026) if (idx.year == y).sum() == 12]
    no_october = [y for y in range(idx.year.min(), 2026) if not ((idx.year == y) & (idx.month == 10)).any()]
    if no_october:
        raise AssertionError(f"No October in {no_october}")
    full_cal = pd.date_range(idx.min(), idx.max(), freq="ME")
    if not idx.equals(full_cal):
        raise AssertionError("Shiller months are not a complete calendar")
    for raw, expect in ((1990.1, "1990-10"), (1990.01, "1990-01"), (1990.11, "1990-11"), (1990.12, "1990-12")):
        if _ym(parse_shiller_date(raw)) != expect:
            raise AssertionError(f"{raw} parses to {_ym(parse_shiller_date(raw))}, expected {expect}")
    show = shiller.loc[pd.to_datetime(["1990-09-30", "1990-10-31", "1990-11-30"])]
    print("Shiller rows 1990-09, 1990-10, 1990-11:")
    print(show.to_string())
    old = pd.read_csv(DATA / "shiller_data.csv", parse_dates=["date"]).set_index("date")
    old.index = _month_end(old.index)
    common = old.index.intersection(shiller.index)
    return {
        "first": _ym(idx.min()),
        "last": _ym(idx.max()),
        "rows": len(shiller),
        "complete_years": (min(complete_years), max(complete_years), len(complete_years)),
        "show": show,
        "old_overlap": (_ym(common.min()), _ym(common.max()), len(common)),
        "old_max_dP": float((old.loc[common, "P"] - shiller.loc[common, "P"]).abs().max()),
        "old_max_dE": float((old.loc[common, "E"] - shiller.loc[common, "E"]).abs().max()),
    }


# ── Controls ─────────────────────────────────────────────────────────────────


def _fred(name: str) -> pd.Series:
    df = pd.read_csv(DATA / f"{name.lower()}.csv", parse_dates=["DATE"])
    s = df.set_index("DATE")[name].astype(float).sort_index()
    s.index = _month_end(s.index)
    if s.index.has_duplicates or not s.index.equals(pd.date_range(s.index.min(), s.index.max(), freq="ME")):
        raise AssertionError(f"{name} is not one row per calendar month")
    return s


def _cay_quarterly() -> pd.Series:
    df = pd.read_csv(DATA / "cay_current.csv", parse_dates=["date"])
    s = df.set_index("date")[CAY_COL].astype(float).sort_index()
    s.index = _month_end(s.index)
    if not set(s.index.month) <= {3, 6, 9, 12}:
        raise AssertionError("cay_current.csv dates are not quarter ends")
    if not s.index.equals(pd.date_range(s.index.min(), s.index.max(), freq="QE-DEC")):
        raise AssertionError("cay_current.csv skips a quarter")
    return s.rename("CAY")


def build_controls(shiller: pd.DataFrame) -> dict:
    """Controls for each timing, and the source month (mcode) behind every value.

    src[timing][control] is a DataFrame of source mcodes, one column per input.
    """
    cal = pd.date_range("1952-01-31", "2026-12-31", freq="ME")
    code = pd.Series(_mcode(cal), index=cal, dtype=float)
    baa, aaa, gs10, tb = _fred("BAA"), _fred("AAA"), _fred("GS10"), _fred("TB3MS")

    P, E = shiller["P"], shiller["E"]
    e_lag = E.shift(3)
    log_pe = {
        "RT": np.log(P / e_lag).reindex(cal),
        "NB": np.log(P / E).reindex(cal),
    }
    pe_src = {
        "RT": pd.DataFrame({"P": code, "E": code - 3}).where(log_pe["RT"].notna(), axis=0),
        "NB": pd.DataFrame({"P": code, "E": code}).where(log_pe["NB"].notna(), axis=0),
    }

    dfsp = (baa - aaa).reindex(cal)
    tmsp = (gs10 - tb).reindex(cal)
    rrel = (tb - tb.rolling(12).mean()).reindex(cal)

    q = _cay_quarterly()
    q_code = pd.Series(_mcode(q.index), index=q.index, dtype=float)
    cay_nb = q.reindex(cal).ffill(limit=2)
    cay_nb_src = q_code.reindex(cal).ffill(limit=2)
    cay = {"NB": cay_nb, "RT": cay_nb.shift(3)}
    cay_src = {"NB": cay_nb_src, "RT": cay_nb_src.shift(3)}

    values, src = {}, {}
    for timing in TIMINGS:
        values[timing] = pd.DataFrame(
            {"log_PE": log_pe[timing], "DFSP": dfsp, "TMSP": tmsp, "RREL": rrel, "CAY": cay[timing]}
        )
        src[timing] = {
            "log_PE": pe_src[timing],
            "DFSP": pd.DataFrame({"BAA": code, "AAA": code}).where(dfsp.notna(), axis=0),
            "TMSP": pd.DataFrame({"GS10": code, "TB3MS": code}).where(tmsp.notna(), axis=0),
            "RREL": pd.DataFrame({"TB3MS_first": code - 11, "TB3MS_last": code}).where(rrel.notna(), axis=0),
            "CAY": pd.DataFrame({"quarter_end": cay_src[timing]}),
        }
    return {"values": values, "src": src, "cay_rt_last": cay["RT"].last_valid_index(),
            "cay_nb_last": cay["NB"].last_valid_index(), "cay_q_last": q.index.max()}


def check_rt_timing(ctrl: dict, months: pd.DatetimeIndex, controls: tuple[str, ...]) -> int:
    """Check 2 on the given months. Returns the number of (month, control) pairs."""
    t = _mcode(months)
    n = 0
    for c in controls:
        s = ctrl["src"]["RT"][c].reindex(months)
        if s.isna().any().any():
            raise AssertionError(f"RT {c} has no source on a block month")
        latest = s.max(axis=1).to_numpy()
        if np.any(latest > t):
            bad = months[latest > t]
            raise AssertionError(f"RT {c} uses a source after t at {_ym(bad[0])}")
        if c == "CAY" and np.any(s["quarter_end"].to_numpy() > t - 3):
            raise AssertionError("RT CAY uses a quarter that ends less than 3 months before t")
        if c == "log_PE" and not np.array_equal(s["E"].to_numpy(), t - 3):
            raise AssertionError("RT log P/E does not use the earnings of month t − 3")
        n += len(months)
    return n


# ── Regressions ──────────────────────────────────────────────────────────────


def regressions(measure: str, controls: tuple[str, ...]) -> list[tuple[str, str, str, list[str]]]:
    """(kind, control, label, regressors) in table order."""
    out = [("a", "", measure, [measure])]
    out += [("b", c, c, [c]) for c in controls]
    out += [("c", c, f"{measure} + {c}", [measure, c]) for c in controls]
    out.append(("d", "all", f"{measure} + all", [measure, *controls]))
    return out


def fit(sample: pd.DataFrame, regs: list[str], name: str) -> dict:
    n = len(sample)
    y = 1200.0 * btz.forward_mean(sample["e"], H)
    frame = pd.concat([y.rename("y"), sample[regs]], axis=1).dropna()
    if not frame.index.equals(pd.DatetimeIndex(sample.index[: n - H])):
        raise AssertionError(f"{name}: rows are not the complete h-month windows of the block")
    for t in frame.index:
        inf.assert_row_timing(sample, t, H, name)
    model = sm.OLS(frame["y"], sm.add_constant(frame[regs], has_constant="add")).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max(1, H - 1), "use_correction": False},
    )
    coef = model.params[regs].to_numpy(dtype=float)
    e = 1200.0 * sample["e"].to_numpy(dtype=float)
    X = sample[regs].to_numpy(dtype=float)
    hod_pos = np.arange(H - 1, n - 1)
    if len(hod_pos) != len(frame):
        raise AssertionError(f"{name}: Hodrick dates and regression rows differ")
    t_1b, _se = inf3a.hodrick_t(e, X, H, hod_pos, coef)
    out = {"index": frame.index, "R2": float(model.rsquared * 100.0), "N": int(model.nobs)}
    for j, r in enumerate(regs):
        out[f"coef_{r}"] = float(coef[j])
        out[f"tNW_{r}"] = float(model.tvalues[r])
        out[f"t1B_{r}"] = float(t_1b[0, j])
    return out


def block_variants(start: str, end: str, cay_end: pd.Timestamp) -> list[tuple[str, str, tuple[str, ...]]]:
    if pd.Timestamp(end) <= cay_end:
        return [(WITH_CAY, end, CONTROLS)]
    return [(WITH_CAY, str(cay_end.date()), CONTROLS), (NO_CAY, end, tuple(c for c in CONTROLS if c != "CAY"))]


def run_blocks(loaded: dict, ctrl: dict) -> tuple[pd.DataFrame, dict]:
    full = loaded["full"].copy()
    full["VRP_fwd"] = loaded["vrp_fwd"]
    rows = []
    meta = {"blocks": [], "n_rt_pairs": 0, "n_rows_checked": 0}
    for timing in TIMINGS:
        values = ctrl["values"][timing]
        for block, measure, start, end in BLOCKS:
            for variant, v_end, controls in block_variants(start, end, CAY_WINDOW_END):
                window = btz.slice_sample(full, start, v_end).join(values[list(controls)])
                needed = ["e", "rv_max", "iv_max", measure, *controls]
                ok = window[needed].notna().all(axis=1)
                months = window.index[ok]
                sample = window.loc[months]
                if not months.equals(pd.date_range(months.min(), months.max(), freq="ME")):
                    raise AssertionError(f"{timing} {block} {measure} {variant}: common months are not contiguous")
                if sample[needed].isna().any().any():
                    raise AssertionError(f"{timing} {block} {measure} {variant}: NaN inside the block")
                if timing == "RT":
                    meta["n_rt_pairs"] += check_rt_timing(ctrl, months, controls)
                index0 = None
                for kind, control, label, regs in regressions(measure, controls):
                    name = f"{timing} {block} {measure} {variant} {label}"
                    res = fit(sample, regs, name)
                    if index0 is None:
                        index0 = res["index"]
                    elif not res["index"].equals(index0):
                        raise AssertionError(f"{name}: rows differ from the block's first regression")
                    meta["n_rows_checked"] += res["N"]
                    rec = {
                        "timing": timing, "block": block, "measure": measure, "variant": variant,
                        "kind": kind, "control": control, "regression": label,
                        "first_month": _ym(months.min()), "last_month": _ym(months.max()),
                        "first_row": _ym(index0.min()), "last_row": _ym(index0.max()),
                    }
                    rec.update({k: v for k, v in res.items() if k != "index"})
                    rows.append(rec)
                meta["blocks"].append(
                    {"timing": timing, "block": block, "measure": measure, "variant": variant,
                     "months": len(months), "first": _ym(months.min()), "last": _ym(months.max()),
                     "N": len(index0), "first_row": _ym(index0.min()), "last_row": _ym(index0.max()),
                     "window_end": _ym(pd.Timestamp(v_end))}
                )
    return pd.DataFrame(rows), meta


def table_columns() -> list[str]:
    cols = ["block", "measure", "variant", "kind", "control", "regression",
            "first_month", "last_month", "first_row", "last_row"]
    for r in REGRESSORS:
        cols += [f"coef_{r}", f"tNW_{r}", f"t1B_{r}"]
    return cols + ["R2", "N"]


def _pick(tab: pd.DataFrame, timing: str, block: str, measure: str, variant: str, kind: str, control: str = "") -> pd.Series:
    sel = tab[(tab["timing"] == timing) & (tab["block"] == block) & (tab["measure"] == measure)
              & (tab["variant"] == variant) & (tab["kind"] == kind) & (tab["control"] == control)]
    if len(sel) != 1:
        raise AssertionError(f"{timing} {block} {measure} {variant} {kind} {control}: {len(sel)} rows")
    return sel.iloc[0]


def check_references(tab: pd.DataFrame) -> list[dict]:
    """Check 3, and row (a) against the spec 003/003a rows on the same sample."""
    t2 = pd.read_csv(btz.RESULTS / "table2_main.csv").set_index(["sample", "h"]).loc[("main", H)]
    ti = pd.read_csv(inf3.RESULTS / "table_inference.csv").set_index(["id", "h"])
    out = []
    for timing in TIMINGS:
        base = _pick(tab, timing, "T4-base", "VRP", WITH_CAY, "a")
        if base["first_month"] != "1990-01" or base["last_month"] != "2007-12":
            raise AssertionError(f"{timing}: CAY or another control shortens T4-base")
        if (abs(base["coef_VRP"] - t2["coef_B"]) > 1e-8 or abs(base["tNW_VRP"] - t2["t_h_minus_1"]) > 1e-8
                or abs(base["R2"] - t2["R2"]) > 1e-8 or base["N"] != int(t2["N"])):
            raise AssertionError(f"{timing} T4-base (a) is not the spec 001 main regression at h=3")
        if (round(base["coef_VRP"], 4), round(base["tNW_VRP"], 2)) != REF_001_H3:
            raise AssertionError(f"{timing} T4-base (a) is not coef 0.4196, t 3.46")
        for (block, measure, variant), ref in REF_003.items():
            r = _pick(tab, timing, block, measure, variant, "a")
            got = ti.loc[(ref, H)]
            diffs = (abs(r[f"coef_{measure}"] - got["coef_B"]), abs(r[f"tNW_{measure}"] - got["t_NW_hm1"]),
                     abs(r[f"t1B_{measure}"] - got["t_1B"]))
            if max(diffs) > 1e-8 or r["N"] != int(got["N"]):
                raise AssertionError(f"{timing} {block} {measure} (a) differs from spec 003 {ref}: {diffs}")
            if timing == "RT":
                out.append({"block": block, "measure": measure, "variant": variant, "ref": ref,
                            "coef": r[f"coef_{measure}"], "tNW": r[f"tNW_{measure}"], "t1B": r[f"t1B_{measure}"],
                            "N": int(r["N"]), "max_diff": max(diffs)})
    return out


def timing_compare(tab: pd.DataFrame) -> pd.DataFrame:
    out = []
    for kind, control, label, _regs in regressions("VRP", CONTROLS):
        rt = _pick(tab, "RT", "T4-base", "VRP", WITH_CAY, kind, control)
        nb = _pick(tab, "NB", "T4-base", "VRP", WITH_CAY, kind, control)
        if rt["first_row"] != nb["first_row"] or rt["last_row"] != nb["last_row"]:
            raise AssertionError("T4-base rows differ between RT and NB")
        out.append({
            "regression": label, "kind": kind, "control": control,
            "R2_RT": rt["R2"], "R2_NB": nb["R2"], "R2_NB_minus_RT": nb["R2"] - rt["R2"],
            "coef_VRP_RT": rt["coef_VRP"], "coef_VRP_NB": nb["coef_VRP"],
            "tNW_VRP_RT": rt["tNW_VRP"], "tNW_VRP_NB": nb["tNW_VRP"],
            "t1B_VRP_RT": rt["t1B_VRP"], "t1B_VRP_NB": nb["t1B_VRP"],
            "N": int(rt["N"]),
        })
    return pd.DataFrame(out)


def flags(tab: pd.DataFrame, cmp: pd.DataFrame) -> dict:
    k1 = {c: _pick(tab, "RT", "T4-base", "VRP", WITH_CAY, "c", c)["t1B_VRP"] for c in CONTROLS}
    k2 = {c: float(cmp.loc[(cmp["kind"] == "c") & (cmp["control"] == c), "R2_NB_minus_RT"].iloc[0])
          for c in ("log_PE", "CAY")}
    k3 = {}
    for c in CONTROLS:
        variant = WITH_CAY if c == "CAY" else NO_CAY
        r = _pick(tab, "RT", "T4-post", "VRP_fwd", variant, "c", c)
        k3[c] = (float(r["t1B_VRP_fwd"]), variant, r["last_month"])
    k4 = {
        "T4-base VRP": float(_pick(tab, "RT", "T4-base", "VRP", WITH_CAY, "d", "all")["t1B_VRP"]),
        "T4-post VRP_fwd": float(_pick(tab, "RT", "T4-post", "VRP_fwd", NO_CAY, "d", "all")["t1B_VRP_fwd"]),
    }
    return {
        "K1": (all(abs(v) > T_CRIT for v in k1.values()), k1),
        "K2": (all(v > K2_GAP for v in k2.values()), k2),
        "K3": (all(abs(v[0]) > T_CRIT for v in k3.values()), k3),
        "K4": (all(abs(v) > T_CRIT for v in k4.values()), k4),
    }


# ── Outputs ──────────────────────────────────────────────────────────────────


def plot_r2(tab: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    panels = (("T4-base", "VRP", "T4-base, 1990–2007"), ("T4-post", "VRP_fwd", "T4-post, 2008–2025"))
    width = 0.38
    pos = np.arange(len(CONTROLS))
    for ax, (block, measure, title) in zip(axes, panels):
        alone, joint, hatch = [], [], []
        for c in CONTROLS:
            variant = WITH_CAY if (block == "T4-base" or c == "CAY") else NO_CAY
            alone.append(_pick(tab, "RT", block, measure, variant, "b", c)["R2"])
            joint.append(_pick(tab, "RT", block, measure, variant, "c", c)["R2"])
            hatch.append(block == "T4-post" and c == "CAY")
        b1 = ax.bar(pos - width / 2, alone, width, color="lightcoral", label="control alone")
        b2 = ax.bar(pos + width / 2, joint, width, color="steelblue", label=f"{measure} + control")
        for bars in (b1, b2):
            for bar, hz in zip(bars, hatch):
                if hz:
                    bar.set_hatch("//")
        base_variant = WITH_CAY if block == "T4-base" else NO_CAY
        r2_a = _pick(tab, "RT", block, measure, base_variant, "a")["R2"]
        ax.axhline(r2_a, color="black", ls="--", lw=1.0, label=f"{measure} alone ({_fmt(r2_a, 2)}%)")
        if block == "T4-post":
            r2_ac = _pick(tab, "RT", block, measure, WITH_CAY, "a")["R2"]
            ax.axhline(r2_ac, color="gray", ls=":", lw=1.0, label=f"{measure} alone, 2008–2019 ({_fmt(r2_ac, 2)}%)")
        for x, v in zip(np.concatenate([pos - width / 2, pos + width / 2]), alone + joint):
            ax.text(x, v + 0.15, _fmt(v, 1), ha="center", va="bottom", fontsize=7)
        labels = list(CONTROLS)
        if block == "T4-post":
            labels[CONTROLS.index("CAY")] = "CAY\n(2008–2019)"
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("R² (%)")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Real-time controls, h = 3", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _block_table(L: list[str], tab: pd.DataFrame, block: str, measure: str, variant: str) -> None:
    sub = tab[(tab["timing"] == "RT") & (tab["block"] == block) & (tab["measure"] == measure) & (tab["variant"] == variant)]
    controls = [c for c in CONTROLS if (sub["control"] == c).any()]
    first = sub.iloc[0]
    L.append(f"**{block}, {measure}, {variant}.** Months {first['first_month']} → {first['last_month']}, "
             f"regression rows {first['first_row']} → {first['last_row']}, N = {int(first['N'])}.")
    L.append("")
    L.append(f"| regression | coef {measure} | t NW | t 1B | coef control | t NW | t 1B | R² |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in sub.iterrows():
        if r["kind"] == "d":
            continue
        c = r["control"]
        L.append(
            f"| {r['regression']} | {_f(r[f'coef_{measure}'], 4)} | {_f(r[f'tNW_{measure}'], 2)} | {_f(r[f't1B_{measure}'], 2)} | "
            f"{_f(r[f'coef_{c}'], 4) if c else ''} | {_f(r[f'tNW_{c}'], 2) if c else ''} | {_f(r[f't1B_{c}'], 2) if c else ''} | "
            f"{_f(r['R2'], 2)} |"
        )
    d = sub[sub["kind"] == "d"].iloc[0]
    L.append("")
    L.append(f"All controls together, {d['regression']} (R² = {_f(d['R2'], 2)}%):")
    L.append("")
    L.append("| regressor | coef | t NW | t 1B |")
    L.append("|---|---|---|---|")
    for r in [measure, *controls]:
        L.append(f"| {r} | {_f(d[f'coef_{r}'], 4)} | {_f(d[f'tNW_{r}'], 2)} | {_f(d[f't1B_{r}'], 2)} |")
    L.append("")


def write_summary(p: dict) -> None:
    tab, meta, sh, ctrl = p["tab"], p["meta"], p["shiller"], p["ctrl"]
    cmp, fl, refs = p["compare"], p["flags"], p["refs"]
    L: list[str] = []
    L.append("# Spec 004 — BTZ Table 4 with real-time controls")
    L.append("")
    L.append(
        "Code: `src/controls.py`, called from notebook section 4.2. It imports the spec 001, 002, 003 and 003a modules and does not modify them. "
        "The panel, units, VRP, VRP_fwd and the multivariate Hodrick 1B function (`inference_003a.hodrick_t`) are reused as they are."
    )
    L.append("")
    L.append("## Data vintage")
    L.append("")
    L.append(
        f"- `data/shiller_2026.csv`: Robert Shiller's `ie_data.xls`, sheet `Data`, columns Date, P and E. Downloaded {SHILLER_DOWNLOAD_DATE} from the link on shillerdata.com "
        f"(`{SHILLER_URL}`). {sh['rows']:,} rows, {sh['first']} → {sh['last']}. "
        "The rows after the last month with earnings (P only) are dropped. P is the monthly average of daily closes. E is 12-month trailing earnings, "
        "interpolated monthly from quarterly totals by Shiller."
    )
    L.append(
        "- The Yale address used in the notebook (`econ.yale.edu/~shiller/data/ie_data.xls`) was also tried the same day. It still serves a file that ends in 2023-09 with E to 2023-06, "
        "the vintage of the existing `data/shiller_data.csv`, which is not modified."
    )
    o0, o1, on = sh["old_overlap"]
    L.append(
        f"- Overlap with `data/shiller_data.csv`, {o0} → {o1} ({on} months): the largest absolute difference is {sh['old_max_dP']:.4g} in P and {sh['old_max_dE']:.4g} in E."
    )
    L.append(
        "- Unchanged frozen files: `data/baa.csv`, `data/aaa.csv`, `data/gs10.csv`, `data/tb3ms.csv` (FRED monthly averages to 2026-08), "
        f"`data/cay_current.csv` (quarterly, last quarter {ctrl['cay_q_last'].strftime('%Y-%m')})."
    )
    L.append("")
    L.append("## Definitions")
    L.append("")
    L.append(
        "h = 3. Dependent variable: Y_t(3) = 1200 × the mean of the excess log returns of months t+1 … t+3, spec 001 specification (B). "
        "VRP and VRP_fwd are in monthly percent squared. NW: statsmodels HAC, Bartlett, maxlags = 2, `use_correction=False`. "
        "Hodrick 1B: `inference_003a.hodrick_t` with z_t = (1, x_t')' over the block's regressors, the null residual, and positions h−1 … n−2 of the block's months. R² in percent."
    )
    L.append("")
    L.append(
        "Controls, RT: log P/E = log(P_t / E_{t−3}); DFSP = BAA_t − AAA_t; TMSP = GS10_t − TB3MS_t; RREL = TB3MS_t − mean(TB3MS_{t−11} … TB3MS_t); "
        "CAY for quarter q ending in month m is assigned to months m+3, m+4, m+5. NB: log P/E = log(P_t / E_t), CAY assigned to months m, m+1, m+2; the other three are the same. "
        "FRED observations dated the first of a month are labelled at the end of that month."
    )
    L.append("")
    L.append(
        "Block sample: the months of the block's window on which e, the VRP measure and every control of the variant are defined. "
        "As in specs 001–003a, Y_t(3) uses only returns inside the block, so the regression rows are the first n − 3 of the block's n months."
    )
    L.append("")
    L.append("## Checks")
    L.append("")
    L.append("### 1. Shiller parsing")
    L.append("")
    cy0, cy1, cyn = sh["complete_years"]
    L.append(
        f"Pass. No duplicate date. Every year from 1990 to 2025 has 12 months, and the file is a complete monthly calendar from {sh['first']} to {sh['last']}, "
        f"so October is present in every year ({cyn} complete years, {cy0}–{cy1}). The date is read on two decimals: 1990.1 → 1990-10, 1990.01 → 1990-01, 1990.11 → 1990-11, asserted."
    )
    L.append("")
    L.append("| month | P | E |")
    L.append("|---|---|---|")
    for t, r in sh["show"].iterrows():
        L.append(f"| {_ym(t)} | {r['P']:.2f} | {r['E']:.4f} |")
    L.append("")
    L.append("### 2. Real-time timing")
    L.append("")
    L.append(
        f"Pass. Every RT control carries the source month of each input. On every month of every RT block ({meta['n_rt_pairs']:,} month × control pairs), "
        "the latest source month is ≤ t. For log P/E the earnings month is exactly t − 3 and the price month is t. "
        "For CAY the quarter used ends at least 3 months before t. "
        f"The last quarter, {_ym(ctrl['cay_q_last'])}, is assigned to {_ym(ctrl['cay_q_last'] + pd.offsets.MonthEnd(3))} … {_ym(ctrl['cay_rt_last'])} under RT "
        f"(NB: through {_ym(ctrl['cay_nb_last'])}). "
        f"The spec 001 look-ahead rule for IV, RV and the return window also passes on all {meta['n_rows_checked']:,} regression rows, both timings."
    )
    L.append("")
    L.append("### 3. T4-base row (a) is the spec 001 main regression")
    L.append("")
    b = _pick(tab, "RT", "T4-base", "VRP", WITH_CAY, "a")
    L.append(
        f"Pass. CAY does not shorten T4-base: the block is 1990-01 → 2007-12 under both timings. Row (a) gives coef(B) = {_fmt(b['coef_VRP'], 4)}, "
        f"t NW = {_fmt(b['tNW_VRP'], 2)}, R² = {_fmt(b['R2'], 2)}%, N = {int(b['N'])}, equal to `results/001-baseline-1990/table2_main.csv` at h = 3 to 1e-8. "
        "Row (a) of the blocks whose sample is a spec 003/003a row also reproduces that row (coef, t NW, t 1B and N, to 1e-8):"
    )
    L.append("")
    L.append("| block | measure | variant | spec 003 row | coef | t NW | t 1B | N |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in refs:
        L.append(f"| {r['block']} | {r['measure']} | {r['variant']} | {r['ref']} | {_fmt(r['coef'], 4)} | {_fmt(r['tNW'], 2)} | {_fmt(r['t1B'], 2)} | {r['N']} |")
    L.append("")
    L.append("### 4 and 5. Same rows, no NaN, N per block")
    L.append("")
    L.append(
        "Pass. Within each block, every regression has the same row index (asserted against the block's first regression), "
        "the block's months are contiguous, and no variable of the block is NaN on them."
    )
    L.append("")
    L.append("| timing | block | measure | variant | months | regression rows | N |")
    L.append("|---|---|---|---|---|---|---|")
    for m in meta["blocks"]:
        L.append(f"| {m['timing']} | {m['block']} | {m['measure']} | {m['variant']} | {m['first']} → {m['last']} | {m['first_row']} → {m['last_row']} | {m['N']} |")
    L.append("")
    L.append("## RT results")
    L.append("")
    L.append("Coefficients in BTZ units. All rows of a table share the same N. Full tables, with NB timing and the T4-95 and T4-full blocks, are in `table4_rt.csv` and `table4_nb.csv`.")
    L.append("")
    _block_table(L, tab, "T4-base", "VRP", WITH_CAY)
    _block_table(L, tab, "T4-post", "VRP_fwd", NO_CAY)
    _block_table(L, tab, "T4-post", "VRP_fwd", WITH_CAY)
    L.append("### Against BTZ Table 4")
    L.append("")
    L.append("The target cells in the spec are not filled, so they are left empty. The two notebook figures quoted in the spec are shown in parentheses.")
    L.append("")
    L.append("| regression | coef VRP | BTZ | t VRP NW | t VRP 1B | BTZ | coef control | BTZ | t control NW | BTZ | R² | BTZ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in CONTROLS:
        r = _pick(tab, "RT", "T4-base", "VRP", WITH_CAY, "c", c)
        tc, tt, tcc, ttc, tr = BTZ_TARGETS[c]
        r2_t = _f(tr, 2) or ("(16.76)" if c == "log_PE" else "")
        tv_t = _f(tt, 2) or ("(3.02)" if c == "CAY" else "")
        L.append(
            f"| VRP + {c} | {_fmt(r['coef_VRP'], 4)} | {_f(tc, 2)} | {_fmt(r['tNW_VRP'], 2)} | {_fmt(r['t1B_VRP'], 2)} | {tv_t} | "
            f"{_fmt(r[f'coef_{c}'], 4)} | {_f(tcc, 2)} | {_fmt(r[f'tNW_{c}'], 2)} | {_f(ttc, 2)} | {_fmt(r['R2'], 2)} | {r2_t} |"
        )
    L.append("")
    L.append("### The VRP measure across blocks, RT")
    L.append("")
    L.append("Hodrick 1B t of the VRP measure: alone (a), with each control (c), and with all controls (d).")
    L.append("")
    L.append("| block | measure | variant | N | (a) | " + " | ".join(f"+ {c}" for c in CONTROLS) + " | (d) |")
    L.append("|---|---|---|---|---|" + "---|" * len(CONTROLS) + "---|")
    for m in [m for m in meta["blocks"] if m["timing"] == "RT"]:
        blk, meas, var = m["block"], m["measure"], m["variant"]
        cells = []
        for c in CONTROLS:
            sel = tab[(tab["timing"] == "RT") & (tab["block"] == blk) & (tab["measure"] == meas) & (tab["variant"] == var)
                      & (tab["kind"] == "c") & (tab["control"] == c)]
            cells.append(_fmt(sel.iloc[0][f"t1B_{meas}"], 2) if len(sel) else "—")
        a = _pick(tab, "RT", blk, meas, var, "a")[f"t1B_{meas}"]
        d = _pick(tab, "RT", blk, meas, var, "d", "all")[f"t1B_{meas}"]
        L.append(f"| {blk} | {meas} | {var} | {m['N']} | {_fmt(a, 2)} | " + " | ".join(cells) + f" | {_fmt(d, 2)} |")
    L.append("")
    L.append("![R², control alone and VRP + control, RT](fig_r2_controls.png)")
    L.append("")
    L.append("## Timing: RT against NB, T4-base")
    L.append("")
    L.append("Same 213 rows under both timings. Only log P/E and CAY change between the timings, so the other rows are identical.")
    L.append("")
    L.append("| regression | R² RT | R² NB | NB − RT | t NW VRP RT | t NW VRP NB | t 1B VRP RT | t 1B VRP NB |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in cmp.iterrows():
        L.append(
            f"| {r['regression']} | {_fmt(r['R2_RT'], 2)} | {_fmt(r['R2_NB'], 2)} | {_fmt(r['R2_NB_minus_RT'], 2)} | "
            f"{_f(r['tNW_VRP_RT'], 2)} | {_f(r['tNW_VRP_NB'], 2)} | {_f(r['t1B_VRP_RT'], 2)} | {_f(r['t1B_VRP_NB'], 2)} |"
        )
    L.append("")
    pe = cmp[(cmp["kind"] == "c") & (cmp["control"] == "log_PE")].iloc[0]
    cy = cmp[(cmp["kind"] == "c") & (cmp["control"] == "CAY")].iloc[0]
    L.append(
        f"Audit reference, direction only (SPY, 1993-12 → 2008-01, old units): VRP + log P/E R² 17.1% NB against 13.6% with E lagged 3 months; "
        f"VRP + CAY 8.6% NB against 5.7% with CAY lagged one quarter. Here: VRP + log P/E {_fmt(pe['R2_NB'], 2)}% NB against {_fmt(pe['R2_RT'], 2)}% RT; "
        f"VRP + CAY {_fmt(cy['R2_NB'], 2)}% NB against {_fmt(cy['R2_RT'], 2)}% RT."
    )
    L.append("")
    L.append("## Flags (h = 3, RT, Hodrick 1B)")
    L.append("")
    L.append("| flag | holds | numbers |")
    L.append("|---|---|---|")
    k1, v1 = fl["K1"]
    L.append(f"| K1 BTZ robustness | {_yn(k1)} | T4-base, t 1B of VRP: " + "; ".join(f"+ {c} {_fmt(v, 2)}" for c, v in v1.items()) + " |")
    k2, v2 = fl["K2"]
    L.append(f"| K2 timing inflates the controls | {_yn(k2)} | T4-base, R² NB − RT of VRP + control: " + "; ".join(f"{c} {_fmt(v, 2)} pp" for c, v in v2.items()) + f" (threshold {_fmt(K2_GAP, 0)} pp) |")
    k3, v3 = fl["K3"]
    L.append(f"| K3 post-2008 robustness | {_yn(k3)} | T4-post, t 1B of VRP_fwd: " + "; ".join(f"+ {c} {_fmt(v[0], 2)} (to {v[2]})" for c, v in v3.items()) + " |")
    k4, v4 = fl["K4"]
    L.append(f"| K4 kitchen sink | {_yn(k4)} | all controls, t 1B: " + "; ".join(f"{k} {_fmt(v, 2)}" for k, v in v4.items()) + " (T4-post without CAY) |")
    L.append("")
    L.append("The flags are read as written in the spec. A flag that fails is reported with its numbers; neither the timing nor the controls were changed after the run.")
    L.append("")
    L.append("## Limitations of the timing")
    L.append("")
    L.append(
        "1. **DFSP, TMSP and RREL.** BAA, AAA, GS10 and TB3MS are monthly averages of month t. FRED publishes them in the first days of month t+1, "
        "so at the end of month t the last few daily values are not yet in the average. Both timings treat them as known at t."
    )
    L.append(
        f"2. **CAY is estimated on the full sample.** The cointegrating coefficients in `cay_current.csv` (c − 0.218 a − 0.801 y + 0.441) are estimated on data to {ctrl['cay_q_last'].year}. "
        "RT timing fixes the publication lag of each quarter, not the look-ahead in the coefficients, which these data cannot remove. "
        f"CAY ends in {ctrl['cay_q_last'].year}Q{(ctrl['cay_q_last'].month - 1) // 3 + 1}, so the CAY versions of T4-post and T4-full end in {_ym(CAY_WINDOW_END)}."
    )
    L.append(
        "3. **Shiller E is interpolated.** Shiller builds monthly E by linear interpolation between quarterly four-quarter totals. "
        "When t − 3 is not a quarter end, E_{t−3} mixes in the next quarter's total, which ends 1 or 2 months before t and can be published after t. "
        "The 3-month lag is the spec's rule and is kept. P is the monthly average of daily closes, known at the end of month t."
    )
    L.append("")
    L.append("## Deviations from this spec")
    L.append("")
    devs = [
        "The Shiller file comes from the shillerdata.com link, not the Yale address in the notebook, because the Yale file ends in 2023-09 and cannot meet check 1 for 2024–2025.",
        f"The spec's two statements about the end of RT CAY disagree. The RT rule assigns {ctrl['cay_q_last'].year}Q3 (m = {_ym(ctrl['cay_q_last'])}) to months m+3 … m+5, "
        f"that is {_ym(ctrl['cay_q_last'] + pd.offsets.MonthEnd(3))} … {_ym(ctrl['cay_rt_last'])}, while the spec says the last usable month is 2019-12. "
        f"Both are kept as written: the RT series follows the rule, and the CAY versions of T4-post and T4-full end in {_ym(CAY_WINDOW_END)}, the end the spec gives three times. "
        "So 2020-01 and 2020-02 are not used.",
        f"Under NB timing, CAY of the last quarter covers months m … m+2, so it ends in {_ym(ctrl['cay_nb_last'])}. "
        "The NB CAY versions of T4-post and T4-full use the block rule (months where every variable is defined) and therefore end in "
        f"{_ym(ctrl['cay_nb_last'])}, one month earlier than the RT versions. T4-base is not affected.",
        "Y_t(3) only uses returns inside the block, as in specs 001–003a, so a block ending in 2019-12 has its last regression row in 2019-09. "
        "This is what check 3 requires (N = 213 on 1990–2007).",
        "The SUMMARY also reports, for row (a), the agreement with the spec 003/003a rows on the same sample, and a compact table of the VRP measure's 1B t across all blocks. These are additions.",
    ]
    for i, text in enumerate(devs, start=1):
        L.append(f"{i}. {text}")
    L.append("")
    (RESULTS / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> dict:
    inf3a._self_checks()
    download_shiller()
    shiller = load_shiller()
    sh = check_shiller(shiller)
    ctrl = build_controls(shiller)
    if ctrl["cay_q_last"] != pd.Timestamp("2019-09-30") or ctrl["cay_rt_last"] < CAY_WINDOW_END:
        raise AssertionError(f"CAY ends in {_ym(ctrl['cay_q_last'])}, last RT month {_ym(ctrl['cay_rt_last'])}")

    loaded = inf3._load()
    tab, meta = run_blocks(loaded, ctrl)
    refs = check_references(tab)
    cmp = timing_compare(tab)
    fl = flags(tab, cmp)

    RESULTS.mkdir(parents=True, exist_ok=True)
    cols = table_columns()
    for timing in TIMINGS:
        tab[tab["timing"] == timing].reindex(columns=cols).to_csv(RESULTS / f"table4_{timing.lower()}.csv", index=False)
    cmp.to_csv(RESULTS / "table4_timing_compare.csv", index=False)
    plot_r2(tab, RESULTS / "fig_r2_controls.png")
    payload = {"tab": tab, "meta": meta, "shiller": sh, "ctrl": ctrl, "compare": cmp, "flags": fl, "refs": refs}
    write_summary(payload)
    for k, (ok, _v) in fl.items():
        print(f"  {k}: {_yn(ok)}")
    print(f"Wrote {RESULTS}")
    return payload


if __name__ == "__main__":
    main()
