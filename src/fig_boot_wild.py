"""Spec 006a — bootstrap histograms of the Newey–West t at h = 3, iid and wild.

Replays the spec 003 iid draws and the spec 003a wild draws for R-base and R-G
with the generators and row order of `src/inference_003a.py`, checks the
p-values against results/003-inference/table_inference.csv, and writes
results/006-findings/fig_boot_h3_iid_wild.png and fig_boot_h3_iid_wild.csv.
Imports the spec 003 and 003a modules and does not modify them.
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

from src import inference as inf3
from src import inference_003a as a3

OUT = inf3.RESULTS.parent / "006-findings"
H = 3
PANEL_ROWS = (("R-base", "1990–2007"), ("R-G", "1995–2025"))
SCHEMES = ("iid", "wild")
COLORS = {"iid": "steelblue", "wild": "seagreen"}
P_TOL = 1e-12


def replay_wild(rows: dict, targets: tuple[str, ...]) -> dict:
    """Wild draws at h = 3 for `targets`, from the 003a stream in its row order."""
    rng = np.random.default_rng(a3.SEED)
    out = {}
    for name in a3.ALL_ROWS:
        e = rows[name]["by_h"][1]["e"]
        x = rows[name]["by_h"][1]["x"]
        if name in a3.NEW_ROWS:
            inf3.simulate_null(rng, e, x, "iid", a3.N_REP)
        e_star, x_star = a3.simulate_wild(rng, e, x, a3.N_REP)
        if name in targets:
            out[name] = a3._boot_stats(rows[name], e_star, x_star, "wild")[H]
        if set(out) == set(targets):
            return out
    raise AssertionError(f"rows {set(targets) - set(out)} are not in the 003a row order")


def collect() -> pd.DataFrame:
    loaded = inf3._load()
    rows, _ = a3.fit_rows(loaded)
    targets = tuple(name for name, _ in PANEL_ROWS)
    iid = a3.replay_003_iid(rows)
    wild = replay_wild(rows, targets)
    table = pd.read_csv(inf3.RESULTS / "table_inference.csv").set_index(["id", "h"])

    records = []
    for name, years in PANEL_ROWS:
        observed = rows[name]["by_h"][H]["t_hm1"]
        ref = table.loc[(name, H)]
        if abs(observed - float(ref["t_NW_hm1"])) > P_TOL:
            raise AssertionError(f"{name}: observed NW t {observed} is not table_inference.csv's {ref['t_NW_hm1']}")
        for scheme, draws in (("iid", iid[name][H]["t_nw"]), ("wild", wild[name]["t_nw"])):
            p = a3._p_value(draws, observed)
            saved = float(ref[f"p_boot_NW_{scheme}"])
            if abs(p - saved) > P_TOL:
                raise AssertionError(f"{name} {scheme}: replayed p = {p}, table_inference.csv has {saved}")
            mean_t = float(draws.mean())
            if abs(mean_t) > inf3.T_MEAN_BAND:
                raise AssertionError(f"{name} {scheme}: mean of the null t is {mean_t}")
            records.append({
                "id": name, "sample": years, "scheme": scheme, "h": H, "n_rep": len(draws),
                "observed_t_NW": observed, "p_boot_NW": p, "mean_t_null": mean_t,
                "sd_t_null": float(draws.std(ddof=1)), "draws": draws,
            })
    return pd.DataFrame(records)


def plot(panels: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), sharey="row")
    for i, (name, years) in enumerate(PANEL_ROWS):
        row = panels[panels["id"] == name].set_index("scheme")
        observed = float(row["observed_t_NW"].iloc[0])
        span = np.concatenate(list(row["draws"]))
        xmax = max(4.0, abs(observed) + 0.6, float(np.quantile(np.abs(span), 0.999)) + 0.4)
        bins = np.linspace(-xmax, xmax, 61)
        for j, scheme in enumerate(SCHEMES):
            ax = axes[i, j]
            r = row.loc[scheme]
            ax.hist(r["draws"], bins=bins, color=COLORS[scheme], alpha=0.75,
                    label=f"null draws, mean {r['mean_t_null']:.2f}")
            ax.axvline(observed, color="black", lw=1.2,
                       label=f"observed t = {observed:.2f}, p = {r['p_boot_NW']:.4f}")
            ax.axvline(0, color="gray", lw=0.6)
            ax.set_xlim(-xmax, xmax)
            ax.set_title(f"{name} ({years}), {scheme}")
            ax.set_xlabel("Newey–West t under no predictability (maxlags = h−1)")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        axes[i, 0].set_ylabel("Replications (of 5,000)")
    fig.suptitle("Bootstrap distribution of the Newey–West t at h = 3, iid and wild schemes", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> pd.DataFrame:
    panels = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    plot(panels, OUT / "fig_boot_h3_iid_wild.png")
    stats = panels.drop(columns="draws")
    stats.to_csv(OUT / "fig_boot_h3_iid_wild.csv", index=False)
    print(stats.to_string(index=False))
    return stats


if __name__ == "__main__":
    main()
