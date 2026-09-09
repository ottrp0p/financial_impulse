"""Driver: fetch universe -> features -> skill gate (dev names) -> sweep -> held-out confirmation.

Usage: .venv/bin/python -m growth.run [--start 2016-01-01] [--workers 8] [--quick] [--features-only]
Outputs under out/growth/: features.parquet, skill_dev.csv, skill_hold.csv, sweep.csv, chosen.json, summary.md
"""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import ROOT
from .book import BookParams, run_book
from .edge import EdgeConfig, fit_predict, gate, skill_table
from .features import panel_features
from .frontier import analytic, bootstrap
from .universe import load_bars, split_names, universe

OUT = ROOT / "out" / "growth"
END = datetime(2026, 8, 26)

GRID = {"tau": [0.05, 0.10, 0.15, 0.20, 0.30], "k": [0.25, 0.5, 0.75, 1.0], "n_max": [3, 5, 10, 20],
        "L_max": [1.0, 2.0, 3.0, 5.0]}
FLOORS = [250e3, 350e3]
TARGET, E0 = 10e6, 500e3


def build_features(a) -> pd.DataFrame:
    fp = OUT / ("features_quick.parquet" if a.quick else "features.parquet")
    if fp.exists() and not a.refresh:
        return pd.read_parquet(fp)
    syms = universe("all")
    if a.quick:
        syms = syms[:20] + universe("etf")[:6] + universe("leveraged")[:2]
    print(f"loading {len(syms)} symbols from {a.start:%Y-%m-%d}")
    bars = load_bars(syms, a.start, END)
    print(f"  {len(bars)} loaded; fitting fields (step={a.step}, workers={a.workers})")
    panel = panel_features(bars, workers=a.workers, step=a.step)
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(fp)
    return panel


def sweep(pred: pd.DataFrame, floors, quick=False) -> pd.DataFrame:
    grid = GRID if not quick else {"tau": [0.1, 0.2], "k": [0.25, 0.5, 1.0], "n_max": [5, 10], "L_max": [1.0, 3.0, 5.0]}
    rows = []
    combos = list(itertools.product(*grid.values()))
    for j, vals in enumerate(combos):
        kw = dict(zip(grid.keys(), vals))
        p = BookParams(**kw, fixed_equity=True)
        res = run_book(pred, p)
        st = res["stats"]
        r = res["daily"]["ret"].to_numpy()
        row = {**kw, **{k: st[k] for k in ["sharpe", "ann_ret", "G", "V", "max_dd", "avg_n", "avg_gross", "avg_n_eff",
                                            "avg_slip_bp", "margin_flags", "days_traded_frac", "universe_start"]}}
        for fl in floors:
            an = analytic(st["G"], st["V"], E0, TARGET, fl)
            bs = bootstrap(r, E0, TARGET, fl, n_paths=3000, block=5, max_days=2520, seed=0)
            tag = f"f{int(fl / 1e3)}"
            row.update({f"{tag}_p_ruin_an": an["p_ruin"], f"{tag}_days_an": an["e_days"],
                        f"{tag}_p_ruin": bs["p_ruin"], f"{tag}_p_target": bs["p_target"], f"{tag}_med_days": bs["med_days"]})
        rows.append(row)
        if j % 20 == 0:
            print(f"  sweep {j + 1}/{len(combos)}  {kw}  sharpe={st['sharpe']:.2f}  p_ruin={row[f'f{int(floors[0]/1e3)}_p_ruin']:.2f}")
    return pd.DataFrame(rows)


def choose(sw: pd.DataFrame, floor_tag="f250", max_ruin=0.20) -> pd.Series | None:
    ok = sw[(sw[f"{floor_tag}_p_ruin"] <= max_ruin) & sw[f"{floor_tag}_med_days"].notna()]
    if len(ok) == 0:
        return None
    return ok.sort_values(f"{floor_tag}_med_days").iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--features-only", action="store_true")
    a = ap.parse_args()
    a.start = datetime.fromisoformat(a.start)
    OUT.mkdir(parents=True, exist_ok=True)

    panel = build_features(a)
    print(f"panel: {len(panel)} rows, {panel['symbol'].nunique()} names, {panel['ts'].min():%Y-%m-%d}..{panel['ts'].max():%Y-%m-%d}")
    if a.features_only:
        return
    dev, hold = split_names(sorted(panel["symbol"].unique()))
    pdev, phold = panel[panel["symbol"].isin(dev)].reset_index(drop=True), panel[panel["symbol"].isin(hold)].reset_index(drop=True)
    cfg = EdgeConfig()

    print("skill gate on dev names")
    tab_dev, preds_dev = skill_table(pdev, cfg)
    tab_dev.to_csv(OUT / "skill_dev.csv")
    print(tab_dev[[c for c in tab_dev.columns if c.startswith("ic") or c == "n_windows"]].round(4).to_string())
    fs, why = gate(tab_dev)
    print(f"gate -> {fs}: {why}")
    tab_hold, preds_hold = skill_table(phold, cfg)
    tab_hold.to_csv(OUT / "skill_hold.csv")

    if fs == "none":
        fs_trade = "both"
        print("no set passes; running the sweep on 'both' for the record")
    else:
        fs_trade = fs
    print(f"sweep on dev names, feature set {fs_trade}")
    sw = sweep(preds_dev[fs_trade], FLOORS, a.quick)
    sw["feature_set"] = fs_trade
    sw.to_csv(OUT / "sweep.csv", index=False)

    best = choose(sw)
    summary = {"gate": {"feature_set": fs, "why": why}, "traded_set": fs_trade, "chosen": None}
    lines = ["# growth: intraday Kelly book — results", "", f"Gate: **{fs}** — {why}", "",
             "## OOS skill (dev names, A–M)", tab_dev.round(4).to_string(), "",
             "## OOS skill (held-out names, N–Z)", tab_hold.round(4).to_string(), ""]
    if best is None:
        lines.append("## Frontier: no cell has P(floor 250k) <= 20% with a reachable target in 10y.")
        top = sw.sort_values("f250_p_ruin").head(10)
        lines += ["Lowest-ruin cells:", top.round(3).to_string(index=False)]
    else:
        kw = {k: (int(best[k]) if k == "n_max" else float(best[k])) for k in GRID}
        conf_hold = run_book(preds_hold[fs_trade], BookParams(**kw, fixed_equity=True))
        p22 = preds_dev[fs_trade]
        p22 = p22[(p22["ts"] >= "2022-01-01") & (p22["ts"] < "2023-01-01")]
        conf_22 = run_book(p22, BookParams(**kw, fixed_equity=True)) if len(p22) else None
        path = run_book(preds_dev[fs_trade], BookParams(**kw, fixed_equity=False))  # path-dependent liquidity
        r_h = conf_hold["daily"]["ret"].to_numpy()
        bs_h = bootstrap(r_h, E0, TARGET, 250e3, n_paths=3000, block=5, max_days=2520)
        summary["chosen"] = {"params": kw, "dev": {k: float(v) for k, v in best.items() if isinstance(v, (int, float, np.floating))},
                             "hold_stats": conf_hold["stats"], "hold_frontier": bs_h,
                             "y2022_stats": conf_22["stats"] if conf_22 else None, "path_stats": path["stats"]}
        path["daily"].to_csv(OUT / "chosen_path_daily.csv", index=False)
        lines += ["## Chosen frontier point (dev, P(floor 250k) <= 20%, min median days)", f"`{kw}`", "",
                  pd.DataFrame([best]).round(3).T.to_string(), "",
                  "## Held-out confirmation (N–Z names)", pd.DataFrame([conf_hold["stats"]]).round(3).T.to_string(),
                  f"bootstrap frontier on held-out: {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in bs_h.items()})}", ""]
        if conf_22:
            lines += ["## 2022 sub-period (dev names)", pd.DataFrame([conf_22["stats"]]).round(3).T.to_string(), ""]
        lines += ["## Path-dependent run (liquidity caps follow realised equity)", pd.DataFrame([path["stats"]]).round(3).T.to_string()]
    lines += ["", "## Frontier table (dev)", sw.round(3).to_string(index=False)]
    (OUT / "summary.md").write_text("\n".join(lines))
    (OUT / "chosen.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n".join(lines[:40]))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
