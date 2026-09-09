"""Layer 2: walk-forward out-of-sample tests of the gravity field.

For each window: fit the field on bars [t-N, t), score every level, then observe
passage behaviour on bars [t, t+M) *using only the fitted grid and band*. Every
metric is computed for three level scores:

  gravity  : g(x) from fit_field
  volume   : log m(x) only (plain volume-profile baseline)
  shuffled : g(x) permuted across valid levels (null)

Hypotheses (see plan): H1 residence top vs bottom quintile, H2 reversal lift,
H3 air-pocket speed, H4 rank correlation with realised residence, H5 field
stability between consecutive windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .bars import Bars
from .grid import PriceGrid
from .passage import episodes, level_stats
from .potential import Field, fit_field

SCORES = ("gravity", "volume", "shuffled")
HYPOTHESES = ("H1", "H2", "H3", "H4", "H5")


@dataclass
class WalkForwardConfig:
    fit_bars: int = 500
    test_bars: int = 100
    step: int | None = None            # default: non-overlapping test windows (= test_bars)
    tau: float | None = None           # default: fit_bars / 2
    band_sigmas: float = 3.0
    min_levels: int = 10               # levels with >=1 test episode needed to form quintiles
    seed: int = 0
    thresholds: dict = field(default_factory=lambda: {
        "H1_ratio": 1.3, "H1_p": 0.01, "H1_frac": 0.70,
        "H2_lift": 1.2, "H2_abs": 0.05,
        "H3_ratio": 1.2, "H3_p": 0.01,
        "H4_rho": 0.15, "H4_frac": 0.70,
        "H5_rho": 0.30,
    })

    def resolved(self, n: int) -> "WalkForwardConfig":
        c = WalkForwardConfig(**{**self.__dict__})
        c.step = self.test_bars if self.step is None else self.step
        c.tau = self.fit_bars / 2 if self.tau is None else self.tau
        return c


def _scores(fld: Field, rng: np.random.Generator) -> dict[str, np.ndarray]:
    g = fld.g.copy()
    valid = np.isfinite(g)
    shuffled = g.copy()
    shuffled[valid] = rng.permutation(g[valid])
    m = fld.table["mass"].to_numpy()
    vol = np.where(valid & (m > 0), np.log(np.where(m > 0, m, 1.0)), np.nan)
    return {"gravity": g, "volume": vol, "shuffled": shuffled}


def _quintile_masks(score: np.ndarray, has_eps: np.ndarray, min_levels: int):
    ok = np.isfinite(score) & has_eps
    if ok.sum() < min_levels:
        return None, None
    q20, q80 = np.nanpercentile(score[ok], [20, 80])
    return ok & (score <= q20), ok & (score >= q80)


@dataclass
class WindowResult:
    window: int
    t: int
    score: str
    h1_ratio: float = np.nan
    h2_top: float = np.nan
    h2_bot: float = np.nan
    h3_ratio: float = np.nan
    h4_rho: float = np.nan
    h5_rho: float = np.nan
    n_levels: int = 0
    n_episodes: int = 0


def _interp_onto(levels_to: np.ndarray, levels_from: np.ndarray, v_from: np.ndarray) -> np.ndarray:
    ok = np.isfinite(v_from)
    if ok.sum() < 2:
        return np.full_like(levels_to, np.nan, dtype=float)
    out = np.interp(levels_to, levels_from[ok], v_from[ok], left=np.nan, right=np.nan)
    return out


def walk_forward(bars: Bars, cfg: WalkForwardConfig | None = None):
    """Returns (per-window DataFrame, pooled episode DataFrame, summary DataFrame)."""
    cfg = (cfg or WalkForwardConfig()).resolved(len(bars))
    rng = np.random.default_rng(cfg.seed)
    N, M, S = cfg.fit_bars, cfg.test_bars, cfg.step
    starts = list(range(N, len(bars) - M + 1, S))
    if len(starts) < 2:
        raise ValueError(f"not enough bars ({len(bars)}) for fit={N} test={M}")

    rows: list[WindowResult] = []
    pooled: list[pd.DataFrame] = []
    prev: dict | None = None  # {"grid": ..., "scores": ..., "results": {score: WindowResult}}

    for wi, t in enumerate(starts):
        fit = bars.slice(t - N, t)
        test = bars.slice(t, t + M)
        grid = PriceGrid.from_bars(fit)
        fld = fit_field(fit, grid, tau=cfg.tau, band_sigmas=cfg.band_sigmas)
        scores = _scores(fld, rng)

        eps = episodes(test, grid, half_width=fld.band_half_width)
        ls = level_stats(eps, grid)
        has_eps = ls["n_episodes"].to_numpy() > 0
        eps_level_idx = eps["level_idx"].to_numpy() if len(eps) else np.array([], dtype=int)

        results: dict[str, WindowResult] = {}
        for name, s in scores.items():
            r = WindowResult(window=wi, t=t, score=name, n_levels=int((np.isfinite(s) & has_eps).sum()),
                             n_episodes=len(eps))
            bot, top = _quintile_masks(s, has_eps, cfg.min_levels)
            if bot is not None and len(eps):
                in_top = top[eps_level_idx]
                in_bot = bot[eps_level_idx]
                e = eps.assign(score=name, window=wi, quintile=np.where(in_top, "top", np.where(in_bot, "bottom", "mid")))
                pooled.append(e)
                rt, rb = eps.loc[in_top, "residence"], eps.loc[in_bot, "residence"]
                if len(rt) and len(rb):
                    r.h1_ratio = float(rt.median() / rb.median())
                    r.h2_top = float(eps.loc[in_top, "reversal"].mean())
                    r.h2_bot = float(eps.loc[in_bot, "reversal"].mean())
                    r.h3_ratio = float(eps.loc[in_bot, "abs_ret"].mean() / eps["abs_ret"].mean())
                ok = np.isfinite(s) & has_eps
                if ok.sum() >= cfg.min_levels:
                    r.h4_rho = float(stats.spearmanr(s[ok], ls["residence_median"].to_numpy()[ok]).statistic)
            results[name] = r

        # H5: stability between consecutive fitted fields (interpolate this field onto previous grid)
        if prev is not None:
            for name in ("gravity", "volume"):
                cur_on_prev = _interp_onto(prev["grid"].levels, grid.levels, scores[name])
                a, b = prev["scores"][name], cur_on_prev
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() >= cfg.min_levels:
                    prev["results"][name].h5_rho = float(stats.spearmanr(a[ok], b[ok]).statistic)
        prev = {"grid": grid, "scores": scores, "results": results}
        rows.extend(results.values())

    per_window = pd.DataFrame([r.__dict__ for r in rows])
    pooled_df = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()
    summary = summarize(per_window, pooled_df, cfg)
    return per_window, pooled_df, summary


def _bootstrap_lift_ci(pooled: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap over windows of the pooled reversal lift top/bottom."""
    rng = np.random.default_rng(seed)
    wins = pooled["window"].unique()
    by = {w: pooled[pooled["window"] == w] for w in wins}
    lifts = []
    for _ in range(n_boot):
        pick = rng.choice(wins, size=len(wins), replace=True)
        d = pd.concat([by[w] for w in pick])
        top, bot = d[d.quintile == "top"], d[d.quintile == "bottom"]
        if len(top) and len(bot) and bot["reversal"].mean() > 0:
            lifts.append(top["reversal"].mean() / bot["reversal"].mean())
    if not lifts:
        return np.nan, np.nan
    return tuple(np.percentile(lifts, [2.5, 97.5]))


def summarize(per_window: pd.DataFrame, pooled: pd.DataFrame, cfg: WalkForwardConfig) -> pd.DataFrame:
    th = cfg.thresholds
    out = []
    for name in SCORES:
        pw = per_window[per_window.score == name]
        pe = pooled[pooled.score == name] if len(pooled) else pooled
        top = pe[pe.quintile == "top"] if len(pe) else pe
        bot = pe[pe.quintile == "bottom"] if len(pe) else pe
        n_win = len(pw)

        # H1
        h1_ratio = float(top["residence"].median() / bot["residence"].median()) if len(top) and len(bot) else np.nan
        h1_p = float(stats.mannwhitneyu(top["residence"], bot["residence"], alternative="greater").pvalue) if len(top) and len(bot) else np.nan
        h1_frac = float((pw["h1_ratio"] > 1).mean()) if n_win else np.nan
        out.append(dict(score=name, hypothesis="H1", metric="residence top/bottom", value=h1_ratio, p=h1_p,
                        frac_windows=h1_frac,
                        passed=bool(h1_ratio >= th["H1_ratio"] and h1_p < th["H1_p"] and h1_frac >= th["H1_frac"])))
        # H2
        h2_top = float(top["reversal"].mean()) if len(top) else np.nan
        h2_bot = float(bot["reversal"].mean()) if len(bot) else np.nan
        lift = h2_top / h2_bot if h2_bot and np.isfinite(h2_bot) and h2_bot > 0 else np.nan
        lo, hi = _bootstrap_lift_ci(pe, seed=cfg.seed) if len(pe) else (np.nan, np.nan)
        out.append(dict(score=name, hypothesis="H2", metric="reversal lift top/bottom", value=lift, p=np.nan,
                        frac_windows=float((pw["h2_top"] > pw["h2_bot"]).mean()) if n_win else np.nan,
                        ci_lo=lo, ci_hi=hi,
                        passed=bool(np.isfinite(lift) and lift >= th["H2_lift"] and (h2_top - h2_bot) >= th["H2_abs"] and lo > 1)))
        # H3
        h3_ratio = float(bot["abs_ret"].mean() / pe["abs_ret"].mean()) if len(bot) else np.nan
        rest = pe[pe.quintile != "bottom"] if len(pe) else pe
        h3_p = float(stats.mannwhitneyu(bot["abs_ret"], rest["abs_ret"], alternative="greater").pvalue) if len(bot) and len(rest) else np.nan
        out.append(dict(score=name, hypothesis="H3", metric="|ret| bottom / all", value=h3_ratio, p=h3_p,
                        frac_windows=float((pw["h3_ratio"] > 1).mean()) if n_win else np.nan,
                        passed=bool(h3_ratio >= th["H3_ratio"] and h3_p < th["H3_p"])))
        # H4
        h4 = pw["h4_rho"].dropna()
        out.append(dict(score=name, hypothesis="H4", metric="spearman(score, residence)", value=float(h4.median()) if len(h4) else np.nan,
                        p=np.nan, frac_windows=float((h4 > 0).mean()) if len(h4) else np.nan,
                        passed=bool(len(h4) and h4.median() >= th["H4_rho"] and (h4 > 0).mean() >= th["H4_frac"])))
        # H5
        h5 = pw["h5_rho"].dropna()
        out.append(dict(score=name, hypothesis="H5", metric="spearman(field_t, field_t+1)", value=float(h5.median()) if len(h5) else np.nan,
                        p=np.nan, frac_windows=float((h5 > 0).mean()) if len(h5) else np.nan,
                        passed=bool(len(h5) and h5.median() >= th["H5_rho"])))
    df = pd.DataFrame(out)
    df.attrs["n_windows"] = int(per_window["window"].nunique()) if len(per_window) else 0
    return df


def verdict(summary: pd.DataFrame) -> dict:
    """Prototype 'works' on this series if gravity passes H1, H4, H5 and beats both nulls on H1 and H4."""
    def val(score, h):
        r = summary[(summary.score == score) & (summary.hypothesis == h)]
        return float(r["value"].iloc[0]) if len(r) else np.nan
    def ok(score, h):
        r = summary[(summary.score == score) & (summary.hypothesis == h)]
        return bool(r["passed"].iloc[0]) if len(r) else False
    beats = all(val("gravity", h) > max(val("volume", h), val("shuffled", h)) for h in ("H1", "H4"))
    core = ok("gravity", "H1") and ok("gravity", "H4") and ok("gravity", "H5")
    return {"H1": ok("gravity", "H1"), "H4": ok("gravity", "H4"), "H5": ok("gravity", "H5"),
            "beats_nulls": beats, "pass": core and beats,
            "H2": ok("gravity", "H2"), "H3": ok("gravity", "H3")}


def format_summary(summary: pd.DataFrame) -> str:
    cols = ["hypothesis", "metric", "score", "value", "p", "frac_windows", "passed"]
    d = summary[cols].copy()
    d["value"] = d["value"].map(lambda v: f"{v:.3f}" if np.isfinite(v) else "-")
    d["p"] = d["p"].map(lambda v: f"{v:.2e}" if np.isfinite(v) else "-")
    d["frac_windows"] = d["frac_windows"].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "-")
    d = d.sort_values(["hypothesis", "score"], key=lambda s: s.map({"gravity": 0, "volume": 1, "shuffled": 2}) if s.name == "score" else s)
    return f"windows: {summary.attrs.get('n_windows', '?')}\n" + d.to_string(index=False)


def split_test(bars: Bars, fit_bars: int, test_bars: int, tau: float | None = None,
               band_sigmas: float = 3.0, min_levels: int = 10, min_n_eff: float = 20.0,
               seed: int = 0) -> dict:
    """Single in-group / out-group split: fit on the first `fit_bars`, test on the next `test_bars`.

    Returns {"field": Field, "grid": PriceGrid, "episodes": DataFrame, "level_stats": DataFrame,
             "metrics": DataFrame(score, H1 ratio/p, H2 lift, H3 ratio/p, H4 rho, n)}.
    """
    if len(bars) < fit_bars + test_bars:
        raise ValueError(f"need {fit_bars + test_bars} bars, have {len(bars)}")
    rng = np.random.default_rng(seed)
    fit = bars.slice(0, fit_bars)
    test = bars.slice(fit_bars, fit_bars + test_bars)
    grid = PriceGrid.from_bars(fit)
    fld = fit_field(fit, grid, tau=fit_bars / 2 if tau is None else tau, band_sigmas=band_sigmas,
                    min_n_eff=min_n_eff)
    scores = _scores(fld, rng)
    eps = episodes(test, grid, half_width=fld.band_half_width)
    ls = level_stats(eps, grid)
    has_eps = ls["n_episodes"].to_numpy() > 0
    idx = eps["level_idx"].to_numpy() if len(eps) else np.array([], dtype=int)
    rows = []
    for name, s in scores.items():
        r = dict(score=name, n_levels=int((np.isfinite(s) & has_eps).sum()), n_episodes=len(eps),
                 h1_ratio=np.nan, h1_p=np.nan, h2_top=np.nan, h2_bot=np.nan, h2_lift=np.nan,
                 h3_ratio=np.nan, h3_p=np.nan, h4_rho=np.nan)
        bot, top = _quintile_masks(s, has_eps, min_levels)
        if bot is not None and len(eps):
            in_top, in_bot = top[idx], bot[idx]
            rt, rb = eps.loc[in_top, "residence"], eps.loc[in_bot, "residence"]
            if len(rt) and len(rb):
                r["h1_ratio"] = float(rt.median() / rb.median())
                r["h1_p"] = float(stats.mannwhitneyu(rt, rb, alternative="greater").pvalue)
                r["h2_top"] = float(eps.loc[in_top, "reversal"].mean())
                r["h2_bot"] = float(eps.loc[in_bot, "reversal"].mean())
                r["h2_lift"] = r["h2_top"] / r["h2_bot"] if r["h2_bot"] > 0 else np.nan
                rest = eps.loc[~in_bot, "abs_ret"]
                r["h3_ratio"] = float(eps.loc[in_bot, "abs_ret"].mean() / eps["abs_ret"].mean())
                r["h3_p"] = float(stats.mannwhitneyu(eps.loc[in_bot, "abs_ret"], rest, alternative="greater").pvalue) if len(rest) else np.nan
            ok = np.isfinite(s) & has_eps
            if ok.sum() >= min_levels:
                r["h4_rho"] = float(stats.spearmanr(s[ok], ls["residence_median"].to_numpy()[ok]).statistic)
        rows.append(r)
    return {"field": fld, "grid": grid, "episodes": eps, "level_stats": ls, "metrics": pd.DataFrame(rows),
            "fit": fit, "test": test}
