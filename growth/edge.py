"""Walk-forward conditional edge: pooled ridge of oc_z on features, out-of-sample skill.

Pre-registered: fit 250 sessions, test 20, step 20, ridge lambda = 0.01 * n_fit on standardised
features, mu shrink by n_fit/(n_fit + 500). Feature sets: field | base | both | shuffled (both, with
the fit-window target permuted -> pure-noise null).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .features import FEATURE_SETS


@dataclass
class EdgeConfig:
    fit_sessions: int = 250
    test_sessions: int = 20
    step: int = 20
    ridge: float = 0.01
    shrink_n0: float = 500.0
    seed: int = 0
    feature_sets: dict = field(default_factory=lambda: dict(FEATURE_SETS))


def _ridge(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y), float(y.mean())


def fit_predict(panel: pd.DataFrame, cfg: EdgeConfig | None = None, feature_set: str = "both") -> pd.DataFrame:
    """Returns panel rows in test windows with mu_hat (log), sigma_hat (log), mu_z_hat, window id."""
    cfg = cfg or EdgeConfig()
    rng = np.random.default_rng(cfg.seed)
    cols = cfg.feature_sets["both"] if feature_set == "shuffled" else cfg.feature_sets[feature_set]
    dates = np.array(sorted(panel["ts"].unique()))
    d_idx = pd.Series(np.arange(len(dates)), index=dates)
    di = d_idx.loc[panel["ts"]].to_numpy()
    X_all = panel[cols].to_numpy(float)
    y_all = panel["oc_z"].to_numpy(float)
    out = []
    w = 0
    for start in range(cfg.fit_sessions, len(dates) - 1, cfg.step):
        fit = (di >= start - cfg.fit_sessions) & (di < start)
        test = (di >= start) & (di < start + cfg.test_sessions)
        if fit.sum() < 50 or test.sum() == 0:
            continue
        Xf, yf = X_all[fit], y_all[fit]
        ok = np.isfinite(Xf).all(1) & np.isfinite(yf)
        Xf, yf = Xf[ok], yf[ok]
        if feature_set == "shuffled":
            yf = rng.permutation(yf)
        m, sd = Xf.mean(0), Xf.std(0) + 1e-12
        Z = (Xf - m) / sd
        n = len(yf)
        beta, y0 = _ridge(Z, yf - yf.mean(), cfg.ridge * n)
        resid = yf - y0 - Z @ beta
        rsd = float(resid.std())
        shrink = n / (n + cfg.shrink_n0)
        Xt = (X_all[test] - m) / sd
        Xt = np.nan_to_num(Xt)
        mu_z = shrink * (y0 + Xt @ beta)
        sub = panel.loc[test, ["ts", "symbol", "oc", "oc_z", "sigma_bar", "adv", "open", "lo_oc", "hi_oc"]].copy()
        sub["mu_z_hat"] = mu_z
        sub["mu_hat"] = mu_z * sub["sigma_bar"]
        sub["sigma_hat"] = rsd * sub["sigma_bar"]
        sub["window"] = w
        out.append(sub)
        w += 1
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def window_skill(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w, d in pred.groupby("window"):
        if len(d) < 30 or d["mu_z_hat"].std() == 0:
            continue
        ic = stats.spearmanr(d["mu_z_hat"], d["oc_z"]).statistic
        q = pd.qcut(d["mu_z_hat"].rank(method="first"), 10, labels=False)
        spread = d["oc_z"][q == 9].mean() - d["oc_z"][q == 0].mean()
        slope = np.polyfit(d["mu_z_hat"], d["oc_z"], 1)[0]
        # signed-edge P&L in sigma units: sign(mu_hat) * oc_z, mean per row
        rows.append({"window": w, "n": len(d), "ic": ic, "decile_spread": spread, "calib_slope": slope,
                     "sign_pnl_z": float((np.sign(d["mu_z_hat"]) * d["oc_z"]).mean())})
    return pd.DataFrame(rows)


def summarize_skill(ws: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {"n_windows": len(ws)}
    for c in ["ic", "decile_spread", "calib_slope", "sign_pnl_z"]:
        v = ws[c].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
        out[c] = float(v.mean())
        out[f"{c}_lo"], out[f"{c}_hi"] = (float(q) for q in np.percentile(boots, [2.5, 97.5]))
        out[f"{c}_frac_pos"] = float((v > 0).mean())
    return out


def skill_table(panel: pd.DataFrame, cfg: EdgeConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Run every feature set through the identical pipeline. Returns the table and the per-set preds."""
    cfg = cfg or EdgeConfig()
    rows, preds = [], {}
    for fs in ["field", "base", "both", "shuffled"]:
        p = fit_predict(panel, cfg, fs)
        preds[fs] = p
        s = summarize_skill(window_skill(p)) if len(p) else {}
        rows.append({"feature_set": fs, **s})
    return pd.DataFrame(rows).set_index("feature_set"), preds


def gate(table: pd.DataFrame, min_frac: float = 0.70) -> tuple[str, str]:
    """Which feature set to trade, and why. Field must beat base and shuffled on mean IC with a CI
    excluding 0 and IC>0 in >= min_frac of windows; else fall back to the best surviving set."""
    def ok(fs):
        r = table.loc[fs]
        return r.get("ic_lo", -1) > 0 and r.get("ic_frac_pos", 0) >= min_frac
    ic = table["ic"]
    if ok("field") and ic["field"] > max(ic["base"], ic["shuffled"]):
        return "field", "field beats base+shuffled on OOS IC with CI>0"
    if ok("both") and ic["both"] > max(ic["base"], ic["shuffled"]):
        return "both", "field adds to base (both > base) with CI>0"
    if ok("base"):
        return "base", "field does not survive; trading baseline (gap/reversal) features only"
    return "none", "no feature set has OOS IC with CI>0 in >=70% of windows"
