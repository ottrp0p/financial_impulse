"""Time-to-target vs probability-of-floor: analytic (drifted Brownian, two barriers) and block bootstrap."""
from __future__ import annotations

import numpy as np


def analytic(G: float, V: float, E0: float, target: float, floor: float) -> dict:
    """Log-equity ~ BM with per-day drift G and variance V, absorbing at +a (target) and -b (floor).
    Scale function s(x) = e^{-2Gx/V}: P(target first) = (s(0) - s(-b)) / (s(a) - s(-b));
    E[T] = (a P_t - b P_r) / G."""
    a, b = np.log(target / E0), np.log(E0 / floor)
    if V <= 0:
        return {"p_ruin": float(G <= 0), "p_target": float(G > 0), "e_days": a / G if G > 0 else np.inf}
    if abs(G) < 1e-12:
        p_r = a / (a + b)
        return {"p_ruin": p_r, "p_target": 1 - p_r, "e_days": a * b / V}
    th = 2 * G / V
    p_t = (1 - np.exp(th * b)) / (np.exp(-th * a) - np.exp(th * b))
    p_t = float(np.clip(p_t, 0, 1))
    p_r = 1 - p_t
    return {"p_ruin": p_r, "p_target": p_t, "e_days": float((a * p_t - b * p_r) / G)}


def bootstrap(daily_ret: np.ndarray, E0: float, target: float, floor: float, n_paths: int = 5000,
              block: int = 5, max_days: int = 2520, seed: int = 0) -> dict:
    """Stationary block bootstrap of realised daily simple returns (whole days, so within-day
    correlation and fat tails survive). First passage to target or floor within max_days."""
    r = np.asarray(daily_ret, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10:
        return {"p_ruin": np.nan, "p_target": np.nan, "med_days": np.nan, "mean_days": np.nan, "p_neither": np.nan}
    rng = np.random.default_rng(seed)
    logr = np.log1p(r)
    a, b = np.log(target / E0), np.log(E0 / floor)
    hit_t = np.full(n_paths, np.inf)
    hit_r = np.full(n_paths, np.inf)
    chunk = 500
    for s in range(0, n_paths, chunk):
        m = min(chunk, n_paths - s)
        nb = int(np.ceil(max_days / block))
        starts = rng.integers(0, n, size=(m, nb))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(m, -1)[:, :max_days] % n
        cum = np.cumsum(logr[idx], axis=1)
        up = cum >= a
        dn = cum <= -b
        t_up = np.where(up.any(1), up.argmax(1), np.inf)
        t_dn = np.where(dn.any(1), dn.argmax(1), np.inf)
        hit_t[s:s + m] = t_up
        hit_r[s:s + m] = t_dn
    ruin = hit_r < hit_t
    targ = hit_t < hit_r
    days = hit_t[targ]
    return {"p_ruin": float(ruin.mean()), "p_target": float(targ.mean()),
            "p_neither": float(1 - ruin.mean() - targ.mean()),
            "med_days": float(np.median(days)) if len(days) else np.nan,
            "mean_days": float(days.mean()) if len(days) else np.nan}
