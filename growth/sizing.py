"""Fractional continuous Kelly with a shrunk covariance and gross/per-name caps."""
from __future__ import annotations

import numpy as np


def ledoit_wolf(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward scaled identity. X: (n, p)."""
    X = np.asarray(X, float)
    n, p = X.shape
    if n < 2:
        return np.eye(p) * (X.var(0).mean() if n else 1.0)
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / n
    mu = np.trace(S) / p
    F = mu * np.eye(p)
    d2 = ((S - F) ** 2).sum()
    b2 = sum(((np.outer(x, x) - S) ** 2).sum() for x in Xc) / n**2
    rho = min(b2, d2) / d2 if d2 > 0 else 1.0
    return rho * F + (1 - rho) * S


def kelly_weights(mu: np.ndarray, Sigma: np.ndarray, k: float = 0.5, L_max: float = 5.0,
                  caps: np.ndarray | None = None, iters: int = 20) -> np.ndarray:
    """f = k * Sigma^-1 mu, then alternately clip |f_i| <= caps_i and scale gross to <= L_max."""
    mu = np.asarray(mu, float)
    p = len(mu)
    if p == 0:
        return mu
    S = np.asarray(Sigma, float) + 1e-10 * np.eye(p)
    f = k * np.linalg.solve(S, mu)
    caps = np.full(p, np.inf) if caps is None else np.asarray(caps, float)
    for _ in range(iters):
        f = np.clip(f, -caps, caps)
        g = np.abs(f).sum()
        if g > L_max:
            f *= L_max / g
        else:
            break
    return f


def growth_rate(f: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
    return float(f @ mu - 0.5 * f @ Sigma @ f)


def n_eff(f: np.ndarray, Sigma: np.ndarray) -> float:
    """Effective number of independent bets: (sum |f_i| sigma_i)^2 / f'Sigma f."""
    if len(f) == 0:
        return 0.0
    s = np.sqrt(np.diag(Sigma))
    v = float(f @ Sigma @ f)
    return float((np.abs(f) * s).sum() ** 2 / v) if v > 0 else 0.0
