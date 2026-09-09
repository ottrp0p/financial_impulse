"""Liquidity screen and slippage model."""
from __future__ import annotations

import numpy as np


def position_caps(adv: np.ndarray, equity: float, participation: float = 0.01) -> np.ndarray:
    """Max |f_i| (fraction of equity) such that the order is <= participation * ADV$."""
    adv = np.asarray(adv, float)
    return np.where(np.isfinite(adv) & (adv > 0), participation * adv / equity, 0.0)


def eligible(adv: np.ndarray, equity: float, participation: float = 0.01, min_adv: float = 5e7,
             min_cap: float = 0.05) -> np.ndarray:
    """A name is tradable if ADV$ >= min_adv and its cap is at least min_cap of equity (a position
    smaller than that is not worth the slot)."""
    caps = position_caps(adv, equity, participation)
    return (np.nan_to_num(adv) >= min_adv) & (caps >= min_cap)


def slippage(order_dollars: np.ndarray, adv: np.ndarray, sigma_daily: np.ndarray,
             half_spread_bp: float = 1.0, impact_coef: float = 0.1) -> np.ndarray:
    """One-way cost in log-return: half-spread + impact_coef * sigma_daily * sqrt(order / ADV$)."""
    part = np.where(np.asarray(adv) > 0, np.abs(order_dollars) / np.asarray(adv), 1.0)
    return half_spread_bp / 1e4 + impact_coef * np.asarray(sigma_daily) * np.sqrt(part)
