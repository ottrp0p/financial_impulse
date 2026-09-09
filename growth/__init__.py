"""growth: intraday (open->close) growth-optimal book on top of the impulse field.

Decision point per session t: field fitted on bars [t-fit, t) (closes through t-1) + today's open.
Trade: fill at open (+slippage), flat at close (-slippage). Sizing: fractional continuous Kelly
f = k * Sigma^-1 mu with Ledoit-Wolf covariance (the correlation safeguard), clipped to a gross
leverage cap and per-name liquidity caps. Output: an E[time to target] vs P(floor) frontier.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
