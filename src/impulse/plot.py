"""Field plot: price history on the left, level-indexed panels on the right sharing the log-price axis."""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .bars import Bars
from .potential import Field

# reference palette (light mode): single-hue panels, text in ink tokens
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

PANELS = [("mass", "m(x)", BLUE), ("sigma", "σ(x)", BLUE), ("mu", "μ(x)", BLUE),
          ("U", "U(x)", BLUE), ("g", "g(x)", ORANGE)]


def plot_field(bars: Bars, fld: Field, title: str | None = None, n_price_bars: int | None = None):
    t = fld.table
    y = fld.grid.levels
    fig = plt.figure(figsize=(16, 7), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 1 + len(PANELS), width_ratios=[4] + [1] * len(PANELS), wspace=0.08)
    ax0 = fig.add_subplot(gs[0, 0])
    n = len(bars) if n_price_bars is None else min(n_price_bars, len(bars))
    ax0.plot(np.arange(n), bars.log_close[-n:], color=INK, lw=1.0)
    ax0.set_xlabel("bar", color=INK2)
    ax0.set_ylabel("log price", color=INK2)
    valid = t["valid"].to_numpy()
    g = t["g"].to_numpy()
    # shade the top-quintile gravity levels across the price panel
    if np.isfinite(g).any():
        q80 = np.nanpercentile(g[valid], 80)
        for lv in y[valid & (g >= q80)]:
            ax0.axhspan(lv - fld.grid.dx / 2, lv + fld.grid.dx / 2, color=ORANGE, alpha=0.12, lw=0)
    axes = [ax0]
    for i, (col, label, color) in enumerate(PANELS, start=1):
        ax = fig.add_subplot(gs[0, i], sharey=ax0)
        v = t[col].to_numpy().astype(float)
        v[~valid] = np.nan
        ax.plot(v, y, color=color, lw=2)
        if col == "g":
            ax.axvline(0, color=GRID, lw=1)
            ax.axvline(2, color=INK2, lw=1, ls=":")
        if col == "mu":
            ax.axvline(0, color=GRID, lw=1)
        ax.set_title(label, color=INK, fontsize=11)
        ax.tick_params(labelleft=False)
        axes.append(ax)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2)
    ax0.set_ylim(np.nanmin(y[valid]) if valid.any() else y[0], np.nanmax(y[valid]) if valid.any() else y[-1])
    fig.suptitle(title or f"{bars.symbol} {bars.interval} — gravity field (last {len(bars)} bars, band ±{fld.band_half_width:.4f})",
                 color=INK, fontsize=13)
    return fig
