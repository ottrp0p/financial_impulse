"""Layer 1: recover known fields from synthetic processes."""
import numpy as np

from impulse import synthetic
from impulse.grid import PriceGrid
from impulse.potential import fit_field
from impulse.passage import episodes, level_stats


def _slope(x, y, w):
    ok = np.isfinite(y) & (w > 0)
    return np.polyfit(x[ok], y[ok], 1, w=np.sqrt(w[ok]))[0]


def test_ou_recovers_well_and_drift_slope():
    k, s_star = 0.05, 0.0
    bars = synthetic.ou(n_bars=20000, k=k, s_star=s_star, sigma=0.02, seed=1)
    f = fit_field(bars)
    t, g = f.table, f.grid
    valid = t["valid"].to_numpy()
    U = t["U"].to_numpy().copy()
    U[~valid] = np.inf
    well = g.levels[np.argmin(U)]
    assert abs(well - s_star) <= g.bandwidth, (well, g.bandwidth)
    # regress mu(x) on x over the well-populated region (within 1.5 sd of s_star)
    sd = 0.02 / np.sqrt(2 * k)
    core = valid & (np.abs(g.levels - s_star) < 1.5 * sd)
    slope = _slope(g.levels[core], t["mu"].to_numpy()[core], t["n_eff"].to_numpy()[core])
    assert abs(slope - (-k)) <= 0.25 * k, slope


def test_double_well_two_minima_and_residence():
    width = 0.1
    bars = synthetic.double_well(n_bars=20000, a=2e-4, width=width, sigma=0.01, seed=2)
    f = fit_field(bars)
    t, g = f.table, f.grid
    valid = t["valid"].to_numpy()
    U = np.where(valid, t["U"].to_numpy(), np.nan)
    L = g.levels
    # local minima of the (smoothed) potential on the valid region
    mins = [i for i in range(1, g.n - 1)
            if np.isfinite(U[i - 1:i + 2]).all() and U[i] < U[i - 1] and U[i] <= U[i + 1]]
    found_pos = any(abs(L[i] - width) <= g.bandwidth for i in mins)
    found_neg = any(abs(L[i] + width) <= g.bandwidth for i in mins)
    assert found_pos and found_neg, [round(L[i], 3) for i in mins]
    # residence in wells >= 2x residence at the barrier (after normalising by sigma)
    ls = level_stats(episodes(bars, g, half_width=f.band_half_width), g)
    res = ls["residence_median"].to_numpy() * t["sigma"].to_numpy()
    at = lambda x: res[g.index_of(np.array([x]))[0]]
    assert min(at(width), at(-width)) >= 2 * at(0.0), (at(width), at(-width), at(0.0))


def test_random_walk_has_no_significant_levels():
    bars = synthetic.random_walk(n_bars=6000, seed=3)
    f = fit_field(bars)
    gz = f.g[f.valid]
    assert np.mean(np.abs(gz) > 2) <= 0.05


def test_volume_spike_raises_mass_but_not_gravity():
    x0 = 0.0
    bars = synthetic.random_walk_with_volume_spike(n_bars=6000, x0=x0, seed=4)
    f = fit_field(bars)
    t, g = f.table, f.grid
    i = g.index_of(np.array([x0]))[0]
    rank = (t["mass"].to_numpy() >= t["mass"].to_numpy()[i]).sum()
    assert rank <= 3, "mass at x0 should be among the top levels"
    assert abs(g.levels[np.nanargmax(t["z_mass"].to_numpy())] - x0) <= g.bandwidth
    assert not (t["g"].iloc[i] > 2), "gravity must not be significant from volume alone"
