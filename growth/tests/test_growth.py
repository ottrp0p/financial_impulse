import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from growth.frontier import analytic, bootstrap  # noqa: E402
from growth.liquidity import position_caps, slippage  # noqa: E402
from growth.sizing import kelly_weights, ledoit_wolf, n_eff  # noqa: E402


def test_kelly_single_asset_is_mu_over_var():
    f = kelly_weights(np.array([0.002]), np.array([[0.01**2]]), k=1.0, L_max=100)
    assert np.isclose(f[0], 0.002 / 0.01**2)


def test_kelly_correlated_pair_shares_budget():
    s = 0.01
    ind = np.array([[s * s, 0], [0, s * s]])
    corr = np.array([[s * s, 0.999 * s * s], [0.999 * s * s, s * s]])
    mu = np.array([0.001, 0.001])
    fi = kelly_weights(mu, ind, 1.0, 1e9)
    fc = kelly_weights(mu, corr, 1.0, 1e9)
    assert np.isclose(fc.sum(), fi.sum() / 2 * 2 / 1.999, rtol=0.05)  # total ~ halves when perfectly correlated
    assert n_eff(fi, ind) > 1.9 and n_eff(fc, corr) < 1.05


def test_kelly_caps_and_gross():
    f = kelly_weights(np.array([0.01, 0.01, -0.01]), np.eye(3) * 1e-4, 1.0, L_max=5.0, caps=np.array([2.0, 2.0, 0.5]))
    assert np.abs(f).sum() <= 5.0 + 1e-9 and abs(f[2]) <= 0.5


def test_ledoit_wolf_is_psd_and_shrinks():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 10))
    S = ledoit_wolf(X)
    assert np.linalg.eigvalsh(S).min() > 0
    assert np.abs(S - np.diag(np.diag(S))).sum() < np.abs(np.cov(X.T, bias=True) - np.diag(np.diag(np.cov(X.T, bias=True)))).sum()


def test_analytic_matches_bootstrap_on_gaussian():
    rng = np.random.default_rng(1)
    G, V = 0.003, 0.02**2
    r = np.expm1(G + np.sqrt(V) * rng.standard_normal(20000))
    a = analytic(G, V, 500e3, 10e6, 250e3)
    b = bootstrap(r, 500e3, 10e6, 250e3, n_paths=4000, block=1, max_days=20000, seed=2)
    assert abs(a["p_ruin"] - b["p_ruin"]) < 0.03
    assert abs(a["e_days"] - b["mean_days"]) / a["e_days"] < 0.15


def test_liquidity_cap_shrinks_with_equity():
    adv = np.array([1e8, 1e9])
    c1, c2 = position_caps(adv, 5e5), position_caps(adv, 5e6)
    assert (c2 < c1).all() and c1[1] > c1[0]
    assert slippage(np.array([1e6]), np.array([1e8]), np.array([0.02]))[0] > slippage(np.array([1e4]), np.array([1e8]), np.array([0.02]))[0]
