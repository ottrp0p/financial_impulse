# growth — intraday Kelly book on the impulse field

Question: can an open→close book sized by fractional continuous Kelly compound 500k → 10M, and at
what tradeoff between expected time and probability of hitting an absolute floor?

## Decision point
Session t: field fitted on bars `[t-120, t)` (closes through t-1) + today's open. Trade open→close,
no intraday stop. Bar low/high are used only to flag days where the marked intraday loss would
exceed `margin_call` (default −25% of equity).

## Pipeline (`python -m growth.run`)
1. `universe.py` — ~150 large caps + 21 ETFs + 8 leveraged ETFs, yfinance daily via the repo's cached source.
2. `features.py` — per session, features at the open in units of the window's per-bar σ:
   * field: `mu_z` (shrunk drift at the open), `g_open`, `z_mass`, `z_restoring`, `z_calm`, `sigma_ratio`, `peak_dist`
   * baseline: `gap_z`, `ret1_z`, `ret5_z`, `oc_prev_z`, `vol_ratio`
   * targets: `oc`, `lo_oc`, `hi_oc` (never features); state: `sigma_bar`, `adv`.
3. `edge.py` — walk-forward pooled ridge of `oc_z` on features (fit 250 / test 20 / step 20, pre-registered
   λ). Feature sets `field | base | both | shuffled` run through the identical code. Skill = OOS Spearman IC,
   decile spread, calibration slope, with bootstrap CIs over windows. `gate()` decides which set (if any) is
   traded: field must beat base and shuffled with IC CI > 0 in ≥ 70% of windows.
4. `sizing.py` — `f = k Σ⁻¹ μ`, Σ = Ledoit-Wolf of open→close returns (last 120 sessions). This is the
   correlation safeguard: correlated names share risk budget. `n_eff = (Σ|f_i|σ_i)² / fᵀΣf` reports the
   effective number of independent bets.
5. `liquidity.py` — cap `|f_i| ≤ participation·ADV$/equity`; eligible if ADV$ ≥ 50M and cap ≥ 5% of
   equity. Slippage = half-spread + 0.1·σ_daily·√(order/ADV$), paid on entry and exit.
6. `book.py` — daily loop: screen → `|μ̂/σ̂| ≥ τ` → top-N → Kelly → fill → compound.
   `fixed_equity=True` (used in the sweep) removes path dependence of caps; the chosen point is re-run
   path-dependent so the universe shrinking with equity is visible.
7. `frontier.py` — analytic two-barrier drifted-BM (P(floor), E[T]) and a block bootstrap of realised
   *whole days* (keeps within-day correlation and fat tails). Sweep over τ × k × N × L × floor.

Development / held-out split: names A–M develop, N–Z confirm; 2022 sub-period reported separately.

## Outputs (`out/growth/`)
`features.parquet`, `skill_dev.csv`, `skill_hold.csv`, `sweep.csv`, `chosen.json`, `summary.md`,
`chosen_path_daily.csv`.

## Tests
`.venv/bin/python -m pytest growth/tests -q`
