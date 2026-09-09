# financial_impulse — market "gravity" from OHLCV

Prototype of an automatic-TA idea: price levels differ in how easily price moves
through them. Some levels are *sticky* (price lingers, reverses), others are *air
pockets* (price traverses fast). This package estimates a **level-indexed field**
from OHLCV bars and tests, out of sample, whether that field predicts how price
behaves at those levels later.

## Physics ↔ finance mapping

| Field | Estimator (all in log-price) | Analogy | Module |
|---|---|---|---|
| mass `m(x)` | decayed volume-at-price density, kernel-smoothed | mass distribution | `mass.py` |
| drift `μ(x)` | Nadaraya–Watson kernel of 1-bar forward return on level | force = −∇U | `dynamics.py` |
| local vol `σ(x)` | kernel second moment at level | local metric ("how fast time runs") | `dynamics.py` |
| potential `U(x)` | `−∫ μ dx`; `p_stat ∝ exp(−2U/σ²)` as self-check vs `m` | gravity well | `potential.py` |
| residence | episode lengths inside band `x ± h` | binding energy | `passage.py` |
| gravity `g(x)` | weighted mean of z-scores: `0.5·mass + calm + restoring + residence` | — | `potential.py` |

`restoring = −dμ/dx` (drift pointing back toward the level). Weights are explicit
in `DEFAULT_WEIGHTS` — the score is a hypothesis, not a fit.

Literature: Market Profile (Steidlmayer), Bouchaud & Cont Langevin model, Florens-Zmirou /
Bandi–Phillips nonparametric diffusion estimation, first-passage processes, latent-order-book /
square-root impact (Donier et al.).

## Data interface

`impulse.bars.Bars` is the only thing estimators consume: UTC `ts` index, float
`open/high/low/close/volume`, validated. `BarSource.fetch(symbol, interval, start, end) -> Bars`
is the whole connector contract; `sources/` holds `yfinance` (parquet-cached) and `csv`.
Adding a broker = one file + one registry line in `sources/__init__.py`.

## Run

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest                         # Layer 1: synthetic recovery tests
.venv/bin/python scripts/run_demo.py --symbol SPY --interval 1h   # field plot + walk-forward table
.venv/bin/python scripts/run_eval.py               # Layer 2 grid -> out/summary.md (GO/NO-GO)
```

## Verification layers

1. **Synthetic** (`tests/test_estimators.py`): OU well location and drift slope; double-well minima
   and residence; random-walk false-positive control; volume spike raises mass but not `g`.
2. **Walk-forward** (`evaluate.py`): fit on `[t−N, t)`, observe `[t, t+M)`. Each hypothesis is
   computed for `gravity`, a `volume`-only baseline, and a `shuffled` null. H1 residence ratio,
   H2 reversal lift, H3 air-pocket speed, H4 rank correlation, H5 field stability. Pass rule and
   thresholds are in `WalkForwardConfig.thresholds` and `verdict()`.
3. **Trading gate**: not built until Layer 2 passes.

Fixed before running the real grid (no tuning on it): Silverman bandwidth, `tau = fit/2`,
passage band `= max(bandwidth, 3·σ_bar)`, windows `(500,100)` for 1h and `(250,60)` for 1d.

## Findings so far (see `out/summary.md`)

- **Passage bands must be several per-bar σ wide.** With a band ≈ one bar's move, every episode
  lasts ~1 bar and drift is invisible (synthetic double-well ratio ≈1.4 → ≈2–4 at 3σ). This is
  the single biggest measurement lesson: stickiness is only observable at a resolution coarser
  than the noise.
- **H5 is a soft bar** when consecutive fit windows overlap (they share `1 − M/N` of their bars):
  even a random walk shows ρ≈0.6. Treat it as a floor, not evidence.
- On synthetic wells the plain volume profile is the *best* predictor (occupancy = residence by
  construction), so beating it on real data is the real test of whether the dynamics add anything.

## Upgrade path

OHLCV lacks order flow, so "stiffness" (Δprice per Δvolume) is only proxied via `m(x)`. With tick
or L2 data, add a `TickSource`/`BookSource` that aggregates to `Bars` and a direct impact-based
stiffness estimator; nothing downstream changes.
