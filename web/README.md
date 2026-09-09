# Impulse Screener (local, single user)

Portfolio screener on top of `impulse`: add names to a long and a short book, pick a strategy per side,
**Generate**, then each day **Refresh** to get the incremental actions and the resulting tracked position.

## Run

```bash
uv pip install -e ".[dev]"            # once (adds fastapi/uvicorn)
./web/run.sh                          # → http://127.0.0.1:8765
```

Data lives in `web/data/` (gitignored): `prices/*.parquet`, `forecasts/*.parquet`, `screens/*.json`.

## How it works

- **Store** (`backend/store.py`): delta‑fetches prices from yfinance, then fits the field at every close that
  doesn't have a forecast yet and appends the 10‑session band (field and Brownian) — append‑only, so a daily
  refresh costs one fit per name.
- **Ledgers** (`backend/ledger.py`): the same stack machine (`impulse.strategy.run`) produces
  - *backtest*: full history, always running, forced close at the end;
  - *live*: starts **flat** at the generate date (or the name's add date), never force‑closes today.
- **Screens** (`backend/screens.py`): an append‑only event log (add / drop / params / generate). Dropping a name
  ends its window at the drop date → liquidation in both ledgers. The live book is a deterministic replay of
  `(prices, params, events)`, so there is no mutable position state to corrupt.
- **Today** = trades the live ledger opened or closed on the last bar, plus the open stack per name.

## Strategy presets

| preset | side | rule |
|---|---|---|
| `long-decel` | long | band reversion, decel ×⅔ stack, `1.0 / 1.5 / 1.0`, no timeout |
| `long-fixed` | long | same entries/exits, fixed 1‑unit stack |
| `short-strict` | short | mirror reversion, 2σ entries, decel ×½, 40‑session timeout |
| `short-capit` | short | mirror reversion, fixed units, capitulate at −2.5u |
| `short-confirm` | short | **confirmation short**: break >1.5σ above the band → first close back below the centre → 0.25u; each 0.75σ lower adds ×⅔; cover when z > 0.25 (`kind: confirm`, `src/impulse/strategy_confirm.py`) |

Any preset can be edited field by field ("custom"); changes apply forward‑only on the live book, and open units are carried across a change — including a change between rule kinds.

## Daily use

1. After the close, hit **Refresh** (or `POST /api/screens/{id}/refresh`; wire it to cron if you like).
2. **Today** tab: actions (BUY/SELL/SHORT/COVER, size in units and dollars, reason) and positions.
3. **Book** tab: backtest (muted) vs live (bold) exposure and equity on the same axes; per‑name table.
4. **Name** tab: price + band + trades (hollow = backtest, filled = live), capital and equity strips.

Notes: yfinance may return a partial bar for the current session before the close — refresh after 16:00 ET.
Units are notional: 1 unit = one full‑size first entry; set the dollar value per unit on the Screen tab.
Costs 1 bp/side, no borrow cost. Changing strategy params applies to the whole history (backtest) and from
inception (live) — it does not liquidate.
