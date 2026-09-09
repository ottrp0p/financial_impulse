# Impulse Screener — plan for a live portfolio website

Turn the research stack in `src/impulse/` into a web app: pick names for a long book and a short book, pick a
strategy variant for each side, hit **Generate**, and get (a) the historical playback you've been looking at in
the artifacts, and (b) a daily recommendation feed for a *tracked* portfolio that starts on the day you generate
it and only ever asks you for incremental actions.

Everything below is a recommendation; the sections marked **decision** are the ones worth pushing back on.

---

## 1. What the product does

1. **Screen** — two lists (long / short). Add or remove tickers. Each side has a strategy preset (or custom params).
2. **Generate** — pulls prices, computes the field, runs the rule, produces:
   - *Backtest* view: the full‑history playback (what the rule *would* have done), with the charts we already built.
   - *Live book*: a tracked portfolio that begins **flat** on the generate date.
3. **Daily update** — each session after the close (or on demand), the app recomputes and shows:
   - **Incremental actions for today**: `BUY AAPL 0.67u`, `COVER XOM all (armed → normalised)`, `CAPITULATE CAT`.
   - **Resulting position**: per‑name units, side, stack age, unrealised P&L; book‑level gross/net.
4. **Drop a name** — removes it from the screen, force‑liquidates its tracked position at the next close, and it
   stops generating signals. History is kept (the liquidation is a real trade in the ledger).
5. **Progress** — equity, exposure and trade history *since inception*, drawn on the same axes as the backtest but
   visually distinct (see §5).

Key semantic: **the backtest assumes it was always running; the live book assumes nothing before inception.** On
day 1 the backtest might be carrying a 2.6‑unit AAPL stack; the live book has zero and will only buy if the rule
fires *today*. The two diverge, on purpose, and converge naturally once the backtest's stack turns over.

---

## 2. Architecture

```
┌──────────────┐   ┌───────────────────┐   ┌────────────────────┐   ┌───────────────────┐
│  yfinance    │ → │  Price store       │ → │  Compute (masks)   │ → │  Ledgers           │
│  (fetch only │   │  parquet / duckdb  │   │  field → band →    │   │  backtest ledger   │
│   the delta) │   │  one row per bar   │   │  signal → intent   │   │  live ledger       │
└──────────────┘   └───────────────────┘   └────────────────────┘   └───────────────────┘
                                                      │                        │
                                              ┌───────▼────────────────────────▼────┐
                                              │  API (FastAPI)  →  UI (React)       │
                                              └─────────────────────────────────────┘
```

**Stack (decision):**
- Backend **Python / FastAPI**, reusing `impulse.*` directly. No rewrite of the estimators.
- Storage **DuckDB over parquet** (single file, columnar, zero ops). Postgres only if multi‑user later.
- Frontend **React + TypeScript**, charts in **uPlot** (canvas, handles 50 names × 250 days × 5 series without
  breaking a sweat; the SVG we used in artifacts won't). Tables with TanStack Table.
- Scheduler: one **cron / APScheduler** job at 16:30 ET that runs the daily update; manual "Refresh now" button
  calls the same function.
- Deploy: single Docker container, volume for the parquet store. Fly.io / Railway / a VPS all fine.

---

## 3. Data layer — masks and overlays

The compute layer is a chain of **aligned 2‑D arrays** indexed `[date, symbol]`. Every stage is a pure function
of the previous one; nothing is stored per‑trade until the ledger. This is what keeps it fast and makes
"drop a name", "change a param", "replay from a date" all trivial (you mask, you don't recompute).

### 3.1 Price panel
```
P  : float[D, N]   log close        (D dates × N symbols; NaN before a symbol's first bar)
H, L, V: same shape (high, low, volume) — only the field needs them
```
Stored as parquet per symbol; loaded into one panel by the API. Fetch is **delta‑only**: last stored date → today.

### 3.2 Field panel (rolling fit, cached)
For each `(date, symbol)` the rolling 120‑bar fit produces the h‑step forecast quantiles. Store only what the
rule needs:
```
Q10, Q50, Q90 : float[D, N]   forecast band targeting each date (issued h sessions earlier)
MU_AT, G_AT   : float[D, N]   drift and g(x) at the close's level (for the trend gate, later)
```
Rolling fits are the only expensive step (~0.1 s per date per symbol). They are **append‑only**: a new day adds
one column of fits; nothing historical changes. Cache key = `(symbol, fit_bars, h, band_sigmas, shrink, sigma_mode)`.
Changing a *strategy* param (z_in, sizing, …) does **not** invalidate this cache — only field params do.

### 3.3 Signal masks (cheap, recomputed on every param change)
```
Z        : float[D, N]   (P − Q50) / (Q90 − Q10) × 2.563
ENTRY    : bool [D, N]   side·Z < −z_in
ARM      : bool [D, N]   side·Z > +z_out
NORM     : bool [D, N]   |Z| < z_norm
```
Three boolean panels fully describe the rule's *intent*. They are ~100 KB for 50 names × a year; regenerate in
milliseconds.

### 3.4 Position panel = the stack machine applied to the masks
The state machine in `impulse.strategy.run` (stack depth, decel size, armed flag, timeout, capitulation, cap)
becomes a vectorised‑over‑symbols pass over dates:
```
UNITS    : float[D, N]   open notional per (date, symbol)
INTENT   : float[D, N]   signed change in units decided at that close (+buy / −sell), before overlays
TRADES   : long table    entry_date, exit_date, symbol, size, reason ∈ {armed, timeout, capitulate, drop, end}
```
Per‑symbol state is tiny (open stack list), so a Python loop over dates with numpy across symbols is fine —
253 × 50 is nothing. If it ever matters, numba.

### 3.5 Overlays (this is where backtest and live differ)
An **overlay** is a `[D, N]` mask or factor applied to `INTENT` before it becomes a ledger entry:
```
ACTIVE   : bool [D, N]   symbol is in the screen on that date (drop → False from drop date)
LIVE     : bool [D, N]   date ≥ inception (per side; long and short books can start on different days)
SIZE     : float[N]      per‑name unit scale (default 1; later: vol‑targeted)
```
- **Backtest ledger** = stack machine on `INTENT ⊙ ACTIVE` with an initial state of "always running".
- **Live ledger**    = stack machine on `INTENT ⊙ ACTIVE ⊙ LIVE` with an initial state of **flat at inception**.
  Because the machine is re‑run from inception each day with the same deterministic inputs, the live ledger is
  reproducible from `(prices, params, screen history)` — no mutable position state to corrupt. The only thing
  persisted about the live book is the **event log**: adds, drops, param changes, manual overrides, each with a
  date. Replay = fold the log.
- A dropped name gets a synthetic `DROP` exit at the next close in both ledgers (backtest as if it had been
  screened out on that date; live as an actual liquidation).

**Today's recommendation** is simply `INTENT[today] ⊙ ACTIVE ⊙ LIVE` for the live ledger, rendered as actions,
plus `UNITS[today]` as the resulting position. If the user has *not* executed yesterday's recommendation there
is no drift to reconcile because the ledger is paper; if you later want a "what I actually did" layer, that is
one more overlay (`FILLS`), not a redesign.

### 3.6 Invalidation table
| change | field cache | signal masks | backtest ledger | live ledger |
|---|---|---|---|---|
| new day of prices | append 1 col | recompute | recompute (fast) | replay log |
| add name | fit that name's history | recompute | recompute | starts at add date |
| drop name | nothing | nothing | DROP exit | DROP exit |
| strategy param | nothing | recompute | recompute | **new version** (see §6) |
| field param | full refit | recompute | recompute | new version |

---

## 4. API surface (FastAPI)

```
GET  /screens                      list saved screens
POST /screens                      {name}
GET  /screens/{id}                 names per side, strategy per side, inception dates, event log
POST /screens/{id}/names           {side, symbols[]}          → event ADD
DELETE /screens/{id}/names/{sym}   → event DROP (liquidation at next close)
PUT  /screens/{id}/strategy        {side, preset | params}     → event PARAMS (creates a live version)
POST /screens/{id}/generate        run everything; returns job id (async) — first run only
POST /screens/{id}/refresh         daily update (delta fetch + recompute), idempotent
GET  /screens/{id}/today           {actions[], positions[], book: {gross_long, gross_short, net}}
GET  /screens/{id}/backtest        summary + per‑name stats + series (paginated by symbol)
GET  /screens/{id}/live            same shape, since inception, plus divergence vs backtest
GET  /symbols/{sym}/series?screen= price, band, z, units(backtest), units(live), trades — one chart's data
GET  /presets                      the strategy presets (below)
```
Responses carry `as_of` (last bar date) and `field_version` so the UI can show staleness.

**Strategy presets** (from what we've tested; each is just a `StrategyParams`):
| preset | side | params |
|---|---|---|
| `long‑decel` | long | decel ×⅔, h=10, 1.0 / 1.5 / 1.0, no timeout |
| `long‑fixed` | long | fixed, h=10, 1.0 / 1.5 / 1.0, no timeout |
| `short‑strict` | short | decel ×½, h=10, 2.0 / 1.5 / 1.5, timeout 40 |
| `short‑capit` | short | fixed, h=10, 1.0 / 0.5 / 1.5, capitulate −2.5 |
| `custom` | either | full param form |
Also expose **band = field | brownian** as a preset flag, since the comparison has been the honest benchmark all along.

---

## 5. UI design (decision — my recommendation)

Single‑page app, one screen at a time, **four tabs** plus a persistent header.

**Header (always visible)**: screen name · as‑of date · staleness pill · `Refresh` · book strip:
`gross long 47.2 · gross short 31.0 · net +16.2 · today P&L +0.84 · since inception +6.3 (Sharpe 1.21)`.

### Tab 1 — Today  (default landing tab)
The reason the app exists. Two columns, long book left, short book right.
- **Actions** list at the top of each column, biggest first: `▲ BUY  AMD  0.44u  (stack 3, z −1.32)`,
  `▼ SELL AAPL  all 1.7u  (armed 08‑12, normalised)`, `✕ LIQUIDATE  XOM  (dropped)`. Each row expands to the
  mini chart (price, band, z, stack) so you can see *why* in one click.
- **Positions** table below: symbol · units · avg entry · unrealised · stack age · armed? · z today.
- Empty state on inception day: "No actions today. The rule fires on the next 1σ dip." — plus what the backtest
  would be holding, greyed, so you know what you're *not* carrying.

### Tab 2 — Screen
Add / drop tickers per side (typeahead against a cached ticker list; validate history ≥ 120 bars on add — CBRS
taught us that). Strategy preset per side with a "customise" disclosure. Inception dates. `Generate` /
`Regenerate`. Drop shows a confirm: "Liquidates 1.7u AAPL at next close."

### Tab 3 — Book  (the pages we've been building, on live data)
- Summary tiles (total, on peak, on day‑weighted, Sharpe, B&H Sharpe, DD, names +ve).
- **Exposure chart** (gross long above / short below / net line) and **equity chart** with **two layers**:
  backtest as a thin muted line from the start of history; live book as a bold line from inception, with the
  inception date marked. Where they diverge you see it immediately.
- Per‑name table sorted by contribution, with the long/short sleeve columns side by side. Click a row → Tab 4.
- Toggle: field band / Brownian band comparator for the backtest layer (it's free — same masks).

### Tab 4 — Name
One name, the full card: price + band + entries/exits (backtest markers hollow, live markers filled),
capital strip, equity strip, z strip, and the estimator panels (m, σ, μ, U, g) at the current fit. This is
the "individual graphs" section, but reached from a row rather than scrolled through 50 at a time.

Visual language: keep what worked in the artifacts (mono numerals, orange for the field, ink for price,
green/red for buy/sell, muted for Brownian). Backtest vs live: **thin & muted vs bold & solid**, plus hollow vs
filled markers — never colour alone.

---

## 6. Live‑book rules that need deciding up front

- **Changing strategy params on a live book** creates a **new version** from that date: the old version's
  positions are carried into the new one as‑is (no forced liquidation), and the event is marked on the charts.
  Alternative is liquidate‑and‑restart; I'd default to carry.
- **Sizing units to money**: the app works in units; the header lets you set `1 unit = $X` per book so the
  Today tab can show dollars and shares (rounded down). Vol‑targeting is a later `SIZE` overlay.
- **Execution timing**: signals are computed on the close and shown after 16:30 ET; the paper ledger fills at
  that same close (as the backtests do). If you want next‑open fills, that's a flag on the ledger, not the
  masks. Either way, say which on screen.
- **Costs**: 1 bp/side default, configurable; no borrow cost modelled (show a warning on hard‑to‑borrow names
  when a borrow‑fee source exists).
- **Corporate actions**: yfinance auto‑adjusts; a split that changes adjusted history will move the stored
  panel. Store the adjusted close *and* a hash; if history changes, refit that symbol and mark the event.

---

## 7. Build phases

**Phase 0 — refactor (1–2 days).** Extract the mask chain from `impulse.strategy.run` into
`impulse/panel.py`: `signals(P, Q10, Q50, Q90, params) → Z, ENTRY, ARM, NORM` and
`positions(masks, params, active, live, init="running"|"flat") → UNITS, INTENT, TRADES`. Keep `run()` as a
thin wrapper so every existing script and test still passes; add tests that the panel version reproduces the
scalar version trade‑for‑trade on the semis basket.

**Phase 1 — store + daily job (2 days).** Parquet/DuckDB price store with delta fetch; field cache keyed by
field params; `refresh()` that appends a day and recomputes. CLI only. Verify idempotence (running refresh
twice changes nothing).

**Phase 2 — API + Today (3 days).** FastAPI with the endpoints above; the event log; live ledger replay;
`/today`. A plain HTML Today page first — this is the MVP you'd actually use.

**Phase 3 — Book + Name tabs (3–4 days).** React app, uPlot charts, tables, backtest/live layering.

**Phase 4 — polish.** Presets UI, money sizing, drop confirm, staleness, export (CSV of trades and today's
actions), auth if it goes beyond one user.

---

## 8. Open questions for you
1. Paper ledger only, or do you want to record actual fills (the `FILLS` overlay) from day one?
2. Close fills or next‑open fills for the live book?
3. Carry‑forward vs liquidate when params change on a live book?
4. One screen or many (e.g. "semis long", "spike shorts") — many is cheap in this design, just a screen id.
5. Is the trend gate (only short below the 60‑day mean / negative μ) a preset flag in v1, given the 50‑name
   result? I'd include it as an optional overlay: `GATE : bool[D, N]` multiplied into `ENTRY`.
