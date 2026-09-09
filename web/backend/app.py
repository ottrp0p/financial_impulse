"""Impulse Screener — local single-user API + static frontend.

    .venv/bin/uvicorn web.backend.app:app --reload --port 8765
"""
from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

from web.backend import ledger as L                      # noqa: E402
from web.backend.screens import DEFAULT_PRESET, PRESETS, Screens, today  # noqa: E402
from web.backend.store import FIELD, Store, session_open_now  # noqa: E402

DATA = ROOT / "web" / "data"
store = Store(DATA); screens = Screens(DATA / "screens")
jobs: dict[str, dict] = {}
app = FastAPI(title="Impulse Screener")


# ---------- models ----------
class NewScreen(BaseModel):
    name: str
    unit_dollars: float = 1000.0

class AddNames(BaseModel):
    side: int
    symbols: list[str]

class Strategy(BaseModel):
    side: int
    preset: str | None = None
    params: dict | None = None

class Units(BaseModel):
    unit_dollars: float


# ---------- helpers ----------
def _screen(sid: str) -> dict:
    try:
        return screens.get(sid)
    except KeyError:
        raise HTTPException(404, "screen not found")


def _job(sid: str, fn):
    if jobs.get(sid, {}).get("running"):
        return jobs[sid]
    jobs[sid] = {"running": True, "error": None, "progress": "starting", "kind": "refresh", "tickers": {}, "result": None}
    def go():
        try:
            fn()
            jobs[sid].update(running=False, progress="done")
        except Exception as e:  # noqa: BLE001
            jobs[sid].update(running=False, error=f"{e}\n{traceback.format_exc()}", progress="failed")
    threading.Thread(target=go, daemon=True).start()
    return jobs[sid]


def _refresh_symbols(sid: str, syms: list[str]):
    jobs[sid]["tickers"] = {x: "queued" for x in syms}
    for i, sym in enumerate(syms):
        jobs[sid]["progress"] = f"{sym} ({i + 1}/{len(syms)})"; jobs[sid]["tickers"][sym] = "fitting"
        store.ensure(sym, refetch=True); jobs[sid]["tickers"][sym] = "ready"


def _compute(s: dict) -> dict:
    """Run both ledgers for every name on both sides. Returns the full book structure."""
    st = Screens.state(s); inception = s.get("generated")
    out = {"as_of": None, "inception": inception, "sides": {}, "names": {}, "today": {"actions": [], "positions": []}}
    last_dates = []
    for side_key in ("1", "-1"):
        side = int(side_key); cfg = s["strategy"][side_key]; p = L.to_params(cfg)
        bt_runs, live_runs = {}, {}
        for sym, meta in st["names"][side_key].items():
            prices, fc = store.load_prices(sym), store.load_forecasts(sym)
            if prices is None or fc is None or not len(fc):
                continue
            last_dates.append(prices.index[-1].strftime("%Y-%m-%d"))
            sf = L.signal_frame(prices, fc, cfg["band"])
            end = meta["dropped"]
            bt = L.run_window(sf, p, None, end, close_at_end=True)
            start = max(inception, meta["added"]) if inception else None
            versions = [(d, L.to_params(c)) for d, c in Screens.versions(s, side)]
            lv = L.run_versioned(sf, versions, start, end, close_at_end=bool(end)) if inception else {"frame": sf.iloc[0:0], "res": None}
            bt_runs[sym], live_runs[sym] = bt, lv
            entry = out["names"].setdefault(sym, {})
            entry[side_key] = {"added": meta["added"], "dropped": meta["dropped"], "backtest": L.symbol_stats(bt), "live": L.symbol_stats(lv),
                               "trades_backtest": L.trades_table(bt["frame"], bt["res"], sym, side, "drop" if end else "end") if bt["res"] else [],
                               "trades_live": L.trades_table(lv["frame"], lv["res"], sym, side, "drop" if end else "end") if lv["res"] else []}
            # today's actions & positions from the live ledger
            if lv["res"] is not None and not end:
                w, res = lv["frame"], lv["res"]; last = w["date"].iloc[-1]; li = len(w) - 1
                for t in res["trades"].itertuples():
                    if t.exit == li:
                        rsn = getattr(t, "reason", None) if "reason" in res["trades"].columns else None
                        entry[side_key].setdefault("_actions", []).append({"action": "COVER" if side < 0 else "SELL", "size": float(t.size), "reason": rsn or ("capitulate" if t.capitulated else ("armed" if t.armed_exit else "timeout")), "date": last})
                opens = res["open"]
                for q in opens:
                    if q["i"] == li:
                        entry[side_key].setdefault("_actions", []).append({"action": "SHORT" if side < 0 else "BUY", "size": float(q["size"]), "reason": f"z {w['z'].iloc[li]:+.2f}", "date": last})
                units = float(sum(q["size"] for q in opens))
                if units > 0:
                    x = w["x"].to_numpy(); px = float(np.exp(x[li]))
                    avg = float(np.exp(sum(q["size"] * x[q["i"]] for q in opens) / units))
                    unreal = float(sum(q["size"] * (x[li] - x[q["i"]]) for q in opens) * side)
                    out["today"]["positions"].append({"symbol": sym, "side": side, "units": units, "avg_px": avg, "px": px, "unreal": unreal,
                                                       "armed": any(q["armed"] for q in opens), "oldest": w["date"].iloc[min(q["i"] for q in opens)], "z": float(w["z"].iloc[li])})
                for a in entry[side_key].pop("_actions", []):
                    out["today"]["actions"].append({"symbol": sym, "side": side, **a})
        out["sides"][side_key] = {"strategy": cfg, "backtest": L.side_book(bt_runs), "live": L.side_book(live_runs),
                                  "names": list(st["names"][side_key])}
    out["as_of"] = max(last_dates) if last_dates else None
    forming, today_et = session_open_now()
    out["session"] = {"forming": forming, "today": today_et, "note": "today's session is still open — recommendations are as of the last close" if forming else None}
    out["book"] = {"backtest": L.combine(out["sides"]["1"]["backtest"], out["sides"]["-1"]["backtest"]),
                   "live": L.combine(out["sides"]["1"]["live"], out["sides"]["-1"]["live"])}
    out["today"]["actions"].sort(key=lambda a: -a["size"])
    return out


# ---------- routes ----------
@app.get("/api/presets")
def presets():
    return {"presets": PRESETS, "default": DEFAULT_PRESET, "field": FIELD}

@app.get("/api/screens")
def list_screens():
    return screens.list()

@app.post("/api/screens")
def create_screen(b: NewScreen):
    return screens.create(b.name, b.unit_dollars)

@app.delete("/api/screens/{sid}")
def delete_screen(sid: str):
    p = screens._path(sid)
    if not p.exists():
        raise HTTPException(404, "screen not found")
    p.unlink(); jobs.pop(sid, None); return {"deleted": sid}

@app.get("/api/screens/{sid}")
def get_screen(sid: str):
    s = _screen(sid); return {**s, "state": Screens.state(s), "job": jobs.get(sid), "symbols": {sym: store.status(sym) for side in Screens.state(s)["names"].values() for sym in side}}

@app.post("/api/screens/{sid}/names")
def add_names(sid: str, b: AddNames):
    """Validate + fetch in the background; the UI polls /job for per-ticker status."""
    s = _screen(sid); syms = [x.strip().upper() for x in b.symbols if x.strip()]
    if not syms:
        raise HTTPException(400, "no symbols")
    if jobs.get(sid, {}).get("running"):
        raise HTTPException(409, "a job is already running for this screen")
    def go():
        ok, bad = [], []
        for i, sym in enumerate(syms):
            jobs[sid]["progress"] = f"validating {sym} ({i + 1}/{len(syms)})"
            jobs[sid]["tickers"][sym] = "checking"
            try:
                store.refresh_prices(sym)
                n = len(store.load_prices(sym))
                if n < FIELD["fit_bars"] + 30:
                    bad.append(f"{sym}: only {n} bars of history (need {FIELD['fit_bars'] + 30})"); jobs[sid]["tickers"][sym] = "rejected"
                else:
                    ok.append(sym); jobs[sid]["tickers"][sym] = "ok"
            except Exception as e:  # noqa: BLE001
                bad.append(f"{sym}: {str(e).splitlines()[0][:120]}"); jobs[sid]["tickers"][sym] = "rejected"
        if ok:
            s2 = screens.event(sid, "add", side=b.side, symbols=ok)
            if s2.get("generated"):
                for i, sym in enumerate(ok):
                    jobs[sid]["progress"] = f"fitting {sym} ({i + 1}/{len(ok)})"; jobs[sid]["tickers"][sym] = "fitting"
                    store.ensure(sym, refetch=False); jobs[sid]["tickers"][sym] = "ready"
        jobs[sid]["result"] = {"added": ok, "rejected": bad}
    j = _job(sid, go); j["tickers"] = {x: "queued" for x in syms}; j["kind"] = "add"; j["result"] = None
    return j


@app.delete("/api/screens/{sid}/names/{sym}")
def drop_name(sid: str, sym: str, side: int):
    s = screens.event(sid, "drop", side=side, symbol=sym.upper())
    return {"screen": s, "note": "liquidated at the next close in both ledgers"}

@app.put("/api/screens/{sid}/strategy")
def set_strategy(sid: str, b: Strategy):
    s = _screen(sid)
    if b.preset:
        if b.preset not in PRESETS:
            raise HTTPException(400, "unknown preset")
        cfg = dict(PRESETS[b.preset], preset=b.preset)
    else:
        cfg = dict(s["strategy"][str(b.side)], **(b.params or {}), preset="custom")
    cfg["side"] = b.side
    s["strategy"][str(b.side)] = cfg; screens.save(s)
    screens.event(sid, "params", side=b.side, cfg=cfg)
    return s

@app.put("/api/screens/{sid}/units")
def set_units(sid: str, b: Units):
    s = _screen(sid); s["unit_dollars"] = b.unit_dollars; screens.save(s); return s

@app.post("/api/screens/{sid}/generate")
def generate(sid: str):
    s = _screen(sid); st = Screens.state(s)
    syms = sorted({sym for side in st["names"].values() for sym in side})
    if not syms:
        raise HTTPException(400, "add some names first")
    def go():
        _refresh_symbols(sid, syms)
        s2 = screens.get(sid)
        if not s2.get("generated"):
            s2["generated"] = today(); screens.save(s2); screens.event(sid, "generate")
    return _job(sid, go)

@app.post("/api/screens/{sid}/refresh")
def refresh(sid: str):
    s = _screen(sid); st = Screens.state(s)
    syms = sorted({sym for side in st["names"].values() for sym, m in side.items() if not m["dropped"]})
    return _job(sid, lambda: _refresh_symbols(sid, syms))

@app.get("/api/screens/{sid}/job")
def job(sid: str):
    return jobs.get(sid, {"running": False, "progress": None, "error": None, "kind": None, "tickers": {}, "result": None})

@app.get("/api/screens/{sid}/book")
def book(sid: str):
    s = _screen(sid)
    return _compute(s)

@app.get("/api/screens/{sid}/series/{sym}")
def series(sid: str, sym: str, side: int):
    s = _screen(sid); cfg = s["strategy"][str(side)]; p = L.to_params(cfg)
    prices, fc = store.load_prices(sym), store.load_forecasts(sym)
    if prices is None or fc is None:
        raise HTTPException(404, "no data yet")
    sf = L.signal_frame(prices, fc, cfg["band"]); st = Screens.state(s); meta = st["names"][str(side)].get(sym, {"added": None, "dropped": None})
    bt = L.run_window(sf, p, None, meta["dropped"], True)
    start = max(s["generated"], meta["added"]) if s.get("generated") and meta["added"] else None
    versions = [(d, L.to_params(c)) for d, c in Screens.versions(s, side)]
    lv = L.run_versioned(sf, versions, start, meta["dropped"], bool(meta["dropped"])) if start else {"frame": sf.iloc[0:0], "res": None}
    def pack(r):
        if r["res"] is None:
            return None
        w = r["frame"]
        return {"dates": w["date"].tolist(), "capital": np.round(r["res"]["capital"], 4).tolist(), "equity": np.round(r["res"]["equity"], 5).tolist(),
                "trades": L.trades_table(w, r["res"], sym, side, "drop" if meta["dropped"] else "end"), "open": [{"date": w["date"].iloc[q["i"]], "size": q["size"], "armed": q["armed"]} for q in r["res"]["open"]]}
    return {"symbol": sym, "side": side, "dates": sf["date"].tolist(), "close": np.round(np.exp(sf["x"]), 3).tolist(),
            "q10": np.round(np.exp(sf["q10"]), 3).tolist(), "q50": np.round(np.exp(sf["q50"]), 3).tolist(), "q90": np.round(np.exp(sf["q90"]), 3).tolist(),
            "z": np.round(sf["z"], 3).tolist(), "backtest": pack(bt), "live": pack(lv), "inception": start,
            "version_dates": [d for d, _ in Screens.versions(s, side)[1:] if start and d > start]}


FRONT = ROOT / "web" / "frontend"


class NoCacheStatic(StaticFiles):
    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        r.headers["Cache-Control"] = "no-store"
        return r


app.mount("/static", NoCacheStatic(directory=FRONT), name="static")


@app.get("/")
def index():
    """index.html with static URLs stamped by file mtime so a changed app.js is never served from cache."""
    html = (FRONT / "index.html").read_text()
    for f in ("app.js", "style.css"):
        html = html.replace(f"/static/{f}", f"/static/{f}?v={int((FRONT / f).stat().st_mtime)}")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})
