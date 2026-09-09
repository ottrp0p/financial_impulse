"""Screens persisted as JSON with an append-only event log; the live book is a fold over the log."""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from pathlib import Path

PRESETS = {
    "long-decel":  {"side": 1,  "band": "field", "h": 10, "z_in": 1.0, "z_out": 1.5, "z_norm": 1.0, "max_hold": 100000, "sizing": "decel", "ratio": 1.5, "max_stack": 4, "max_units": None, "capitulate": None, "cost_bp": 1.0},
    "long-fixed":  {"side": 1,  "band": "field", "h": 10, "z_in": 1.0, "z_out": 1.5, "z_norm": 1.0, "max_hold": 100000, "sizing": "fixed", "ratio": 1.0, "max_stack": 4, "max_units": None, "capitulate": None, "cost_bp": 1.0},
    "short-strict":{"side": -1, "band": "field", "h": 10, "z_in": 2.0, "z_out": 1.5, "z_norm": 1.5, "max_hold": 40,     "sizing": "decel", "ratio": 2.0, "max_stack": 4, "max_units": None, "capitulate": None, "cost_bp": 1.0},
    "short-capit": {"side": -1, "band": "field", "h": 10, "z_in": 1.0, "z_out": 0.5, "z_norm": 1.5, "max_hold": 100000, "sizing": "fixed", "ratio": 1.0, "max_stack": 6, "max_units": None, "capitulate": 2.5, "cost_bp": 1.0},
    # confirmation short: break above the band -> first close back in the lower half -> 0.25u, decelerating adds, cover above the centre
    "short-confirm": {"side": -1, "kind": "confirm", "band": "field", "h": 10, "z_break": 1.5, "z_enter": 0.0, "step": 0.75, "z_exit": 0.25, "start": 0.25, "accel": 0.6667, "max_adds": 5, "lookback": 20, "max_hold": 60, "capitulate": None, "rearm": False, "cost_bp": 1.0},
}
DEFAULT_PRESET = {1: "long-decel", -1: "short-capit"}


def today() -> str:
    return date.today().isoformat()


class Screens:
    def __init__(self, root: str | Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sid: str) -> Path:
        return self.root / f"{sid}.json"

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*.json")):
            s = json.loads(p.read_text())
            out.append({"id": s["id"], "name": s["name"], "created": s["created"], "names": {k: len(v) for k, v in self.state(s)["names"].items()}})
        return out

    def get(self, sid: str) -> dict:
        p = self._path(sid)
        if not p.exists():
            raise KeyError(sid)
        s = json.loads(p.read_text())
        # migration: screens created before strategy versioning get an initial params event at creation
        changed = False
        for side in ("1", "-1"):
            if not any(e["kind"] == "params" and str(e.get("side", 1)) == side for e in s["events"]):
                s["events"].insert(0, {"kind": "params", "date": s["created"], "ts": s["created"] + "T00:00:00", "side": int(side), "cfg": s["strategy"][side], "note": "backfilled"})
                changed = True
        if changed:
            self.save(s)
        return s

    def save(self, s: dict) -> None:
        self._path(s["id"]).write_text(json.dumps(s, indent=1))

    def create(self, name: str, unit_dollars: float = 1000.0) -> dict:
        s = {"id": uuid.uuid4().hex[:8], "name": name, "created": today(), "unit_dollars": unit_dollars, "events": [],
             "strategy": {"1": dict(PRESETS["long-decel"], preset="long-decel"), "-1": dict(PRESETS["short-capit"], preset="short-capit")},
             "generated": None}
        for side in ("1", "-1"):   # log the initial strategy so versions can be replayed
            s["events"].append({"kind": "params", "date": today(), "ts": datetime.utcnow().isoformat(timespec="seconds"), "side": int(side), "cfg": s["strategy"][side]})
        self.save(s); return s

    def event(self, sid: str, kind: str, **data) -> dict:
        s = self.get(sid)
        s["events"].append({"kind": kind, "date": data.pop("date", today()), "ts": datetime.utcnow().isoformat(timespec="seconds"), **data})
        self.save(s); return s

    @staticmethod
    def state(s: dict, as_of: str | None = None) -> dict:
        """Fold the event log: which names are active per side, when each was added/dropped."""
        names: dict[str, dict[str, dict]] = {"1": {}, "-1": {}}
        for e in s["events"]:
            if as_of and e["date"] > as_of:
                continue
            side = str(e.get("side", 1))
            if e["kind"] == "add":
                for sym in e["symbols"]:
                    if sym not in names[side] or names[side][sym].get("dropped"):
                        names[side][sym] = {"added": e["date"], "dropped": None}
            elif e["kind"] == "drop":
                if e["symbol"] in names[side] and not names[side][e["symbol"]]["dropped"]:
                    names[side][e["symbol"]]["dropped"] = e["date"]
        return {"names": names, "inception": s.get("generated")}

    @staticmethod
    def versions(s: dict, side: int) -> list[tuple[str, dict]]:
        """Strategy versions for a side as [(effective_date, cfg)], oldest first; falls back to the current cfg."""
        out = [(e["date"], e["cfg"]) for e in s["events"] if e["kind"] == "params" and int(e.get("side", 1)) == side and e.get("cfg")]
        if not out:
            out = [("0000-00-00", s["strategy"][str(side)])]
        out.sort(key=lambda x: x[0])
        # same-day changes: last wins
        dedup = {}
        for d, c in out:
            dedup[d] = c
        return sorted(dedup.items())
