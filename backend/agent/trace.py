"""Trace — the agent's observable, step-by-step log.

Observability requirement: for each step show reasoning -> tool/action -> inputs
-> result -> next decision. Entries are append-only dicts; the trace renders to a
live console stream AND to `trace.json`. The loop, the tools, and the guardrail
all log through this single object so the run is fully auditable.
"""
from __future__ import annotations

import json
from pathlib import Path


def _short(obj, n: int = 220) -> str:
    """Compact one-line preview of an arbitrary value for the console."""
    if isinstance(obj, (dict, list)):
        s = json.dumps(obj, ensure_ascii=False, default=str)
    else:
        s = str(obj)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


class Trace:
    def __init__(self, verbose: bool = True):
        self.entries: list[dict] = []
        self.verbose = verbose

    # --- generic ----------------------------------------------------------
    def event(self, step: int, etype: str, **data) -> dict:
        entry = {"step": step, "type": etype, **data}
        self.entries.append(entry)
        if self.verbose:
            self._print(entry)
        return entry

    # --- canonical helpers (the required think -> act -> observe shape) ----
    def think(self, step: int, phase: str, reasoning: str, next_decision: str = "") -> dict:
        return self.event(step, "think", phase=phase, reasoning=reasoning,
                          next_decision=next_decision)

    def tool(self, step: int, name: str, inputs: dict, result_summary: str,
             ok: bool = True, phase: str = "") -> dict:
        return self.event(step, "tool", phase=phase, tool=name, inputs=inputs,
                          ok=ok, result_summary=result_summary)

    def guardrail(self, step: int, field: str, decision: str, reason: str) -> dict:
        return self.event(step, "guardrail", field=field, decision=decision, reason=reason)

    def replan(self, step: int, reason: str) -> dict:
        return self.event(step, "replan", reason=reason)

    def finish(self, step: int, reason: str) -> dict:
        return self.event(step, "finish", reason=reason)

    # --- rendering --------------------------------------------------------
    def _print(self, e: dict) -> None:
        step = e.get("step", "-")
        t = e["type"]
        if t == "think":
            print(f"\n[step {step}] 🧠 ({e.get('phase','')}) {_short(e.get('reasoning',''))}")
            if e.get("next_decision"):
                print(f"          ↳ next: {_short(e['next_decision'])}")
        elif t == "tool":
            mark = "✓" if e.get("ok", True) else "✗"
            print(f"[step {step}]  🔧 {e['tool']}({_short(e.get('inputs',{}),120)}) "
                  f"{mark} → {_short(e.get('result_summary',''))}")
        elif t == "guardrail":
            print(f"[step {step}]  🛡  {e['field']}: {e['decision']} — {_short(e.get('reason',''))}")
        elif t == "replan":
            print(f"[step {step}] 🔁 re-plan: {_short(e.get('reason',''))}")
        elif t == "finish":
            print(f"[step {step}] 🏁 finish: {_short(e.get('reason',''))}")
        else:
            print(f"[step {step}] · {t}: {_short({k: v for k, v in e.items() if k not in ('step','type')})}")

    def to_json(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.entries, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
