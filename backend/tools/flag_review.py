"""flag_review — record a clinician-review escalation (append-only).

The safe default for anything uncertain, conflicting, or clinically risky:
surface it loudly rather than bury it. Flags are rendered at the top of the
discharge draft so a reviewer sees them first.
"""
from __future__ import annotations

from ..agent.state import AgentState

_SEVERITIES = {"info", "warning", "critical"}


def flag_review(state: AgentState, field: str, reason: str,
                severity: str = "warning") -> dict:
    sev = severity if severity in _SEVERITIES else "warning"
    state.add_flag(field, reason, sev)
    return {
        "ok": True,
        "flag": {"field": field, "reason": reason, "severity": sev},
        "total_flags": len(state.flags),
    }
