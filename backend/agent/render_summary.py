"""render_summary — turn the validated AgentState into the clinician-facing draft.

This produces `discharge_summary.md`: the actual Part-1 deliverable a clinician
reads. It is rendered from the SAME serialized state the guardrail produced
(`state.to_dict()` / `agent_state.json`), so the document can NEVER contain a fact
the guardrail did not accept — the renderer adds no content of its own.

Design principles (all in service of the no-fabrication rule):
  * A prominent DRAFT / not-finalized header — this is never an auto-finalized doc.
  * A review BANNER up top listing every conflict / pending / missing field and the
    flag count, so the unknown is impossible to miss.
  * Every section shows its status badge and its citations (page + verbatim quote).
  * Medication reconciliation is shown with changes lacking a documented reason
    called out explicitly.
  * Clinician-review flags are listed in full at the end.

`render_summary` takes the dict form so it works both from a live run and by
re-reading a saved `agent_state.json` (no agent re-run needed).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .state import FIELD_ORDER, FIELD_LABELS, Status

_BADGE = {
    Status.PRESENT.value: "✓ present",
    Status.MISSING.value: "❗ MISSING",
    Status.PENDING.value: "⏳ PENDING",
    Status.CONFLICT.value: "⚠️ CONFLICT",
}
_SEV_RANK = {"critical": 0, "warning": 1, "info": 2}
_MAX_QUOTE = 300


def _clip(s: str, n: int = _MAX_QUOTE) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _citation_lines(citations: list[dict]) -> list[str]:
    out = []
    for c in citations or []:
        doc = c.get("doc") or ""
        tag = f"p{c.get('page')}" + (f", {doc}" if doc else "")
        out.append(f"> _[{tag}]_ “{_clip(c.get('quote', ''))}”")
    return out


def _banner(fields: dict, flags: list[dict]) -> list[str]:
    buckets: dict[str, list[str]] = {"conflict": [], "pending": [], "missing": []}
    for name in FIELD_ORDER:
        st = fields.get(name, {}).get("status")
        if st in buckets:
            buckets[st].append(FIELD_LABELS[name])

    lines = ["## ⚑ Review banner", ""]
    none_flagged = not any(buckets.values()) and not flags
    if none_flagged:
        lines += ["All 12 fields are sourced and cited, with no flags. "
                  "**This is still a draft and must be reviewed before finalizing.**", ""]
        return lines

    def row(label: str, items: list[str]) -> str:
        return f"- **{label} ({len(items)}):** " + (", ".join(items) if items else "—")

    lines += [
        row("⚠️ Conflicts — two sources disagree, NOT resolved", buckets["conflict"]),
        row("⏳ Pending — documented as awaited", buckets["pending"]),
        row("❗ Missing — not found in any source", buckets["missing"]),
        f"- **⚑ Clinician-review flags ({len(flags)}):** see *Clinician Review Flags* at the end.",
        "",
        "> These items were **deliberately not invented**. Each requires clinician attention.",
        "",
    ]
    return lines


def _field_section(idx: int, name: str, fv: dict) -> list[str]:
    status = fv.get("status", "missing")
    badge = _BADGE.get(status, status)
    lines = [f"## {idx}. {FIELD_LABELS[name]}  ·  {badge}", ""]

    value = (fv.get("value") or "").strip()
    notes = (fv.get("notes") or "").strip()
    cits = fv.get("citations") or []

    if status == Status.CONFLICT.value:
        lines.append("**⚠️ Conflicting values — surfaced for clinician resolution (the agent did "
                     "NOT pick one):**")
        lines.append("")
        for cand in [c.strip() for c in value.split("|")] if value else []:
            lines.append(f"- {cand}")
        lines.append("")
    elif status == Status.MISSING.value:
        lines.append("_Not documented in the source notes — flagged for clinician review. "
                     "No value has been inferred._")
        lines.append("")
    elif status == Status.PENDING.value:
        lines.append("_Documented as pending / awaited — not imputed._")
        lines.append("")
        if value:
            lines.append(value)
            lines.append("")
    else:  # present
        lines.append(value if value else "_(no value)_")
        lines.append("")

    if notes:
        lines.append(f"**Notes:** {notes}")
        lines.append("")
    if cits:
        lines.append("**Source citations:**")
        lines += _citation_lines(cits)
        lines.append("")
    return lines


def _med_reconciliation(rec: dict | None) -> list[str]:
    lines = ["## Medication Reconciliation", ""]
    if not rec:
        lines += ["_Medication reconciliation was not completed for this draft._", ""]
        return lines

    def med_str(m: dict) -> str:
        parts = [m.get("name", "")]
        for k in ("dose", "route", "freq"):
            if m.get(k):
                parts.append(str(m[k]))
        reason = f" — reason: {m['reason']}" if m.get("reason") else ""
        return " · ".join(p for p in parts if p) + reason

    def block(title: str, items: list, fmt) -> None:
        lines.append(f"**{title} ({len(items)}):**")
        if items:
            for it in items:
                lines.append(f"- {fmt(it)}")
        else:
            lines.append("- —")
        lines.append("")

    block("Added at discharge", rec.get("added", []), med_str)
    block("Stopped since admission", rec.get("stopped", []), med_str)
    block("Changed (dose/route/frequency)", rec.get("changed", []),
          lambda c: f"{med_str(c.get('from', {}))}  →  {med_str(c.get('to', {}))}")
    block("Continued", rec.get("continued", []), med_str)

    no_reason = rec.get("changes_without_documented_reason", [])
    if no_reason:
        lines.append("**⚑ Changes WITHOUT a documented reason (flagged for reconciliation — "
                     "the agent did not invent a rationale):**")
        for c in no_reason:
            if c.get("change") == "changed":
                lines.append(f"- changed: {med_str(c.get('from', {}))} → {med_str(c.get('to', {}))}")
            else:
                lines.append(f"- {c.get('change', 'change')}: {med_str(c)}")
        lines.append("")
    return lines


def _flags(flags: list[dict]) -> list[str]:
    lines = ["## ⚑ Clinician Review Flags", ""]
    if not flags:
        lines += ["_No flags raised._", ""]
        return lines
    ordered = sorted(flags, key=lambda f: _SEV_RANK.get(f.get("severity", "warning"), 1))
    lines += ["| Severity | Field | Reason |", "|---|---|---|"]
    for f in ordered:
        sev = f.get("severity", "warning").upper()
        field = f.get("field", "")
        reason = _clip(f.get("reason", ""), 240).replace("|", "\\|")
        lines.append(f"| {sev} | {field} | {reason} |")
    lines.append("")
    return lines


def render_summary(state_dict: dict) -> str:
    """Render the clinician-facing discharge-summary markdown from serialized state."""
    fields = state_dict.get("fields", {})
    flags = state_dict.get("flags", [])
    pid = state_dict.get("patient_id", "unknown")
    steps = state_dict.get("step_count", "?")

    out: list[str] = []
    out += [
        f"# Discharge Summary — DRAFT FOR CLINICIAN REVIEW",
        "",
        "> **⚠️ AUTO-GENERATED DRAFT — NOT a finalized clinical document.**",
        "> Every field below is either sourced with a verbatim citation, or explicitly "
        "marked **MISSING / PENDING / CONFLICT** and flagged. Nothing has been invented or "
        "guessed. A clinician must review and finalize before any clinical use.",
        "",
        f"**Patient:** `{pid}`  ·  **Agent steps used:** {steps}  ·  "
        f"**Generated:** {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "---",
        "",
    ]
    out += _banner(fields, flags)
    out += ["---", ""]

    for i, name in enumerate(FIELD_ORDER, start=1):
        out += _field_section(i, name, fields.get(name, {}))

    out += ["---", ""]
    out += _med_reconciliation(state_dict.get("med_reconciliation"))
    out += ["---", ""]
    out += _flags(flags)

    out += [
        "---",
        "",
        "_Draft produced by an agentic system under a no-fabrication guardrail. "
        "Citations reference OCR'd source pages by global page number. "
        "See `trace.json` for the full step-by-step reasoning trace._",
        "",
    ]
    return "\n".join(out)


def render_summary_file(state_dict: dict, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_summary(state_dict), encoding="utf-8")
    return out_path


def render_from_json(agent_state_json: Path, out_path: Path | None = None) -> Path:
    """Regenerate discharge_summary.md from a saved agent_state.json (no agent re-run)."""
    agent_state_json = Path(agent_state_json)
    data = json.loads(agent_state_json.read_text(encoding="utf-8"))
    out_path = out_path or (agent_state_json.parent / "discharge_summary.md")
    return render_summary_file(data, out_path)
