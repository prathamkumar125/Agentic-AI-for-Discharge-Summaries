"""reconcile — medication reconciliation: admission meds vs discharge meds.

Deterministic diff (the agent extracts the two med lists first, then calls this).
Classifies each drug as added / stopped / changed / continued, and surfaces any
change that lacks a documented reason — which the agent must then flag for
reconciliation rather than invent a rationale for.
"""
from __future__ import annotations

import re

# Dosage-form / route / unit tokens that prefix or trail a drug name on a chart
# (e.g. "INJ PAN 40 mg", "TAB DOLO", "INS LANTUS 10 U"). They are NOT the drug
# identity, so they must be stripped before matching across lists — otherwise
# every injectable collapses to the key "inj" and the diff is meaningless.
_FORM_ROUTE = {
    "inj", "injection", "tab", "tabs", "tablet", "cap", "caps", "capsule",
    "syp", "syr", "syrup", "ins", "insulin", "sol", "soln", "susp", "oint",
    "ointment", "neb", "nebule", "iv", "im", "sc", "po", "sl", "pr", "s",
}
_UNITS = {
    "mg", "g", "gm", "kg", "ml", "l", "mcg", "ug", "u", "iu", "unit", "units",
    "amp", "ampoule", "cc", "tsp", "puff", "puffs", "drop", "drops", "od", "bd",
    "tds", "qid", "sos", "stat", "hs",
}


def _key(med) -> str:
    """Drug-identity key: name with form/route/dose tokens stripped.

    "INJ PAN 40 mg" -> "pan";  "INS LANTUS 10 U" -> "lantus";
    "INJ H. ACTRAPID" -> "actrapid";  "INJ HAPPY NERVE PLUS" -> "happy nerve plus".
    """
    if isinstance(med, dict):
        name = med.get("name", "")
    else:
        name = str(med)
    toks = re.findall(r"[a-z0-9]+", name.lower())
    core: list[str] = []
    for t in toks:
        if t in _FORM_ROUTE or t in _UNITS:
            continue
        if any(c.isdigit() for c in t):  # dose/strength tokens like "40", "650", "10u"
            continue
        if len(t) <= 1:                  # single-letter noise ("h" in "H. ACTRAPID")
            continue
        core.append(t)
    if core:
        return " ".join(core)
    # Fallback: name was only form/dose tokens — keep first raw token so distinct
    # un-named entries don't all collapse to "".
    return toks[0] if toks else ""


def _detail(med) -> dict:
    if isinstance(med, dict):
        return {k: med.get(k) for k in ("name", "dose", "route", "freq", "reason") if med.get(k)}
    return {"name": str(med)}


def _changed(a: dict, b: dict) -> bool:
    return any((a.get(k) or "") != (b.get(k) or "") for k in ("dose", "route", "freq"))


def reconcile(adm_meds, dis_meds) -> dict:
    """Return a structured reconciliation of admission vs discharge medications."""
    adm = {_key(m): _detail(m) for m in (adm_meds or []) if _key(m)}
    dis = {_key(m): _detail(m) for m in (dis_meds or []) if _key(m)}

    added, stopped, changed, continued = [], [], [], []
    for k, d in dis.items():
        if k not in adm:
            added.append(d)
        elif _changed(adm[k], d):
            changed.append({"from": adm[k], "to": d})
        else:
            continued.append(d)
    for k, a in adm.items():
        if k not in dis:
            stopped.append(a)

    # Changes (added/stopped/dose-changed) that carry no documented reason.
    def _no_reason(item: dict) -> bool:
        target = item.get("to", item)
        return not (target.get("reason"))

    needs_reason = (
        [{"change": "added", **m} for m in added if _no_reason(m)]
        + [{"change": "stopped", **m} for m in stopped if _no_reason(m)]
        + [{"change": "changed", **c} for c in changed if _no_reason(c)]
    )

    return {
        "ok": True,
        "added": added,
        "stopped": stopped,
        "changed": changed,
        "continued": continued,
        "changes_without_documented_reason": needs_reason,
        "note": ("Flag each undocumented change for reconciliation; do NOT invent a reason."
                 if needs_reason else "All changes have a documented reason or none detected."),
    }
