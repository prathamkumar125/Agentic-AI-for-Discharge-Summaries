"""Simulated reviewer — a stand-in "doctor" with a CONSISTENT, HIDDEN editing policy.

In production a clinician edits the agent's draft before finalizing; those edits are
the learning signal. We have no real clinician, so this module manufactures the signal:
it applies a deterministic, GENERAL editing policy to a draft section and returns the
"edited" version, yielding `(draft, edited)` pairs.

Two rules make this faithful to the task:
  * The policy is **hidden from the agent and the learner** — it lives only here; the
    learner (`memory.py`) sees only the resulting pairs and must INDUCE the rules.
  * The policy is **general clinical editing** (expand standard abbreviations, standardize
    recurring phrasings), applied to ANY text — never patient-specific find/replace. It
    operates purely on the value TEXT, so citations / flags / MISSING-PENDING-CONFLICT
    markers (which live in separate state metadata) are inherently preserved.
"""
from __future__ import annotations

import re

# General clinical abbreviation / phrasing expansions a reviewer would standardize.
# These are common medical shorthand, NOT tied to any particular patient's notes.
_EXPANSIONS: dict[str, str] = {
    "k/c/o": "known case of",
    "c/o": "complains of",
    "h/o": "history of",
    "s/p": "status post",
    "b/l": "bilateral",
    "adv": "advised",
    "rx": "treatment",
    "dx": "diagnosis",
    "tx": "treatment",
    "fu": "follow-up",
    "f/u": "follow-up",
    "sr.": "serum",
    "sr": "serum",
    "grbs": "random blood glucose",
    "rbs": "random blood glucose",
    "t2dm": "type 2 diabetes mellitus",
    "t2 dm": "type 2 diabetes mellitus",
    "aki": "acute kidney injury",
    "dka": "diabetic ketoacidosis",
    "uti": "urinary tract infection",
    "usg": "ultrasonography",
    "iv": "intravenous",
    "inj": "injection",
    "tab": "tablet",
    "od": "once daily",
    "bd": "twice daily",
    "tds": "three times daily",
    "sos": "as needed",
    "f/c": "Foley catheter",
    "nbm": "nil by mouth",
}

# Recurring phrasings a reviewer standardizes (full-phrase, order matters: longest first).
_PHRASES: list[tuple[str, str]] = [
    ("report awaited", "result pending"),
    ("reports awaited", "results pending"),
    ("not known", "none known"),
    ("pt is stable", "patient is stable"),
    ("pt ", "patient "),
]


def _normalize_ws(text: str) -> str:
    """Collapse runs of whitespace; trim. (Whitespace is normalized on BOTH sides
    before edit distance is measured, so it is never the learning signal.)"""
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", (text or "").strip()))


def _expand_token(match: re.Match) -> str:
    """Expand one matched abbreviation, preserving leading capitalization."""
    raw = match.group(0)
    repl = _EXPANSIONS[raw.lower()]
    return repl[0].upper() + repl[1:] if raw[:1].isupper() else repl


# One regex of all abbreviations, longest-first, with sensible boundaries (handles the
# slash forms like "k/c/o" that \b would not delimit cleanly).
_ABBR_RE = re.compile(
    r"(?<![A-Za-z0-9])(" +
    "|".join(re.escape(a) for a in sorted(_EXPANSIONS, key=len, reverse=True)) +
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def apply_policy(text: str) -> str:
    """Return the reviewer's edited version of one draft section (deterministic)."""
    if not text:
        return text
    out = _normalize_ws(text)
    out = _ABBR_RE.sub(_expand_token, out)
    for src, tgt in _PHRASES:
        out = re.sub(re.escape(src), tgt, out, flags=re.IGNORECASE)
    return _normalize_ws(out)


def make_pairs(state_dict: dict) -> list[dict]:
    """Build `(draft, edited)` pairs from a validated agent_state's field values.

    Only fields that carry text are used (missing fields have no value to edit). The
    citations/flags in the state are untouched — the reviewer edits prose only.
    """
    pairs: list[dict] = []
    for field, fv in state_dict.get("fields", {}).items():
        draft = (fv.get("value") or "").strip()
        if not draft:
            continue
        pairs.append({
            "field": field,
            "label": fv.get("label", field),
            "status": fv.get("status"),
            "draft": _normalize_ws(draft),
            "edited": apply_policy(draft),
        })
    return pairs
