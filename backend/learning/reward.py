"""reward — the edit-derived signal, plus the anti-gaming safety penalty.

Less editing ⇒ higher reward. But "fewer edits" can be gamed by getting vaguer or
dropping clinical facts, so the reward subtracts a **safety penalty** whenever a
candidate draft loses clinical content (numbers/doses, drug & diagnosis tokens, or
citation-quote content) relative to the validated baseline. A method that lowers edit
distance by eroding content therefore scores WORSE, not better.

Pure functions, no API — fully reproducible.
"""
from __future__ import annotations

import difflib
import re


def _norm(s: str) -> str:
    return " ".join((s or "").split())


def similarity(draft: str, edited: str) -> float:
    """1.0 = identical, 0.0 = nothing in common (whitespace-normalized)."""
    return difflib.SequenceMatcher(None, _norm(draft), _norm(edited)).ratio()


def edit_distance(draft: str, edited: str) -> float:
    """Normalized edit burden in [0, 1]; lower is better."""
    return 1.0 - similarity(draft, edited)


def is_match(draft: str, edited: str) -> bool:
    """Section needed no edits at all."""
    return _norm(draft) == _norm(edited)


# A "clinical token" = something a clinician must not lose: numbers/doses, or a
# content word (>=4 chars) that is capitalized or all-caps (drug/diagnosis names).
_NUM = re.compile(r"\d[\d.,/:-]*")
_WORD = re.compile(r"[A-Za-z][A-Za-z]{3,}")


def _clinical_tokens(text: str) -> set[str]:
    """Must-keep content in a value: numbers/doses + capitalized/all-caps content words
    (drug & diagnosis names). Citations/flags are separate state metadata that the learning
    never touches, so they are preserved by construction and are not measured here."""
    toks: set[str] = set(m.group(0) for m in _NUM.finditer(text or ""))
    for m in _WORD.finditer(text or ""):
        w = m.group(0)
        if w[:1].isupper() or w.isupper():
            toks.add(w.lower())
    return toks


def safety_penalty(baseline_value: str, candidate: str) -> float:
    """Penalty in [0, 1]: fraction of the baseline value's clinical tokens missing from the
    candidate, plus a vagueness term if the candidate collapsed in length."""
    base = _clinical_tokens(baseline_value)
    if not base:
        return 0.0
    cand = _clinical_tokens(candidate)
    lost = base - cand
    content_penalty = len(lost) / len(base)

    # Vagueness: a big length drop with content loss is the classic gaming move.
    b_len, c_len = len(_norm(baseline_value)), len(_norm(candidate))
    vagueness = 0.0
    if b_len > 0 and c_len < 0.5 * b_len and content_penalty > 0:
        vagueness = 0.25
    return min(1.0, content_penalty + vagueness)


def section_reward(draft: str, candidate: str, edited: str) -> dict:
    """Reward for a (possibly memory-improved) candidate vs the reviewer's edited target.

    reward = similarity(candidate, edited) - safety_penalty(draft, candidate).
    """
    sim = similarity(candidate, edited)
    pen = safety_penalty(draft, candidate)
    return {
        "similarity": sim,
        "edit_distance": 1.0 - sim,
        "safety_penalty": pen,
        "reward": sim - pen,
        "match": is_match(candidate, edited),
    }
