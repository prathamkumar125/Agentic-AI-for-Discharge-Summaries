"""memory — structured correction-memory: the learning mechanism (Part 2).

It LEARNS from accumulated `(draft, edited)` pairs by inducing reusable substitution
rules straight from the edit diffs (token-level `difflib` alignment) — it never sees the
reviewer's hidden policy, it infers it. A rule learned on one section generalizes to any
other section that contains the same phrasing, which is exactly what drives the held-out
improvement curve.

Why this mechanism (justification, per the task):
  * DPO/SFT need training infra we don't have (API-only, no GPU) and a cold start with one
    patient; a learned reward model + best-of-N is heavier and still needs eval data.
  * A structured correction-memory is **online, inspectable, cheap, and structurally safe**:
    it only rewrites *presentation* (abbreviations/phrasings) and is applied either
    deterministically (here, for reproducible measurement) or injected into the agent's
    prompt in production via `as_prompt_block()`. The validated clinical values it reshapes
    came from the no-fab guardrail and are never re-fabricated.

Rules that would not generalize (anything containing a digit — dates, doses) are NOT
learned, so the memory can't memorize patient-specific values.
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

_MAX_RULE_TOKENS = 4  # only learn short phrasings/abbreviations (these generalize)


def _tokens(text: str) -> list[str]:
    return (text or "").split()


def _has_digit(s: str) -> bool:
    return any(c.isdigit() for c in s)


class CorrectionMemory:
    """A learned set of (source phrase -> standardized replacement) rules."""

    def __init__(self) -> None:
        # src_lower -> {target_lower_canonical: support_count}
        self._rules: dict[str, dict[str, int]] = {}

    # --- learning ----------------------------------------------------------
    def learn(self, pairs: list[tuple[str, str]]) -> None:
        """Induce substitution rules from `(draft, edited)` text pairs."""
        for draft, edited in pairs:
            self._learn_pair(draft, edited)

    def _learn_pair(self, draft: str, edited: str) -> None:
        a, b = _tokens(draft), _tokens(edited)
        sm = difflib.SequenceMatcher(None, [t.lower() for t in a], [t.lower() for t in b])
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != "replace":
                continue
            src = " ".join(a[i1:i2])
            tgt = " ".join(b[j1:j2])
            if not src or not tgt:
                continue
            if (i2 - i1) > _MAX_RULE_TOKENS or (j2 - j1) > 2 * _MAX_RULE_TOKENS:
                continue
            if _has_digit(src) or _has_digit(tgt):   # don't memorize dates/doses
                continue
            if src.lower() == tgt.lower():
                continue
            bucket = self._rules.setdefault(src.lower(), {})
            bucket[tgt] = bucket.get(tgt, 0) + 1

    # --- the learned rule table (most-supported target per source) ---------
    def rules(self) -> dict[str, str]:
        return {src: max(tgts, key=tgts.get) for src, tgts in self._rules.items()}

    def __len__(self) -> int:
        return len(self._rules)

    # --- applying (deterministic; reproducible measurement) ----------------
    def apply(self, text: str) -> str:
        """Rewrite `text` by applying the learned substitutions (longest source first)."""
        if not text:
            return text
        out = text
        for src, tgt in sorted(self.rules().items(), key=lambda kv: -len(kv[0])):
            pattern = re.compile(
                r"(?<![A-Za-z0-9])" + re.escape(src).replace(r"\ ", r"\s+") + r"(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
            out = pattern.sub(lambda m, t=tgt: (t[0].upper() + t[1:] if m.group(0)[:1].isupper() else t), out)
        return " ".join(out.split())

    # --- production path: inject into the agent's prompt -------------------
    def as_prompt_block(self) -> str:
        """A compact style guide to inject into the render/extraction prompt in production.

        The agent would apply these to PRESENTATION only — never to clinical values or
        citations. (Measurement here uses `apply()` for reproducibility / to isolate the
        learning signal from LLM non-determinism.)"""
        rules = self.rules()
        if not rules:
            return ""
        lines = ["LEARNED STYLE CORRECTIONS (apply to wording/formatting only — never change a "
                 "clinical value, name, date, dose, or citation):"]
        for src, tgt in sorted(rules.items()):
            lines.append(f"  - write \"{src}\" as \"{tgt}\"")
        return "\n".join(lines)

    # --- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {"rules": {src: dict(tgts) for src, tgts in self._rules.items()}}

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                              encoding="utf-8")

    @classmethod
    def load(cls, path) -> "CorrectionMemory":
        m = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        m._rules = {src: dict(tgts) for src, tgts in data.get("rules", {}).items()}
        return m
