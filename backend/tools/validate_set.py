"""validate_set — THE no-fabrication guardrail and the only field write path.

Nothing else in the system sets a field directly. Every write is routed here so
the guarantee holds in one auditable place:

  * status="present"  is accepted ONLY IF value is non-empty AND at least one
    citation is VERIFIED (its quote actually occurs in the cited page's OCR text).
    Otherwise the write is rejected and the field is kept MISSING. This makes a
    fabricated "present" value structurally impossible — not merely discouraged.
  * status="conflict" requires >=2 documented candidate values; it is never
    auto-resolved, and it raises a clinician-review flag.
  * status="pending"/"missing" are always allowed (they carry a reason in notes).
  * Citations whose quote cannot be found on the cited page are DROPPED (anti
    hallucinated-citation check), and the drop is logged.

It is a plain function (also exposed as a tool in the registry) so it is fully
unit-testable without any API calls.
"""
from __future__ import annotations

import re

from ..agent.state import AgentState, Citation, FieldValue, Status, FIELD_ORDER, _norm

_MIN_QUOTE_CHARS = 4

# Below this fraction of a 'present' value's informative tokens being backed by its
# verified citation quotes, we keep the value but FLAG it: the no-fab guardrail
# verifies each quote exists on its page, but cannot by itself prove every clause of
# the value is supported (field-level, not claim-level). Low coverage = the value
# carries content not found in any cited quote -> surface for clinician review.
_COVERAGE_FLAG_THRESHOLD = 0.55
# Boilerplate / structural words ignored when measuring coverage.
_STOPISH = {
    "the", "and", "with", "from", "this", "that", "were", "was", "for", "not",
    "documented", "noted", "value", "values", "field", "page", "pages", "report",
    "reports", "indicating", "including", "advice", "plan", "used", "list", "chart",
    "drug", "medication", "medications", "discharge", "patient", "status", "given",
}


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _informative_tokens(text: str) -> set[str]:
    """Lowercased alnum tokens >=4 chars, minus boilerplate — the 'content' of a value."""
    return {t for t in re.findall(r"[a-z0-9]{4,}", (text or "").lower()) if t not in _STOPISH}


def _citation_coverage(value: str, citations: list[Citation]) -> float:
    """Fraction of the value's informative tokens that appear in the cited quotes.

    1.0 means every content word of the value is present in some verified quote;
    low values mean the value asserts content the quotes don't support. Returns 1.0
    for short/empty values (nothing to measure).
    """
    vtok = _informative_tokens(value)
    if not vtok:
        return 1.0
    qtok: set[str] = set()
    for c in citations:
        qtok |= _informative_tokens(c.quote)
    return len(vtok & qtok) / len(vtok)


def _verify_citation(state: AgentState, cit: Citation) -> bool:
    """True iff the quote genuinely occurs on the cited page (robust to spacing/punct)."""
    if cit.page not in state.ocr_index:
        return False
    quote = (cit.quote or "").strip()
    if len(quote) < _MIN_QUOTE_CHARS:
        return False
    page_norm = state.page_norm(cit.page)
    if _norm(quote) in page_norm:
        return True
    # Fallback: ignore punctuation/spacing entirely (OCR spacing is noisy).
    qa = _alnum(quote)
    return len(qa) >= _MIN_QUOTE_CHARS and qa in _alnum(state.page_text(cit.page))


def _coerce_citations(raw) -> list[Citation]:
    out: list[Citation] = []
    for c in raw or []:
        if isinstance(c, Citation):
            out.append(c)
        elif isinstance(c, dict):
            out.append(Citation(
                doc=str(c.get("doc", "")),
                page=int(c.get("page", -1)),
                quote=str(c.get("quote", "")),
            ))
    return out


def validate_and_set(
    state: AgentState,
    trace,
    step: int,
    field_name: str,
    value,
    status: str = Status.PRESENT.value,
    citations=None,
    notes: str = "",
    candidates: list | None = None,
) -> dict:
    """Validate and (if it passes) commit a field write. Returns a result dict."""
    if field_name not in FIELD_ORDER:
        result = {"ok": False, "status_set": None, "reason": f"unknown field '{field_name}'"}
        trace.guardrail(step, field_name, "reject", result["reason"])
        return result

    if isinstance(value, (list, tuple)):
        value = "; ".join(str(v) for v in value)
    value = (value or "").strip() if isinstance(value, str) else value

    cites = _coerce_citations(citations)
    verified = [c for c in cites if _verify_citation(state, c)]
    dropped = [c for c in cites if c not in verified]

    def commit(fv: FieldValue, decision: str, reason: str) -> dict:
        existing = state.fields.get(field_name)
        # Anti-regression guard: a field that is already PRESENT with a verified
        # citation must NOT silently fall back to MISSING because of a later
        # failed/empty re-write (e.g. the brain redundantly re-records and its new
        # quotes don't verify). Refining to another PRESENT, or escalating to
        # CONFLICT, is still allowed — only the data-losing drop to MISSING is blocked.
        if (existing is not None
                and existing.status == Status.PRESENT.value and existing.citations
                and fv.status == Status.MISSING.value):
            trace.guardrail(step, field_name, "keep-present",
                            f"blocked regression to missing; kept existing verified value [{reason}]")
            return {
                "ok": True,
                "status_set": Status.PRESENT.value,
                "verified_citations": len(existing.citations),
                "dropped_citations": len(dropped),
                "reason": "kept existing verified value; ignored regression to missing",
                "regression_blocked": True,
            }
        state.fields[field_name] = fv
        trace.guardrail(step, field_name, decision, reason)
        return {
            "ok": fv.status != Status.MISSING.value or status == Status.MISSING.value,
            "status_set": fv.status,
            "verified_citations": len(verified),
            "dropped_citations": len(dropped),
            "reason": reason,
        }

    # --- CONFLICT: never auto-pick; require >=2 candidates; flag for review ---
    if status == Status.CONFLICT.value:
        cand = candidates or []
        if len(cand) < 2:
            return commit(
                FieldValue(value=value or None, status=Status.MISSING.value,
                           citations=verified, notes=notes or "conflict claimed without 2+ candidates"),
                "reject", "conflict needs >=2 candidates; kept MISSING",
            )
        summary = " | ".join(str(c) for c in cand)
        state.add_flag(field_name, f"Conflicting values: {summary}", severity="critical")
        return commit(
            FieldValue(value=summary, status=Status.CONFLICT.value,
                       citations=verified,
                       notes=(notes + f"  CANDIDATES: {summary}").strip()),
            "conflict", f"surfaced {len(cand)} candidates + flagged for review",
        )

    # --- PENDING: documented-as-awaited; allowed (citation verified if given) ---
    if status == Status.PENDING.value:
        return commit(
            FieldValue(value=value or None, status=Status.PENDING.value,
                       citations=verified, notes=notes or "documented as pending/awaited"),
            "set-pending", "marked PENDING (not imputed)",
        )

    # --- MISSING: explicit "not found" ---
    if status == Status.MISSING.value:
        return commit(
            FieldValue(value=None, status=Status.MISSING.value, citations=[],
                       notes=notes or "not found in source documents"),
            "set-missing", "marked MISSING",
        )

    # --- PRESENT: the guarded path — value AND a verified citation required ---
    if not value:
        return commit(
            FieldValue(value=None, status=Status.MISSING.value, citations=[],
                       notes=notes or "no value provided"),
            "reject", "PRESENT rejected: empty value -> kept MISSING",
        )
    if not verified:
        why = ("all citations failed verification (quote not found on cited page)"
               if cites else "no citations supplied")
        return commit(
            FieldValue(value=None, status=Status.MISSING.value, citations=[],
                       notes=f"{notes}  [guardrail: {why}]".strip()),
            "reject", f"PRESENT rejected: {why} -> kept MISSING",
        )
    coverage = _citation_coverage(value, verified)
    note_out = notes
    if coverage < _COVERAGE_FLAG_THRESHOLD:
        pct = round(coverage * 100)
        note_out = (f"{notes}  [guardrail: only {pct}% of the value's content is backed by "
                    f"the cited quotes — verify the unsupported parts]").strip()
        state.add_flag(field_name,
                       f"Low citation coverage ({pct}%): value contains content not found in "
                       f"its cited quotes; clinician should verify.", severity="warning")
    return commit(
        FieldValue(value=value, status=Status.PRESENT.value, citations=verified, notes=note_out),
        "accept", f"PRESENT accepted with {len(verified)} verified citation(s)"
                  + (f"; dropped {len(dropped)} unverified" if dropped else "")
                  + (f"; LOW coverage {round(coverage*100)}%" if coverage < _COVERAGE_FLAG_THRESHOLD else ""),
    )
