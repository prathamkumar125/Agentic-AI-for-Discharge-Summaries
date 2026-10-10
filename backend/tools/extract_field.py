"""extract_field — focused, grounded extraction of one field from given pages.

The agent decides WHICH field and WHICH pages; this tool does the careful reading:
it makes one constrained LLM call (gpt-5-mini, forced function output) over only
the supplied page text and proposes {value, status, citations, candidates}. Every
citation must be a verbatim quote with its page number.

This is a *proposal* — the registry commits it via `validate_and_set`, which
re-verifies each quote against the page. So even if this extractor were to
hallucinate a quote, the guardrail would reject it and keep the field MISSING.
Two layers, one guarantee.
"""
from __future__ import annotations

import json

from ..config import BRAIN_MODEL
from ..agent.state import AgentState
from .parse_doc import parse_page_spec

# Per-field guidance: what the value should contain.
FIELD_HINTS: dict[str, str] = {
    "patient_demographics": (
        "Capture ONLY the explicit demographic identifier fields: Name, Age, Sex/Gender, "
        "MRN/IP No, Ward/Bed — exactly as written. Do NOT infer sex/gender from narrative "
        "pronouns (he/she/his/her): pronouns in templated notes are unreliable and are NOT "
        "a demographic value. If the identifier fields are blank/redacted, return "
        "status='missing' with a note that identifiers are redacted in the source. Only use "
        "status='conflict' if two EXPLICIT demographic values disagree (e.g. two different "
        "ages or two different stated sexes in a 'Sex:' field)."),
    "admission_date": (
        "The admission date, from explicit 'Date of Admission' fields. Dates in this record are "
        "inconsistently transcribed across pages. Gather EVERY distinct admission-date value you "
        "see; if two or more DIFFERENT values appear, return status='conflict' with all of them as "
        "candidates (never pick one). A single value whose YEAR is implausibly far from the rest of "
        "the record (e.g. 2023 when other pages say 2026) is suspect — treat it as a conflict "
        "candidate or flag it, do NOT assert it alone as the admission date."),
    "discharge_date": (
        "The discharge date, from explicit 'Date of Discharge' fields or an explicit discharge event "
        "(e.g. 'patient got discharge' dated on a nursing note). Gather EVERY distinct discharge-date "
        "value; if they disagree, return status='conflict' with all candidates (never pick one)."),
    "principal_diagnosis": "The single main diagnosis treated this admission.",
    "secondary_diagnoses": "Other diagnoses / comorbidities (e.g. known thyroid disorder).",
    "hospital_course": "A concise narrative of the hospital stay: presentation, key findings, treatment, response.",
    "procedures": (
        "Procedures and investigations performed/documented (e.g. USG abdomen, CT KUB, 2D echo, "
        "catheter insertion, ECG). If any are documented as done, the field is 'present'. Do NOT "
        "mark this field 'pending' just because a lab RESULT is awaited — a pending result belongs "
        "to the pending_results field, not here."),
    "discharge_medications": "Medications to take after discharge, with dose/route/frequency. If no explicit discharge prescription exists, use the most recent inpatient DRUG CHART / medication chart as the list and note it is from the drug chart (not a formal discharge Rx); these may be inpatient IV meds.",
    "allergies": "Known drug allergies. 'Not known'/'No' counts as a documented value.",
    "follow_up_instructions": "Advice/instructions given on discharge and any follow-up plan.",
    "pending_results": (
        "Tests/results explicitly documented as awaited or not-yet-resulted (e.g. 'culture report "
        "awaited'). IMPORTANT: a result is pending ONLY if no final value for it appears ANYWHERE "
        "in the provided pages. If an earlier note says a test is 'awaited' but a LATER page actually "
        "contains its result, the test is NOT pending — set status='present', report the result, and "
        "in 'notes' flag the discrepancy (earlier page said awaited, later page has the result) for review. "
        "If nothing is genuinely outstanding, status='missing' with a note that no results remain pending."),
    "discharge_condition": "The patient's condition/status at discharge (prefer an explicit discharge note; nursing notes are supporting evidence — say so if that is the only source).",
}

_PROPOSE_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_field",
        "description": "Propose a value for one discharge-summary field, grounded in the provided pages.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["present", "missing", "pending", "conflict"],
                    "description": "present=found & quotable; missing=not in text; pending=documented as awaited; conflict=2+ contradictory values.",
                },
                "value": {"type": "string", "description": "The ACTUAL field content for the discharge summary (real names/dates/diagnoses/med lines/narrative in the source's wording) — NOT a description of where it came from. Empty if missing."},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "page": {"type": "integer"},
                            "quote": {"type": "string", "description": "VERBATIM text copied from that page."},
                        },
                        "required": ["page", "quote"],
                    },
                },
                "candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "If status=conflict, the 2+ contradictory values found.",
                },
                "notes": {"type": "string", "description": "Why pending/conflict/missing, if applicable."},
            },
            "required": ["status", "value", "citations"],
        },
    },
}

_SYSTEM = (
    "You extract a SINGLE discharge-summary field from scanned clinical pages. "
    "Rules you MUST follow:\n"
    "1. Use ONLY the provided page text. Never use outside knowledge or guess.\n"
    "2. 'value' MUST be the ACTUAL field content that a clinician would read in the "
    "discharge summary — the real names, dates, diagnosis text, medication lines, or "
    "the hospital-course narrative — using the source's own wording (lightly cleaned). "
    "NEVER write a meta-description such as 'summary from page 1' or 'details present "
    "on page X'. If you cannot state the real content, the field is 'missing'.\n"
    "3. Every citation quote must be copied VERBATIM from the page and tagged with "
    "that page's number (shown as 'PAGE n'); the quote is the EVIDENCE for 'value'. "
    "Every factual clause in 'value' MUST be supported by one of your quotes — do NOT "
    "add names, dates, doses, or findings that are not present in the quotes you cite.\n"
    "4. If the field is not present, return status='missing' with empty value.\n"
    "5. status='pending' applies ONLY when THIS field's own content is explicitly "
    "documented as awaited/not-yet-resulted. Do NOT mark a field 'pending' merely "
    "because some related item in a DIFFERENT field is pending. If a result that was "
    "once 'awaited' actually appears resulted on any provided page, it is NOT pending.\n"
    "6. status='conflict' ONLY when sources DISAGREE about the SAME explicit fact (e.g. "
    "two different admission dates, or two different stated ages). A list of co-existing "
    "diagnoses is NOT a conflict — the first is principal, the rest are secondary. Narrative "
    "pronouns (he/she) are NOT an explicit fact and must NEVER by themselves create a sex "
    "'conflict'. Put real candidates in 'candidates' with a citation for each; never pick one.\n"
    "7. Call propose_field exactly once."
)

_MAX_PAGE_CHARS = 6000  # cap per page to control cost on a runaway page


def _build_context(state: AgentState, pages: list[int]) -> str:
    blocks = []
    for p in pages:
        meta = state.ocr_index.get(p)
        if not meta:
            continue
        text = (meta.get("text") or "")[:_MAX_PAGE_CHARS]
        warn = f" [WARNING: {meta['warning']}]" if meta.get("warning") else ""
        blocks.append(f"=== PAGE {p} ({meta.get('doc_type')}){warn} ===\n{text}")
    return "\n\n".join(blocks)


def extract_field(client, state: AgentState, field_name: str, pages) -> dict:
    """Propose a value for `field_name` from `pages`. Returns a proposal dict."""
    max_page = max(state.ocr_index) if state.ocr_index else 0
    nums = parse_page_spec(pages, max_page)
    if not nums:
        return {"status": "missing", "value": "", "citations": [], "candidates": [],
                "notes": f"no valid pages in {pages!r}", "pages_used": []}

    context = _build_context(state, nums)
    hint = FIELD_HINTS.get(field_name, "")
    user = (f"FIELD: {field_name}\nWHAT TO FIND: {hint}\n\n"
            f"PAGES:\n{context}\n\nCall propose_field for this field only.")

    resp = client.chat.completions.create(
        model=BRAIN_MODEL,
        messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        tools=[_PROPOSE_TOOL],
        tool_choice={"type": "function", "function": {"name": "propose_field"}},
    )
    msg = resp.choices[0].message
    if not msg.tool_calls:
        return {"status": "missing", "value": "", "citations": [], "candidates": [],
                "notes": "extractor returned no proposal", "pages_used": nums}
    try:
        args = json.loads(msg.tool_calls[0].function.arguments)
    except (json.JSONDecodeError, TypeError):
        return {"status": "missing", "value": "", "citations": [], "candidates": [],
                "notes": "extractor returned unparseable proposal", "pages_used": nums}

    args.setdefault("citations", [])
    args.setdefault("candidates", [])
    args.setdefault("notes", "")
    args["pages_used"] = nums
    return args
