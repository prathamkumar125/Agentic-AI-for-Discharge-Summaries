"""Segment OCR'd pages into logical document types.

The source is one combined ~71-page scan holding many document types, and —
crucially — almost every page carries a clear TITLE on its first line
(`NURSES NOTES`, `DRUG CHART`, `INVESTIGATION ... REFERENCE RANGE`,
`USG ABDOMEN`, `CONSULTATION SHEET`, ...). So classification is **title-first**:
match the heading; only fall back to body keywords when the title is
uninformative (e.g. the discharge-summary narrative page, which opens with
`DIAGNOSIS:` rather than a doc title).

This is a transparent, deterministic ORGANIZING aid only. The agent always
reasons over the raw OCR text + page numbers, so a mis-label here can never
cause a wrong clinical fact (that is the no-fab guardrail's job downstream).
Each result carries its method ('title' / 'body' / 'none') and scores so the
grouping is explainable.
"""
from __future__ import annotations

# Canonical types, ordered most-specific-first (used for tie-breaks + display).
DOC_TYPES = [
    "discharge", "admission", "consultation", "labs",
    "imaging", "medication", "er", "nursing", "progress",
]

# Title cues, checked in order; first hit wins. Matched as substrings against the
# uppercased first two non-empty lines of the page.
_TITLE_RULES: list[tuple[str, str]] = [
    ("DISCHARGE SUMMARY", "discharge"),
    ("DISCHARGE CHECK", "discharge"),
    ("DISCHARGE ADVICE", "discharge"),
    ("ADMISSION RECORD", "admission"),
    ("CASE RECORD", "admission"),
    ("ADMISSION NOTE", "admission"),
    ("NURSING ASSESSMENT", "nursing"),
    ("NURSING DOCUMENTATION", "nursing"),
    ("NURSES NOTES", "nursing"),
    ("NURSE'S NOTES", "nursing"),
    ("BED SORE", "nursing"),
    ("INTAKE/OUTPUT", "nursing"),
    ("INTAKE OUTPUT", "nursing"),
    ("GRAPHIC (TPR)", "nursing"),
    ("TPR", "nursing"),
    ("MONITORING CHART", "nursing"),
    ("PROCEDURE CHART", "nursing"),
    ("CAUTI", "nursing"),
    ("CONSULTATION", "consultation"),
    ("DRUG CHART", "medication"),
    ("MEDICATION CHART", "medication"),
    ("ICU-CHART", "medication"),
    ("ICU CHART", "medication"),
    ("INVESTIGATION CHECKLIST", "labs"),
    ("INVESTIGATION CHECK LIST", "labs"),
    ("ER OBSERVATION", "er"),
    ("USG", "imaging"),
    ("ULTRASOUND", "imaging"),
    ("ECHO REPORT", "imaging"),
    ("2D ECHO", "imaging"),
    ("TRANS-THORACIC", "imaging"),
    ("ECHO", "imaging"),
    ("X-RAY", "imaging"),
    ("X RAY", "imaging"),
    ("CT KUB", "imaging"),
    ("C.T. KUB", "imaging"),
    ("KUB (PLAIN)", "imaging"),
    ("CT SCAN", "imaging"),
    ("MRI", "imaging"),
    ("DOPPLER", "imaging"),
    ("INVESTIGATION", "labs"),
    ("TEST DESCRIPTION", "labs"),
    ("BIOCHEMISTRY", "labs"),
    ("HAEMATOLOGY", "labs"),
    ("HEMATOLOGY", "labs"),
]

# Weighted body cues for title-less pages. Tuned to catch the discharge-summary
# narrative (hospital course) and imaging-report continuation pages.
_BODY_KEYWORDS: dict[str, list[tuple[str, int]]] = {
    "discharge": [
        ("course in the hospital", 3), ("condition at discharge", 3),
        ("advised on discharge", 3), ("discharge medication", 2),
        ("to be discharged", 2),
    ],
    "imaging": [
        ("pericardium", 2), ("doppler", 2), ("pulmonary artery", 2),
        ("ejection fraction", 2), ("aortic", 1), ("ventricle", 1),
        ("fatty infiltration", 1), ("echocardio", 2), ("aortic arch", 2),
    ],
    "labs": [
        ("reference range", 2), ("result value", 2), ("biochemistry", 2),
        ("haematology", 2), ("haemoglobin", 1), ("creatinine", 1),
    ],
    "medication": [
        ("nebulisation", 2), ("drug name", 2), (" inj ", 1), (" tab ", 1),
    ],
    "nursing": [
        ("cauti", 3), ("catheter associated", 3), ("date of insertion", 2),
    ],
}


def _title(text: str) -> str:
    # First 3 non-empty lines: pages often carry a hospital letterhead on lines
    # 1-2 with the real document title on line 3.
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return " ".join(lines[:3]).upper()


def _body_scores(text: str) -> dict[str, int]:
    low = (text or "").lower()
    scores: dict[str, int] = {}
    for dt, kws in _BODY_KEYWORDS.items():
        s = sum(low.count(kw) * w for kw, w in kws)
        if s:
            scores[dt] = s
    return scores


def classify_page(text: str) -> tuple[str, dict]:
    """Return (doc_type, info). info = {method, title, scores}.

    method: 'title' (matched a heading), 'body' (keyword fallback), or 'none'.
    """
    title = _title(text)
    for cue, dt in _TITLE_RULES:
        if cue in title:
            return dt, {"method": "title", "title": title[:80], "matched": cue}

    scores = _body_scores(text)
    if scores:
        best = max(scores, key=lambda dt: (scores[dt], -DOC_TYPES.index(dt)))
        return best, {"method": "body", "title": title[:80], "scores": scores}

    return "other", {"method": "none", "title": title[:80], "scores": {}}
