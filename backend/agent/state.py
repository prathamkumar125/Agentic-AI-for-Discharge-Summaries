"""AgentState — the structured, cited discharge-summary state.

The 12 required output sections ARE the state fields. Every field is a
`FieldValue` carrying its status and provenance, so the no-fabrication rule is
expressible in the data model itself: a field can be `present` only when it has a
value AND at least one citation (enforced in `tools/validate_set.py`).

This module is pure data + serialization — no LLM/API calls — so it is fully
unit-testable offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from enum import Enum


class Status(str, Enum):
    PRESENT = "present"    # value found AND cited
    MISSING = "missing"    # not found / not documented
    PENDING = "pending"    # documented as awaited (e.g. "report awaited")
    CONFLICT = "conflict"  # >=2 contradictory candidates; never auto-picked


# The 12 sections, in output order. These keys ARE the AgentState fields.
FIELD_ORDER: list[str] = [
    "patient_demographics",
    "admission_date",
    "discharge_date",
    "principal_diagnosis",
    "secondary_diagnoses",
    "hospital_course",
    "procedures",
    "discharge_medications",
    "allergies",
    "follow_up_instructions",
    "pending_results",
    "discharge_condition",
]

FIELD_LABELS: dict[str, str] = {
    "patient_demographics": "Patient Demographics",
    "admission_date": "Admission Date",
    "discharge_date": "Discharge Date",
    "principal_diagnosis": "Principal Diagnosis",
    "secondary_diagnoses": "Secondary Diagnoses",
    "hospital_course": "Hospital Course",
    "procedures": "Procedures",
    "discharge_medications": "Discharge Medications",
    "allergies": "Allergies",
    "follow_up_instructions": "Follow-up Instructions",
    "pending_results": "Pending Results",
    "discharge_condition": "Discharge Condition",
}


@dataclass
class Citation:
    """Provenance for a field value: a quote anchored to a corpus page."""
    doc: str
    page: int      # global_page in the ingested corpus
    quote: str


@dataclass
class FieldValue:
    value: str | None = None
    status: str = Status.MISSING.value
    citations: list[Citation] = field(default_factory=list)
    notes: str = ""  # conflict candidates / why pending / why flagged


@dataclass
class Flag:
    """A clinician-review escalation."""
    field: str
    reason: str
    severity: str = "warning"  # info | warning | critical


def _norm(s: str) -> str:
    """Lowercase + collapse whitespace — for robust quote/substring matching."""
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


@dataclass
class AgentState:
    patient_id: str
    fields: dict[str, FieldValue] = field(default_factory=dict)
    med_reconciliation: dict | None = None
    flags: list[Flag] = field(default_factory=list)
    step_count: int = 0
    # Corpus index: page -> {text, doc_type, source}. Held for citation
    # verification and on-demand reads; excluded from serialized output.
    ocr_index: dict[int, dict] = field(default_factory=dict)
    _page_norm: dict[int, str] = field(default_factory=dict)  # cache of normalized page text

    # --- construction ------------------------------------------------------
    @classmethod
    def from_corpus(cls, corpus: dict) -> "AgentState":
        """Build an empty state (all 12 fields MISSING) from an ingested corpus."""
        index: dict[int, dict] = {}
        for p in corpus.get("pages", []):
            index[p["global_page"]] = {
                "text": (p.get("ocr") or {}).get("text", "") or "",
                "doc_type": p.get("doc_type", "other"),
                "source": p.get("source", ""),
                "warning": (p.get("ocr") or {}).get("warning"),
            }
        state = cls(patient_id=corpus.get("patient_id", "unknown"), ocr_index=index)
        for name in FIELD_ORDER:
            state.fields[name] = FieldValue()
        return state

    # --- corpus access -----------------------------------------------------
    def page_text(self, page: int) -> str:
        return (self.ocr_index.get(page) or {}).get("text", "")

    def page_norm(self, page: int) -> str:
        if page not in self._page_norm:
            self._page_norm[page] = _norm(self.page_text(page))
        return self._page_norm[page]

    def doc_type_index(self) -> dict[str, list[int]]:
        idx: dict[str, list[int]] = {}
        for page, meta in sorted(self.ocr_index.items()):
            idx.setdefault(meta.get("doc_type", "other"), []).append(page)
        return idx

    # --- helpers -----------------------------------------------------------
    def add_flag(self, field_name: str, reason: str, severity: str = "warning") -> None:
        self.flags.append(Flag(field=field_name, reason=reason, severity=severity))

    def summary(self) -> dict[str, str]:
        """Compact {field: status} view — cheap to feed the brain each turn."""
        return {name: fv.status for name, fv in self.fields.items()}

    def unresolved(self) -> list[str]:
        """Fields not yet PRESENT (still missing/pending/conflict)."""
        return [n for n, fv in self.fields.items() if fv.status != Status.PRESENT.value]

    # --- serialization (output-facing; excludes bulky corpus) --------------
    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "step_count": self.step_count,
            "fields": {
                name: {
                    "label": FIELD_LABELS[name],
                    "value": fv.value,
                    "status": fv.status,
                    "citations": [asdict(c) for c in fv.citations],
                    "notes": fv.notes,
                }
                for name, fv in self.fields.items()
            },
            "med_reconciliation": self.med_reconciliation,
            "flags": [asdict(f) for f in self.flags],
        }
