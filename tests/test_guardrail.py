"""Unit tests for the no-fabrication guardrail (validate_and_set).

These are the safety-critical tests: they prove a fabricated "present" value is
structurally impossible, conflicts are never auto-resolved, and a verified value
cannot silently regress to MISSING. Pure offline — no API calls.

Run:  python -m unittest tests.test_guardrail -v
"""
from __future__ import annotations

import unittest

from backend.agent.state import AgentState, Status
from backend.agent.trace import Trace
from backend.tools.validate_set import validate_and_set


def _state() -> AgentState:
    corpus = {
        "patient_id": "test",
        "pages": [
            {"global_page": 1, "doc_type": "discharge", "source": "d.pdf",
             "ocr": {"text": "DIAGNOSIS: SEPSIS\nDate of Admission: 01/01/2026"}},
            {"global_page": 2, "doc_type": "labs", "source": "d.pdf",
             "ocr": {"text": "Hemoglobin 11.2 g/dL\nUrine culture report awaited"}},
        ],
    }
    return AgentState.from_corpus(corpus)


class GuardrailTests(unittest.TestCase):
    def setUp(self):
        self.state = _state()
        self.trace = Trace(verbose=False)

    def vs(self, **kw):
        return validate_and_set(self.state, self.trace, 1, **kw)

    # --- PRESENT path -----------------------------------------------------
    def test_present_with_verifying_citation_accepted(self):
        res = self.vs(field_name="principal_diagnosis", value="SEPSIS",
                      status="present", citations=[{"page": 1, "quote": "DIAGNOSIS: SEPSIS"}])
        self.assertTrue(res["ok"])
        self.assertEqual(res["status_set"], Status.PRESENT.value)
        self.assertEqual(self.state.fields["principal_diagnosis"].status, Status.PRESENT.value)

    def test_present_without_citation_rejected_to_missing(self):
        res = self.vs(field_name="principal_diagnosis", value="SEPSIS",
                      status="present", citations=[])
        self.assertEqual(res["status_set"], Status.MISSING.value)
        self.assertIsNone(self.state.fields["principal_diagnosis"].value)

    def test_present_with_fabricated_citation_rejected(self):
        # Quote does not occur on page 1 -> citation fails verification -> MISSING.
        res = self.vs(field_name="principal_diagnosis", value="PNEUMONIA",
                      status="present", citations=[{"page": 1, "quote": "DIAGNOSIS: PNEUMONIA"}])
        self.assertEqual(res["status_set"], Status.MISSING.value)
        self.assertEqual(res["verified_citations"], 0)

    def test_citation_on_wrong_page_rejected(self):
        # Quote is real text but cited to the wrong page -> not verified.
        res = self.vs(field_name="principal_diagnosis", value="SEPSIS",
                      status="present", citations=[{"page": 2, "quote": "DIAGNOSIS: SEPSIS"}])
        self.assertEqual(res["status_set"], Status.MISSING.value)

    # --- PENDING / MISSING ------------------------------------------------
    def test_pending_allowed(self):
        res = self.vs(field_name="pending_results", value="urine culture awaited",
                      status="pending", citations=[{"page": 2, "quote": "Urine culture report awaited"}])
        self.assertEqual(res["status_set"], Status.PENDING.value)

    def test_missing_allowed(self):
        res = self.vs(field_name="allergies", value=None, status="missing")
        self.assertEqual(res["status_set"], Status.MISSING.value)

    # --- CONFLICT ---------------------------------------------------------
    def test_conflict_with_two_candidates_sets_conflict_and_flags(self):
        res = self.vs(field_name="admission_date", value="", status="conflict",
                      candidates=["01/01/2026", "02/01/2026"])
        self.assertEqual(res["status_set"], Status.CONFLICT.value)
        self.assertTrue(any(f.field == "admission_date" and f.severity == "critical"
                            for f in self.state.flags))

    def test_conflict_with_one_candidate_rejected_to_missing(self):
        res = self.vs(field_name="admission_date", value="", status="conflict",
                      candidates=["01/01/2026"])
        self.assertEqual(res["status_set"], Status.MISSING.value)

    # --- anti-regression --------------------------------------------------
    def test_present_cannot_regress_to_missing(self):
        self.vs(field_name="principal_diagnosis", value="SEPSIS", status="present",
                citations=[{"page": 1, "quote": "DIAGNOSIS: SEPSIS"}])
        res = self.vs(field_name="principal_diagnosis", value="", status="missing")
        self.assertTrue(res.get("regression_blocked"))
        self.assertEqual(self.state.fields["principal_diagnosis"].status, Status.PRESENT.value)

    def test_present_can_escalate_to_conflict(self):
        self.vs(field_name="admission_date", value="01/01/2026", status="present",
                citations=[{"page": 1, "quote": "Date of Admission: 01/01/2026"}])
        res = self.vs(field_name="admission_date", value="", status="conflict",
                      candidates=["01/01/2026", "02/01/2026"])
        self.assertEqual(res["status_set"], Status.CONFLICT.value)

    # --- citation-coverage backstop --------------------------------------
    def test_low_coverage_value_is_flagged_but_kept(self):
        # Value asserts much more than the single verifying quote supports.
        res = self.vs(field_name="hospital_course",
                      value="SEPSIS with pneumonia, meningitis, endocarditis and pancreatitis",
                      status="present", citations=[{"page": 1, "quote": "DIAGNOSIS: SEPSIS"}])
        self.assertEqual(res["status_set"], Status.PRESENT.value)
        self.assertTrue(any("coverage" in f.reason.lower() for f in self.state.flags))

    def test_unknown_field_rejected(self):
        res = self.vs(field_name="not_a_field", value="x", status="present")
        self.assertFalse(res["ok"])


if __name__ == "__main__":
    unittest.main()
