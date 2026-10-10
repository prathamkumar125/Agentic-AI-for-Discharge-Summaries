"""Unit tests for the mock drug-interaction tool.

Proves brand names actually resolve to generics (so the check is real, not a
no-op), that a catalogued dangerous pair fires, and that "none found" is always
qualified as a mock-DB limitation rather than a clinical all-clear.

Run:  python -m unittest tests.test_drug_interaction -v
"""
from __future__ import annotations

import unittest

from backend.tools.drug_interaction import _normalize, drug_interaction


class NormalizeTests(unittest.TestCase):
    def test_brand_names_resolve_to_generics(self):
        self.assertEqual(_normalize("INJ MEROMAC"), "meropenem")
        self.assertEqual(_normalize("INJ PAN"), "pantoprazole")
        self.assertEqual(_normalize("INJ EMESET"), "ondansetron")
        self.assertEqual(_normalize("INS LANTUS"), "insulin")
        self.assertEqual(_normalize("INJ H. ACTRAPID"), "insulin")
        self.assertEqual(_normalize("TAB DOLO 650"), "paracetamol")

    def test_form_prefix_not_used_as_name(self):
        self.assertNotEqual(_normalize("INJ MEROMAC"), "inj")


class InteractionTests(unittest.TestCase):
    def test_catalogued_pair_fires(self):
        r = drug_interaction(["Tab Ecosprin", "Warfarin"])
        self.assertEqual(len(r["interactions"]), 1)
        self.assertEqual(r["interactions"][0]["severity"], "major")

    def test_this_patients_meds_have_no_catalogued_pair(self):
        r = drug_interaction(["INJ MEROMAC", "INJ PAN", "INJ EMESET", "INS LANTUS",
                              "INJ H. ACTRAPID", "TAB DOLO"])
        self.assertEqual(r["interactions"], [])
        # ...but the meds genuinely resolved (the check was not a no-op):
        self.assertIn("meropenem", r["checked"])
        self.assertIn("insulin", r["checked"])

    def test_none_found_is_qualified_not_all_clear(self):
        r = drug_interaction(["paracetamol"])
        self.assertIn("not", r["note"].lower())  # "does NOT mean none exist"
        self.assertIn("MOCK", r["source"])

    def test_string_input_accepted(self):
        r = drug_interaction("aspirin, warfarin")
        self.assertEqual(len(r["interactions"]), 1)


if __name__ == "__main__":
    unittest.main()
