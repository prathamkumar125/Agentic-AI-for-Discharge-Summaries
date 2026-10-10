"""Unit tests for medication reconciliation key-matching.

Guards against the form-prefix bug: chart entries like "INJ PAN" and
"INJ H. ACTRAPID" must NOT collapse to the same key ("inj") and be reported as a
single drug "changing" into an unrelated one.

Run:  python -m unittest tests.test_reconcile -v
"""
from __future__ import annotations

import unittest

from backend.tools.reconcile import _key, reconcile


class KeyTests(unittest.TestCase):
    def test_strips_form_route_dose_tokens(self):
        self.assertEqual(_key("INJ MEROMAC"), "meromac")
        self.assertEqual(_key("INJ PAN 40 mg"), "pan")
        self.assertEqual(_key("INS LANTUS 10 U"), "lantus")
        self.assertEqual(_key("TAB DOLO 650"), "dolo")
        self.assertEqual(_key("INJ H. ACTRAPID"), "actrapid")

    def test_distinct_drugs_get_distinct_keys(self):
        self.assertNotEqual(_key("INJ PAN"), _key("INJ H. ACTRAPID"))
        self.assertNotEqual(_key("INJ MEROMAC"), _key("INJ EMESET"))


class ReconcileTests(unittest.TestCase):
    def test_unrelated_drugs_not_reported_as_changed(self):
        adm = [{"name": "INJ PAN", "dose": "40 mg", "route": "IV"},
               {"name": "INJ MEROMAC", "dose": "1 g"},
               {"name": "INS LANTUS", "dose": "10 U"}]
        dis = [{"name": "INJ H. ACTRAPID", "dose": "10 u", "route": "S/C"},
               {"name": "INS LANTUS", "dose": "10 U"}]
        r = reconcile(adm, dis)
        self.assertEqual(r["changed"], [])  # PAN does not "change into" ACTRAPID
        names_stopped = {m["name"] for m in r["stopped"]}
        self.assertIn("INJ PAN", names_stopped)
        self.assertIn("INJ MEROMAC", names_stopped)
        self.assertEqual([m["name"] for m in r["added"]], ["INJ H. ACTRAPID"])
        self.assertEqual([m["name"] for m in r["continued"]], ["INS LANTUS"])

    def test_real_dose_change_detected(self):
        adm = [{"name": "INJ PAN", "dose": "40 mg", "route": "IV", "freq": "1-0-0"}]
        dis = [{"name": "INJ PAN", "dose": "80 mg", "route": "IV", "freq": "1-0-0"}]
        r = reconcile(adm, dis)
        self.assertEqual(len(r["changed"]), 1)
        self.assertEqual(r["changed"][0]["from"]["dose"], "40 mg")
        self.assertEqual(r["changed"][0]["to"]["dose"], "80 mg")

    def test_undocumented_change_surfaced(self):
        adm = [{"name": "TAB DOLO", "dose": "650"}]
        dis = []
        r = reconcile(adm, dis)
        self.assertTrue(any(c["change"] == "stopped" for c in r["changes_without_documented_reason"]))

    def test_documented_reason_not_flagged(self):
        adm = []
        dis = [{"name": "TAB METFORMIN", "dose": "500", "reason": "new T2DM diagnosis"}]
        r = reconcile(adm, dis)
        self.assertEqual(r["changes_without_documented_reason"], [])


if __name__ == "__main__":
    unittest.main()
