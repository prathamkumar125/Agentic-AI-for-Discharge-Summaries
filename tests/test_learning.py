"""Unit tests for Part 2 — the learning-from-doctor-edits loop.

Covers: the reviewer applies its hidden policy (and preserves content); the reward's
safety penalty catches content loss / vagueness; the correction-memory induces rules from
edits and GENERALIZES them to unseen text without learning patient-specific values; and the
end-to-end eval lowers held-out edit distance while the safety invariant holds.

Run:  python -m unittest tests.test_learning -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.learning.reviewer import apply_policy, make_pairs
from backend.learning.reward import similarity, edit_distance, safety_penalty, section_reward
from backend.learning.memory import CorrectionMemory
from backend.learning.evaluate import run_curve


class ReviewerTests(unittest.TestCase):
    def test_expands_abbreviations(self):
        self.assertEqual(apply_policy("K/C/O diabetes"), "Known case of diabetes")
        self.assertIn("result pending", apply_policy("culture report awaited"))

    def test_leaves_clean_text_unchanged(self):
        self.assertEqual(apply_policy("Patient is comfortable"), "Patient is comfortable")

    def test_make_pairs_skips_empty_values(self):
        state = {"fields": {
            "a": {"label": "A", "value": "Adv USG", "status": "present", "citations": []},
            "b": {"label": "B", "value": "", "status": "missing", "citations": []},
        }}
        fields = [p["field"] for p in make_pairs(state)]
        self.assertEqual(fields, ["a"])


class RewardTests(unittest.TestCase):
    def test_similarity_and_distance(self):
        self.assertEqual(similarity("abc", "abc"), 1.0)
        self.assertGreater(edit_distance("a b c", "x y z"), 0.5)

    def test_safety_penalty_zero_when_content_kept(self):
        base = "Serum creatinine 1.65 elevated"
        self.assertEqual(safety_penalty(base, base), 0.0)
        # expanding an abbreviation keeps all clinical tokens -> still safe
        self.assertEqual(safety_penalty("Adv USG; creatinine 1.65", "Advised USG; creatinine 1.65"), 0.0)

    def test_safety_penalty_flags_dropped_content(self):
        base = "DIAGNOSIS SEPSIS; creatinine 1.65"
        vague = "labs noted"   # dropped SEPSIS, DIAGNOSIS, 1.65
        self.assertGreater(safety_penalty(base, vague), 0.0)

    def test_section_reward_combines_similarity_and_penalty(self):
        r = section_reward("Adv USG", "Advised USG", "Advised USG")
        self.assertAlmostEqual(r["similarity"], 1.0)
        self.assertEqual(r["safety_penalty"], 0.0)
        self.assertTrue(r["match"])


class MemoryTests(unittest.TestCase):
    def test_learns_and_generalizes_a_rule(self):
        mem = CorrectionMemory()
        mem.learn([("K/C/O diabetes", "known case of diabetes")])
        # rule generalizes to a DIFFERENT string containing the same source phrase
        self.assertEqual(mem.apply("K/C/O hypertension"), "Known case of hypertension")

    def test_does_not_learn_digit_rules(self):
        mem = CorrectionMemory()
        mem.learn([("date 26-2-26 noted", "date 26 Feb 2026 noted")])
        self.assertEqual(len(mem), 0)  # nothing with a digit is memorized

    def test_prompt_block_lists_rules(self):
        mem = CorrectionMemory()
        mem.learn([("Adv rest", "advised rest")])
        self.assertIn("adv", mem.as_prompt_block().lower())


class EvaluateTests(unittest.TestCase):
    def _state(self):
        # Sections sharing abbreviations so a rule learned on one transfers to others.
        def f(v):
            return {"label": "L", "value": v, "status": "present", "citations": []}
        return {"patient_id": "t", "flags": [], "fields": {
            "a": f("K/C/O diabetes. Adv USG abdomen."),
            "b": f("c/o fever. Adv CT scan."),
            "c": f("K/C/O hypertension on Rx."),
            "d": f("Adv USG pelvis. c/o pain."),
        }}

    def test_run_curve_improves_and_stays_safe(self):
        with tempfile.TemporaryDirectory() as d:
            res = run_curve(self._state(), Path(d), seed=1, n_splits=10)
            self.assertTrue(res["safety_invariant_held"])
            self.assertEqual(res["max_safety_penalty"], 0.0)
            # learning should not increase held-out edit distance
            self.assertLessEqual(res["after"]["mean_edit_distance"],
                                 res["before"]["mean_edit_distance"])
            self.assertTrue((Path(d) / "improvement_curve.png").exists())
            self.assertTrue((Path(d) / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
