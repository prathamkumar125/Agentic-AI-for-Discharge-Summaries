"""Robustness: the tool failure contract (Hard Requirement #8).

A failed tool must become a structured error the agent OBSERVES — never an
exception that crashes the run, and never reported as success. Deterministic and
offline (sleep is patched out so backoff doesn't slow the test).

Run:  python -m unittest tests.test_registry_failure -v
"""
from __future__ import annotations

import unittest

from backend.agent.state import AgentState
from backend.agent.trace import Trace
from backend.tools import registry as registry_mod
from backend.tools.registry import ToolRegistry


def _state() -> AgentState:
    corpus = {"patient_id": "test", "pages": [
        {"global_page": 1, "doc_type": "discharge", "source": "d.pdf",
         "ocr": {"text": "DIAGNOSIS: SEPSIS"}},
    ]}
    return AgentState.from_corpus(corpus)


class FailureContractTests(unittest.TestCase):
    def setUp(self):
        self._orig_sleep = registry_mod.time.sleep
        registry_mod.time.sleep = lambda *a, **k: None  # no real backoff delay
        self.state = _state()
        self.reg = ToolRegistry(self.state, Trace(verbose=False), client=None,
                                fail_tools={"parse_doc"})

    def tearDown(self):
        registry_mod.time.sleep = self._orig_sleep

    def test_injected_failure_returns_structured_error_not_exception(self):
        res = self.reg.dispatch("parse_doc", {"pages": "1"}, step=1)
        self.assertFalse(res["ok"])         # NOT reported as success
        self.assertIn("error", res)         # explicit, observable error
        self.assertEqual(res.get("tool"), "parse_doc")

    def test_working_tool_still_succeeds(self):
        res = self.reg.dispatch("flag_review",
                                {"field": "allergies", "reason": "verify", "severity": "info"},
                                step=1)
        self.assertTrue(res["ok"])
        self.assertEqual(len(self.state.flags), 1)

    def test_unknown_tool_reported_not_raised(self):
        res = self.reg.dispatch("does_not_exist", {}, step=1)
        self.assertFalse(res["ok"])
        self.assertIn("unknown tool", res["error"])


if __name__ == "__main__":
    unittest.main()
