"""Tool registry — OpenAI function schemas + dispatch + the failure contract.

One place wires the 6 tools (plus a `finish` signal) to the brain:
  * `schemas` — the function definitions the brain sees.
  * `dispatch` — runs a tool by name with the shared failure contract
    (retry x TOOL_RETRIES with backoff -> structured error). A failure becomes an
    error the agent OBSERVES; it never crashes the run and is never reported as
    success.
  * `extract_field` results are committed through `validate_and_set`, so the
    no-fab guardrail governs every write.

`fail_tools` lets a test inject a forced failure to prove graceful recovery.
"""
from __future__ import annotations

import time

from ..config import TOOL_RETRIES, RETRY_BACKOFF
from ..agent.state import AgentState, FIELD_ORDER
from .validate_set import validate_and_set
from .extract_field import extract_field as _extract_field
from .parse_doc import parse_doc as _parse_doc
from .reconcile import reconcile as _reconcile
from .drug_interaction import drug_interaction as _drug_interaction
from .flag_review import flag_review as _flag_review

_STATUS_ENUM = ["present", "missing", "pending", "conflict"]

SCHEMAS: list[dict] = [
    {"type": "function", "function": {
        "name": "extract_field",
        "description": "Read the given pages and extract one field WITH verbatim citations. "
                       "The result is committed through the no-fabrication guardrail.",
        "parameters": {"type": "object", "properties": {
            "field": {"type": "string", "enum": FIELD_ORDER},
            "pages": {"type": "string", "description": "Page numbers, e.g. '1', '1-3', or '1,5,7'."},
        }, "required": ["field", "pages"]}}},
    {"type": "function", "function": {
        "name": "parse_doc",
        "description": "Fetch the raw OCR text for specific pages (to read/verify before deciding).",
        "parameters": {"type": "object", "properties": {
            "pages": {"type": "string", "description": "Page numbers, e.g. '1-3' or '1,5,7'."},
        }, "required": ["pages"]}}},
    {"type": "function", "function": {
        "name": "validate_set",
        "description": "Directly set a field through the guardrail (e.g. mark pending/missing, "
                       "or record a conflict with candidates). 'present' requires a verifiable citation.",
        "parameters": {"type": "object", "properties": {
            "field": {"type": "string", "enum": FIELD_ORDER},
            "value": {"type": "string", "description": "The ACTUAL field content (real names/dates/diagnoses/narrative), NOT a description of it."},
            "status": {"type": "string", "enum": _STATUS_ENUM},
            "citations": {"type": "array", "items": {"type": "object", "properties": {
                "doc": {"type": "string"}, "page": {"type": "integer"}, "quote": {"type": "string"}},
                "required": ["page", "quote"]}},
            "candidates": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
        }, "required": ["field", "status"]}}},
    {"type": "function", "function": {
        "name": "reconcile",
        "description": "Reconcile admission vs discharge medications (added/stopped/changed/continued "
                       "+ changes lacking a documented reason).",
        "parameters": {"type": "object", "properties": {
            "admission_meds": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "dose": {"type": "string"},
                "route": {"type": "string"}, "freq": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["name"]}},
            "discharge_meds": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "dose": {"type": "string"},
                "route": {"type": "string"}, "freq": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["name"]}},
        }, "required": ["admission_meds", "discharge_meds"]}}},
    {"type": "function", "function": {
        "name": "drug_interaction",
        "description": "Check a list of drug names for known interactions (MOCK DB; never assume 'none').",
        "parameters": {"type": "object", "properties": {
            "medications": {"type": "array", "items": {"type": "string"}},
        }, "required": ["medications"]}}},
    {"type": "function", "function": {
        "name": "flag_review",
        "description": "Escalate something uncertain/conflicting/risky for clinician review.",
        "parameters": {"type": "object", "properties": {
            "field": {"type": "string"},
            "reason": {"type": "string"},
            "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
        }, "required": ["field", "reason"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Signal that the draft is as complete as the sources allow (all fields resolved "
                       "to present/pending/missing/conflict).",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string"}}, "required": ["reason"]}}},
]


class ToolRegistry:
    def __init__(self, state: AgentState, trace, client, fail_tools=None):
        self.state = state
        self.trace = trace
        self.client = client
        self.fail_tools = set(fail_tools or [])

    @property
    def schemas(self) -> list[dict]:
        return SCHEMAS

    # --- dispatch with the failure contract -------------------------------
    def dispatch(self, name: str, args: dict, step: int) -> dict:
        handler = getattr(self, f"_h_{name}", None)
        if handler is None:
            res = {"ok": False, "error": f"unknown tool: {name}"}
            self.trace.tool(step, name, args, res["error"], ok=False)
            return res

        last = None
        for attempt in range(TOOL_RETRIES + 1):
            try:
                if name in self.fail_tools:
                    raise RuntimeError(f"injected failure for '{name}'")
                res = handler(args, step)
                # Log the action; for extract_field the guardrail ALSO logs the
                # write decision (complementary, not duplicate).
                self.trace.tool(step, name, args, _summarize(name, res),
                                ok=res.get("ok", True), phase="")
                return res
            except Exception as e:  # transient -> retry; final -> report
                last = e
                if attempt < TOOL_RETRIES:
                    time.sleep(RETRY_BACKOFF * (2 ** attempt))
        res = {"ok": False, "error": f"{type(last).__name__}: {last}", "tool": name}
        self.trace.tool(step, name, args, f"FAILED after {TOOL_RETRIES} retries: {last}", ok=False)
        return res

    # --- handlers ---------------------------------------------------------
    def _h_extract_field(self, args: dict, step: int) -> dict:
        field = args["field"]
        prop = _extract_field(self.client, self.state, field, args.get("pages"))
        res = validate_and_set(
            self.state, self.trace, step, field,
            value=prop.get("value"), status=prop.get("status", "missing"),
            citations=prop.get("citations"), notes=prop.get("notes", ""),
            candidates=prop.get("candidates"),
        )
        res["proposal_status"] = prop.get("status")
        res["pages_used"] = prop.get("pages_used")
        committed = res.get("status_set") in ("present", "pending", "conflict")
        res["committed"] = committed
        if committed:
            res["message"] = (
                f"'{field}' is now COMMITTED to the draft as {res.get('status_set')} "
                f"with its citations. Do NOT re-record it with validate_set — it is "
                f"already saved. Work only on fields still 'missing'."
            )
        else:
            res["message"] = (
                f"'{field}' is still missing. Try other pages if you believe it exists; "
                f"otherwise leave it missing."
            )
        return res

    def _h_validate_set(self, args: dict, step: int) -> dict:
        return validate_and_set(
            self.state, self.trace, step, args["field"],
            value=args.get("value"), status=args.get("status", "present"),
            citations=args.get("citations"), notes=args.get("notes", ""),
            candidates=args.get("candidates"),
        )

    def _h_parse_doc(self, args: dict, step: int) -> dict:
        return _parse_doc(self.state, args.get("pages"))

    def _h_reconcile(self, args: dict, step: int) -> dict:
        res = _reconcile(args.get("admission_meds"), args.get("discharge_meds"))
        self.state.med_reconciliation = {
            k: res[k] for k in ("added", "stopped", "changed", "continued",
                                "changes_without_documented_reason")
        }
        return res

    def _h_drug_interaction(self, args: dict, step: int) -> dict:
        return _drug_interaction(args.get("medications"))

    def _h_flag_review(self, args: dict, step: int) -> dict:
        return _flag_review(self.state, args.get("field", ""), args.get("reason", ""),
                            args.get("severity", "warning"))

    def _h_finish(self, args: dict, step: int) -> dict:
        return {"ok": True, "finish": True, "reason": args.get("reason", "")}


def _summarize(name: str, res: dict) -> str:
    if not isinstance(res, dict):
        return str(res)[:200]
    if name in ("validate_set", "extract_field"):
        return f"status={res.get('status_set')} ({res.get('reason')})"
    if name == "parse_doc":
        return f"returned {len(res.get('pages', []))} page(s)"
    if name == "reconcile":
        return (f"added={len(res.get('added', []))} stopped={len(res.get('stopped', []))} "
                f"changed={len(res.get('changed', []))} no_reason={len(res.get('changes_without_documented_reason', []))}")
    if name == "drug_interaction":
        return f"{len(res.get('interactions', []))} interaction(s) [{res.get('source')}]"
    if name == "flag_review":
        return f"flagged ({res.get('flag', {}).get('severity')}); total={res.get('total_flags')}"
    if name == "finish":
        return f"finish: {res.get('reason')}"
    return str({k: v for k, v in res.items() if k != "pages"})[:200]
