"""The ReAct loop — a real plan / act / observe agent (not a fixed pipeline).

Each turn the brain (gpt-5-mini) sees the document-type index + current field
status and DECIDES the next tool call(s). Tool results are observed and fed back,
so the brain re-plans from what it reads. Key properties:

  * Hard step cap (MAX_STEPS) — never runs forever; on cap it finalizes gracefully
    with whatever statuses exist (unresolved stay missing/pending).
  * Re-plan edge — if the brain tries to finish while fields were never attempted,
    we reject the finish once, nudge it back to extraction (Validate -> Extract),
    and continue. This feedback edge is what makes it agentic.
  * Every write goes through the guardrail; failures are observed, never fatal.

Outputs `trace.json` + `agent_state.json`. The pretty discharge_summary.md is
rendered in Phase 3 (M5).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import BRAIN_MODEL, MAX_STEPS, OUTPUT_DIR, require_api_key
from ..agent.state import AgentState, FIELD_ORDER, Status
from ..agent.trace import Trace
from ..agent.prompts import SYSTEM_PROMPT, planning_context
from ..tools.registry import ToolRegistry

MAX_REPLANS = 2
_TOOL_RESULT_CLIP = 6000  # cap tool-result text fed back to the brain (bounds context)


def _assistant_dict(msg) -> dict:
    d: dict = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [{
            "id": tc.id, "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        } for tc in msg.tool_calls]
    return d


def _parse_args(raw: str) -> dict:
    try:
        return json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def _clip(s: str, n: int = _TOOL_RESULT_CLIP) -> str:
    return s if len(s) <= n else s[:n] + " …[truncated]"


def _phase(state: AgentState, attempted: set, finishing: bool = False) -> str:
    if finishing:
        return "validate&output"
    if not attempted:
        return "ingest&plan"
    if state.med_reconciliation is None:
        return "extract"
    if state.unresolved():
        return "conflict&gap"
    return "reconcile&draft"


def run_agent(corpus: dict, verbose: bool = True, fail_tools=None,
              max_steps: int = MAX_STEPS) -> tuple[AgentState, Trace]:
    from openai import OpenAI
    client = OpenAI(api_key=require_api_key())

    state = AgentState.from_corpus(corpus)
    trace = Trace(verbose=verbose)
    registry = ToolRegistry(state, trace, client, fail_tools=fail_tools)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": planning_context(
            state, header="Begin. Plan from the index, then extract each field with citations.")},
    ]

    attempted: set[str] = set()
    replans = 0
    finished = False
    step = 0

    while step < max_steps:
        step += 1
        state.step_count = step
        resp = client.chat.completions.create(
            model=BRAIN_MODEL, messages=messages,
            tools=registry.schemas, tool_choice="auto",
        )
        msg = resp.choices[0].message
        trace.think(step, _phase(state, attempted), msg.content or "(no narration)")

        # No tool calls: the brain is done or stuck. Re-plan once if work remains.
        if not msg.tool_calls:
            messages.append(_assistant_dict(msg))
            never = [f for f in FIELD_ORDER if f not in attempted]
            if never and replans < MAX_REPLANS:
                replans += 1
                trace.replan(step, f"no action taken; fields never attempted: {never}")
                messages.append({"role": "user", "content": planning_context(
                    state, header=f"You took no action. Attempt these fields from the most "
                                  f"relevant pages, then call finish: {never}")})
                continue
            trace.finish(step, "brain produced no further actions")
            finished = True
            break

        messages.append(_assistant_dict(msg))
        finish_req = None
        for tc in msg.tool_calls:
            name = tc.function.name
            args = _parse_args(tc.function.arguments)
            if name in ("extract_field", "validate_set") and args.get("field"):
                attempted.add(args["field"])
            res = registry.dispatch(name, args, step)
            if name == "finish":
                finish_req = res
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": _clip(json.dumps(res, ensure_ascii=False, default=str))})

        if finish_req is not None:
            still_missing = [f for f in FIELD_ORDER
                             if state.fields[f].status == Status.MISSING.value]
            if still_missing and replans < MAX_REPLANS:
                replans += 1
                trace.replan(step, f"finish requested but fields still missing: {still_missing}")
                messages.append({"role": "user", "content": planning_context(
                    state, header=(
                        f"Before finishing, these fields are still MISSING: {still_missing}. "
                        f"Try extracting EACH from OTHER relevant document types per the index "
                        f"(e.g. follow-up/discharge condition from consultation & nursing pages; "
                        f"dates from admission / ICU / nursing pages — and if two sources disagree, "
                        f"record a conflict). If a field truly appears in no source, leave it missing. "
                        f"Do NOT touch fields already present/pending/conflict."))})
                continue
            trace.finish(step, finish_req.get("reason", ""))
            finished = True
            break

    if not finished:
        trace.event(step, "cap", reason=f"MAX_STEPS={max_steps} reached")
        trace.finish(step, "step cap reached — finalizing with current statuses")

    _write_outputs(state, trace)
    if verbose:
        _print_summary(state, step)
    return state, trace


def _write_outputs(state: AgentState, trace: Trace) -> None:
    out_dir = OUTPUT_DIR / state.patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dict = state.to_dict()
    trace.to_json(out_dir / "trace.json")
    (out_dir / "agent_state.json").write_text(
        json.dumps(state_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    # The clinician-facing deliverable, rendered from the SAME validated state.
    from .render_summary import render_summary_file
    render_summary_file(state_dict, out_dir / "discharge_summary.md")


def _print_summary(state: AgentState, step: int) -> None:
    counts: dict[str, int] = {}
    for fv in state.fields.values():
        counts[fv.status] = counts.get(fv.status, 0) + 1
    print(f"\n[agent done] steps used: {step}/{MAX_STEPS}")
    print("[field statuses] " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"[flags] {len(state.flags)}")
    out = OUTPUT_DIR / state.patient_id
    print(f"[written] {out / 'trace.json'}")
    print(f"[written] {out / 'agent_state.json'}")
    print(f"[written] {out / 'discharge_summary.md'}")
