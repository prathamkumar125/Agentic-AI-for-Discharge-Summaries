# Phase 2 — Safe Agent Core (Plan)

> **Scope.** This is the plan for **Phase 2 of Plan.md ("Safe Agent Core")**, which
> bundles three sub-milestones: **M2** (state + guardrail + trace), **M3** (the 6
> tools), **M4** (the ReAct loop). It mirrors how *Milestone 1* covered all of
> Phase 1 (M0 scaffold + M1 ingestion). This is a **forward-looking to-do** — what
> we have to build — not a results write-up.

**Objective:** a *real* agent loop that reads the Phase-1 OCR corpus and can only
ever write **cited** clinical facts. Everything unknown, pending, or contradictory
is flagged — never guessed.

**Exit criterion (the bar for this milestone):** the loop runs within the step
budget, **chooses its own tools**, writes only fields that pass the no-fab
guardrail, and emits a readable per-step trace.

---

## 1. Where this fits

| | Input | What it does | Output |
|---|---|---|---|
| **Phase 1 (done)** | image-only PDF | render → OCR → segment | `ocr_corpus.json` |
| **Phase 2 (this)** | `ocr_corpus.json` | plan → extract → detect conflicts → reconcile → validate | filled `AgentState` + trace |
| Phase 3 (next) | filled `AgentState` | render the draft + harden | `discharge_summary.md` + `trace.json` |

Phase 1 gave us trustworthy text with page-level provenance (`global_page`). Phase 2
builds the brain that turns that text into a **structured, cited, review-ready**
state — safely.

---

## 2. The principle we are enforcing (restate before building)

**No fabrication, enforced structurally — not by prompt-hope.**
Every field in the final state must be either:

- **(a)** sourced from the corpus *with a citation*, **or**
- **(b)** explicitly `missing` / `pending` / `conflict` and flagged for review.

There is no third option. The whole point of Phase 2 is to make (a) *impossible
without a citation*. That is the `validate_set` guardrail (M2).

---

## 3. M2 — State, Guardrail, Trace

### 3.1 `AgentState` — the 12 clinical fields (each cited)

The 12 required output sections **are** the state fields:

```
patient_demographics · admission_date · discharge_date · principal_diagnosis ·
secondary_diagnoses · hospital_course · procedures · discharge_medications ·
allergies · follow_up_instructions · pending_results · discharge_condition
```

Each field is a structured **`FieldValue`**:

```python
FieldValue = {
  "value":     str | None,                 # the content, or None
  "status":    "present" | "missing" | "pending" | "conflict",
  "citations": [ {"doc": str, "page": int, "quote": str} ],
  "notes":     str,                         # conflict description / candidates / why flagged
}
```

Plus **meta** on the state: `med_reconciliation`, `flags[]`, `trace[]`,
`step_count`, `ocr_index` (the loaded corpus).

**To build:**
- [ ] `src/agent/state.py`: `Status` enum, `FieldValue`, `AgentState` (dataclasses or
      TypedDicts), with a clean constructor that seeds all 12 fields as `missing`.
- [ ] Serialization: `AgentState → dict/JSON` for the final draft + trace.

### 3.2 `validate_set` — the guardrail write path

**Single chokepoint:** every field write goes through `validate_set`. No code sets a
`FieldValue` directly.

**Rules (reject the write, keep the field safe, if violated):**
- [ ] `status="present"` is allowed **only if** `value` is non-empty **and**
      `citations` is non-empty. Otherwise → reject (field stays `missing`).
- [ ] `pending` / `missing` writes are always allowed (and should carry a `notes`
      reason).
- [ ] `conflict` requires **≥2 candidate values** recorded in `notes`/citations and
      auto-triggers a `flag_review`.
- [ ] Citations must reference real corpus pages (validate `page` against
      `ocr_index`); a quote that isn't a substring of that page's OCR text is
      rejected or downgraded (anti-hallucinated-citation check).
- [ ] Every accept/reject is appended to the trace with a reason.

**To build:**
- [ ] `src/tools/validate_set.py` implementing the above as the *only* setter.
- [ ] Unit-style check: an unsourced "present" write is rejected.

### 3.3 Trace

Per-step, human-and-machine readable. Each entry records:

```
step# · phase · reasoning · tool · inputs · result-summary · next-decision
```

**To build:**
- [ ] `src/agent/trace.py`: append-only trace; pretty console printer + `to_json()`.
- [ ] Wire the guardrail and (M3) tools to log through it.

---

## 4. M3 — Tool registry (the 6 tools)

The agent decides **when** to call each (function calling). All tools are wrapped
with the **failure contract**: `timeout → retry (2×, backoff) → structured fallback
→ report` (never crash, never fake success). Config already provides
`TOOL_RETRIES`, `RETRY_BACKOFF`.

| Tool | Signature (intent) | Purpose | Failure behavior |
|---|---|---|---|
| `parse_doc` | `parse_doc(page_range)` | fetch OCR text for specific pages from the corpus (re-read on demand) | missing/empty page → report, agent re-plans |
| `extract_field` | `extract_field(field, context_pages)` | propose a value **with supporting quotes/citations** from OCR text | nothing found → return "not found" → agent marks `missing` |
| `reconcile` | `reconcile(adm_meds, dis_meds)` | diff meds: added / stopped / changed / continued + **documented-reason** flag | — |
| `drug_interaction` | `drug_interaction(meds)` | **mock** interaction lookup (severity) | timeout → retry → report; **never** assume "no interaction" |
| `flag_review` | `flag_review(field, reason, severity)` | record an escalation for the clinician | append-only |
| `validate_set` | `validate_set(field, value, status, citations)` | **guardrail write path** (§3.2) | rejects unsourced "present" |

**To build:**
- [ ] `src/tools/{parse_doc,extract_field,reconcile,drug_interaction,flag_review,validate_set}.py`.
- [ ] `src/tools/registry.py`: the OpenAI **function schemas** (name, description,
      JSON params) + a dispatch map `name → callable`.
- [ ] A shared `with_retries(...)` wrapper so every tool inherits the failure contract.
- [ ] `extract_field` must return quotes; **no quote ⇒ cannot become "present"** (it
      feeds `validate_set`, which enforces this).

**Mockable tools:** `drug_interaction` and `flag_review` are explicitly mocks/side
tools — they branch off the conflict/reconcile phases. Mock realistically (e.g.
a small known-interactions table) so the *escalation behavior* is demonstrable.

---

## 5. M4 — The ReAct loop (what makes it an agent)

A **from-scratch** Think → Act → Observe loop driven by the brain
(`gpt-5-mini`) via function calling. **Not** a hardcoded pipeline — the brain picks
the next action each step, and a **re-plan edge** feeds new findings back into
extraction.

**The 5 phases the loop moves through** (from `discharge_agent_loop.svg`):

1. **Ingest & plan** — load `ocr_corpus.json`; brain builds an *agenda*: which
   fields to source from which doc-type pages (uses the Phase-1 `doc_type_index`).
2. **Extract loop** — per field/doc: `extract_field` → write via `validate_set`.
3. **Conflict & gap analysis** — find contradictions across notes + missing/pending
   fields; set `conflict` / `missing` / `pending`; `flag_review`.
4. **Reconcile & draft** — `reconcile` admission vs discharge meds; call
   `drug_interaction` when warranted; assemble draft content.
5. **Validate & output** — step-cap check; finalize unresolved as `missing`/`pending`.

**The re-plan edge:** Validate → back to Extract when new gaps/conflicts appear.
This feedback loop is the difference between an agent and a script — call it out
explicitly in the trace.

**Control & safety:**
- [ ] **`MAX_STEPS = 25`** hard stop (already in config). On cap → finalize
      gracefully (unresolved = `missing`/`pending`), never loop forever.
- [ ] Brain temperature: **gpt-5 reasoning models reject `temperature ≠ 1`** — do
      not pass a custom temperature for the brain (this differs from the OCR model,
      which uses `temperature=0`). *Flagged so we don't trip on it.*

**To build:**
- [ ] `src/agent/prompts.py`: system prompt (role + the inviolable no-fab rules) +
      planner guidance.
- [ ] `src/agent/loop.py`: the loop — call brain → get tool call(s) → dispatch via
      registry → append observation → repeat until brain says "done" or cap hit.
- [ ] Extend `src/main.py` with a `run` subcommand:
      `python -m src.main run patient_2`.

---

## 6. Files to create this milestone

```
src/agent/
├── state.py     # Status, FieldValue, AgentState (12 fields + meta)
├── prompts.py   # system + planner prompts (no-fab rules)
├── loop.py      # ReAct loop, planner call, MAX_STEPS cap, re-plan edge
└── trace.py     # trace entries → console + trace.json
src/tools/
├── registry.py          # OpenAI function schemas + dispatch + with_retries wrapper
├── validate_set.py      # the guardrail write path (the no-fab chokepoint)
├── extract_field.py
├── parse_doc.py
├── reconcile.py
├── drug_interaction.py  # mock
└── flag_review.py        # mock/side tool
src/main.py      # + `run` subcommand
```

---

## 7. Design decisions to make (and our leaning)

| Decision | Options | Leaning |
|---|---|---|
| State representation | dataclass vs TypedDict vs pydantic | **dataclasses** (stdlib, simple, JSON via `asdict`) |
| Citation integrity | trust the model vs verify quote ∈ page text | **verify** — reject/▼downgrade quotes not found in the cited page |
| How `extract_field` reads text | pass full corpus vs only agenda pages | **agenda pages** (cheaper, focused); `parse_doc` for on-demand re-reads |
| Tool-call style | OpenAI `tools`/function-calling vs hand-rolled JSON | **function-calling** (native, robust parsing) |
| Loop termination | brain "finish" tool vs heuristic | **explicit finish** signal + `MAX_STEPS` backstop |

---

## 8. Acceptance criteria (testable — use as the checklist)

- [ ] Loop runs end-to-end on `patient_2` within `MAX_STEPS=25`.
- [ ] The brain **chooses** tools (visible in the trace) — not a fixed call order.
- [ ] **No field is `present` without a value *and* a citation** (guardrail proven by
      a rejected unsourced write).
- [ ] At least one **pending** result is surfaced (e.g. "urine culture report
      awaited") — not imputed.
- [ ] A **medication change with no documented reason** is flagged for reconciliation.
- [ ] An injected/simulated tool failure is **retried then reported** — no crash, no
      fake success.
- [ ] The trace shows, per step: reasoning → tool → inputs → result → next decision.
- [ ] At least one **re-plan** (Validate → Extract) is demonstrable.

---

## 9. Risks & open questions

- **gpt-5-mini call mechanics** — confirm function-calling shape and the
  no-custom-temperature constraint early (cheap smoke test before wiring the full loop).
- **Conflict detection quality** — contradictions across 71 pages are subtle; we may
  need targeted prompts per field rather than one generic pass.
- **Citation verification strictness** — too strict rejects valid paraphrased values;
  too loose lets hallucinated citations through. Start strict (substring match),
  relax only with evidence.
- **Cost/latency of the brain** — many tool round-trips per run; mitigate by scoping
  `extract_field` to agenda pages and caching nothing risky.
- **Step budget vs completeness** — 25 steps must cover 12 fields + conflicts +
  reconciliation; if tight, batch extractions per agenda group.

---

## 10. Explicitly out of scope for this milestone (deferred)

- Final `discharge_summary.md` rendering + full robustness hardening → **Phase 3 (M5–M6)**.
- The learning loop (simulated reviewer, reward, improvement curve) → **Phase 4 (M7)**.
- README write-up + demo video + diagram sync → **Phase 4 (M8)**.

---

*Source of truth:* `Plan.md` §6 (architecture), §7 (guardrail), §8 (failure
handling), §10 (Phase 2). The task PDF remains the requirement source of truth.
