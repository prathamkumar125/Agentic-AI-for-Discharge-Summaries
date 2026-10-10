# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status: Part 1 built; web-app delivery in progress

**Part 1 is built and runs end-to-end** (ingestion → ReAct agent → no-fab guardrail →
`discharge_summary.md` + `trace.json`), with a CLI and a unit-test suite. **Part 2** (the
learning loop) is **built and measured** in `backend/learning/` (run `python -m backend.main
learn patient_2`). Keep the diagrams, docs, and code in sync as you change things.

**Delivery decision — Part 1 is delivered as a small FastAPI web app.** The user-facing
flow is: **upload the patient's source-note PDF(s) + a patient name → the app creates that
patient's folder under `data/<patient_id>/` → ingest → agent → render → the discharge-summary
draft is shown and is downloadable.** The CLI (`python -m src.main …`) remains the underlying
engine the web app calls; the web app is the front door. (A UI was previously out of scope —
this supersedes that: a minimal upload/download web app is now in scope for Part 1.)

Environment: Python 3.12.10. Stack: OpenAI SDK (`openai`), `pdfplumber` (render-to-image only
— see Ingestion), and **FastAPI + uvicorn** for the web app. Dependencies are in
`requirements.txt`; tests run via `python -m unittest discover -s tests`.

**LLM provider: OpenAI only.** Brain/planner = `gpt-5-mini`. Vision OCR = `gpt-4.1-mini`. (No Anthropic, no Gemini — earlier diagrams referencing Claude Sonnet / Gemini Flash are superseded by this note. Free-tier Gemini was rejected because it retains/trains on inputs, which conflicts with the task's "do not upload to a service that would retain it" rule; paid OpenAI API does not train on data by default.) `OPENAI_API_KEY` comes from the environment / a gitignored `.env` — never commit it.

## What this is

An **agentic AI system** that reads a patient's messy, incomplete source-note PDFs and produces a **structured discharge-summary draft for clinician review**. The hard part is judgment under uncertainty: the documents are verbose, incomplete, have pending labs, undocumented medication changes, and notes that contradict each other.

The single most important property — above feature completeness, above polish — is **no fabrication**. See the guardrail section below.

## The inviolable rule: no fabrication

The agent must **never invent or guess a clinical fact.** This is the core evaluation criterion.

- Every field in the output must be either (a) sourced from a document with a citation/provenance, or (b) explicitly marked `MISSING` / `PENDING` and flagged for clinician review. There is no third option — no plausible-sounding fill-ins.
- The output is **always a draft for review**, never an auto-finalized clinical document.
- Pending lab → say "pending", do not impute a value. Absent note → say "absent".
- Conflicting notes (e.g. two different discharge diagnoses) → **surface both and flag the conflict**; never silently pick one.
- Medication change with no documented reason → **flag for reconciliation**; never invent a rationale.
- Drug-interaction / safety concern found → **surface and escalate**, never bury.

In the intended design this is centralized in a `validate_and_set()` function (the "no-fab guardrail" node) — route every field write through it rather than setting state fields ad hoc, so the guarantee holds in one place.

## Intended architecture

### Ingestion (PDF → text)
**The provided data is image-only.** `patient 2 (1).pdf` is 71 pages, every page a full-page scanned image with **zero embedded text layer**. The content is mixed: most pages are *typed text rendered as an image* (clean printed admission/progress notes, labs), and some pages are *forms with genuine handwriting* (e.g. a "BED SORES" body diagram with handwritten annotations, vitals charts, nurse signatures).

Consequences for the build:
- **`pdfplumber` extracts no text here — do not use it as a text extractor.** Use it (or an equivalent) only to **render each page to a PNG**, then send that image to vision OCR.
- **Every page goes through vision OCR (`gpt-4.1-mini`)** — there is no typed-vs-handwritten routing for this dataset; the typed pages have no text layer either.
- Handwriting that the model cannot read confidently must be left `MISSING` / flagged, never guessed (see guardrail). Most required discharge fields live in the typed pages anyway; the handwritten nursing charts are largely supporting data.

### Agent core
- **Brain / planner:** OpenAI `gpt-5-mini` (chosen over `gpt-5-nano` — nano is too weak for the planning, conflict-detection, and reconciliation reasoning this task grades).
- **Loop:** a from-scratch **ReAct engine** — Think → Act → Observe. This must be a *real* agent loop that plans and **re-plans** from what it reads and what tools return, **not a fixed hardcoded pipeline**. (The task slightly prefers a clear from-scratch loop over frameworks like LangGraph/CrewAI, and you must be able to justify every decision the agent makes.)
- **AgentState:** ~12 fields, each carrying citations/provenance.
- **Control:** hard iteration cap (`MAX = 25` in the diagram) so the agent cannot run forever; plus a readable trace log.

### Tool registry (6 callable tools)
The agent decides *when* to call each:
- `parse_doc` — ingest/extract a source document.
- `extract_field` — pull a specific field from parsed content.
- `flag_review` — escalate / "flag for clinician review" (mockable).
- `reconcile` — medication reconciliation (admission vs. discharge).
- `validate_set` — the guardrail write path (`validate_and_set()`).
- `drug_interaction` — drug-interaction lookup (mockable).

`flag_review` and `drug_interaction` are side tools that branch off the conflict/gap and reconcile/draft phases.

### The five-phase loop (`discharge_agent_loop.svg`)
1. **Ingest & plan** — read PDFs, build the agenda.
2. **Extract loop** — per document, per section.
3. **Conflict & gap analysis** — find missing fields and contradictions.
4. **Reconcile & draft** — medications, procedures, hospital-course summary.
5. **Validate & output** — cap check, emit trace.

A **re-plan arrow loops from Validate back into the Extract loop** — that feedback edge is what makes this an agent rather than a pipeline.

### Outputs (produce these for every patient)
- `discharge_summary.md` — the structured draft for review.
- `trace.json` — step-by-step agent log.

The trace must be readable and, for each step, show: **reasoning → tool/action chosen → inputs → result → next decision** (Observability requirement).

### Delivery: FastAPI web app (Part 1 front door)
- **Intake:** an upload endpoint accepts the patient PDF(s) + a patient name, validates the
  file, derives a unique `patient_id`, creates `data/<patient_id>/`, and saves the PDF there
  (multiple PDFs per patient allowed).
- **Pipeline:** the same intake call chains ingest → `run_agent` → renderer, reusing the CLI
  engine (no logic forked into the web layer).
- **Output:** the rendered `discharge_summary.md` is shown and is **downloadable** (the draft,
  optionally a zip with `trace.json` + `agent_state.json`).
- Nothing patient-specific is hardcoded — the same flow works for any uploaded patient.

## Required output sections

The discharge summary must contain: patient demographics · admission & discharge dates · principal and secondary diagnoses · hospital course · procedures · **discharge medications with changes from admission clearly noted** · allergies · follow-up instructions · **pending results** · discharge condition.

## Robustness requirements (build these in, don't bolt on)

- Tool calls and document reads can fail, time out, or return empty. The agent must **retry, fall back, or report — never crash, and never behave as if a failed call succeeded.**
- Enforce the step cap as a hard stop, not a suggestion.

## Part 2 — Learning from doctor edits

The task labels Part 2 a **stretch** ("a strong Part 1 alone is a valid, complete submission");
**this project nonetheless commits to it** as a required deliverable. Premise: in production a
clinician edits the draft before finalizing, and those edits are *signal* — the system should
learn from them so future drafts need fewer corrections. We have **no real doctor-edited data**,
so part of the task is manufacturing the feedback loop ourselves and proving it works.

The five required pieces (verbatim intent from the task PDF):
1. **Reward / accuracy signal derived from edits** — e.g. section-level match rate or normalized
   edit distance between the draft and the corrected version (less editing ⇒ higher reward).
2. **Simulated reviewer** — a stand-in "doctor" applying a **consistent editing policy that is
   hidden from the agent**, producing `(draft, edited)` pairs (so we can train/measure without a
   real clinician).
3. **A learning mechanism** that uses accumulated edits to improve future drafts — choose and
   justify (contextual bandit over prompt/template strategies; DPO/SFT on pairs; learned
   reward model + best-of-N; or structured correction-memory injected into prompts).
4. **Measurable improvement** — a before/after metric on a **held-out** set, with an improvement
   curve over iterations.
5. **Limitations** — the cold-start / limited-data problem, and the gaming risk (an agent can
   lower edit distance by getting vaguer or mimicking style instead of getting the medicine
   right). Explain how the loop is kept from degrading the Part 1 guarantees.

Decisions for this project (consistent with the no-hardcoding rule):
- **Chosen mechanism: structured correction-memory.** Corrections are *derived from the observed
  edit diffs* and injected into the render/extraction prompts — chosen over DPO/SFT (no training
  infra, API-only) and reward-model+best-of-N; a contextual bandit over prompt strategies is an
  optional extension. It must **genuinely generalize** — a learned signal, not a hand-written
  style guide, and carrying **no patient-specific values**.
- **The reviewer's policy is general editing rules** (normalize dates, enforce a med-line schema,
  expand abbreviations, de-hedge prose, ensure pending results are listed) — never patient-specific
  find/replace, and it never deletes a citation, a flag, or a MISSING/PENDING/CONFLICT marker. It
  stands in for the clinician edits a production system would capture (in our web app, §Delivery,
  that is where real edits would eventually come from).
- **Held-out = per-section within the available patient.** We have only `patient_2`; **do NOT
  fabricate extra patients.** Learn corrections from a train subset of sections, measure on a
  disjoint held-out subset, and report the improvement curve over iterations — documenting the
  one-patient / cold-start constraint honestly, exactly as the task asks.

**Safety invariant (inviolable):** learning may only change *presentation/format/structure* of an
already-validated field; every clinical value still flows through `validate_and_set()`, so the loop
can **never** relax the no-fabrication guarantee. The reward carries a **safety penalty** (lost
citations / dropped flags / downgraded missing-pending-conflict / increased vagueness score
*worse*, not better), and the held-out evaluation **asserts** the invariants are unchanged before
vs after learning. Per the task: a clean, well-measured method that demonstrably lowers edit burden
beats a sophisticated one that isn't evaluated.

## Hard constraints

- **Keep API keys out of the repo.** `OPENAI_API_KEY` via environment variable / a gitignored `.env`. The task permits any provider; this project commits to OpenAI only (see the LLM provider note above).
- **Synthetic data only.** Do not upload the patient PDFs to any third-party service that would retain them, and never introduce real patient data.
- The task nominally suggests ~one working day, but **both Part 1 and Part 2 must be completed** for this project — treat neither as optional. The README must still state clearly what is done and any known limitations.
