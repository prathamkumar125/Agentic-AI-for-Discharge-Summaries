# Plan.md — Discharge Summary Agent

Build roadmap for the Dscribe take-home. This is the executable plan; `CLAUDE.md` holds the binding design constraints, and the task PDF is the source of truth for requirements. **Both Part 1 and Part 2 are required, committed deliverables** — completing both is necessary to advance in the hiring process, so neither is optional. The task suggests ~6–10h; completing both parts realistically takes ~11–13h across the 4 phases below.

---

## 1. Goal & success criteria

Build an **agentic AI system** that ingests a patient's source-note PDFs and produces a **structured discharge-summary draft for clinician review** — plus a readable step trace.

Graded, in priority order:
1. **Clinical safety** — never fabricate; always flag the unknown, pending, and conflicting.
2. **Agentic design** — real plan/re-plan loop, tool use, recovery, control.
3. **Handling the messy cases** — missing data, undocumented med changes, contradictions.
4. **Part 2 learning loop** (required) — measurable reduction in edit burden.
5. **Clarity** of code, traces, and reasoning.

---

## 2. Scope

**In scope — Part 1 (required):** the agent loop, PDF ingestion via OCR, the 12 output sections, no-fabrication guardrail, medication reconciliation, conflict handling, mock tools (drug-interaction + flag-for-review), robust failure handling, step cap, observable trace. Output drafts + traces for every patient provided. **Delivered via a small FastAPI web app** (§14): the user uploads the patient's PDF(s) + a patient name, the app creates that patient's folder and runs the pipeline, and the discharge-summary draft is shown and downloadable.

**In scope — Part 2 (required):** simulated reviewer, edit-based reward, a learning mechanism, before/after metric + curve, limitations writeup. Both parts are committed deliverables (see note at top).

**Out of scope:** real EHR integration, auth/multi-user, real patient data, fine-tuning infra (API-only — see §9). *(A minimal upload/download web UI is now IN scope — see §14 — but anything beyond that single flow, e.g. dashboards/accounts, is not.)*

---

## 3. Tech stack (locked)

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.12 | available |
| Brain / planner | OpenAI `gpt-5-mini` | ReAct loop via function calling |
| Vision OCR | OpenAI `gpt-4.1-mini` | every page (data has no text layer) |
| PDF rendering | `pdfplumber` | page → PNG only; **not** a text extractor here |
| Agent loop | from-scratch ReAct | preferred over frameworks; justify each decision |
| Web app (Part 1 delivery) | `FastAPI` + `uvicorn` | upload PDF + name → pipeline → view/download summary (§14) |
| Plotting (Part 2) | `matplotlib` | improvement curve |
| Secrets | `OPENAI_API_KEY` via `.env` (gitignored) | never committed |

No Anthropic, no Gemini (free-tier Gemini retains/trains on inputs → conflicts with the task's data-handling rule).

---

## 4. Data reality (verified)

`data/patient_2/` source PDF = **71 pages, every page a full-page scanned image, zero embedded text.** Mixed content:
- **Most pages:** typed text rendered as an image (admission note, progress notes, labs) — trivial for vision OCR.
- **Some pages:** forms with **genuine handwriting** (bed-sore body diagram, vitals charts, nurse signatures) — harder; unreadable → mark `MISSING`/flag, never guess.

Implication: it's effectively **one combined PDF holding many document types**, so ingestion must **OCR all pages and then segment/classify** them into logical documents (admission / progress / labs / meds / nursing) before the agent reasons over them.

> Only `patient_2` is present today. Build to accept **N patients** (one folder each). For Part 2 held-out evaluation this is thin — see §9 and §12.

---

## 5. Repository layout (target)

```
.
├── CLAUDE.md  README.md  Plan.md
├── requirements.txt   .env.example   .gitignore
├── data/patient_2/<source>.pdf
├── src/
│   ├── main.py              # CLI: ingest | run | render | add  (the underlying engine)
│   ├── config.py           # model names, MAX_STEPS, paths
│   ├── intake.py           # NEW: upload → validate PDF → create data/<id>/ → chain pipeline
│   ├── web/                # NEW: FastAPI app (Part 1 front door)
│   │   ├── app.py          #   GET / (upload form) · POST /upload · GET /download/<id>
│   │   └── templates/      #   minimal upload + result pages
│   ├── ingestion/
│   │   ├── render.py        # pdf → page PNGs (pdfplumber)
│   │   ├── ocr.py           # gpt-4.1-mini OCR + on-disk cache
│   │   ├── segment.py       # group OCR'd pages into doc types
│   │   └── pipeline.py      # orchestrator → ocr_corpus.json + ocr_dump.md
│   ├── agent/
│   │   ├── state.py         # AgentState, FieldValue, Status
│   │   ├── loop.py          # ReAct loop, planner, step cap
│   │   ├── prompts.py       # system + planner prompts
│   │   ├── trace.py         # trace → trace.json + console
│   │   └── render_summary.py # AgentState → discharge_summary.md (banner + citations)
│   ├── tools/
│   │   ├── registry.py      # OpenAI function schemas
│   │   ├── parse_doc.py  extract_field.py  reconcile.py
│   │   ├── drug_interaction.py  flag_review.py  validate_set.py
│   └── learning/            # Part 2
│       ├── reviewer.py  reward.py  memory.py  bandit.py  evaluate.py
├── outputs/patient_2/{ocr_corpus.json, ocr_dump.md, discharge_summary.md, trace.json}
└── cache/{pages/, ocr/}    # rendered page PNGs + OCR JSON keyed by image hash
```

---

## 6. Architecture

### 6.1 AgentState — 12 clinical fields (each cited)
The 12 required output sections **are** the state fields:

`patient_demographics · admission_date · discharge_date · principal_diagnosis · secondary_diagnoses · hospital_course · procedures · discharge_medications · allergies · follow_up_instructions · pending_results · discharge_condition`

Each field is a structured `FieldValue`:
```python
FieldValue = {
  "value":    str | None,
  "status":   "present" | "missing" | "pending" | "conflict",
  "citations": [ {"doc": str, "page": int, "quote": str} ],
  "notes":    str   # conflict description / why flagged / candidates
}
```
Plus meta: `med_reconciliation`, `flags[]`, `trace[]`, `step_count`, `ocr_index`.

### 6.2 The agent loop (ReAct, 5 phases)
A real plan/re-plan loop — **not** a fixed pipeline. The brain decides the next action each step via function calling; the **re-plan edge** (Validate → Extract) is what makes it agentic.

1. **Ingest & plan** — OCR all pages (cached) → segment into docs → brain builds an *agenda* of which fields to source from which docs.
2. **Extract loop** — per field/doc, call `extract_field`; write via `validate_set`.
3. **Conflict & gap analysis** — detect contradictions across notes and missing/pending fields; mark `conflict`/`missing`/`pending`.
4. **Reconcile & draft** — `reconcile` admission vs discharge meds; call `drug_interaction` when warranted; assemble the draft.
5. **Validate & output** — step-cap check, finalize unresolved as `missing`/`pending`, emit `discharge_summary.md` + `trace.json`.

`flag_review` and `drug_interaction` branch off phases 3–4. On new conflicts/gaps the brain loops back to Extract (re-plan).

> **Honest design note:** OCR-all-upfront is deterministic (cheap, cached) — the genuine *agency* lives in planning the agenda, extraction, conflict resolution, reconciliation, and validation. `parse_doc` remains a tool so the agent can re-read specific pages on demand. This tradeoff is documented in the README.

### 6.3 Tools (6) — OpenAI function calling
| Tool | Purpose | Failure behavior |
|---|---|---|
| `parse_doc(page_range)` | render + OCR pages (cached) | retry → fallback (lower res / per-page) → report |
| `extract_field(field, context)` | propose value **with citations** from OCR text | empty → return "not found", agent marks missing |
| `reconcile(adm_meds, dis_meds)` | diff: added/stopped/changed/continued + documented-reason flag | — |
| `drug_interaction(meds)` | **mock** interaction lookup (severity) | timeout → retry → report, never assume "no interaction" |
| `flag_review(field, reason, severity)` | record escalation | append-only |
| `validate_set(field, value, status, citations)` | **guardrail write path** (§7) | rejects unsourced "present" writes |

### 6.4 Control & observability
- **`MAX_STEPS = 25`** hard cap in `loop.py`; on cap → finalize gracefully (unresolved = missing), never run forever.
- **Trace** per step: `step# · phase · reasoning · tool · inputs · result-summary · next-decision` → `trace.json` + human-readable console.

---

## 7. No-fabrication guardrail (the core)

Enforced **structurally**, not by prompt-hope:
- Every field write goes through `validate_set`. A field may be `status="present"` **only if** `value` is non-null **and** `citations` is non-empty. Otherwise it is rejected and the field stays `missing`/`pending`/`conflict`.
- `extract_field` must return supporting quotes; no quote ⇒ no "present".
- Conflicts: store **all** candidate values + their citations in `notes`, set `status="conflict"`, and `flag_review`. Never auto-pick.
- Med changes with no documented reason ⇒ surfaced in `med_reconciliation` and `flag_review`, never silently resolved.
- Output is **always** a draft for review; header states this and lists all flags/missing/pending up top.

---

## 8. Failure handling
- Every tool/LLM call wrapped: **timeout → retry (2×, backoff) → structured fallback → report**.
- A failed call returns an explicit error the agent *observes* and re-plans around (e.g., page unreadable ⇒ field `missing` + flag). **Never** treat failure as success; never crash the run.
- OCR cached on disk (keyed by page-image hash) so reruns/demo are fast and deterministic.
- Inject a forced tool failure in testing to prove graceful recovery (good for the demo video).

---

## 9. Part 2 — learning from doctor edits

> The task labels Part 2 a **stretch**; **this project commits to it** as required. The five
> "You must" pieces from the task map to the modules below: (1) reward signal, (2) simulated
> reviewer, (3) learning mechanism + justification, (4) measurable improvement on a **held-out**
> set + curve, (5) limitations (cold-start + gaming + safety). Built under the **no-hardcoding**
> rule: the learning must genuinely generalize and carry no patient-specific values.

- **Simulated reviewer** (`reviewer.py`): a deterministic, **agent-hidden** editing policy of
  **general** rules — standardize med lines (`drug — dose — route — freq`), normalize dates,
  expand abbreviations from a reference map, de-hedge prose, ensure pending results are listed.
  General transformations, **never patient-specific find/replace**; it **never deletes** a
  citation, flag, or MISSING/PENDING/CONFLICT marker. Produces `(draft, edited)` pairs. It stands
  in for the clinician edits a production system — our web app (§14) — would eventually capture.
- **Reward** (`reward.py`): per-section normalized edit distance (`difflib`) + section match rate;
  lower edits ⇒ higher reward. Plus a **safety penalty** so the metric can't be gamed — lost
  citations / dropped flags / downgraded missing-pending-conflict / increased vagueness score
  *worse*, not better.
- **Learning mechanism** (`memory.py`) — **chosen: structured correction-memory** *derived from the
  edit diffs* and injected at the **render/format layer** (so already-validated values stay intact).
  It is learned from the data and **generalizes** — not a hand-written style guide. Justification:
  DPO/SFT impractical (no training infra, API-only); a learned reward model + best-of-N is heavier
  and still needs eval data; correction-memory is online, inspectable, and structurally safe.
  Optional extension: a contextual bandit (`bandit.py`) over formatting strategies.
- **Evaluation** (`evaluate.py`): before/after edit distance + section accuracy on a **held-out**
  split, with an improvement curve over iterations (matplotlib). Each iteration also **asserts the
  safety invariants are unchanged**, so improvement can't be bought by eroding safety.
- **Held-out — DECIDED: per-section within the available patient.** Learn corrections from a train
  subset of sections; measure on a disjoint held-out subset; the curve is over iterations. We have
  only `patient_2` — **we do NOT fabricate extra patients**; instead we document the cold-start /
  limited-data constraint honestly, exactly as the task requires.
- **Safety invariant (inviolable):** learned changes adjust *presentation/format/structure* only;
  every clinical value still passes `validate_set` — the loop can never relax the no-fab guardrail.

---

## 10. Project phases & milestones

Build in **4 phases**; each ends in something runnable/demoable. Milestones (M#) nest inside phases. **All four phases are required** — Phase 3 completes the Part 1 safety core (a clean checkpoint), and Phase 4 delivers Part 2 plus the submission package; both are mandatory for this project. Don't start a phase until the previous phase's exit criteria are met.

### Phase 1 — Foundation & Ingestion  ·  M0–M1  ·  ~2h
Goal: turn the raw 71-page image-only PDF into a clean, cached, segmented OCR corpus.
- **M0 Scaffold** — repo layout, `requirements.txt`, `.env.example`, `.gitignore`, `config.py`.
- **M1 Ingestion** — render pages (pdfplumber) → OCR (`gpt-4.1-mini`) → on-disk cache → segment pages into doc types (admission/progress/labs/meds/nursing).
- **Exit:** `main.py` runs ingestion on `patient_2` and dumps structured OCR text grouped by doc type; reruns hit the cache.

### Phase 2 — Safe Agent Core  ·  M2–M4  ·  ~4.5h
Goal: a real agent loop that can only ever write **cited** facts.
- **M2 State + guardrail + trace** — `AgentState`/`FieldValue`; `validate_set` rejects unsourced "present" writes; trace → `trace.json` + console.
- **M3 Tools** — all 6 tools with failure wrappers (retry→fallback→report) + OpenAI function schemas.
- **M4 Agent loop** — ReAct (think/act/observe) + planner + `MAX_STEPS=25` cap + re-plan edge, wired to the tools.
- **Exit:** the loop runs within the step budget, chooses tools itself, writes only cited fields, and emits a readable per-step trace.

### Phase 3 — End-to-End Part 1 (the bar)  ·  M5–M6  ·  ~1.5h
Goal: the Part 1 safety core, complete and runnable end-to-end.
- **M5 End-to-end** — produce `discharge_summary.md` + `trace.json` for **every** patient; conflicts/pending/missing all flagged; med reconciliation surfaced.
- **M6 Robustness** — retries/fallback verified; injected-failure test passes; no crashes; step cap enforced.
- **Exit:** runs on every patient with **no fabrication** and every unknown flagged. ✅ **Part 1 complete** (clinical-safety core done) — Part 2 still required, see Phase 4.

### Phase 4 — Learning Loop + Submission Package  ·  M7–M8  ·  ~3–4h  ·  *required*
Goal: Part 2 plus everything needed to submit.
- **M7 Part 2** — simulated reviewer (hidden policy), edit-based reward + safety penalty, correction-memory (+ optional bandit), before/after metric + improvement curve.
- **M8 Writeup & demo** — README (1–2 pp), 3–5 min video, sync the two SVG diagrams (drop Claude/Gemini labels).
- **Exit:** full submission (Part 1 + Part 2) with a measurable reduction in edit burden on a held-out set.

---

## 11. Hard-requirement compliance map

| # | Requirement | Where satisfied |
|---|---|---|
| 1 | Real agent loop (plan/re-plan) | §6.2 ReAct + Validate→Extract re-plan |
| 2 | PDF ingestion | §6.2 / `ingestion/` OCR all pages |
| 3 | No fabrication | §7 `validate_set` structural guardrail |
| 4 | Pending / missing | `FieldValue.status` = pending/missing |
| 5 | Medication reconciliation | `reconcile` + `med_reconciliation` + flag |
| 6 | Conflicts | status=`conflict`, keep all candidates, flag |
| 7 | Tools, decide when | 6-tool registry; brain chooses; escalate on safety |
| 8 | Robust failures | §8 retry/fallback/report wrappers |
| 9 | Control | `MAX_STEPS=25` hard cap |
| 10 | Observability | §6.4 per-step trace → `trace.json` |

---

## 12. Risks & open questions
- **One patient only** — confirm whether more patients will be provided; affects Part 2 held-out eval (§9).
- **Combined 71-page PDF** vs the spec's "folder of PDFs" — segmentation step handles it; verify doc-type boundaries are detected sanely.
- **Handwriting OCR fidelity** on nursing charts — acceptable to leave `missing`/flagged; most required fields are in typed pages.
- **Cost/latency** of 71 OCR calls — mitigated by on-disk cache; OCR uses the cheap `gpt-4.1-mini`.
- **Update the two SVG diagrams** to drop Claude/Gemini labels, or keep them as historical artifacts (note in README).

---

## 13. Deliverables checklist (maps to "What to Submit")
- [ ] Source code + run instructions (README)
- [ ] `discharge_summary.md` + `trace.json` for **every** patient
- [ ] Part 2 (required): reviewer + learning mechanism + before/after metric & curve
- [ ] 3–5 min video demo (2 patients incl. a missing/pending/conflict case; show a flag/escalate moment; show Part 2 before/after)
- [ ] README (1–2 pp): loop design · no-fab guardrail · failure/conflict handling · Part 2 reward + results · limitations · what's next
- [ ] FastAPI web app: upload patient PDF + name → view & download the discharge summary (§14)

---

## 14. Web app — Part 1 delivery (upload → summary → download)

**Goal:** a single-flow **FastAPI** app so a user uploads a patient report and gets a
**downloadable** discharge summary — no CLI needed, nothing patient-specific hardcoded.

**Flow:** upload PDF(s) + patient name → `intake` validates & creates `data/<patient_id>/`
→ ingest → `run_agent` → render → show + **download** `discharge_summary.md`.

### M9 — Intake + orchestration (`src/intake.py`)
- [ ] `intake(pdf, patient_name)`: validate it is a real, readable PDF; derive a **unique**
      `patient_id` (slug of the name, deduped with a suffix/uuid); create `data/<id>/`; save
      the PDF (allow multiple PDFs per patient).
- [ ] `process_upload(pdf, patient_name)`: intake → `ingest_patient` → `run_agent` → renderer;
      returns `{patient_id, summary_path, status_counts, flags}`. **Reuses the CLI engine — no
      logic forked into the web layer.**
- [ ] CLI parity: `python -m src.main add <path.pdf> [--name "..."]`.

### M10 — FastAPI app (`src/web/app.py`)
- [ ] `GET /` — minimal upload form (file + patient name).
- [ ] `POST /upload` — runs `process_upload`; renders a result page (summary + the
      conflicts/pending/missing/flags banner) with a **Download** button.
- [ ] `GET /download/<patient_id>` — serves `discharge_summary.md` as an attachment;
      optional `?zip=1` bundles `trace.json` + `agent_state.json`.
- [ ] Errors surfaced to the user (bad/empty PDF, OCR/agent failure) — never a raw 500,
      never a faked success.
- [ ] Run: `uvicorn src.web.app:app --reload`.

### Acceptance criteria
- [ ] Upload an arbitrary patient PDF (not just patient_2) → get its discharge summary,
      viewable + downloadable, with **no code change**.
- [ ] Synthetic-data notice shown; the API key stays server-side (`.env`), never exposed.
- [ ] First request OCRs + caches; the existing no-fab guardrail and trace are unchanged.

### Decisions
- **Sync vs async:** a run takes time (OCR + brain). Start **synchronous** (simplest); add a
  "processing…" state if needed. Background jobs/queues are out of scope.
- **Formats:** Markdown always; optional HTML/PDF export of the summary as a later nicety.
