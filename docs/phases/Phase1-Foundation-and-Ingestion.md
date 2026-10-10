# Phase 1 — Foundation & Ingestion

> **Milestone 1 write-up.** Covers **M0 (Scaffold)** and **M1 (Ingestion)** of the
> Discharge Summary Agent. Goal of this phase: turn the raw, image-only patient
> PDF into a **clean, cached, segmented OCR corpus** that the agent (Phase 2) can
> reason over — without ever fabricating text.

**Status:** ✅ Complete and verified. 71/71 pages OCR'd, 0 errors, reruns are free (fully cached).

---

## 1. What Phase 1 had to achieve

The assignment input is a patient's source notes. Before any "agent" can draft a
discharge summary, we must reliably **get the text out of the documents**. For
this dataset that is the hard, easy-to-underestimate part, because of what the
data actually is (next section).

Exit criteria for the phase (all met):

1. `main.py` runs ingestion on a patient folder.
2. It produces **structured OCR text grouped by document type**.
3. **Reruns hit the cache** (no repeated API cost, deterministic output).

---

## 2. The data reality (the discovery that drove every decision)

`data/patient_2/patient_notes.pdf` is **71 pages, and every page is a full-page
scanned image with ZERO embedded text layer.** Content is mixed:

| Kind | Examples | Notes |
|---|---|---|
| **Typed text rendered as an image** | admission/discharge narrative, lab reports, drug charts | clean, but still has *no* selectable text |
| **Genuine handwriting** | bed-sore body diagrams, nursing notes, vitals/TPR charts, signatures | harder; some values not legible |

**Consequences:**

- A normal PDF text extractor returns **nothing** here. We verified page 1 has
  `0` text characters. So `pdfplumber` is used **only to rasterize pages to PNG**,
  never as a text extractor.
- **Every page must go through vision OCR** — there is no "typed vs handwritten"
  shortcut, because the typed pages have no text layer either.
- It is effectively **one combined PDF holding many document types**, so after
  OCR we must **segment** pages back into logical documents (admission / labs /
  meds / nursing / …) so the agent knows where to look for each field.

---

## 3. Approach & architecture

The ingestion pipeline is a simple, deterministic 4-stage flow. Determinism here
is deliberate: the *agency* (planning, conflict resolution, reconciliation) lives
in Phase 2 — ingestion should be cheap, repeatable, and trustworthy.

```
PDF ──► [1] Render ──► PNG ──► [2] OCR ──► text ──► [3] Segment ──► [4] Orchestrate
        pdfplumber           gpt-4.1-mini         title-first       corpus + dump
        (cached)             (cached)             classifier
```

### Stage 1 — Render (`src/ingestion/render.py`)

- `pdfplumber` (pypdfium2 backend, **no system dependencies** like poppler/ImageMagick)
  renders each page at **150 DPI**.
- Native page rasterizes to ~3376×5400 px. We **downscale the long edge to
  ≤ 2048 px** with LANCZOS before saving.
  - *Why:* OpenAI vision in `detail:"high"` internally caps the image anyway, so
    sending anything larger wastes upload size and money for **no extra detail**.
    2048 px is the effective sweet spot.
- PNGs are cached on disk; the filename embeds DPI + max-edge, so changing render
  settings **auto-invalidates** stale renders instead of silently reusing them.

### Stage 2 — OCR (`src/ingestion/ocr.py`)

- Model: **`gpt-4.1-mini`** (cheap, strong vision), `temperature=0` for stable
  transcription.
- **On-disk cache** keyed by `sha256(image bytes) + model + prompt-version`. An
  identical page is never OCR'd twice; a fully-cached run **needs no API key at all**.
- **No-fabrication starts at OCR.** The system prompt instructs: transcribe only
  what is visible, preserve structure, and write `[illegible]` for anything not
  clearly readable — **never guess** a value/name/dose/date. Blank pages return
  `[no text]`. *(The model actually used `[illegible]` / `[redacted]` on hard pages
  — see Results.)*
- **Robustness:** each call retries on transient failure (**2 retries, exponential
  backoff**). On final failure it returns a structured error with `text=""` — the
  page is recorded as unreadable; the run **never crashes and never treats failure
  as success**. A **missing API key fails fast** with an actionable message instead
  of burning retries.

### Stage 3 — Segment (`src/ingestion/segment.py`)

- **Title-first classification.** We observed that almost every page carries a
  clear heading on its first lines (`NURSES NOTES`, `DRUG CHART`,
  `INVESTIGATION … REFERENCE RANGE`, `USG ABDOMEN`, `CONSULTATION SHEET`, …). So
  we match the **first 3 non-empty lines** against an ordered list of title cues.
  - 3 lines (not 1) because pages often have a `SUPER SPECIALITY HOSPITAL`
    letterhead on lines 1–2 and the real document title on line 3.
- **Body-keyword fallback** for title-less pages (e.g. the discharge-summary
  narrative that opens with `DIAGNOSIS:`, or imaging-report continuation pages).
- 9 doc types: `discharge · admission · consultation · labs · imaging ·
  medication · er · nursing · progress` (+ `other`).
- Each result records its **method** (`title` / `body` / `none`) and matched cue,
  so every grouping is **explainable**.
- **Important:** segmentation is an *organizing aid only*. The agent always reads
  the raw OCR text + page numbers, so a mis-label here **cannot** cause a wrong
  clinical fact — that guarantee is the guardrail's job in Phase 2.

### Stage 4 — Orchestrate (`src/ingestion/pipeline.py`, `src/main.py`)

- Walks every PDF in a patient folder, runs stages 1–3, and assigns each page a
  stable **`global_page`** number — the anchor used later for citations.
- Emits two artifacts (see §6).

---

## 4. Key design decisions & trade-offs

| Decision | Why | Trade-off accepted |
|---|---|---|
| **OpenAI only** (`gpt-4.1-mini` OCR, `gpt-5-mini` brain) | No Anthropic key; free-tier Gemini **retains/trains on inputs**, violating the data-handling rule. Paid OpenAI does not train on API data by default. | Single-vendor dependency. |
| **`pdfplumber` as renderer only** | Data has no text layer; extraction is impossible. pypdfium2 backend needs no system installs (poppler/ImageMagick). | None meaningful for this data. |
| **OCR every page** | Typed pages also lack a text layer — no cheap routing exists. | 71 vision calls — mitigated by caching (≈ a few cents, once). |
| **Downscale to ≤2048 px** | API caps high-detail images anyway; larger = wasted cost. | Slightly less raw detail than the full scan (the API would discard it regardless). |
| **On-disk caching (render + OCR)** | Free, deterministic reruns; demo/dev don't re-pay; graders can rerun cheaply. | Cache is large (~67 MB PNGs) → gitignored. |
| **Title-first segmentation (heuristic, not LLM)** | Cheap, deterministic, explainable; titles are highly reliable here. | A few continuation/fragment pages land in `other`. Acceptable — agent reads raw text. |
| **Deterministic ingestion** (agency deferred to Phase 2) | Keeps the safety-critical extraction simple and auditable. | `parse_doc` stays a tool so the agent can still re-read specific pages on demand. |

---

## 5. Robustness findings handled (built in, not bolted on)

1. **Vision-OCR repetition loop (the big one).** Page 39 is a dense ICU
   medication grid. The OCR model fell into a degenerate loop and emitted
   `"Inj 1 cc "` **73,729 characters** long (real content: ~5 lines). Passing that
   to the agent would blow context/cost and risk polluting extraction with
   hallucinated bulk.
   - **Fix:** a post-OCR guard detects runaway length (`> 5000` chars), **collapses**
     the repeated run to a single instance + a marker, hard-caps at 10,000 chars,
     keeps the **raw** transcription in the cache, and **flags the page**
     (`warning: repetition_collapsed`, `raw_chars: 73729`). Result: **73,729 → 167
     chars**, surfaced transparently in the dump. *Hallucinated bulk never reaches
     the agent.*
2. **Missing API key → fail fast.** A cache miss requires the key; we check it
   *once, up front* with a clear message ("copy `.env.example` to `.env` …")
   instead of retrying 71×3 times and mislabeling every page as an error.
3. **Transient OCR failure → retry/report.** Retry with backoff; on exhaustion,
   record an unreadable page rather than crash. (Fully-cached pages need no key.)

---

## 6. Results

Run: `python -m src.main ingest patient_2`

| Metric | Value |
|---|---|
| Pages processed | **71 / 71** |
| OCR errors | **0** |
| OCR warnings (contained) | **1** (page 39 repetition loop) |
| Rerun API cost | **0** (render + OCR fully cached) |
| Page 39 size after guard | **73,729 → 167 chars** |

**Document-type distribution:**

| Doc type | Pages | Where the key clinical content lives |
|---|---:|---|
| `nursing` | 31 | nursing docs, bed-sore charts, intake/output, vitals/TPR, CAUTI |
| `consultation` | 10 | specialist consultation sheets |
| `labs` | 9 | investigation reports (CBC, creatinine, electrolytes, urine) |
| `medication` | 7 | **DRUG CHARTs** (admission vs discharge meds → reconciliation) |
| `imaging` | 5 | USG abdomen, 2D Echo, CT KUB |
| `admission` | 3 | CASE RECORD / ADMISSION RECORD |
| `discharge` | 2 | **discharge-summary narrative** (diagnoses, hospital course) |
| `er` | 2 | ER observation charts |
| `other` | 2 | 1 blank (`[no text]`), 1 ultrasound-machine readout fragment |

**OCR quality evidence — page 1 (faithful, structured, no invention):**

```text
DIAGNOSIS:
1) ACUTE GASTROENTERITIS WITH DEHYDRATION
2) URINARY TRACT INFECTION
...
COURSE IN THE HOSPITAL:
... Serum creatinine (1.65mg/dl) done was elevated. Serum electrolytes done
showed low serum sodium(128.00mmol/L). ... urine culture and sensitivity sent-
report awaited. ...
```

Note "**report awaited**" — a genuinely **pending** result the agent must surface
(not impute) in Phase 2.

---

## 7. Outputs produced

| Path | Purpose |
|---|---|
| `outputs/patient_2/ocr_corpus.json` | Machine-readable corpus — the agent's input. Per page: `global_page`, `source`, `doc_type` (+ method/scores), and the OCR record (`text`, `cached`, `error`, `warning`, `raw_chars`). |
| `outputs/patient_2/ocr_dump.md` | Human-readable transcript, **grouped by doc type**, with per-page provenance and warning markers (Phase-1 exit artifact). |
| `cache/pages/patient_2/*.png` | Rendered page images (gitignored, regenerable). |
| `cache/ocr/*.json` | Cached OCR keyed by image hash (gitignored, regenerable). |

---

## 8. How to run

```bash
# 1. Install deps
python -m pip install -r requirements.txt

# 2. Provide the key (never committed)
#    copy .env.example -> .env and paste your OPENAI_API_KEY

# 3. Run ingestion (first run OCRs all pages; later runs hit the cache)
python -m src.main ingest patient_2     # or omit the id to ingest all of data/*
python -m src.main ingest -q patient_2  # quiet (no per-page log)
```

First run ≈ a few cents of OCR and a few minutes; every run after is instant and free.

---

## 9. Requirements this phase satisfies

| Task requirement | How Phase 1 addresses it |
|---|---|
| **PDF ingestion** | Render → OCR every page; image-only handled. |
| **No fabrication** (begins here) | OCR prompt forbids guessing; `[illegible]`/`[no text]`; repetition guard flags unreliable pages. |
| **Robust failure handling** | Retry/backoff, fail-fast on missing key, never crash, never treat failure as success. |
| **Handling messy data** | Mixed typed/handwritten, blank pages, degenerate OCR — all handled or flagged. |
| **Observability** | Per-page console log + explainable segmentation + warning surfacing in the dump. |

---

## 10. Known limitations & what's deferred

- **Handwriting fidelity.** Some handwritten nursing values may be `[illegible]`.
  This is *correct by design* — left unreadable/flagged, never guessed. Most
  required discharge fields live in the **typed** pages anyway.
- **A few `other`/continuation pages.** 2 pages remain `other` (1 blank, 1 device
  fragment); a couple of narrative-continuation pages inherit a sibling's type
  imperfectly. No clinical impact — the agent reads raw text regardless.
- **Single patient available.** Pipeline already accepts N patients (one folder
  each); only `patient_2` is present today. Relevant to Part 2's held-out eval.
- **README** (full project write-up) is intentionally deferred to Phase 4 / M8.

---

## 11. What's next — Phase 2 (Safe Agent Core)

- **M2** — `AgentState` / `FieldValue` + the `validate_set` **no-fab guardrail**
  (a field can be `present` only if it has a value **and** a citation) + trace log.
- **M3** — the 6 tools (`parse_doc`, `extract_field`, `reconcile`,
  `drug_interaction`, `flag_review`, `validate_set`) with failure wrappers.
- **M4** — the from-scratch **ReAct loop** (think → act → observe), planner, the
  `MAX_STEPS = 25` hard cap, and the re-plan edge that makes it a real agent.

---

*Files added in Phase 1:* `requirements.txt`, `.gitignore`, `.env.example`,
`src/config.py`, `src/ingestion/{render,ocr,segment,pipeline}.py`, `src/main.py`,
package markers under `src/`.
