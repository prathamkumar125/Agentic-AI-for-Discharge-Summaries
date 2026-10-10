# Phase 4 — Learning Loop + Submission Package (Plan)

> **STATUS (M7 — Part 2): BUILT & MEASURED.** `backend/learning/{reviewer,reward,memory,evaluate}.py`
> + `python -m backend.main learn patient_2`. Held-out (per-section, 25-split Monte-Carlo) mean
> edit distance **0.189 → 0.148 (−21.6%)**, match rate **0 → 0.11**, **safety penalty 0.0** every
> split (citations/flags untouched). Curve: `outputs/patient_2/learning/improvement_curve.png`.
> Remaining in Phase 4: **M8** (video demo). 11 learning unit tests pass.

> **Scope.** This is the plan for **Phase 4 of Plan.md ("Learning Loop + Submission
> Package")**, which bundles two sub-milestones: **M7** (Part 2 — learning from doctor
> edits) and **M8** (submission package: video demo + final diagram/README sync). It is
> a **forward-looking to-do** — what we still have to build — not a results write-up.

**Objective:** manufacture a feedback loop from *simulated* doctor edits and prove the
agent's drafts need **fewer corrections over iterations** — **without ever degrading the
Part-1 no-fabrication guarantees.**

**Exit criterion (the bar for this phase):** a **before/after metric on a held-out set**
with an **improvement curve over iterations** showing reduced edit burden, while the
safety invariants (no uncited `present`; every conflict/missing/pending preserved and
flagged) are demonstrably unchanged.

---

## 1. Where this fits

| | Input | What it does | Output |
|---|---|---|---|
| Phase 1 (done) | image-only PDF | render → OCR → segment | `ocr_corpus.json` |
| Phase 2 (done) | `ocr_corpus.json` | plan → extract → conflict → reconcile → validate | filled `AgentState` + trace |
| Phase 3 (done) | filled `AgentState` | render the draft + harden + tests | `discharge_summary.md` + `trace.json` + 27 unit tests |
| **Phase 4 (this)** | `(draft, edited)` pairs | reward → learn → measure | improvement curve + Part-2 writeup |

Part 1 is complete and runnable end-to-end under the guardrail. Phase 4 sits **on top**
of it: it observes how a (simulated) clinician edits the draft, derives a reward, and
feeds learned **style/format corrections** back into future drafts — never touching the
clinical-fact guarantee.

---

## 2. The principle we are enforcing (restate before building)

**Learning may only change presentation, never the medicine.** Two hard rules:

1. **The guardrail stays sovereign.** Every clinical value still flows through
   `validate_and_set`; learned changes adjust *formatting / structure / phrasing* of an
   already-validated value (or the renderer), never the citation requirement. A learned
   prompt can never make an unsourced fact "present."
2. **The metric must resist gaming.** Optimizing "fewer edits" can be cheated by getting
   vaguer, mimicking style, or dropping flags. So the reward carries an explicit **safety
   penalty** (§3.2), and the evaluation **asserts the invariants** before/after (§3.4, §7).

> This is the part the task most wants to see reasoned about. We state the gaming risk
> up front and design against it, rather than bolting safety on after.

---

## 3. M7 — Part 2 components

### 3.1 Simulated reviewer — the (hidden) editing policy  `src/learning/reviewer.py`
A **deterministic** stand-in "doctor" that applies a **consistent policy the agent never
sees**, turning a draft section into an "edited" section → producing `(draft, edited)`
pairs. Candidate edit rules (clinically plausible, purely presentational):

- **Medication lines** → canonical `DRUG — dose — route — frequency` (one per line).
- **Dates** → a single fixed format (e.g. `DD-Mon-YYYY`); collapse obvious OCR variants.
- **Abbreviation expansion** — a small fixed map (`K/C/O` → "known case of", `c/o` →
  "complains of", `Adv` → "Advised", …).
- **De-hedging / trimming** — remove filler, keep one clean sentence per fact.
- **Pending results** — ensure each awaited test is listed as its own line.
- **Structure** — enforce section ordering / bullet style.

**Invariant the reviewer must respect (and we test):** it **never deletes** a citation, a
flag, or a `MISSING/PENDING/CONFLICT` marker — a real clinician keeps the safety scaffolding.

- [ ] `reviewer.py`: `apply_policy(field_name, value) -> edited_value`; `make_pairs(state_dict) -> [{field, draft, edited}]`.
- [ ] Policy lives **only** here — never injected into any agent prompt.

### 3.2 Reward signal  `src/learning/reward.py`
Derived from the edits — **less editing ⇒ higher reward.**

- **Per-section similarity:** normalized edit distance via `difflib.SequenceMatcher` ratio
  (stdlib, no deps) → `section_reward = ratio(draft, edited)`; also a section **match rate**
  (fraction of sections needing zero edits).
- **Aggregate:** mean section reward across the 12 fields.
- **Safety penalty (anti-gaming):** subtract a penalty when the *agent's* draft, relative
  to the validated state, **loses citations, drops a flag, downgrades a MISSING/PENDING/
  CONFLICT marker, or becomes vacuous** (large length drop with lost content tokens).
  `total = mean_section_reward − safety_penalty`.

- [ ] `reward.py`: `section_reward`, `match_rate`, `safety_penalty(state_before, draft)`, `total_reward`.

### 3.3 Learning mechanism — structured correction-memory  `src/learning/memory.py`
**Primary approach (chosen): structured correction-memory injected into prompts.**

- Accumulate recurring `(field → correction pattern)` from the edit pairs (e.g. "for
  `discharge_medications`, format each line as `DRUG — dose — route — freq`"; "expand
  `K/C/O`").
- Summarize into a compact, per-field **style note** that is appended to the **renderer /
  extraction prompt** on the *next* run — so future drafts pre-apply the corrections.
- **Injection point matters for safety:** corrections target the **rendered presentation
  of already-validated fields**, not the citation logic. The value still came from the
  guardrail; memory only reshapes how it's written.

**Justification (why not DPO/SFT or a reward model):** preference fine-tuning needs training
infra we don't have (API-only, no GPU) and a cold start with one patient; correction-memory
is **online, cheap, inspectable, and structurally safe** (it cannot relax the guardrail). It
also degrades gracefully — a bad correction shows up as a *worse* reward next iteration and
can be dropped.

- **Optional stretch:** `bandit.py` — a contextual bandit (ε-greedy / Thompson) over a small
  set of per-section formatting strategies, rewarded by reduced edit burden. Document it as
  an extension if time allows.

- [ ] `memory.py`: `record(pairs)`, `corrections_for(field) -> str`, `as_prompt_block() -> str`; persistence to `outputs/learning/memory.json`.
- [ ] Wire `as_prompt_block()` into the renderer/extractor prompt behind a flag (`--use-memory`).

### 3.4 Evaluation  `src/learning/evaluate.py`
Show **measurable improvement on a held-out set with a curve.**

- **Held-out unit (DECIDED):** **per-section hold-out within the one available patient** —
  split the 12 fields/sub-items into a *train* split (memory is learned from these) and a
  disjoint *held-out* split (only measured, never learned from). **We do not fabricate extra
  patients;** the one-patient / cold-start constraint is documented honestly instead (§9).
- **Iteration loop:** iter 0 = no memory → measure mean edit distance on held-out; then learn
  corrections from train → re-render → re-measure; repeat N iterations, accumulating memory.
- **Outputs:** a before/after table + an **improvement curve** (matplotlib PNG).
- **Safety assertion each iteration:** held-out drafts keep all citations/flags and never
  downgrade a MISSING/PENDING/CONFLICT — printed alongside the reward so improvement can't be
  bought by eroding safety.

- [ ] `evaluate.py`: run iterations, compute metrics, assert invariants, plot curve → `outputs/learning/improvement_curve.png` + `metrics.json`.

---

## 4. M8 — Submission package

- [x] **README** (1–2 pp) — loop design · guardrail · failure/conflict handling · Part 2
      design · limitations · what's next. *(Done in Phase 3; refresh with Part-2 results.)*
- [x] **System-architecture SVG** synced to the real stack (gpt-5-mini brain, gpt-4.1-mini
      OCR-every-page, 6 tools). *(Done.)* Loop SVG already accurate — keep.
- [ ] **Generated drafts + traces for every patient** — currently `patient_2` only; produce
      for any additional patients added.
- [ ] **Video demo (3–5 min, required):** run live on **≥2 patients** incl. a
      missing/pending/conflict case; walk a `trace.json` and point to a **flag/escalate**
      moment (e.g. undocumented med change, date conflict, low-coverage flag); show the
      Part-2 **before/after** improvement curve.

---

## 5. Files to create this phase

```
src/learning/
├── reviewer.py     # hidden, deterministic editing policy -> (draft, edited) pairs
├── reward.py       # per-section edit distance + match rate + safety penalty
├── memory.py       # structured correction-memory; prompt-block injector
├── bandit.py       # OPTIONAL: contextual bandit over formatting strategies
└── evaluate.py     # iteration loop, before/after metric, improvement curve
outputs/learning/
├── pairs.json              # (draft, edited) pairs
├── memory.json             # accumulated corrections
├── metrics.json            # per-iteration reward / edit distance / safety checks
└── improvement_curve.png   # the curve
src/main.py         # + `learn` / `evaluate` subcommand; `--use-memory` on `render`
requirements.txt    # + matplotlib, python-Levenshtein optional (difflib suffices)
```

---

## 6. Design decisions to make (and our leaning)

| Decision | Options | Leaning |
|---|---|---|
| Reward metric | char Levenshtein vs token vs `difflib` ratio | **`difflib` ratio per section** (stdlib, no deps) + match rate |
| Learning mechanism | correction-memory vs bandit vs DPO/SFT | **correction-memory** (online, cheap, structurally safe); bandit optional |
| Memory injection point | extractor prompt vs **renderer/format layer** | **format layer** — keeps guardrail-verified values intact (safety) |
| Held-out unit | per-section vs synthetic patients | **per-section within patient_2** (DECIDED); **no synthetic patients** — document cold-start instead |
| Reviewer policy source | LLM vs deterministic rules | **deterministic rules** (consistent, hidden, reproducible) |
| Base-agent noise | single run vs averaged | render from a **fixed `agent_state.json`** so reward measures formatting, not run variance |

---

## 7. The safety guarantee under learning (the part that must not break)

1. Corrections only reshape the **presentation of already-validated fields**; the value's
   provenance came from `validate_and_set` and is untouched.
2. The reward's **safety penalty** makes "fewer edits via dropped flags / lost citations /
   vagueness" score *worse*, not better.
3. `evaluate.py` **asserts** every held-out iteration: citations preserved, flags preserved,
   no MISSING/PENDING/CONFLICT silently downgraded. A violation fails the run, not just the score.
4. Cold-start / limited-data is acknowledged: with N=1 we measure per-section and report the
   curve's confidence honestly rather than overclaiming.

---

## 8. Acceptance criteria (testable — use as the checklist)

- [ ] `(draft, edited)` pairs produced by a **consistent, agent-hidden** policy.
- [ ] Reward = section similarity + match rate, **minus a safety penalty**, computed and reported.
- [ ] Correction-memory **measurably lowers mean edit distance on a held-out split over
      iterations** (improvement curve saved).
- [ ] **Safety invariants verified unchanged** before vs after learning (asserted in code).
- [ ] Limitations written up: cold-start, gaming risk, and how the loop is prevented from
      degrading Part-1 safety.
- [ ] Video demo recorded; drafts + traces exist for every patient in the set.

---

## 9. Risks & open questions

- **One patient only (the big one).** Held-out is thin. **Decided: per-section hold-out within
  `patient_2`; no synthetic patients.** We document the cold-start / limited-data limitation
  honestly (as the task requires) rather than manufacturing datasets.
- **Gaming.** Must actively watch the safety penalty and a vagueness check; report any case
  where edit distance dropped but safety did too.
- **Reviewer realism.** Keep the policy clinically plausible but fully deterministic, so the
  signal is learnable and reproducible.
- **Base-agent non-determinism** (already a documented Part-1 limitation) would add noise to
  reward; mitigate by learning/measuring against a **fixed rendered `agent_state.json`**.
- **Time-box.** M7 is the bulk of remaining effort; M8 (video) is quick once M7 lands.

---

## 10. Explicitly out of scope for this phase (deferred / non-goals)

- Real DPO/SFT or a trained reward model (no training infra; API-only).
- Real EHR integration, auth, UI, real clinician-edited data.
- Changing the Part-1 extraction/guardrail behavior — Phase 4 sits *around* it, not inside it.

---

*Source of truth:* `Plan.md` §9 (Part 2 design), §10 (Phase 4 / M7–M8). The task PDF's
"Part 2 — Learning from Doctor Edits" remains the requirement source of truth.
