"""intake — turn an uploaded patient PDF into a fully processed patient.

Both the web API (`backend/api/app.py`) and the CLI (`backend.main add`) call into
here, so NO pipeline logic is forked into the web layer. Flow:

    validate the PDF -> derive a unique patient_id -> save under data/<id>/
    -> ingest (OCR) -> run the agent -> render the draft

Everything downstream is the existing engine, reused unchanged; nothing is hardcoded
to a particular patient.
"""
from __future__ import annotations

import io
import re
import uuid
from pathlib import Path

from .config import DATA_DIR, OUTPUT_DIR, MAX_STEPS
from .ingestion.pipeline import ingest_patient

_PDF_MAGIC = b"%PDF-"


def _slugify(text: str) -> str:
    """A filesystem-safe, lowercase slug; empty input -> 'patient'."""
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return s or "patient"


def validate_pdf(data: bytes) -> None:
    """Raise ValueError if `data` is not a usable PDF (empty / wrong type / unreadable)."""
    if not data:
        raise ValueError("Uploaded file is empty.")
    if _PDF_MAGIC not in data[:1024]:
        raise ValueError("File does not look like a PDF (missing %PDF header).")
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if len(pdf.pages) == 0:
                raise ValueError("PDF has no pages.")
    except ValueError:
        raise
    except Exception as e:  # corrupt/encrypted/etc.
        raise ValueError(f"Could not open the PDF: {e}") from e


def _existing_ids() -> set[str]:
    ids: set[str] = set()
    for root in (DATA_DIR, OUTPUT_DIR):
        if root.is_dir():
            ids |= {p.name for p in root.iterdir() if p.is_dir()}
    return ids


def unique_patient_id(patient_name: str) -> str:
    """A unique, readable patient_id derived from the name (deduped against existing)."""
    base = _slugify(patient_name)
    existing = _existing_ids()
    if base not in existing:
        return base
    while True:
        cand = f"{base}_{uuid.uuid4().hex[:6]}"
        if cand not in existing:
            return cand


def intake(file_bytes: bytes, filename: str, patient_name: str,
           *, patient_id: str | None = None) -> str:
    """Validate + store an uploaded PDF under data/<patient_id>/. Returns the patient_id.

    Multiple PDFs may be staged for the same patient_id (re-upload) without clobbering.
    """
    validate_pdf(file_bytes)
    pid = patient_id or unique_patient_id(patient_name)
    pdir = DATA_DIR / pid
    pdir.mkdir(parents=True, exist_ok=True)

    stem = _slugify(Path(filename or "report").stem)
    dest = pdir / f"{stem}.pdf"
    i = 1
    while dest.exists():
        dest = pdir / f"{stem}_{i}.pdf"
        i += 1
    dest.write_bytes(file_bytes)
    return pid


def intake_path(pdf_path, patient_name: str | None = None,
                patient_id: str | None = None) -> str:
    """CLI helper: stage a PDF from a filesystem path. Returns the patient_id."""
    p = Path(pdf_path)
    if not p.is_file():
        raise FileNotFoundError(f"No such file: {p}")
    return intake(p.read_bytes(), p.name, patient_name or p.stem, patient_id=patient_id)


def process_patient(patient_id: str, *, verbose: bool = False,
                    max_steps: int | None = None) -> dict:
    """Run the full pipeline on an already-staged patient: ingest -> agent -> render.

    `run_agent` writes outputs/<id>/{discharge_summary.md, trace.json, agent_state.json}.
    Returns a compact result summary.
    """
    from .agent.loop import run_agent
    corpus = ingest_patient(patient_id, verbose=verbose)
    state, _trace = run_agent(corpus, verbose=verbose,
                              max_steps=max_steps or MAX_STEPS)
    counts: dict[str, int] = {}
    for fv in state.fields.values():
        counts[fv.status] = counts.get(fv.status, 0) + 1
    out_dir = OUTPUT_DIR / patient_id
    return {
        "patient_id": patient_id,
        "steps_used": state.step_count,
        "status_counts": counts,
        "n_flags": len(state.flags),
        "summary_path": str(out_dir / "discharge_summary.md"),
    }


def process_upload(file_bytes: bytes, filename: str, patient_name: str,
                   *, verbose: bool = False) -> dict:
    """End-to-end for an upload: intake + full pipeline. Returns the result summary."""
    pid = intake(file_bytes, filename, patient_name)
    return process_patient(pid, verbose=verbose)
