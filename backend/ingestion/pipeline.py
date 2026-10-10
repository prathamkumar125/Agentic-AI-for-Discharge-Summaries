"""Ingestion orchestrator: PDF folder -> rendered PNGs -> OCR -> segmented corpus.

Produces, for one patient:
  - outputs/<patient_id>/ocr_corpus.json  (machine-readable; the agent's input)
  - outputs/<patient_id>/ocr_dump.md      (human-readable, grouped by doc type)

The on-disk page + OCR caches make reruns instant. `global_page` (1-based across
all PDFs in the folder) is the stable anchor used later for citations.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import DATA_DIR, OUTPUT_DIR, ROOT, ensure_dirs
from . import render, segment
from .ocr import ocr_image
from .segment import DOC_TYPES


def _rel(p: Path) -> str:
    """Path relative to repo root, with forward slashes (portable in JSON)."""
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def ingest_patient(patient_id: str, verbose: bool = True) -> dict:
    """Render -> OCR -> segment every page for one patient; write corpus + dump."""
    ensure_dirs()
    patient_dir = DATA_DIR / patient_id
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"No data folder for patient '{patient_id}': {patient_dir}")

    pdfs = render.find_pdfs(patient_dir)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {patient_dir}")

    pages: list[dict] = []
    global_page = 0
    n_errors = 0
    n_warnings = 0

    for pdf_path in pdfs:
        if verbose:
            print(f"[render] {pdf_path.name} ...", flush=True)
        png_paths = render.render_pdf(pdf_path, patient_id)

        for src_idx, png in enumerate(png_paths):
            global_page += 1
            rec = ocr_image(png)
            if rec.get("error"):
                n_errors += 1
            if rec.get("warning"):
                n_warnings += 1
            doc_type, info = segment.classify_page(rec["text"])
            pages.append({
                "global_page": global_page,
                "source": pdf_path.name,
                "src_page": src_idx + 1,
                "png": _rel(png),
                "doc_type": doc_type,
                "doc_type_detail": info,
                "ocr": rec,
            })
            if verbose:
                tag = "cache" if rec.get("cached") else ("ERROR" if rec.get("error") else "ocr")
                n_chars = len(rec.get("text") or "")
                warn = f"  ⚠ {rec['warning']}" if rec.get("warning") else ""
                print(
                    f"  p{global_page:03d} [{tag:>5}] {doc_type:<12} {n_chars:>5} chars{warn}",
                    flush=True,
                )

    # Index: doc_type -> [global_page, ...]
    doc_index: dict[str, list[int]] = {}
    for p in pages:
        doc_index.setdefault(p["doc_type"], []).append(p["global_page"])

    corpus = {
        "patient_id": patient_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_pages": len(pages),
        "n_ocr_errors": n_errors,
        "n_ocr_warnings": n_warnings,
        "sources": [p.name for p in pdfs],
        "doc_type_index": doc_index,
        "pages": pages,
    }

    out_dir = OUTPUT_DIR / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ocr_corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_dump(out_dir / "ocr_dump.md", corpus)

    if verbose:
        print(f"\n[done] {len(pages)} pages, {n_errors} OCR error(s), "
              f"{n_warnings} OCR warning(s).")
        print("[doc-type counts] " + ", ".join(
            f"{dt}={len(doc_index.get(dt, []))}" for dt in DOC_TYPES + ["other"]
            if dt in doc_index
        ))
        print(f"[written] {_rel(out_dir / 'ocr_corpus.json')}")
        print(f"[written] {_rel(out_dir / 'ocr_dump.md')}")
    return corpus


def _write_dump(path: Path, corpus: dict) -> None:
    """Human-readable OCR dump, grouped by document type (Phase-1 exit artifact)."""
    by_type: dict[str, list[dict]] = {}
    for p in corpus["pages"]:
        by_type.setdefault(p["doc_type"], []).append(p)

    lines: list[str] = [
        f"# OCR corpus — {corpus['patient_id']}",
        "",
        f"- Generated: {corpus['generated_at']}",
        f"- Pages: {corpus['n_pages']}  ·  OCR errors: {corpus['n_ocr_errors']}"
        f"  ·  OCR warnings: {corpus.get('n_ocr_warnings', 0)}",
        f"- Sources: {', '.join(corpus['sources'])}",
        "",
        "Document-type counts: " + ", ".join(
            f"`{dt}`={len(v)}" for dt, v in by_type.items()
        ),
        "",
    ]
    for dt in DOC_TYPES + ["other"]:
        group = by_type.get(dt)
        if not group:
            continue
        lines.append(f"\n## {dt}  ({len(group)} page(s))\n")
        for p in group:
            notes = ""
            if p["ocr"].get("error"):
                notes += f"  ⚠️ OCR ERROR: {p['ocr']['error']}"
            if p["ocr"].get("warning"):
                notes += (f"  ⚠️ OCR {p['ocr']['warning']}"
                          f" (raw {p['ocr'].get('raw_chars', '?')} chars)")
            lines.append(
                f"### page {p['global_page']} "
                f"(`{p['source']}` p{p['src_page']}){notes}\n"
            )
            lines.append("```text")
            lines.append((p["ocr"].get("text") or "").rstrip())
            lines.append("```\n")
    path.write_text("\n".join(lines), encoding="utf-8")
