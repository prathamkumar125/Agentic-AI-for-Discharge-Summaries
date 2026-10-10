"""FastAPI JSON API — upload a patient PDF, run the pipeline, view & download the draft.

A thin HTTP layer over `backend.intake` and the existing engine; it forks no logic.
The pipeline (OCR + agent) is long-running, so a POST kicks it off in a background
thread and the client polls `/status`. Job state is in-memory and single-process
(no DB/queue) — adequate for this app; documented as a limitation. Every failure is
surfaced as a job `error` (or a 4xx) — never a raw 500, never a faked success.

Run:  uvicorn backend.api.app:app --reload
"""
from __future__ import annotations

import io
import json
import threading
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from ..config import OUTPUT_DIR
from ..intake import intake, process_patient

app = FastAPI(title="Discharge Summary Agent", version="1.0")

# Dev: allow the Vite frontend origin. Tighten allow_origins for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job registry: patient_id -> {status, error, result}. Single-process only.
_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _set(pid: str, **kw) -> None:
    with _LOCK:
        _JOBS.setdefault(pid, {}).update(kw)


def _get(pid: str) -> dict | None:
    with _LOCK:
        job = _JOBS.get(pid)
        return dict(job) if job is not None else None


def _run_pipeline(pid: str) -> None:
    """Background worker — never raises; failures land in the job's `error`."""
    try:
        _set(pid, status="processing", error=None)
        result = process_patient(pid, verbose=False)
        _set(pid, status="done", result=result, error=None)
    except Exception as e:  # noqa: BLE001 — surface, don't crash the server
        _set(pid, status="error", error=f"{type(e).__name__}: {e}")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/patients")
async def create_patient(file: UploadFile = File(...), name: str = Form(...)) -> dict:
    """Upload a patient PDF + name → stage it and start processing in the background."""
    data = await file.read()
    try:
        pid = intake(data, file.filename or "report.pdf", name)
    except ValueError as e:  # bad/empty/unreadable PDF
        raise HTTPException(status_code=400, detail=str(e))
    _set(pid, status="processing", error=None, result=None)
    threading.Thread(target=_run_pipeline, args=(pid,), daemon=True).start()
    return {"patient_id": pid, "status": "processing"}


@app.get("/api/patients/{pid}/status")
def status(pid: str) -> dict:
    job = _get(pid)
    if job is None:
        # Processed in an earlier server run? Infer "done" from disk.
        if (OUTPUT_DIR / pid / "discharge_summary.md").exists():
            return {"patient_id": pid, "status": "done"}
        raise HTTPException(status_code=404, detail="Unknown patient.")
    return {"patient_id": pid,
            **{k: job[k] for k in ("status", "error", "result") if k in job}}


@app.get("/api/patients/{pid}/summary")
def summary(pid: str) -> dict:
    out = OUTPUT_DIR / pid
    md, state = out / "discharge_summary.md", out / "agent_state.json"
    if not md.exists() or not state.exists():
        raise HTTPException(status_code=404, detail="Summary not ready for this patient.")
    return {
        "patient_id": pid,
        "markdown": md.read_text(encoding="utf-8"),
        "state": json.loads(state.read_text(encoding="utf-8")),
    }


@app.get("/api/patients/{pid}/download")
def download(pid: str, format: str = "md") -> Response:
    out = OUTPUT_DIR / pid
    md = out / "discharge_summary.md"
    if not md.exists():
        raise HTTPException(status_code=404, detail="Summary not ready for this patient.")
    if format == "zip":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for fn in ("discharge_summary.md", "trace.json", "agent_state.json"):
                f = out / fn
                if f.exists():
                    z.write(f, arcname=fn)
        return Response(
            content=buf.getvalue(), media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{pid}_discharge.zip"'},
        )
    return FileResponse(md, media_type="text/markdown",
                        filename=f"{pid}_discharge_summary.md")


@app.get("/api/patients")
def list_patients() -> dict:
    if not OUTPUT_DIR.is_dir():
        return {"patients": []}
    pts = [d.name for d in sorted(OUTPUT_DIR.iterdir())
           if d.is_dir() and (d / "discharge_summary.md").exists()]
    return {"patients": pts}
