"""Central configuration: models, paths, and control limits.

Every model name, path, and tunable limit lives here so nothing is hardcoded
elsewhere. Values may be overridden via environment variables (loaded from a
gitignored `.env`). The OpenAI API key is read from the environment ONLY — it
is never stored in code or committed.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths -----------------------------------------------------------------
# Repo root = the directory above this file's package (backend/).
ROOT = Path(__file__).resolve().parents[1]

# Populate os.environ from a .env at the repo root, if present.
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"          # data/<patient_id>/*.pdf
OUTPUT_DIR = ROOT / "outputs"     # outputs/<patient_id>/{discharge_summary.md, trace.json}
CACHE_DIR = ROOT / "cache"
PAGE_CACHE_DIR = CACHE_DIR / "pages"  # rendered page PNGs
OCR_CACHE_DIR = CACHE_DIR / "ocr"     # OCR text JSON, keyed by image hash

# --- Models (OpenAI only) --------------------------------------------------
BRAIN_MODEL = os.getenv("OPENAI_BRAIN_MODEL", "gpt-5-mini")   # planner / reasoner
OCR_MODEL = os.getenv("OPENAI_OCR_MODEL", "gpt-4.1-mini")     # vision OCR

# --- Secrets ---------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Ingestion / rendering -------------------------------------------------
RENDER_DPI = int(os.getenv("RENDER_DPI", "150"))            # pdfplumber render resolution
MAX_IMAGE_EDGE = int(os.getenv("MAX_IMAGE_EDGE", "2048"))   # downscale long edge before OCR
OCR_DETAIL = os.getenv("OCR_DETAIL", "high")               # OpenAI vision detail level

# OCR degeneration guards. A real clinical page OCRs to < ~2.5k chars; far more
# than this means the vision model fell into a repetition loop (a known failure
# on dense grid/table forms). We collapse the repetition and, past a hard cap,
# truncate and flag the page as unreliable — never pass hallucinated bulk on.
OCR_SUSPECT_CHARS = int(os.getenv("OCR_SUSPECT_CHARS", "5000"))  # above this: inspect/collapse
OCR_HARD_CAP = int(os.getenv("OCR_HARD_CAP", "10000"))          # absolute max chars kept

# --- Agent control ---------------------------------------------------------
MAX_STEPS = int(os.getenv("MAX_STEPS", "25"))              # hard iteration cap (never exceed)
TOOL_RETRIES = int(os.getenv("TOOL_RETRIES", "2"))         # retry attempts on tool/LLM failure
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "2.0"))   # seconds; exponential base


def require_api_key() -> str:
    """Return the OpenAI API key, or raise a clear, actionable error."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your "
            "key (or export the environment variable). It is never read from code."
        )
    return OPENAI_API_KEY


def ensure_dirs() -> None:
    """Create the output/cache directory tree if missing."""
    for d in (OUTPUT_DIR, CACHE_DIR, PAGE_CACHE_DIR, OCR_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
