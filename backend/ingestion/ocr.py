"""Vision OCR: PNG -> text via OpenAI gpt-4.1-mini, cached on disk.

Cache key = sha256(image bytes) + model + prompt version, so an identical page
is never OCR'd twice and reruns are free and deterministic.

No-fabrication starts here: the prompt tells the model to transcribe ONLY what
is legible and to write [illegible] for anything it cannot read, rather than
guessing a clinical value, name, number, or date. A failed call is retried with
backoff and, if it still fails, returns a structured error (text="") that the
pipeline records as an unreadable page — it never crashes the run and never
fabricates text.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from pathlib import Path

from ..config import (
    OCR_CACHE_DIR, OCR_MODEL, OCR_DETAIL, TOOL_RETRIES, RETRY_BACKOFF,
    OCR_SUSPECT_CHARS, OCR_HARD_CAP, require_api_key,
)

_PROMPT_VERSION = "v1"

OCR_SYSTEM = (
    "You are a careful OCR transcriber for scanned clinical documents. "
    "Transcribe ALL visible text faithfully, preserving structure (headings, "
    "lists, key: value pairs, and tables rendered as plain text). Do NOT "
    "summarize, interpret, correct spellings, or infer missing content. "
    "If a word, number, or value is not clearly legible, write the token "
    "[illegible] in its place — never guess a clinical value, name, dose, or "
    "date. If the page contains no readable text, output exactly: [no text]."
)

_client_singleton = None


def _client():
    global _client_singleton
    if _client_singleton is None:
        from openai import OpenAI
        _client_singleton = OpenAI(api_key=require_api_key())
    return _client_singleton


def _cache_path(img_bytes: bytes) -> Path:
    h = hashlib.sha256(img_bytes).hexdigest()[:32]
    return OCR_CACHE_DIR / f"{h}_{OCR_MODEL}_{_PROMPT_VERSION}.json"


# A 3-60 char unit repeated 4+ times in a row = a vision-OCR repetition loop.
_REPEAT_RE = re.compile(r"(.{3,60}?)\1{3,}", re.DOTALL)


def _collapse_repetition(text: str) -> tuple[str, int]:
    """Collapse runs of a short repeated unit to one instance + a marker.

    Returns (cleaned_text, n_runs_collapsed).
    """
    n = 0

    def _repl(m: "re.Match[str]") -> str:
        nonlocal n
        n += 1
        return m.group(1) + " [... repeated content omitted (OCR repetition loop) ...] "

    return _REPEAT_RE.sub(_repl, text), n


def _postprocess(rec: dict) -> dict:
    """Guard against degenerate OCR (repetition loops / runaway length).

    A normal clinical page OCRs to well under OCR_SUSPECT_CHARS. Past that we
    treat the output as suspect: collapse repeated runs and, beyond OCR_HARD_CAP,
    truncate. The page is then flagged (`warning`) and its original size kept in
    `raw_chars`, so downstream never trusts hallucinated bulk — consistent with
    the no-fabrication rule. Applied on every return (fresh AND cached) so the
    raw text stays in the cache while consumers always see the cleaned version.
    """
    rec.setdefault("warning", None)
    text = rec.get("text") or ""
    if len(text) <= OCR_SUSPECT_CHARS:
        return rec

    raw_len = len(text)
    warnings: list[str] = []
    cleaned, n = _collapse_repetition(text)
    if n and len(cleaned) < len(text):
        text = cleaned
        warnings.append("repetition_collapsed")
    if len(text) > OCR_HARD_CAP:
        text = text[:OCR_HARD_CAP] + (
            "\n[... truncated: OCR exceeded cap; treat this page as unreliable ...]"
        )
        warnings.append("truncated")
    if warnings:
        rec["raw_chars"] = raw_len
        rec["text"] = text
        rec["warning"] = "+".join(warnings)
    return rec


def _call_openai(b64: str) -> str:
    resp = _client().chat.completions.create(
        model=OCR_MODEL,
        temperature=0,  # deterministic transcription (gpt-4.1-mini supports it)
        messages=[
            {"role": "system", "content": OCR_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Transcribe this page."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}", "detail": OCR_DETAIL}},
            ]},
        ],
    )
    return resp.choices[0].message.content or ""


def ocr_image(png_path: Path) -> dict:
    """OCR one PNG. Returns {text, model, cached, error}; cached on disk.

    Retries transient failures (TOOL_RETRIES x, exponential backoff). On final
    failure returns text="" with a populated `error` — the caller treats this as
    an unreadable page (never as success, never crashes).
    """
    png_path = Path(png_path)
    img_bytes = png_path.read_bytes()
    cache_file = _cache_path(img_bytes)

    if cache_file.exists():
        rec = json.loads(cache_file.read_text(encoding="utf-8"))
        rec["cached"] = True
        return _postprocess(rec)

    # Cache miss: we will call the API, so the key is required NOW. A missing key
    # is not transient — fail fast with the clear message instead of retrying.
    # (Fully-cached reruns above never reach here and need no key at all.)
    _client()

    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    b64 = base64.b64encode(img_bytes).decode("ascii")

    last_err = None
    for attempt in range(TOOL_RETRIES + 1):
        try:
            text = _call_openai(b64)
            rec = {"text": text, "model": OCR_MODEL, "cached": False, "error": None}
            # Cache the RAW transcription; clean only the copy we hand back.
            cache_file.write_text(
                json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return _postprocess(dict(rec))
        except Exception as e:  # transient API/network error -> retry with backoff
            last_err = e
            if attempt < TOOL_RETRIES:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))

    # Exhausted retries: report, do not crash, do not fabricate. Not cached, so a
    # later rerun will try again.
    return _postprocess(
        {"text": "", "model": OCR_MODEL, "cached": False, "error": str(last_err)}
    )
