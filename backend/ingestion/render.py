"""Render PDF pages to PNGs — the ONLY job pdfplumber does in this project.

The patient data is image-only (no embedded text layer), so pdfplumber cannot
extract text. We use it purely to rasterize each page to a PNG, which is then
sent to the vision-OCR model. Rendered PNGs are cached on disk (the cache key
embeds the render settings) so reruns are instant and deterministic.
"""
from __future__ import annotations

from pathlib import Path

import pdfplumber
from PIL import Image

from ..config import PAGE_CACHE_DIR, RENDER_DPI, MAX_IMAGE_EDGE


def _downscale(img: Image.Image, max_edge: int) -> Image.Image:
    """Downscale so the longest edge <= max_edge (never upscale). LANCZOS for text."""
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= max_edge:
        return img
    scale = max_edge / long_edge
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def find_pdfs(patient_dir: Path) -> list[Path]:
    """All PDFs in a patient folder, sorted by name (stable, reproducible ordering)."""
    return sorted(Path(patient_dir).glob("*.pdf"))


def render_pdf(pdf_path: Path, patient_id: str) -> list[Path]:
    """Render every page of one PDF to a cached PNG; return ordered PNG paths.

    The filename embeds DPI + max-edge so changing render settings invalidates
    stale renders automatically instead of silently reusing them.
    """
    pdf_path = Path(pdf_path)
    out_dir = PAGE_CACHE_DIR / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pages: list[Path] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            name = f"{pdf_path.stem}_p{i + 1:03d}_dpi{RENDER_DPI}_edge{MAX_IMAGE_EDGE}.png"
            png_path = out_dir / name
            if not png_path.exists():
                pil = page.to_image(resolution=RENDER_DPI).original  # base PIL image
                pil = _downscale(pil.convert("RGB"), MAX_IMAGE_EDGE)
                pil.save(png_path, format="PNG")
            pages.append(png_path)
    return pages
