"""parse_doc — fetch OCR text for specific corpus pages, on demand.

Phase 1 already OCR'd every page, so this is a read against the loaded corpus
(not a re-OCR). It lets the agent re-read specific pages when planning or
resolving a conflict. Returns a structured result; a bad/empty page is reported,
never silently treated as success.
"""
from __future__ import annotations

from ..agent.state import AgentState


def parse_page_spec(spec, max_page: int) -> list[int]:
    """Accept [1,2], '1-3', '1,2,5', '4', or 4 -> sorted unique page numbers."""
    pages: set[int] = set()
    if isinstance(spec, int):
        pages.add(spec)
    elif isinstance(spec, (list, tuple)):
        for x in spec:
            pages.update(parse_page_spec(x, max_page))
    elif isinstance(spec, str):
        for part in spec.replace(" ", "").split(","):
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                if a.isdigit() and b.isdigit():
                    pages.update(range(int(a), int(b) + 1))
            elif part.isdigit():
                pages.add(int(part))
    return sorted(p for p in pages if 1 <= p <= max_page)


def parse_doc(state: AgentState, pages) -> dict:
    """Return OCR text for the requested pages."""
    max_page = max(state.ocr_index) if state.ocr_index else 0
    nums = parse_page_spec(pages, max_page)
    if not nums:
        return {"ok": False, "error": f"no valid pages in {pages!r} (1..{max_page})", "pages": []}
    out = []
    for p in nums:
        meta = state.ocr_index.get(p)
        if not meta:
            out.append({"page": p, "error": "page not found"})
            continue
        out.append({
            "page": p,
            "doc_type": meta.get("doc_type"),
            "source": meta.get("source"),
            "warning": meta.get("warning"),
            "text": meta.get("text", ""),
        })
    return {"ok": True, "pages": out}
