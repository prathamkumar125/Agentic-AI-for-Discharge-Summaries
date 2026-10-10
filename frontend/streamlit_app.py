"""Streamlit frontend for the discharge-summary agent.

Run from the repository root with:
    streamlit run streamlit_app.py

The app calls the same intake and processing functions as the CLI and FastAPI
frontends. It does not contain a second implementation of the agent pipeline.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import streamlit as st

from backend.config import OUTPUT_DIR
from backend.intake import process_upload


STATUS_META = {
    "conflict": ("Conflict", "#b91c1c", "#fde8e8"),
    "pending": ("Pending", "#b45309", "#fdf0db"),
    "missing": ("Missing", "#4b5563", "#eef0f3"),
    "present": ("Present", "#047857", "#e3f5ec"),
}

STEPS = [
    ("Enter the patient's name", "Used only to label and organize the generated draft."),
    ("Upload the source-note PDF", "One patient's scanned admission, progress, lab, or medication notes."),
    ("Generate the discharge summary", "The agent OCRs every page and reasons over it. Cached pages make later runs faster."),
    ("Review the draft and status banner", "Every field is cited, or marked Missing, Pending, or Conflict and flagged."),
    ("Download the result", "Export Markdown, or a ZIP that also includes the reasoning trace."),
]

LEGEND = [
    ("present", "found in the notes with a verbatim citation"),
    ("pending", "documented as awaited, such as a lab not yet resulted"),
    ("missing", "not found anywhere — never guessed"),
    ("conflict", "two notes disagree — both are shown, not resolved"),
    ("flag", "items escalated for clinician review"),
]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root { --accent: #2563eb; --border: #e3e6ea; --muted: #6b7280; }
        .block-container { max-width: 920px; padding-top: 1.5rem; padding-bottom: 5rem; }
        .app-title { font-size: 2rem; font-weight: 700; letter-spacing: -.025em; margin: 0; }
        .sub { color: #6b7280; line-height: 1.55; max-width: 70ch; }
        .draft-tag { display: inline-block; color: #b91c1c; background: #fde8e8;
          padding: .2rem .55rem; border-radius: .4rem; font-size: .75rem;
          font-weight: 700; letter-spacing: .03em; }
        .status-banner { border: 1px solid #e3e6ea; border-left: 4px solid #2563eb;
          border-radius: .65rem; padding: .85rem 1rem; margin: 1rem 0; }
        .chip { display: inline-block; border-radius: 999px; padding: .2rem .65rem;
          margin: 0 .35rem .35rem 0; font-size: .8rem; font-weight: 600; }
        .warning-note { margin-top: .45rem; line-height: 1.5; }
        .guide-step { border: 1px solid #e3e6ea; border-radius: .75rem;
          padding: .9rem 1rem; margin: .65rem 0; }
        .step-num { display: inline-flex; width: 1.7rem; height: 1.7rem; border-radius: 50%;
          background: #2563eb; color: white; align-items: center; justify-content: center;
          font-weight: 700; margin-right: .55rem; }
        .safety-note { border-left: 4px solid #b91c1c; padding: .8rem 1rem;
          background: #fff; margin: 1.25rem 0; line-height: 1.55; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_nav() -> str:
    st.markdown("## 🩺 Discharge Summary Agent")
    current_view = st.session_state.get("current_view", "Generate")
    view = st.radio(
        "Navigate",
        ("Generate", "Guide"),
        index=0 if current_view == "Generate" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.current_view = view
    return view


def render_guide() -> None:
    st.header("How to use this")
    st.markdown(
        '<p class="sub">Upload a patient\'s messy source notes and get a structured, '
        '<b>citation-backed discharge-summary draft to review</b> — in a few clicks.</p>',
        unsafe_allow_html=True,
    )
    for index, (title, detail) in enumerate(STEPS, 1):
        st.markdown(
            f'<div class="guide-step"><span class="step-num">{index}</span>'
            f'<b>{title}</b><br><span class="sub">{detail}</span></div>',
            unsafe_allow_html=True,
        )
    st.subheader("What the status labels mean")
    for status, description in LEGEND:
        if status == "flag":
            color, background = "#6d28d9", "#efe9fd"
            label = "Flags"
        else:
            label, color, background = STATUS_META[status]
        st.markdown(
            f'<span class="chip" style="color:{color};background:{background}">{label}</span> '
            f'{description}',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="safety-note"><b>⚠️ Always a draft for clinician review.</b> '
        'The agent never invents a clinical fact — anything it cannot source is flagged, '
        'not filled in. Use <b>synthetic data only</b>; do not upload real patient information.</div>',
        unsafe_allow_html=True,
    )
    if st.button("Start now →", type="primary"):
        st.session_state.current_view = "Generate"
        st.rerun()


def status_counts(state: dict) -> dict[str, list[str]]:
    counts = {key: [] for key in STATUS_META}
    for field in state.get("fields", {}).values():
        status = field.get("status", "missing")
        counts.setdefault(status, []).append(field.get("label", "Unknown field"))
    return counts


def render_banner(state: dict) -> None:
    counts = status_counts(state)
    chips = []
    for status in ("conflict", "pending", "missing", "present"):
        label, color, background = STATUS_META[status]
        chips.append(
            f'<span class="chip" style="color:{color};background:{background}">'
            f'{label}: {len(counts[status])}</span>'
        )
    flags = state.get("flags", [])
    chips.append(
        f'<span class="chip" style="color:#6d28d9;background:#efe9fd">Flags: {len(flags)}</span>'
    )
    details = []
    for status in ("conflict", "pending", "missing"):
        if counts[status]:
            details.append(f"{STATUS_META[status][0]}: <b>{', '.join(counts[status])}</b>")
    note = ""
    if details:
        note = (
            '<div class="warning-note">⚠️ '
            + ". ".join(details)
            + ". These were deliberately not invented — each needs clinician review.</div>"
        )
    st.markdown(
        f'<div class="status-banner">{" ".join(chips)}{note}</div>',
        unsafe_allow_html=True,
    )


def make_zip(patient_id: str) -> bytes:
    output_dir = OUTPUT_DIR / patient_id
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in ("discharge_summary.md", "trace.json", "agent_state.json"):
            path = output_dir / filename
            if path.exists():
                archive.writestr(filename, path.read_bytes())
    return buffer.getvalue()


def render_generate() -> None:
    st.markdown('<h1 class="app-title">Discharge Summary Agent</h1>', unsafe_allow_html=True)
    st.markdown(
        ' <p class="sub">Upload a patient\'s source-note PDF and a name. The agent drafts a '
        'structured discharge summary <b>for clinician review</b> — nothing is invented; '
        'unknowns are flagged. <b style="color:#b91c1c">Synthetic data only.</b></p>',
        unsafe_allow_html=True,
    )

    result = st.session_state.get("result")
    if result:
        patient_id = result["patient_id"]
        state = result["state"]
        st.subheader(f"Draft for `{patient_id}`")
        st.markdown('<span class="draft-tag">DRAFT — not a finalized clinical document</span>',
                    unsafe_allow_html=True)
        render_banner(state)
        col1, col2, col3 = st.columns(3)
        markdown = result["markdown"].encode("utf-8")
        with col1:
            st.download_button("⬇ Markdown", markdown, f"{patient_id}_discharge_summary.md",
                               "text/markdown", use_container_width=True)
        with col2:
            st.download_button("⬇ Zip (+ trace)", make_zip(patient_id),
                               f"{patient_id}_discharge.zip", "application/zip",
                               use_container_width=True)
        with col3:
            if st.button("New summary", use_container_width=True):
                st.session_state.pop("result", None)
                st.rerun()
        st.markdown(result["markdown"])
        return

    with st.form("upload"):
        name = st.text_input("Patient name", placeholder="e.g. Jane Doe")
        uploaded = st.file_uploader("Source-note PDF", type=["pdf"])
        submitted = st.form_submit_button("Generate discharge summary", type="primary")
    st.caption("First run OCRs every page (cached afterward) — this can take a minute.")

    if submitted:
        if not name.strip() or uploaded is None:
            st.error("Enter a patient name and choose a PDF before generating.")
            return
        try:
            with st.spinner(f"Processing {name.strip()}… rendering pages, OCR, and running the agent."):
                summary = process_upload(uploaded.getvalue(), uploaded.name, name.strip())
            summary_path = Path(summary["summary_path"])
            state_path = summary_path.parent / "agent_state.json"
            st.session_state.result = {
                "patient_id": summary["patient_id"],
                "markdown": summary_path.read_text(encoding="utf-8"),
                "state": json.loads(state_path.read_text(encoding="utf-8")),
            }
            st.rerun()
        except Exception as exc:  # surface processing failures in the UI
            st.error(f"{type(exc).__name__}: {exc}")


def main() -> None:
    st.set_page_config(page_title="Discharge Summary Agent", page_icon="🩺", layout="centered")
    inject_css()
    view = render_nav()
    if view == "Guide":
        render_guide()
    else:
        render_generate()


if __name__ == "__main__":
    main()
