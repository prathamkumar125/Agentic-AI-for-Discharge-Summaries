"""CLI entrypoint.

    python -m backend.main ingest [patient_id ...]   # render + OCR + segment
    python -m backend.main run    [patient_id ...]   # run the agent -> draft + trace
    python -m backend.main render [patient_id ...]   # re-render the draft from saved state
    python -m backend.main add    <path.pdf> --name "..."  # intake an upload + full pipeline

With no patient id, every folder under data/ is processed.
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import DATA_DIR, OUTPUT_DIR, MAX_STEPS
from .ingestion.pipeline import ingest_patient


def _all_patients() -> list[str]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())


def cmd_ingest(args: argparse.Namespace) -> int:
    patients = args.patient_ids or _all_patients()
    if not patients:
        print(f"No patient folders found under {DATA_DIR}.", file=sys.stderr)
        return 1
    for pid in patients:
        print(f"\n=== Ingesting {pid} ===")
        try:
            ingest_patient(pid, verbose=not args.quiet)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .agent.loop import run_agent
    patients = args.patient_ids or _all_patients()
    if not patients:
        print(f"No patient folders found under {DATA_DIR}.", file=sys.stderr)
        return 1
    for pid in patients:
        corpus_path = OUTPUT_DIR / pid / "ocr_corpus.json"
        if not corpus_path.exists():
            print(f"[run] no corpus for {pid}; ingesting first ...")
            try:
                ingest_patient(pid, verbose=not args.quiet)
            except (RuntimeError, FileNotFoundError) as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 1
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        print(f"\n=== Running agent on {pid} ===")
        try:
            run_agent(corpus, verbose=not args.quiet,
                      fail_tools=args.fail_tool, max_steps=args.max_steps)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from .agent.render_summary import render_from_json
    patients = args.patient_ids or _all_patients()
    if not patients:
        print(f"No patient folders found under {DATA_DIR}.", file=sys.stderr)
        return 1
    rc = 0
    for pid in patients:
        state_path = OUTPUT_DIR / pid / "agent_state.json"
        if not state_path.exists():
            print(f"[render] no agent_state.json for {pid} — run the agent first.", file=sys.stderr)
            rc = 1
            continue
        out = render_from_json(state_path)
        print(f"[written] {out}")
    return rc


def cmd_add(args: argparse.Namespace) -> int:
    """Stage an uploaded PDF as a new patient and run the full pipeline (CLI parity)."""
    from .intake import intake_path, process_patient
    try:
        pid = intake_path(args.pdf, patient_name=args.name)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"[intake] staged '{args.pdf}' as patient '{pid}'")
    try:
        res = process_patient(pid, verbose=not args.quiet, max_steps=args.max_steps)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"[done] {res}")
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    """Part 2: learn correction-memory from simulated doctor edits; emit the curve."""
    from .learning.evaluate import run_curve
    patients = args.patient_ids or _all_patients_with_state()
    if not patients:
        print("No agent_state.json found — run the agent first.", file=sys.stderr)
        return 1
    rc = 0
    for pid in patients:
        state_path = OUTPUT_DIR / pid / "agent_state.json"
        if not state_path.exists():
            print(f"[learn] no agent_state.json for {pid} — run the agent first.", file=sys.stderr)
            rc = 1
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        out_dir = OUTPUT_DIR / pid / "learning"
        try:
            res = run_curve(state, out_dir, seed=args.seed)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            rc = 1
            continue
        print(f"\n=== Part 2 learning — {pid} ===")
        print(f"  splits averaged : {res['n_splits']}  ·  {res['n_train']} train / "
              f"{res['n_held_out']} held-out of {res['n_editable']} editable sections")
        print(f"  edit distance   : {res['before']['mean_edit_distance']} -> "
              f"{res['after']['mean_edit_distance']}  (−{res['edit_distance_reduction']}, "
              f"{res['edit_distance_reduction_pct']}% lower)")
        print(f"  match rate      : {res['match_rate_before']} -> {res['match_rate_after']}")
        print(f"  rules learned   : {res['rules_learned_representative']} (representative split)")
        print(f"  safety held     : {res['safety_invariant_held']} (max penalty {res['max_safety_penalty']}; "
              f"{res['citations_preserved']} citations + {res['flags_preserved']} flags untouched)")
        print(f"  [written] {out_dir / 'improvement_curve.png'}")
        print(f"  [written] {out_dir / 'metrics.json'}")
    return rc


def _all_patients_with_state() -> list[str]:
    if not OUTPUT_DIR.is_dir():
        return []
    return sorted(p.name for p in OUTPUT_DIR.iterdir()
                  if (p / "agent_state.json").exists())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="discharge-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="render + OCR + segment a patient's PDFs")
    p_ing.add_argument("patient_ids", nargs="*", help="e.g. patient_2 (default: all)")
    p_ing.add_argument("-q", "--quiet", action="store_true", help="suppress per-page log")
    p_ing.set_defaults(func=cmd_ingest)

    p_run = sub.add_parser("run", help="run the discharge-summary agent on a patient")
    p_run.add_argument("patient_ids", nargs="*", help="e.g. patient_2 (default: all)")
    p_run.add_argument("-q", "--quiet", action="store_true", help="suppress live trace")
    p_run.add_argument("--max-steps", type=int, default=MAX_STEPS,
                       help=f"hard reasoning-turn cap (default {MAX_STEPS})")
    p_run.add_argument("--fail-tool", action="append", metavar="TOOL",
                       help="inject a forced failure for a tool (robustness testing)")
    p_run.set_defaults(func=cmd_run)

    p_rnd = sub.add_parser("render", help="re-render discharge_summary.md from saved agent_state.json")
    p_rnd.add_argument("patient_ids", nargs="*", help="e.g. patient_2 (default: all)")
    p_rnd.set_defaults(func=cmd_render)

    p_add = sub.add_parser("add", help="intake an uploaded PDF as a new patient + run the pipeline")
    p_add.add_argument("pdf", help="path to the patient's source-note PDF")
    p_add.add_argument("--name", default=None, help="patient name (default: the file name)")
    p_add.add_argument("-q", "--quiet", action="store_true", help="suppress live trace")
    p_add.add_argument("--max-steps", type=int, default=MAX_STEPS,
                       help=f"hard reasoning-turn cap (default {MAX_STEPS})")
    p_add.set_defaults(func=cmd_add)

    p_learn = sub.add_parser("learn", help="Part 2: learn from simulated doctor edits + emit curve")
    p_learn.add_argument("patient_ids", nargs="*", help="e.g. patient_2 (default: all with state)")
    p_learn.add_argument("--seed", type=int, default=7, help="train/held-out split seed")
    p_learn.set_defaults(func=cmd_learn)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
