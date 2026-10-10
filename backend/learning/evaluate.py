"""evaluate — measure that the correction-memory lowers edit burden on a HELD-OUT split,
with an improvement curve, while asserting the Part-1 safety invariants are untouched.

Held-out unit = per-section within the one available patient (we do not fabricate
patients). The memory is learned ONLY from train sections; the held-out sections are
measured but never learned from — so improvement reflects generalization, not memorization.

Because there is only one patient (few sections), a single train/held-out split is noisy
and order-dependent. We therefore report the curve **averaged over many random splits**
(Monte-Carlo cross-validation) with a min–max band, which is a more honest estimate of
"expected edit reduction vs. amount of training signal." Everything is offline and
deterministic given the seeds.

Safety: the learning only reshapes prose; it never touches the state's citations/flags, and
every held-out candidate is checked to retain all baseline clinical tokens (safety penalty
must stay 0 across every split).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from .reviewer import make_pairs
from .reward import edit_distance, section_reward
from .memory import CorrectionMemory

_N_SPLITS = 25  # Monte-Carlo random splits to average the curve over


def _measure(mem: CorrectionMemory, held_out: list[dict]) -> dict:
    dists, rewards, pens, matches = [], [], [], 0
    for p in held_out:
        cand = mem.apply(p["draft"])
        sec = section_reward(p["draft"], cand, p["edited"])
        dists.append(sec["edit_distance"]); rewards.append(sec["reward"])
        pens.append(sec["safety_penalty"]); matches += int(sec["match"])
    n = len(held_out)
    return {
        "mean_edit_distance": sum(dists) / n,
        "mean_reward": sum(rewards) / n,
        "match_rate": matches / n,
        "max_safety_penalty": max(pens),
    }


def _split(editable: list[dict], frac: float, rng: random.Random):
    items = list(editable)
    rng.shuffle(items)
    n_ho = max(1, min(len(items) - 1, round(len(items) * frac)))
    return items[:n_ho], items[n_ho:]   # held_out, train


def run_curve(state_dict: dict, out_dir: Path, *, held_out_frac: float = 0.4,
              seed: int = 7, n_splits: int = _N_SPLITS) -> dict:
    """Run the learning loop over many random splits; write metrics + averaged curve."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = make_pairs(state_dict)
    (out_dir / "pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    editable = [p for p in pairs if edit_distance(p["draft"], p["edited"]) > 1e-9]
    if len(editable) < 2:
        raise RuntimeError(
            f"Not enough editable sections to form a held-out split (found {len(editable)}). "
            "The reviewer made too few edits to learn from.")

    n_ho = max(1, min(len(editable) - 1, round(len(editable) * held_out_frac)))
    n_train = len(editable) - n_ho

    # Per-n_train accumulators across splits.
    by_n: dict[int, dict[str, list]] = defaultdict(lambda: {"ed": [], "match": [], "pen": []})
    rep_mem: CorrectionMemory | None = None
    rep_held_out: list[str] = []

    for s in range(n_splits):
        held_out, train = _split(editable, held_out_frac, random.Random(seed + s))
        mem = CorrectionMemory()
        for k in range(0, n_train + 1):
            if k > 0:
                mem.learn([(train[k - 1]["draft"], train[k - 1]["edited"])])
            m = _measure(mem, held_out)
            by_n[k]["ed"].append(m["mean_edit_distance"])
            by_n[k]["match"].append(m["match_rate"])
            by_n[k]["pen"].append(m["max_safety_penalty"])
        if s == 0:  # representative split -> saved memory + reported fields
            rep_mem, rep_held_out = mem, [p["field"] for p in held_out]

    curve = []
    for k in range(0, n_train + 1):
        ed = by_n[k]["ed"]
        curve.append({
            "n_train": k,
            "mean_edit_distance": round(sum(ed) / len(ed), 4),
            "edit_distance_min": round(min(ed), 4),
            "edit_distance_max": round(max(ed), 4),
            "match_rate": round(sum(by_n[k]["match"]) / len(by_n[k]["match"]), 4),
        })

    max_pen = max(max(v["pen"]) for v in by_n.values())
    assert max_pen == 0.0, f"SAFETY VIOLATION: learning lost clinical content (penalty={max_pen})"

    before, after = curve[0], curve[-1]
    n_citations = sum(len(fv.get("citations", [])) for fv in state_dict.get("fields", {}).values())
    n_flags = len(state_dict.get("flags", []))

    assert rep_mem is not None
    rep_mem.save(out_dir / "memory.json")
    _plot(curve, out_dir / "improvement_curve.png")

    summary = {
        "patient_id": state_dict.get("patient_id"),
        "seed": seed,
        "n_splits": n_splits,
        "n_sections_total": len(pairs),
        "n_editable": len(editable),
        "n_train": n_train,
        "n_held_out": n_ho,
        "representative_held_out_fields": rep_held_out,
        "before": {"mean_edit_distance": before["mean_edit_distance"], "match_rate": before["match_rate"]},
        "after": {"mean_edit_distance": after["mean_edit_distance"], "match_rate": after["match_rate"]},
        "edit_distance_reduction": round(before["mean_edit_distance"] - after["mean_edit_distance"], 4),
        "edit_distance_reduction_pct": round(
            100 * (before["mean_edit_distance"] - after["mean_edit_distance"])
            / before["mean_edit_distance"], 1) if before["mean_edit_distance"] else 0.0,
        "match_rate_before": before["match_rate"],
        "match_rate_after": after["match_rate"],
        "rules_learned_representative": len(rep_mem),
        "max_safety_penalty": max_pen,
        "safety_invariant_held": max_pen == 0.0,
        "citations_preserved": n_citations,   # untouched by learning (metadata)
        "flags_preserved": n_flags,           # untouched by learning (metadata)
        "curve": curve,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return summary


def _plot(curve: list[dict], path: Path) -> None:
    xs = [pt["n_train"] for pt in curve]
    dist = [pt["mean_edit_distance"] for pt in curve]
    dmin = [pt["edit_distance_min"] for pt in curve]
    dmax = [pt["edit_distance_max"] for pt in curve]
    acc = [pt["match_rate"] for pt in curve]

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.fill_between(xs, dmin, dmax, color="#b91c1c", alpha=0.12, label="edit-distance range")
    ax1.plot(xs, dist, "o-", color="#b91c1c", label="mean edit distance (held-out)")
    ax1.set_xlabel("# training sections learned from")
    ax1.set_ylabel("mean edit distance  (lower = better)", color="#b91c1c")
    ax1.tick_params(axis="y", labelcolor="#b91c1c")
    ax1.set_ylim(0, max(0.001, max(dmax) * 1.15))

    ax2 = ax1.twinx()
    ax2.plot(xs, acc, "s--", color="#047857", label="section match rate")
    ax2.set_ylabel("section match rate (held-out)", color="#047857")
    ax2.tick_params(axis="y", labelcolor="#047857")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    plt.title("Part 2 — learning from doctor edits: held-out improvement")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
