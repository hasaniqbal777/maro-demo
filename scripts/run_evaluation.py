"""Evaluate MARO on held-out PHEME items (cross-event, paper Appendix G.4).

For each PHEME event used as the held-out target, this script loads the
`data/rules_<target>.json` that was produced by `run_optimization.py
--target-event <target>` (rules optimized WITHOUT seeing the target), then
runs inference on the target event's items and records accuracy/F1.

Outputs:
  - data/results/eval.csv              (per-item predictions)
  - data/results/summary.json          (per-event + averaged accuracy/F1, ablation diff)
  - data/results/confidence_plot.png   (accuracy vs confidence)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console

from maro.config import (
    DEFAULT_MODEL,
    INITIAL_DECISION_RULE,
    PHEME_EVENTS,
    PROJECT_ROOT,
)
from maro.dataset.pheme_loader import load_target
from maro.optimization import load_rules
from maro.pipeline import analyze, infer

console = Console()
RESULTS = PROJECT_ROOT / "data" / "results"


def _eval_one(*, item, rules, model: str, use_trust: bool) -> dict:
    report = analyze(
        news=item.text,
        comments=item.comments,
        model=model,
        use_trust_weighting=use_trust,
    )
    result = infer(report=report, rules=rules, demonstrations=[], model=model)
    verdicts = report.implicit_verdicts()
    n_real = sum(1 for v in verdicts.values() if v == "REAL")
    n_fake = sum(1 for v in verdicts.values() if v == "FAKE")
    total_v = max(1, n_real + n_fake)
    agent_fake_rate = n_fake / total_v
    return {
        "id": item.id,
        "event": item.event,
        "label": item.label,
        "pred": result.label,
        "confidence": result.confidence,
        "agent_fake_rate": agent_fake_rate,
        "agents_real": n_real,
        "agents_fake": n_fake,
        "trust_weighting": use_trust,
    }


def _rules_for(target_event: str) -> tuple[list[str], str]:
    path = PROJECT_ROOT / "data" / f"rules_{target_event}.json"
    if path.exists():
        return load_rules(path), path.name
    return [INITIAL_DECISION_RULE], "seed (r_0 — optimization NOT run)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--per-event",
        type=int,
        default=10,
        help="items per held-out event to evaluate (paper uses the full event)",
    )
    ap.add_argument(
        "--events",
        nargs="+",
        default=list(PHEME_EVENTS),
        choices=list(PHEME_EVENTS),
        help="which events to hold out as target (default: all 5)",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--ablation",
        action="store_true",
        help="also run with trust weighting OFF on the same items",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict] = []
    for target in args.events:
        rules, rules_src = _rules_for(target)
        console.rule(f"[bold]Target: {target}[/bold]  ·  rules: {rules_src} ({len(rules)})")
        items = load_target(target)
        rng.shuffle(items)
        sample = items[: args.per_event]
        console.print(f"Evaluating {len(sample)} items on {args.model}. ablation={args.ablation}.")

        for i, item in enumerate(sample, start=1):
            console.print(f"  [{i}/{len(sample)}] {item.id} (label={item.label})")
            rows.append(_eval_one(item=item, rules=rules, model=args.model, use_trust=True))
            if args.ablation:
                rows.append(_eval_one(item=item, rules=rules, model=args.model, use_trust=False))

    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "eval.csv", index=False)

    summary: dict = {"model": args.model, "n": args.n, "by_event": {}}
    for (event, trust), grp in df.groupby(["event", "trust_weighting"]):
        key = f"{event}__trust={trust}"
        tp = ((grp.pred == 1) & (grp.label == 1)).sum()
        fp = ((grp.pred == 1) & (grp.label == 0)).sum()
        fn = ((grp.pred == 0) & (grp.label == 1)).sum()
        acc = (grp.pred == grp.label).mean()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        summary["by_event"][key] = {
            "n": int(len(grp)),
            "accuracy": float(acc),
            "f1": float(f1),
        }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    console.print_json(data=summary)

    # Confidence-calibration plot.
    trust_df = df[df.trust_weighting].copy()
    trust_df["correct"] = (trust_df.pred == trust_df.label).astype(int)
    buckets = defaultdict(list)
    for _, row in trust_df.iterrows():
        buckets[round(row.confidence, 2)].append(row.correct)
    xs = sorted(buckets)
    ys = [sum(buckets[x]) / len(buckets[x]) for x in xs]
    ns = [len(buckets[x]) for x in xs]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(xs, ys, "o-", label="accuracy")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("confidence (votes_majority / K)")
    ax.set_ylabel("accuracy")
    ax.set_title("Confidence calibration")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "confidence_plot.png", dpi=150)
    console.print(f"Wrote {RESULTS / 'confidence_plot.png'}")


if __name__ == "__main__":
    main()
