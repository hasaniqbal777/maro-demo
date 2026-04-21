"""One-shot inference on a single news item (smoke test for the demo)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel

from maro.config import DEFAULT_MODEL, INITIAL_DECISION_RULE, PROJECT_ROOT
from maro.optimization import load_rules
from maro.pipeline import analyze, infer
from maro.trace import trace_context

console = Console()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", required=True, help="news text")
    ap.add_argument("--comments", nargs="*", default=[])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--target-event",
        default=None,
        help="If set, load data/rules_<target-event>.json (paper-faithful, "
             "rules optimized without seeing this event). Otherwise uses r_0.",
    )
    ap.add_argument(
        "--rules",
        default=None,
        help="Explicit path to a rules file; overrides --target-event.",
    )
    ap.add_argument("--no-reflection", action="store_true")
    ap.add_argument("--no-trust", action="store_true")
    args = ap.parse_args()

    if args.rules:
        rules_path = Path(args.rules)
    elif args.target_event:
        rules_path = PROJECT_ROOT / "data" / f"rules_{args.target_event}.json"
    else:
        rules_path = None

    if rules_path is not None and rules_path.exists():
        rules = load_rules(rules_path)
        src = str(rules_path)
    else:
        rules = [INITIAL_DECISION_RULE]
        src = "seed r_0"
    console.print(f"[dim]Using {len(rules)} rule(s) from {src}[/dim]")

    with trace_context() as trace:
        report = analyze(
            news=args.news,
            comments=args.comments,
            model=args.model,
            use_reflection=not args.no_reflection,
            use_trust_weighting=not args.no_trust,
        )
        result = infer(report=report, rules=rules, demonstrations=[], model=args.model)

    console.print(Panel(report.linguistic.text, title="Linguistic"))
    if report.comment is not None:
        console.print(Panel(report.comment.text, title="Comment"))
    console.print(Panel(report.fact.text, title="Fact-Checking"))

    verdict = "FAKE" if result.label == 1 else "REAL"
    console.print(
        Panel(
            f"[bold]Verdict:[/bold] {verdict}    "
            f"[bold]Confidence:[/bold] {result.confidence:.2f}\n"
            + "\n".join(
                f"  rule {i}: {'FAKE' if r.label == 1 else 'REAL'}  ({r.reasoning})"
                for i, (_, r) in enumerate(result.per_rule)
            ),
            title="Final",
        )
    )

    tin, tout = trace.total_tokens
    console.print(f"[dim]Total tokens: in={tin} out={tout}  calls={len(trace.calls)}[/dim]")

    dump_path = PROJECT_ROOT / "data" / "last_trace.json"
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    dump_path.write_text(
        json.dumps(
            [
                {
                    "agent": c.agent,
                    "step": c.step,
                    "model": c.model,
                    "tokens_in": c.tokens_in,
                    "tokens_out": c.tokens_out,
                    "latency_ms": c.latency_ms,
                    "response": c.response,
                    "tool_calls": [
                        {"tool": t.tool, "query": t.query, "latency_ms": t.latency_ms}
                        for t in c.tool_calls
                    ],
                }
                for c in trace.calls
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[dim]Full trace written to {dump_path}[/dim]")


if __name__ == "__main__":
    main()
