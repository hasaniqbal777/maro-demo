"""Run the decision-rule optimization loop offline (paper Algorithm 1).

Cross-event faithful to paper Appendix G.4: rules are optimized on SOURCE events
only (events != target). The held-out target event is never seen by the
optimizer — evaluation on that target then uses the resulting rules.

Writes data/rules_<target_event>.json. Run once per target event to cover the
full 5-fold cross-event evaluation; the average across targets is comparable
to paper Table 17.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console

from maro.config import (
    DEFAULT_MODEL,
    MAX_PARALLEL,
    N_ATT,
    N_ITER,
    N_VAL_TASKS,
    PHEME_EVENTS,
    PROJECT_ROOT,
    TOP_K_RULES,
)
from maro.dataset.pheme_loader import load_source
from maro.dataset.validation_tasks import build_tasks
from maro.optimization import optimize
from maro.pipeline import analyze

console = Console()


def run_for_target(target_event: str, args: argparse.Namespace) -> None:
    console.rule(f"[bold]Target event held out: {target_event}[/bold]")
    items = load_source(target_event, limit_per_event=args.limit_per_event)
    console.print(f"Loaded {len(items)} items from {len(PHEME_EVENTS) - 1} source events.")

    tasks = build_tasks(items, n_tasks=args.tasks)
    console.print(f"Built {len(tasks)} cross-event validation tasks (source-only).")

    # Pre-compute & cache multi-dim analyses once per unique news item (paper-scale trick).
    # Each item's analysis is independent — run in parallel to dominate wall-clock time.
    # Individual items can hang (network, rate limit); per-item timeout + skip-on-failure
    # ensures one bad item can't stall the whole batch.
    unique: dict[str, str] = {}
    all_items = {it.id: it for t in tasks for it in [t.query] + t.demos}
    console.print(
        f"Running multi-dimensional analysis on {len(all_items)} unique items "
        f"(in parallel, max {MAX_PARALLEL} at a time, timeout={args.item_timeout}s/item)…"
    )

    def _analyze(nid: str, it) -> tuple[str, str]:
        report = analyze(news=it.text, comments=it.comments, model=args.model)
        return nid, report.as_text()

    completed = 0
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(_analyze, nid, it): (nid, it) for nid, it in all_items.items()}
        for fut in as_completed(futures, timeout=args.item_timeout * len(all_items)):
            nid, it = futures[fut]
            try:
                _, text = fut.result(timeout=args.item_timeout)
                unique[nid] = text
                completed += 1
                console.print(f"  [{completed}/{len(all_items)}] {nid} ({it.event})")
            except FuturesTimeout:
                failed.append(nid)
                console.print(f"  [TIMEOUT] {nid} ({it.event}) — skipping")
                fut.cancel()
            except Exception as exc:
                failed.append(nid)
                console.print(f"  [FAIL] {nid} ({it.event}) — {type(exc).__name__}: {exc}")

    if failed:
        console.print(f"[yellow]Skipped {len(failed)} item(s) due to timeout/error.[/yellow]")
        # Drop any validation task that references a skipped item — otherwise the Judge
        # gets blank context and the rule-accuracy signal is corrupted.
        original = len(tasks)
        tasks = [t for t in tasks if t.query.id in unique and all(d.id in unique for d in t.demos)]
        if len(tasks) < original:
            console.print(
                f"[yellow]Dropped {original - len(tasks)} validation task(s) that referenced skipped items. "
                f"{len(tasks)} tasks remaining.[/yellow]"
            )
        if not tasks:
            console.print("[red]No usable validation tasks after drops — aborting.[/red]")
            return

    out_path = PROJECT_ROOT / "data" / f"rules_{target_event}.json"
    console.print("Starting optimization loop …")
    top_k, state = optimize(
        tasks=tasks,
        analyses=unique,
        model=args.model,
        n_iter=args.iters,
        n_att=args.attempts,
        k=args.k,
        out_path=out_path,
    )
    console.print(
        f"Done. Best acc on source validation: {state.best_acc:.3f}. "
        f"Top-{args.k} rules → {out_path.relative_to(PROJECT_ROOT)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target-event",
        choices=list(PHEME_EVENTS) + ["all"],
        required=True,
        help="which event to hold out as the target. 'all' runs all 5 events sequentially.",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--iters", type=int, default=N_ITER)
    ap.add_argument("--attempts", type=int, default=N_ATT)
    ap.add_argument("--tasks", type=int, default=N_VAL_TASKS)
    ap.add_argument("--k", type=int, default=TOP_K_RULES)
    ap.add_argument(
        "--limit-per-event",
        type=int,
        default=6,
        help="cap items loaded per source event to keep costs bounded",
    )
    ap.add_argument(
        "--item-timeout",
        type=float,
        default=180.0,
        help="seconds before a single-item analysis is abandoned and skipped "
             "(protects against a hung Serper/Wikipedia/OpenAI call).",
    )
    args = ap.parse_args()

    targets = list(PHEME_EVENTS) if args.target_event == "all" else [args.target_event]
    for t in targets:
        run_for_target(t, args)


if __name__ == "__main__":
    main()
