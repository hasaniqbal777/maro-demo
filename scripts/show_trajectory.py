"""Print the Decision Rule Optimization trajectory in presentation-friendly form.

Reads `data/rules_<target>.json` produced by `run_optimization.py` and renders
the seed r_0 + each candidate rule with its validation accuracy. Designed for
screenshotting directly into slides.

Usage:
    uv run python scripts/show_trajectory.py --target-event charliehebdo
    uv run python scripts/show_trajectory.py --target-event charliehebdo --markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from maro.config import INITIAL_DECISION_RULE, PHEME_EVENTS, PROJECT_ROOT

console = Console()


def _shorten(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _spark(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[min(7, int((v - lo) / span * 7))] for v in values)


def render_rich(data: dict, target: str) -> None:
    traj = data["trajectory"]
    top_k = data["top_k"]
    best_acc = data["best_acc"]
    accs = [rs["accuracy"] for rs in traj]

    console.rule(f"[bold]Decision Rule Optimization — target event held out: {target}[/bold]")

    header = Table.grid(padding=(0, 2))
    header.add_row(
        f"Iterations run: [bold]{len(traj) - 1}[/bold]",
        f"Seed r₀ accuracy: [dim]{accs[0]:.3f}[/dim]",
        f"Best accuracy reached: [bold green]{best_acc:.3f}[/bold green]",
        f"Improvement: [bold]{(best_acc - accs[0]) * 100:+.1f} points[/bold]",
    )
    console.print(header)
    console.print(f"Accuracy trajectory: [cyan]{_spark(accs)}[/cyan]  "
                  f"[dim]({' → '.join(f'{a:.2f}' for a in accs)})[/dim]")
    console.print()

    table = Table(show_lines=True, header_style="bold")
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Accuracy", justify="right", width=10)
    table.add_column("Decision rule", min_width=80, overflow="fold")
    table.add_column("Top-K?", justify="center", width=8)

    for i, rs in enumerate(traj):
        acc = rs["accuracy"]
        is_seed = i == 0
        in_topk = rs["rule"] in top_k
        label = "r₀" if is_seed else f"r{i}"
        mark = "★" if in_topk else ""
        style = "green bold" if rs["rule"] == data.get("best_rule") else ""
        table.add_row(
            label, f"[{style}]{acc:.3f}[/{style}]" if style else f"{acc:.3f}",
            _shorten(rs["rule"], 240), mark,
        )
    console.print(table)

    console.print(Panel(
        "\n\n".join(
            f"[bold]Top-{i + 1}:[/bold]\n{rule}" for i, rule in enumerate(top_k)
        ),
        title=f"Top-{len(top_k)} rules used at inference (majority vote)",
        border_style="green",
    ))


def render_markdown(data: dict, target: str) -> None:
    traj = data["trajectory"]
    top_k = data["top_k"]
    accs = [rs["accuracy"] for rs in traj]

    print(f"## MARO Decision Rule Optimization — `{target}` held out\n")
    print(f"- Iterations run: **{len(traj) - 1}**")
    print(f"- Seed r₀ accuracy: `{accs[0]:.3f}`")
    print(f"- Best accuracy: **`{data['best_acc']:.3f}`** "
          f"(`{(data['best_acc'] - accs[0]) * 100:+.1f}` points over seed)\n")
    print(f"Accuracy trajectory: {_spark(accs)}  "
          f"({' → '.join(f'{a:.2f}' for a in accs)})\n")

    print("| # | Acc | Rule | Top-K |")
    print("|---|-----|------|:-----:|")
    top_set = set(top_k)
    for i, rs in enumerate(traj):
        label = "r₀" if i == 0 else f"r{i}"
        mark = "★" if rs["rule"] in top_set else ""
        print(f"| {label} | `{rs['accuracy']:.3f}` | {_shorten(rs['rule'], 180)} | {mark} |")

    print(f"\n### Top-{len(top_k)} rules used at inference\n")
    for i, rule in enumerate(top_k, 1):
        print(f"**Top-{i}**\n\n> {rule}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target-event",
        choices=list(PHEME_EVENTS),
        help="which target event's rules file to show",
    )
    ap.add_argument("--file", help="explicit path to a rules_*.json file")
    ap.add_argument("--markdown", action="store_true", help="output as markdown")
    args = ap.parse_args()

    if args.file:
        path = Path(args.file)
    elif args.target_event:
        path = PROJECT_ROOT / "data" / f"rules_{args.target_event}.json"
    else:
        sys.exit("need --target-event or --file")
    if not path.exists():
        sys.exit(
            f"{path} does not exist — run `uv run python scripts/run_optimization.py "
            f"--target-event {args.target_event or '<event>'}` first."
        )

    data = json.loads(path.read_text())
    # Store best_rule for highlighting (not always in JSON; derive from top_k[0]).
    if "best_rule" not in data and data.get("top_k"):
        data["best_rule"] = data["top_k"][0]

    target = args.target_event or path.stem.replace("rules_", "")
    if args.markdown:
        render_markdown(data, target)
    else:
        render_rich(data, target)
        console.print(
            f"\n[dim]Source file: {path.relative_to(PROJECT_ROOT)}  ·  "
            f"seed rule r₀ is defined in src/maro/config.py[/dim]"
        )


if __name__ == "__main__":
    main()
