"""Decision Rule Optimization — paper Algorithm 1, at course-project scale."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agents import JudgeAgent
from .agents.judge import Demonstration
from .agents.rule_optimizer import DecisionRuleOptimizationAgent, RuleScore
from .config import (
    DEFAULT_MODEL,
    INITIAL_DECISION_RULE,
    MAX_PARALLEL,
    N_ATT,
    N_ITER,
    TOP_K_RULES,
)
from .dataset.pheme_loader import NewsItem
from .dataset.validation_tasks import ValidationTask


@dataclass
class CachedAnalysis:
    """A per-news multi-dimensional report, serialised to plain text."""
    news_id: str
    analysis_text: str


@dataclass
class OptimizationState:
    trajectory: list[RuleScore] = field(default_factory=list)
    best_rule: str = ""
    best_acc: float = 0.0


def _evaluate(
    *,
    rule: str,
    tasks: list[ValidationTask],
    analyses: dict[str, str],
    model: str,
) -> float:
    judge = JudgeAgent()

    def _score_one(t: ValidationTask) -> bool:
        demos = [
            Demonstration(news=d.text, label=d.label, analysis_report=analyses.get(d.id, ""))
            for d in t.demos
        ]
        result = judge.judge(
            news=t.query.text,
            analysis_report=analyses.get(t.query.id, ""),
            decision_rule=rule,
            demonstrations=demos,
            model=model,
            step="validation",
        )
        return result.label == t.query.label

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        correct = sum(pool.map(_score_one, tasks))
    return correct / max(1, len(tasks))


def _example_block(task: ValidationTask, analyses: dict[str, str]) -> str:
    return (
        f"Input news: {task.query.text}\n"
        f"Multi-dimensional analysis: {analyses.get(task.query.id, '')}\n"
        f"<DECISION RULE>\n"
        f"Ground-truth label: {'fake' if task.query.label == 1 else 'real'}"
    )


def optimize(
    *,
    tasks: list[ValidationTask],
    analyses: dict[str, str],
    model: str = DEFAULT_MODEL,
    n_iter: int = N_ITER,
    n_att: int = N_ATT,
    k: int = TOP_K_RULES,
    out_path: Path | None = None,
    seed_rule: str = INITIAL_DECISION_RULE,
) -> tuple[list[str], OptimizationState]:
    """Iteratively refine decision rules, return top-K and the full state."""
    state = OptimizationState()
    acc0 = _evaluate(rule=seed_rule, tasks=tasks, analyses=analyses, model=model)
    state.trajectory.append(RuleScore(rule=seed_rule, accuracy=acc0))
    state.best_rule, state.best_acc = seed_rule, acc0

    optimizer = DecisionRuleOptimizationAgent()
    sample_examples = [_example_block(t, analyses) for t in tasks[:3]]

    attempts = 0
    for i in range(n_iter):
        if attempts >= n_att:
            break
        top10 = sorted(state.trajectory, key=lambda r: r.accuracy, reverse=True)[:10]
        candidate = optimizer.propose(
            trajectory=list(reversed(top10)),
            examples=sample_examples,
            model=model,
        )
        acc = _evaluate(rule=candidate, tasks=tasks, analyses=analyses, model=model)
        state.trajectory.append(RuleScore(rule=candidate, accuracy=acc))
        if acc > state.best_acc:
            state.best_rule, state.best_acc = candidate, acc
            attempts = 0
        else:
            attempts += 1
        print(f"[iter {i + 1:>3}] acc={acc:.3f}  best={state.best_acc:.3f}  attempts={attempts}")

    top_k = [rs.rule for rs in sorted(state.trajectory, key=lambda r: r.accuracy, reverse=True)[:k]]
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "top_k": top_k,
                    "trajectory": [asdict(rs) for rs in state.trajectory],
                    "best_acc": state.best_acc,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return top_k, state


def load_rules(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["top_k"])
