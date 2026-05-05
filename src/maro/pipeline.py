"""MARO Module 1 orchestration + inference (majority vote over top-K rules).

Inference-path parallelism: independent LLM calls and tool calls are dispatched
concurrently via ThreadPoolExecutor (they are I/O bound). Worker threads need
an explicit, freshly-copied `Context` object per submission so our module-level
`current_trace()` contextvar is visible and the Trace receives every agent
call. Each submission gets its OWN copied context — sharing one across workers
raises "Context already entered" when workers run concurrently.

The progress callback is always invoked from the main thread so Streamlit UI
updates are thread-safe.
"""

from __future__ import annotations

import contextvars
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .agents import (
    CommentAnalysisAgent,
    DecisionRuleOptimizationAgent,  # re-exported for convenience
    FactCheckingAgent,
    FactQuestioningAgent,
    ImplicitVerdict,
    JudgeAgent,
    LinguisticFeatureAgent,
    QuestioningAgent,
)
from .config import MAX_PARALLEL
from .agents.fact_checking import FactReport, format_evidence
from .agents.judge import Demonstration, JudgeResult
from .agents.linguistic import LinguisticReport
from .agents.comment import CommentReport
from .config import DEFAULT_MODEL
from .tools.serper import SearchHit, search as serper_search
from .tools.wikipedia import WikipediaHit, lookup as wiki_lookup

__all__ = [
    "MultiDimReport",
    "InferenceResult",
    "analyze",
    "infer",
    "DecisionRuleOptimizationAgent",
]


@dataclass
class MultiDimReport:
    news: str
    comments: list[str]
    linguistic: LinguisticReport
    comment: CommentReport | None
    fact: FactReport
    fact_questions: list[str]
    serper_hits: list[SearchHit] = field(default_factory=list)
    wiki_hits: list[WikipediaHit] = field(default_factory=list)

    def as_text(self) -> str:
        parts = ["## Linguistic analysis", self.linguistic.text]
        if self.comment is not None:
            parts += ["\n## Comment analysis", self.comment.text]
        parts += ["\n## Fact-checking analysis", self.fact.text]
        return "\n".join(parts)

    def implicit_verdicts(self) -> dict[str, ImplicitVerdict]:
        out: dict[str, ImplicitVerdict] = {
            "linguistic": self.linguistic.implicit_verdict,
            "fact_checking": self.fact.implicit_verdict,
        }
        if self.comment is not None:
            out["comment"] = self.comment.implicit_verdict
        return out


@dataclass
class InferenceResult:
    label: int  # 0 real, 1 fake
    confidence: float  # fraction of top-K rules agreeing with majority
    per_rule: list[tuple[str, JudgeResult]]
    report: MultiDimReport


ProgressCb = Callable[[str, str, Any], None]
# signature: (stage, status, data) where status ∈ {"running", "complete", "skipped"}


def _noop(stage: str, status: str, data: Any = None) -> None:
    return None


def _submit(pool: ThreadPoolExecutor, fn: Callable, /, *args: Any, **kwargs: Any) -> Future:
    """Submit `fn` to `pool` in a fresh per-call context copy.

    This is required on Python 3.14 because ThreadPoolExecutor does not
    auto-propagate contextvars to worker threads. Each submission gets its own
    copied context — never share a single Context across workers (Context
    objects can only be entered once at a time)."""
    ctx = contextvars.copy_context()
    return pool.submit(ctx.run, fn, *args, **kwargs)


def analyze(
    *,
    news: str,
    comments: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    use_reflection: bool = True,
    use_trust_weighting: bool = True,
    progress: ProgressCb | None = None,
) -> MultiDimReport:
    """Run the full multi-dimensional analysis with optional question-reflection.

    `progress(stage, status, data)` is invoked before/after each major stage so a UI can
    stream updates live. Stages: linguistic, comment, fact_questioning, fact_search,
    fact_checking, questioning_linguistic, refine_linguistic, questioning_comment,
    refine_comment, questioning_fact, refine_fact.
    """
    notify = progress or _noop
    comments = comments or []
    has_comments = bool(comments)

    lf = LinguisticFeatureAgent()
    ca = CommentAnalysisAgent()
    fq = FactQuestioningAgent()
    fc = FactCheckingAgent()
    qa = QuestioningAgent()

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        # Phase 1: three independent initial analyses fire in parallel.
        notify("linguistic", "running", None)
        if has_comments:
            notify("comment", "running", None)
        notify("fact_questioning", "running", None)

        f_ling = _submit(pool, lf.analyze, news=news, model=model)
        f_comment = (
            _submit(pool, ca.analyze, news=news, comments=comments, model=model)
            if has_comments else None
        )
        f_factq = _submit(pool, fq.generate, news=news, model=model)

        linguistic = f_ling.result()
        notify("linguistic", "complete", linguistic)
        if has_comments:
            comment_report = f_comment.result()
            notify("comment", "complete", comment_report)
        else:
            comment_report = None
            notify("comment", "skipped", None)
        fact_qs = f_factq.result()
        notify("fact_questioning", "complete", fact_qs)

        # Phase 2: search in parallel across every question × {serper, wikipedia}.
        notify("fact_search", "running", None)
        serper_hits: list[SearchHit] = []
        wiki_hits: list[WikipediaHit] = []
        search_futures: list[tuple[str, object]] = []
        for q in fact_qs[:5]:
            search_futures.append(("serper", _submit(pool, serper_search, q, num=10)))
            search_futures.append(("wiki",   _submit(pool, wiki_lookup, q)))
        for kind, fut in search_futures:
            r = fut.result()
            if kind == "serper":
                serper_hits.extend(r)
            elif r is not None:
                wiki_hits.append(r)
        notify("fact_search", "complete", {"serper": serper_hits, "wiki": wiki_hits})

        # Phase 3: fact-check LLM call (needs all evidence).
        evidence = format_evidence(
            serper_hits, wiki_hits, use_trust_weighting=use_trust_weighting
        )
        notify("fact_checking", "running", None)
        fact = fc.analyze(news=news, evidence=evidence, model=model)
        notify("fact_checking", "complete", fact)

        # Phase 4: three independent reflection+refine chains run in parallel.
        if use_reflection:
            notify("questioning_linguistic", "running", None)
            if comment_report is not None:
                notify("questioning_comment", "running", None)
            else:
                notify("questioning_comment", "skipped", None)
                notify("refine_comment", "skipped", None)
            notify("questioning_fact", "running", None)

            f_qling = _submit(
                pool,
                qa.review, news=news, report=linguistic.text, target="linguistic", model=model
            )
            f_qcom = (
                _submit(
                    pool,
                    qa.review,
                    news=news, report=comment_report.text, target="comment",
                    comments=comments, model=model,
                )
                if comment_report is not None else None
            )
            f_qfact = _submit(
                pool,
                qa.review,
                news=news, report=fact.text, target="fact", evidence=evidence, model=model,
            )

            ling_qs = f_qling.result()
            notify("questioning_linguistic", "complete", ling_qs)
            com_qs = f_qcom.result() if f_qcom is not None else None
            if comment_report is not None:
                notify("questioning_comment", "complete", com_qs)
            fact_qs_reflect = f_qfact.result()
            notify("questioning_fact", "complete", fact_qs_reflect)

            # Refinements — also in parallel.
            f_refine_ling = (
                _submit(
                    pool,
                    lf.refine, news=news, prior_report=linguistic.text,
                    questions=ling_qs, model=model,
                )
                if ling_qs else None
            )
            f_refine_com = (
                _submit(
                    pool,
                    ca.refine, news=news, comments=comments,
                    prior_report=comment_report.text, questions=com_qs or [], model=model,
                )
                if (comment_report is not None and com_qs) else None
            )
            f_refine_fact = (
                _submit(
                    pool,
                    fc.refine, news=news, evidence=evidence, prior_report=fact.text,
                    questions=fact_qs_reflect, model=model,
                )
                if fact_qs_reflect else None
            )

            if f_refine_ling is not None:
                notify("refine_linguistic", "running", None)
                linguistic = f_refine_ling.result()
                notify("refine_linguistic", "complete", linguistic)
            else:
                notify("refine_linguistic", "skipped", None)

            if f_refine_com is not None:
                notify("refine_comment", "running", None)
                comment_report = f_refine_com.result()
                notify("refine_comment", "complete", comment_report)
            elif comment_report is not None:
                notify("refine_comment", "skipped", None)

            if f_refine_fact is not None:
                notify("refine_fact", "running", None)
                fact = f_refine_fact.result()
                notify("refine_fact", "complete", fact)
            else:
                notify("refine_fact", "skipped", None)

    return MultiDimReport(
        news=news,
        comments=comments,
        linguistic=linguistic,
        comment=comment_report,
        fact=fact,
        fact_questions=fact_qs,
        serper_hits=serper_hits,
        wiki_hits=wiki_hits,
    )


def infer(
    *,
    report: MultiDimReport,
    rules: list[str],
    demonstrations: list[Demonstration],
    model: str = DEFAULT_MODEL,
) -> InferenceResult:
    """Run the JudgeAgent under each rule; majority vote with confidence.

    The K rule votes are dispatched in parallel — they are independent LLM calls.
    """
    judge = JudgeAgent()

    def _vote(indexed: tuple[int, str]) -> tuple[str, JudgeResult]:
        i, rule = indexed
        return rule, judge.judge(
            news=report.news,
            analysis_report=report.as_text(),
            decision_rule=rule,
            demonstrations=demonstrations,
            model=model,
            step=f"judge-rule-{i}",
        )

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = [_submit(pool, _vote, indexed) for indexed in enumerate(rules)]
        per_rule: list[tuple[str, JudgeResult]] = [f.result() for f in futures]

    labels = [r.label for _, r in per_rule]
    counter = Counter(labels)
    majority_label, majority_count = counter.most_common(1)[0]
    confidence = majority_count / len(labels) if labels else 0.0
    return InferenceResult(
        label=majority_label,
        confidence=confidence,
        per_rule=per_rule,
        report=report,
    )
