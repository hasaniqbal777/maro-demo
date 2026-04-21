"""End-to-end pipeline test with a mocked LLM and mocked tools.

Verifies: trace records every expected agent, majority vote + confidence compute,
implicit-verdict extraction flows end-to-end.
"""

from __future__ import annotations

from unittest.mock import patch

from maro.pipeline import analyze, infer
from maro.trace import trace_context


def _fake_chat(*, agent, step, system_prompt, user_prompt, model, temperature):
    from maro.llm import current_trace  # noqa: F401 imported for parity
    from maro.trace import AgentCall, current_trace as ct
    # Generate plausible responses per agent to keep parsers happy.
    if agent == "fact_questioning":
        resp = "1. Did event X happen?\n2. Was person Y there?"
    elif agent == "questioning":
        resp = "1. What about aspect Z?"
    elif agent == "judge":
        resp = "reasoning: evidence contradicts claim\njudgment: 1"
    elif agent == "rule_optimizer":
        resp = "A new rule.\nOutput requirements: - Output format: judgment: <'1' or '0'>"
    else:
        resp = f"Analysis text for {agent}.\nImplicit verdict: FAKE"
    trace = ct()
    if trace is not None:
        trace.add(
            AgentCall(
                agent=agent,
                step=step,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=resp,
                tokens_in=10,
                tokens_out=20,
                latency_ms=1.0,
            )
        )
    return resp


def test_pipeline_end_to_end():
    news = "Some news item claiming X happened."
    with patch("maro.llm.chat", side_effect=_fake_chat), \
         patch("maro.agents.linguistic.chat", side_effect=_fake_chat), \
         patch("maro.agents.comment.chat", side_effect=_fake_chat), \
         patch("maro.agents.fact_questioning.chat", side_effect=_fake_chat), \
         patch("maro.agents.fact_checking.chat", side_effect=_fake_chat), \
         patch("maro.agents.questioning.chat", side_effect=_fake_chat), \
         patch("maro.agents.judge.chat", side_effect=_fake_chat), \
         patch("maro.tools.serper.search", return_value=[]), \
         patch("maro.tools.wikipedia.lookup", return_value=None):
        with trace_context() as trace:
            report = analyze(news=news, comments=["comment 1", "comment 2"])
            result = infer(report=report, rules=["rule A", "rule B", "rule C"], demonstrations=[])

    assert report.linguistic.implicit_verdict == "FAKE"
    assert report.comment is not None and report.comment.implicit_verdict == "FAKE"
    assert report.fact.implicit_verdict == "FAKE"
    assert result.label == 1
    assert result.confidence == 1.0
    agents_seen = {c.agent for c in trace.calls}
    # Every expected agent fired.
    assert {"linguistic", "comment", "fact_questioning", "fact_checking", "questioning", "judge"} <= agents_seen
