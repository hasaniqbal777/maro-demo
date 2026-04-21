"""Comment Analysis Agent."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import TEMP_ANALYSIS
from ..llm import chat
from .base import BaseAgent, ImplicitVerdict, extract_implicit_verdict


@dataclass
class CommentReport:
    text: str
    implicit_verdict: ImplicitVerdict


def _format_comments(comments: list[str]) -> str:
    if not comments:
        return "(no comments available)"
    return "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(comments))


class CommentAnalysisAgent(BaseAgent):
    prompt_name = "comment"

    def analyze(
        self, *, news: str, comments: list[str], model: str, step: str = "initial"
    ) -> CommentReport:
        user_prompt = (
            f"News:\n{news}\n\n"
            f"Comments:\n{_format_comments(comments)}\n\n"
            "Produce the comment analysis report."
        )
        resp = chat(
            agent="comment",
            step=step,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=TEMP_ANALYSIS,
        )
        return CommentReport(text=resp, implicit_verdict=extract_implicit_verdict(resp))

    def refine(
        self,
        *,
        news: str,
        comments: list[str],
        prior_report: str,
        questions: list[str],
        model: str,
    ) -> CommentReport:
        user_prompt = (
            f"News:\n{news}\n\n"
            f"Comments:\n{_format_comments(comments)}\n\n"
            f"Your prior analysis:\n{prior_report}\n\n"
            "Follow-up questions raised by the Questioning Agent:\n"
            + "\n".join(f"- {q}" for q in questions)
            + "\n\nRevise and extend your analysis to address these."
        )
        resp = chat(
            agent="comment",
            step="reflection",
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=TEMP_ANALYSIS,
        )
        return CommentReport(text=resp, implicit_verdict=extract_implicit_verdict(resp))
