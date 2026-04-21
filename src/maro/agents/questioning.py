"""Questioning Agent — reviews an analysis report and poses follow-up questions."""

from __future__ import annotations

import re

from ..config import TEMP_ANALYSIS
from ..llm import chat
from .base import BaseAgent


class QuestioningAgent(BaseAgent):
    prompt_name = "questioning"

    def review(
        self,
        *,
        news: str,
        report: str,
        target: str,  # "linguistic" | "comment" | "fact"
        comments: list[str] | None = None,
        evidence: str | None = None,
        model: str,
    ) -> list[str]:
        extra = ""
        if target == "comment" and comments is not None:
            formatted = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(comments)) or "(none)"
            extra = f"\nComments:\n{formatted}\n"
        elif target == "fact" and evidence is not None:
            extra = f"\nEvidence:\n{evidence}\n"

        user_prompt = (
            f"News:\n{news}{extra}\n\n"
            f"Analysis report (target={target}):\n{report}\n\n"
            "Generate 2-4 follow-up questions."
        )
        resp = chat(
            agent="questioning",
            step=f"review-{target}",
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=TEMP_ANALYSIS,
        )
        return _parse_numbered(resp)


def _parse_numbered(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        m = re.match(r"\s*\d+[\.\)]\s*(.+)", line)
        if m:
            out.append(m.group(1).strip())
    return out
