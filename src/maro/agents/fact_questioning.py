"""Fact-Questioning Agent — emits yes/no questions for evidence retrieval."""

from __future__ import annotations

import re

from ..config import TEMP_ANALYSIS
from ..llm import chat
from .base import BaseAgent


class FactQuestioningAgent(BaseAgent):
    prompt_name = "fact_questioning"

    def generate(self, *, news: str, model: str) -> list[str]:
        user_prompt = (
            f"News:\n{news}\n\nGenerate 3-6 focused yes/no fact-checking questions."
        )
        resp = chat(
            agent="fact_questioning",
            step="initial",
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=TEMP_ANALYSIS,
        )
        return _parse_numbered(resp)


def _parse_numbered(text: str) -> list[str]:
    questions: list[str] = []
    for line in text.splitlines():
        m = re.match(r"\s*\d+[\.\)]\s*(.+)", line)
        if m:
            questions.append(m.group(1).strip())
    return questions
