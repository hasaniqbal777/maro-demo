"""Linguistic Feature Analysis Agent."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import TEMP_ANALYSIS
from ..llm import chat
from .base import BaseAgent, ImplicitVerdict, extract_implicit_verdict


@dataclass
class LinguisticReport:
    text: str
    implicit_verdict: ImplicitVerdict


class LinguisticFeatureAgent(BaseAgent):
    prompt_name = "linguistic"

    def analyze(self, *, news: str, model: str, step: str = "initial") -> LinguisticReport:
        user_prompt = f"News:\n{news}\n\nProduce the linguistic feature analysis report."
        resp = chat(
            agent="linguistic",
            step=step,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=TEMP_ANALYSIS,
        )
        return LinguisticReport(text=resp, implicit_verdict=extract_implicit_verdict(resp))

    def refine(
        self,
        *,
        news: str,
        prior_report: str,
        questions: list[str],
        model: str,
    ) -> LinguisticReport:
        user_prompt = (
            f"News:\n{news}\n\n"
            f"Your prior analysis:\n{prior_report}\n\n"
            "Follow-up questions raised by the Questioning Agent:\n"
            + "\n".join(f"- {q}" for q in questions)
            + "\n\nRevise and extend your analysis to address these."
        )
        resp = chat(
            agent="linguistic",
            step="reflection",
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=TEMP_ANALYSIS,
        )
        return LinguisticReport(text=resp, implicit_verdict=extract_implicit_verdict(resp))
