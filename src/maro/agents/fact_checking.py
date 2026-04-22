"""Fact-Checking Agent — consumes evidence with trust tiers (improvement #1)."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import TEMP_ANALYSIS
from ..llm import chat
from ..tools.serper import SearchHit
from ..tools.wikipedia import WikipediaHit
from .base import BaseAgent, ImplicitVerdict, extract_implicit_verdict


@dataclass
class FactReport:
    text: str
    implicit_verdict: ImplicitVerdict


def format_evidence(
    serper_hits: list[SearchHit],
    wiki_hits: list[WikipediaHit],
    *,
    use_trust_weighting: bool = True,
) -> str:
    lines: list[str] = []
    if wiki_hits:
        lines.append("## Wikipedia evidence")
        for w in wiki_hits:
            tag = "[TRUSTED · Wikipedia] " if use_trust_weighting else ""
            lines.append(f"- {tag}{w.title} — {w.summary}  (source: {w.url})")
    if serper_hits:
        lines.append("\n## Web search evidence")
        for h in serper_hits:
            if use_trust_weighting:
                note = f" · {h.trust_note}" if h.trust_note else ""
                tag = f"[{h.trust.upper()}{note}] "
            else:
                tag = ""
            lines.append(f"- {tag}{h.title} — {h.snippet}  (source: {h.url})")
    if not lines:
        return "(no evidence retrieved)"
    return "\n".join(lines)


class FactCheckingAgent(BaseAgent):
    prompt_name = "fact_checking"

    def analyze(
        self,
        *,
        news: str,
        evidence: str,
        model: str,
        step: str = "initial",
    ) -> FactReport:
        user_prompt = (
            f"News:\n{news}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Produce the fact-checking analysis report."
        )
        resp = chat(
            agent="fact_checking",
            step=step,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=TEMP_ANALYSIS,
        )
        return FactReport(text=resp, implicit_verdict=extract_implicit_verdict(resp))

    def refine(
        self,
        *,
        news: str,
        evidence: str,
        prior_report: str,
        questions: list[str],
        model: str,
    ) -> FactReport:
        user_prompt = (
            f"News:\n{news}\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Your prior analysis:\n{prior_report}\n\n"
            "Follow-up questions raised by the Questioning Agent:\n"
            + "\n".join(f"- {q}" for q in questions)
            + "\n\nRevise and extend your analysis to address these."
        )
        resp = chat(
            agent="fact_checking",
            step="reflection",
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=TEMP_ANALYSIS,
        )
        return FactReport(text=resp, implicit_verdict=extract_implicit_verdict(resp))
