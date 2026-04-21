"""Judge Agent — applies a decision rule to decide real/fake."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import TEMP_JUDGE
from ..llm import chat
from .base import BaseAgent


@dataclass
class Demonstration:
    news: str
    label: int  # 0=real, 1=fake
    analysis_report: str


@dataclass
class JudgeResult:
    label: int  # 0 or 1
    reasoning: str
    raw: str


class JudgeAgent(BaseAgent):
    prompt_name = "judge"

    def judge(
        self,
        *,
        news: str,
        analysis_report: str,
        decision_rule: str,
        demonstrations: list[Demonstration],
        model: str,
        step: str = "judge",
    ) -> JudgeResult:
        demo_blocks = []
        for i, d in enumerate(demonstrations, start=1):
            demo_blocks.append(
                f"--- Demonstration {i} (label={'fake' if d.label == 1 else 'real'}) ---\n"
                f"News: {d.news}\n"
                f"Analysis: {d.analysis_report}"
            )
        demo_text = "\n\n".join(demo_blocks) if demo_blocks else "(no demonstrations)"

        user_prompt = (
            f"DECISION RULE:\n{decision_rule}\n\n"
            f"Demonstrations from other domains:\n{demo_text}\n\n"
            f"Target news:\n{news}\n\n"
            f"Multi-dimensional analysis report:\n{analysis_report}\n\n"
            "Apply the decision rule and output your judgment."
        )
        resp = chat(
            agent="judge",
            step=step,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=TEMP_JUDGE,
        )
        return _parse_judge_output(resp)


def _parse_judge_output(text: str) -> JudgeResult:
    reasoning = ""
    label: int | None = None
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()
        elif low.startswith("judgment:") or low.startswith("judgement:"):
            m = re.search(r"[01]", line)
            if m:
                label = int(m.group())
    if label is None:
        m = re.search(r"\b[01]\b", text)
        label = int(m.group()) if m else 0
    return JudgeResult(label=label, reasoning=reasoning, raw=text)
