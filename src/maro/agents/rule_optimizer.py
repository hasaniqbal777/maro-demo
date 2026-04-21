"""Decision Rule Optimization Agent (paper Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import TEMP_OPTIMIZER
from ..llm import chat
from .base import BaseAgent, load_prompt


@dataclass
class RuleScore:
    rule: str
    accuracy: float


class DecisionRuleOptimizationAgent(BaseAgent):
    """Uses the rule_optimizer prompt template — not plain string prompt."""

    prompt_name = "rule_optimizer"

    def __init__(self) -> None:
        # The rule_optimizer file is a template, not a ready system prompt.
        self.prompt_template = load_prompt("rule_optimizer")
        # Use a short generic system message; the full template goes into the user message.
        self.system_prompt = (
            "You design decision rules for a multi-agent misinformation detection system."
        )

    def propose(
        self,
        *,
        trajectory: list[RuleScore],
        examples: list[str],
        model: str,
    ) -> str:
        trajectory_sorted = sorted(trajectory, key=lambda r: r.accuracy)
        traj_text = "\n\n".join(
            f"<decision rule {i + 1}, accuracy = {rs.accuracy:.2f}>\n{rs.rule}"
            for i, rs in enumerate(trajectory_sorted)
        )
        examples_text = "\n\n".join(f"Example {i + 1}:\n{ex}" for i, ex in enumerate(examples))
        user_prompt = self.prompt_template.format(
            trajectory=traj_text, examples=examples_text
        )
        return chat(
            agent="rule_optimizer",
            step=f"propose-{len(trajectory)}",
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=TEMP_OPTIMIZER,
        ).strip()
