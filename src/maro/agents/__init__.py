"""Agent implementations — one class per expert role defined in paper §2.2."""

from .base import BaseAgent, ImplicitVerdict, extract_implicit_verdict, parse_key_findings
from .comment import CommentAnalysisAgent
from .fact_checking import FactCheckingAgent
from .fact_questioning import FactQuestioningAgent
from .judge import JudgeAgent
from .linguistic import LinguisticFeatureAgent
from .questioning import QuestioningAgent
from .rule_optimizer import DecisionRuleOptimizationAgent

__all__ = [
    "BaseAgent",
    "ImplicitVerdict",
    "extract_implicit_verdict",
    "parse_key_findings",
    "CommentAnalysisAgent",
    "FactCheckingAgent",
    "FactQuestioningAgent",
    "JudgeAgent",
    "LinguisticFeatureAgent",
    "QuestioningAgent",
    "DecisionRuleOptimizationAgent",
]
