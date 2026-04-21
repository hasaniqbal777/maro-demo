"""Agent base class + helpers for implicit-verdict parsing (improvement #3)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

ImplicitVerdict = Literal["REAL", "FAKE", "UNKNOWN"]

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


_VERDICT_RE = re.compile(r"implicit\s+verdict\s*:\s*(real|fake)", re.IGNORECASE)


def extract_implicit_verdict(text: str) -> ImplicitVerdict:
    match = _VERDICT_RE.search(text)
    if not match:
        return "UNKNOWN"
    return "REAL" if match.group(1).upper() == "REAL" else "FAKE"


_KEY_FINDINGS_RE = re.compile(
    r"key\s+findings\s*:\s*\n(.*?)(?:\n\s*implicit\s+verdict|\n\s*\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_BULLET_RE = re.compile(r"^\s*[-•*]\s*([^:]+?)\s*:\s*(.+?)\s*$")


def parse_key_findings(text: str) -> dict[str, str]:
    """Pull out the ``KEY FINDINGS:`` block that expert agent prompts emit."""
    block = _KEY_FINDINGS_RE.search(text)
    if not block:
        return {}
    out: dict[str, str] = {}
    for line in block.group(1).splitlines():
        m = _BULLET_RE.match(line)
        if m:
            out[m.group(1).strip()] = m.group(2).strip()
    return out


class BaseAgent:
    """Every concrete agent is a thin wrapper around a system prompt."""

    prompt_name: str

    def __init__(self) -> None:
        self.system_prompt = load_prompt(self.prompt_name)
