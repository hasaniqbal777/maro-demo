"""MARO Agent-I/O Inspector — Streamlit demo.

Launch with:  uv run streamlit run demo/app.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

from maro.config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEMO_EXAMPLES,
    INITIAL_DECISION_RULE,
    PROJECT_ROOT,
)
from maro.agents.base import parse_key_findings
from maro.optimization import load_rules
from maro.pipeline import analyze, infer
from maro.trace import AgentCall, trace_context

# ------------------------------------------------------------------ Page setup

st.set_page_config(
    page_title="MARO Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_CSS = """
<style>
.main .block-container { padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1350px; }
.maro-hero {
  display: flex; align-items: center; gap: 24px;
  padding: 24px 28px; border-radius: 18px;
  background: linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 100%);
  color: white; margin: 12px 0 24px 0;
  box-shadow: 0 10px 25px rgba(0,0,0,0.12);
}
.maro-hero-label { font-size: 54px; font-weight: 800; letter-spacing: 1px; line-height: 1; }
.maro-hero-sub   { font-size: 15px; opacity: 0.92; margin-top: 6px; }
.maro-hero-votes { margin-top: 10px; font-size: 13px; opacity: 0.92; }
.maro-vote-pill  {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  background: rgba(255,255,255,0.18); margin: 0 4px 4px 0; font-weight: 600;
}

.maro-phase { margin: 6px 0 10px 0; }
.maro-phase-title {
  font-size: 12px; font-weight: 700; letter-spacing: 0.6px; text-transform: uppercase;
  color: #6B7280; margin-bottom: 6px;
}
.maro-pipeline { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
.maro-stage {
  padding: 7px 12px; border-radius: 9px; font-size: 12.5px; font-weight: 600;
  background: #F3F4F6; color: #6B7280; border: 1px solid #E5E7EB;
  white-space: nowrap;
}
.maro-stage.running { background: #DBEAFE; color: #1D4ED8; border-color: #93C5FD; }
.maro-stage.complete { background: #D1FAE5; color: #065F46; border-color: #6EE7B7; }
.maro-stage.skipped { background: #F3F4F6; color: #9CA3AF; text-decoration: line-through; }
.maro-arrow { color: #D1D5DB; font-size: 16px; padding: 0 2px; }

.maro-card {
  border-radius: 14px; padding: 14px 16px; margin-bottom: 10px;
  border-left: 5px solid var(--accent); background: #FAFAFA;
  box-shadow: 0 2px 6px rgba(0,0,0,0.04);
  min-height: 170px;
}
.maro-card h4 {
  margin: 0 0 8px 0; font-size: 13px; letter-spacing: 0.4px; text-transform: uppercase;
  color: var(--accent); font-weight: 700;
}
.maro-card .summary { font-size: 13.5px; color: #111827; line-height: 1.45; margin-bottom: 6px; }
.maro-card .meta { font-size: 11px; color: #6B7280; margin-top: 6px; }
.maro-card.pending { opacity: 0.55; }
.maro-card.pending h4::after { content: "  ⏳"; }

.maro-kv { margin: 6px 0 2px 0; font-size: 12.5px; }
.maro-kv-row { display: flex; gap: 8px; align-items: baseline; padding: 2px 0; }
.maro-kv-key {
  flex: 0 0 82px; color: #6B7280; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.4px; font-size: 10.5px;
}
.maro-kv-val { flex: 1; color: #111827; line-height: 1.35; }

.maro-verdict-chip {
  display: inline-block; padding: 2px 9px; border-radius: 999px;
  font-size: 10.5px; font-weight: 700; letter-spacing: 0.5px; margin-top: 4px;
}
.maro-verdict-chip.REAL  { background: #D1FAE5; color: #065F46; }
.maro-verdict-chip.FAKE  { background: #FEE2E2; color: #991B1B; }
.maro-verdict-chip.UNKNOWN { background: #E5E7EB; color: #4B5563; }

.maro-evidence-card {
  border: 1px solid #E5E7EB; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;
  background: white;
}
.maro-evidence-card .title a { color: #111827; font-weight: 600; text-decoration: none; }
.maro-evidence-card .title a:hover { text-decoration: underline; }
.maro-evidence-card .snippet { color: #374151; font-size: 13px; margin-top: 4px; line-height: 1.45; }
.maro-evidence-card .host { color: #6B7280; font-size: 12px; margin-top: 4px; }

.maro-badge {
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 11px; font-weight: 700; letter-spacing: 0.4px; margin-right: 6px;
}
.maro-badge.trusted { background: #D1FAE5; color: #065F46; }
.maro-badge.neutral { background: #E5E7EB; color: #374151; }
.maro-badge.untrusted { background: #FEE2E2; color: #991B1B; }

/* MBFC label pills — one per dimension (Fact / Bias / Cred). Use severity colors. */
.maro-mbfc {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 2px 7px; border-radius: 999px;
  font-size: 10.5px; font-weight: 600; letter-spacing: 0.2px;
  margin-right: 5px; margin-top: 4px; color: #111827; background: #F3F4F6;
  border: 1px solid #E5E7EB;
}
.maro-mbfc-key { font-weight: 700; color: #6B7280; font-size: 9.5px; letter-spacing: 0.5px; text-transform: uppercase; }
/* Factual-reporting severity — green for high, red for very-low */
.maro-mbfc.fact-VH, .maro-mbfc.fact-H  { background: #D1FAE5; border-color: #6EE7B7; color: #065F46; }
.maro-mbfc.fact-MF                     { background: #ECFDF5; border-color: #A7F3D0; color: #047857; }
.maro-mbfc.fact-M                      { background: #FEF3C7; border-color: #FDE68A; color: #92400E; }
.maro-mbfc.fact-L                      { background: #FED7AA; border-color: #FDBA74; color: #9A3412; }
.maro-mbfc.fact-VL                     { background: #FEE2E2; border-color: #FCA5A5; color: #991B1B; }
/* Bias — red for fringe/fakenews/conspiracy, gray-blue for normal spectrum */
.maro-mbfc.bias-L,  .maro-mbfc.bias-LC { background: #DBEAFE; border-color: #93C5FD; color: #1E3A8A; }
.maro-mbfc.bias-C,  .maro-mbfc.bias-PS { background: #E5E7EB; border-color: #D1D5DB; color: #1F2937; }
.maro-mbfc.bias-RC, .maro-mbfc.bias-R  { background: #FEE2E2; border-color: #FCA5A5; color: #7F1D1D; }
.maro-mbfc.bias-QS                     { background: #FEF3C7; border-color: #FDE68A; color: #78350F; }
.maro-mbfc.bias-CP, .maro-mbfc.bias-FN { background: #7F1D1D; border-color: #7F1D1D; color: #FEE2E2; }
.maro-mbfc.bias-SA                     { background: #EDE9FE; border-color: #C4B5FD; color: #4C1D95; }
/* Credibility severity */
.maro-mbfc.cred-H                      { background: #D1FAE5; border-color: #6EE7B7; color: #065F46; }
.maro-mbfc.cred-M                      { background: #FEF3C7; border-color: #FDE68A; color: #78350F; }
.maro-mbfc.cred-L                      { background: #FEE2E2; border-color: #FCA5A5; color: #991B1B; }

.maro-verdict-real { --bg1: #059669; --bg2: #10B981; }
.maro-verdict-fake { --bg1: #DC2626; --bg2: #F97316; }
.maro-verdict-unknown { --bg1: #6B7280; --bg2: #9CA3AF; }

.maro-truth {
  display: inline-block; margin-top: 10px; padding: 6px 12px;
  border-radius: 999px; font-size: 12px; font-weight: 700;
  background: rgba(255,255,255,0.22); letter-spacing: 0.3px;
}
.maro-truth.wrong { background: rgba(0,0,0,0.3); }

.maro-ext-panel {
  border: 1px dashed #D1D5DB; border-radius: 12px; padding: 14px 18px;
  margin: 6px 0 18px 0; background: #FFFCF5;
}
.maro-ext-title {
  font-size: 11px; font-weight: 700; letter-spacing: 0.6px;
  text-transform: uppercase; color: #9A7B1F; margin-bottom: 6px;
}
.maro-ext-sub { font-size: 12px; color: #6B7280; margin-bottom: 10px; }

.maro-trust-bar {
  display: flex; height: 22px; border-radius: 6px; overflow: hidden;
  background: #F3F4F6; border: 1px solid #E5E7EB;
}
.maro-trust-bar > div {
  display: flex; align-items: center; justify-content: center;
  color: white; font-size: 11px; font-weight: 700; letter-spacing: 0.3px;
}
.maro-trust-trusted   { background: #10B981; }
.maro-trust-neutral   { background: #9CA3AF; }
.maro-trust-untrusted { background: #EF4444; }
.maro-trust-legend { font-size: 11px; color: #6B7280; margin-top: 6px; }

.maro-vote-row { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.maro-vote-dot {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
  background: #F3F4F6; color: #374151;
}
.maro-vote-dot .swatch {
  width: 10px; height: 10px; border-radius: 999px; background: #9CA3AF;
}
.maro-vote-dot.REAL    .swatch { background: #10B981; }
.maro-vote-dot.FAKE    .swatch { background: #EF4444; }
.maro-vote-dot.UNKNOWN .swatch { background: #9CA3AF; }
.maro-vote-dot.judge { border: 1px solid #111827; background: #FAFAFA; font-weight: 700; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

st.title("🔍 MARO — Multi-Agent Misinformation Detection")
st.markdown(
    "**Three expert AI agents analyze the news from different angles "
    "(language, comments, facts). A Questioning Agent makes them reflect "
    "and revise. A Judge votes on whether the story is misinformation.**"
)
st.caption(
    "Faithful replication of the EMNLP-2025 paper "
    "**[“A Multi-Agent Framework with Automated Decision Rule Optimization for "
    "Cross-Domain Misinformation Detection”](https://aclanthology.org/2025.emnlp-main.291/)** "
    "by Hui Li, Ante Wang, Kunquan Li, Zhihao Wang, Liang Zhang, Delai Qiu, "
    "Qingsong Liu, and Jinsong Su  —  with two course-project extensions "
    "(evidence-source trust weighting + confidence / agent-disagreement analysis). "
    "All credit for the MARO framework belongs to the original authors."
)

# Paper-faithful cross-event replication: per-target rules files live at
# data/rules_<event>.json. Presence of any rules file flips the demo to
# MARO mode; for any target without its rules file we fall back to r_0.
from maro.config import PHEME_EVENTS as _PHEME_EVENTS  # local alias

_RULES_BY_EVENT = {
    ev: PROJECT_ROOT / "data" / f"rules_{ev}.json" for ev in _PHEME_EVENTS
}
_available_events = [ev for ev, p in _RULES_BY_EVENT.items() if p.exists()]
if not _available_events:
    st.warning(
        "**Seed-rule mode.** No `data/rules_<event>.json` files found, so the "
        "Judge is using only the manually-defined seed rule **r₀**. This is "
        "the *starting point* of Algorithm 1, not the final MARO behaviour. "
        "To faithfully replicate the paper (Appendix G.4), run Decision Rule "
        "Optimization once per held-out event:\n\n"
        "```\n"
        "uv run python scripts/download_pheme.py                              # one-time\n"
        "uv run python scripts/run_optimization.py --target-event all         # 5× optimization\n"
        "uv run python scripts/run_evaluation.py --per-event 20 --ablation    # eval\n"
        "```",
        icon="⚠️",
    )
else:
    missing = [ev for ev in _PHEME_EVENTS if ev not in _available_events]
    if missing:
        st.info(
            f"**Partial MARO mode.** Rules available for "
            f"{len(_available_events)}/{len(_PHEME_EVENTS)} events: "
            f"`{', '.join(_available_events)}`. Events still in seed-rule mode: "
            f"`{', '.join(missing)}`.",
            icon="⚠️",
        )
    else:
        st.success(
            f"**MARO mode.** Per-event optimized rule sets loaded for all "
            f"{len(_PHEME_EVENTS)} PHEME events.",
            icon="✅",
        )

# ------------------------------------------------------------------ Agent metadata

AGENT_META = {
    "linguistic":    {"color": "#3B82F6", "icon": "🗣️",  "label": "Linguistic"},
    "comment":       {"color": "#F59E0B", "icon": "💬",  "label": "Comments"},
    "fact_checking": {"color": "#10B981", "icon": "🔬",  "label": "Fact-Check"},
    "judge":         {"color": "#EF4444", "icon": "⚖️",  "label": "Judge"},
    # kept for raw-trace rendering only
    "questioning":   {"color": "#8B5CF6", "icon": "❓",  "label": "Questioning"},
}

# Pipeline, grouped into 3 phases. 9 stages (reflect/revise pairs collapsed).
PIPELINE_PHASES: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Analyze",
        [
            ("linguistic",       "Read writing style"),
            ("comment",          "Read comments"),
            ("fact_questioning", "Draft fact-check questions"),
            ("fact_search",      "Search web + Wikipedia"),
            ("fact_checking",    "Check facts against evidence"),
        ],
    ),
    (
        "Reflect & revise",
        [
            ("reflect_linguistic", "Reflect & revise language"),
            ("reflect_comment",    "Reflect & revise comments"),
            ("reflect_fact",       "Reflect & revise facts"),
        ],
    ),
    (
        "Judge",
        [
            ("judge", "Vote with top-K rules"),
        ],
    ),
]

# Synthetic combined stages map to the two raw events that feed them.
COMBINED_STAGES = {
    "reflect_linguistic": ("questioning_linguistic", "refine_linguistic"),
    "reflect_comment":    ("questioning_comment",    "refine_comment"),
    "reflect_fact":       ("questioning_fact",       "refine_fact"),
}


def _status_of(state: dict[str, str], key: str) -> str:
    if key not in COMBINED_STAGES:
        return state.get(key, "pending")
    subs = COMBINED_STAGES[key]
    statuses = [state.get(s, "pending") for s in subs]
    if all(s in ("complete", "skipped") for s in statuses):
        return "complete" if any(s == "complete" for s in statuses) else "skipped"
    if any(s == "running" for s in statuses) or any(s == "complete" for s in statuses):
        return "running"
    return "pending"

# ------------------------------------------------------------------ Helpers

def _verdict_color_class(label: int) -> str:
    return "maro-verdict-fake" if label == 1 else "maro-verdict-real"


def _render_pipeline(placeholder, state: dict[str, str]) -> None:
    phase_blocks: list[str] = []
    for phase_title, stages in PIPELINE_PHASES:
        items: list[str] = []
        for i, (key, label) in enumerate(stages):
            status = _status_of(state, key)
            items.append(f'<div class="maro-stage {status}">{label}</div>')
            if i != len(stages) - 1:
                items.append('<span class="maro-arrow">→</span>')
        phase_blocks.append(
            f'<div class="maro-phase">'
            f'  <div class="maro-phase-title">{html.escape(phase_title)}</div>'
            f'  <div class="maro-pipeline">{"".join(items)}</div>'
            f'</div>'
        )
    placeholder.markdown("".join(phase_blocks), unsafe_allow_html=True)


def _agent_card_html(
    *,
    agent_key: str,
    summary: str = "",
    meta: str = "",
    pending: bool = False,
    findings: dict[str, str] | None = None,
    verdict: str | None = None,
) -> str:
    info = AGENT_META[agent_key]
    cls = "maro-card" + (" pending" if pending else "")
    parts: list[str] = [f'<h4>{info["icon"]} {info["label"]}</h4>']
    if summary:
        parts.append(f'<div class="summary">{html.escape(summary)}</div>')
    if findings:
        rows = "".join(
            f'<div class="maro-kv-row">'
            f'  <div class="maro-kv-key">{html.escape(k)}</div>'
            f'  <div class="maro-kv-val">{html.escape(v)}</div>'
            f'</div>'
            for k, v in findings.items()
        )
        parts.append(f'<div class="maro-kv">{rows}</div>')
    if verdict:
        parts.append(
            f'<span class="maro-verdict-chip {verdict}">leans {verdict.lower()}</span>'
        )
    if meta:
        parts.append(f'<div class="meta">{html.escape(meta)}</div>')
    return (
        f'<div class="{cls}" style="--accent: {info["color"]}">'
        + "".join(parts)
        + "</div>"
    )


def _verdict_pill(verdict: str) -> str:
    color = {"REAL": "#10B981", "FAKE": "#EF4444", "UNKNOWN": "#9CA3AF"}.get(verdict, "#9CA3AF")
    return f'<span class="maro-vote-pill" style="background: {color}">{verdict}</span>'


def _hero(
    placeholder,
    *,
    label: int,
    confidence: float,
    verdicts: dict[str, str],
    n_rules: int,
    truth_is_rumour: bool | None = None,
) -> None:
    label_str = "FAKE" if label == 1 else "REAL"
    subtitle = "Likely misinformation" if label == 1 else "Likely credible"
    majority_fake = label == 1
    n_agents_agree = sum(
        1 for v in verdicts.values() if (v == "FAKE") == majority_fake
    )

    # Honest confidence: with only 1 rule, "100%" is meaningless.
    if n_rules <= 1:
        ring_pct = n_agents_agree / max(1, len(verdicts))
        ring_label = f"{n_agents_agree}/{len(verdicts)}"
        conf_line = (
            f"Based on 1 seed rule &middot; "
            f"{n_agents_agree} of {len(verdicts)} expert agents agree "
            f"(run optimization for calibrated top-K voting)"
        )
    else:
        ring_pct = confidence
        ring_label = f"{int(confidence * 100)}%"
        conf_line = (
            f"Majority vote of {n_rules} decision rules &middot; "
            f"{n_agents_agree} of {len(verdicts)} expert agents agree"
        )

    truth_banner = ""
    if truth_is_rumour is not None:
        correct = (majority_fake == truth_is_rumour)
        truth_str = "RUMOUR" if truth_is_rumour else "CONFIRMED"
        if correct:
            truth_banner = (
                f'<div class="maro-truth correct">✓ Matches ground truth '
                f'({truth_str})</div>'
            )
        else:
            truth_banner = (
                f'<div class="maro-truth wrong">✗ Model disagrees with ground '
                f'truth — this example is {truth_str}</div>'
            )

    cls = _verdict_color_class(label)
    html_block = f"""
    <div class="maro-hero {cls}">
      <div>{_confidence_ring_custom(ring_pct, ring_label)}</div>
      <div style="flex: 1;">
        <div class="maro-hero-label">{label_str}</div>
        <div class="maro-hero-sub">{subtitle}</div>
        <div class="maro-hero-votes" style="font-size: 12.5px;">{conf_line}</div>
        {truth_banner}
      </div>
    </div>
    """
    placeholder.markdown(html_block, unsafe_allow_html=True)


def _confidence_ring_custom(pct: float, label: str) -> str:
    r, stroke = 48, 10
    circumference = 2 * 3.14159 * r
    dash = max(0.001, min(1.0, pct)) * circumference
    return f"""
    <svg width="120" height="120" viewBox="0 0 120 120">
      <circle cx="60" cy="60" r="{r}" fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="{stroke}"/>
      <circle cx="60" cy="60" r="{r}" fill="none" stroke="white" stroke-width="{stroke}"
              stroke-dasharray="{dash} {circumference}" stroke-dashoffset="0"
              stroke-linecap="round" transform="rotate(-90 60 60)"/>
      <text x="60" y="66" text-anchor="middle" font-size="22" font-weight="700" fill="white">
        {label}
      </text>
    </svg>
    """


def _render_trust_bar(serper_hits, wiki_hits) -> str:
    """Stacked horizontal bar + per-untrusted MBFC breakdown."""
    trusted = sum(1 for h in serper_hits if h.trust == "trusted") + len(wiki_hits)
    neutral = sum(1 for h in serper_hits if h.trust == "neutral")
    untrusted_hits = [h for h in serper_hits if h.trust == "untrusted"]
    untrusted = len(untrusted_hits)
    total = trusted + neutral + untrusted
    if total == 0:
        return '<div class="maro-ext-sub">No external evidence retrieved yet.</div>'

    pct_t = 100 * trusted / total
    pct_n = 100 * neutral / total
    pct_u = 100 * untrusted / total
    segs: list[str] = []
    if trusted:
        segs.append(f'<div class="maro-trust-trusted" style="width:{pct_t:.1f}%">{trusted}</div>')
    if neutral:
        segs.append(f'<div class="maro-trust-neutral" style="width:{pct_n:.1f}%">{neutral}</div>')
    if untrusted:
        segs.append(f'<div class="maro-trust-untrusted" style="width:{pct_u:.1f}%">{untrusted}</div>')

    # Per-untrusted MBFC breakdown — shows what made each untrusted hit untrusted.
    breakdown_html = ""
    if untrusted_hits:
        rows: list[str] = []
        from urllib.parse import urlparse
        for h in untrusted_hits:
            host = urlparse(h.url).netloc.lower().lstrip("www.")
            pills: list[str] = []
            fact_code = getattr(h, "mbfc_fact", "")
            fact_label = getattr(h, "mbfc_fact_label", "")
            bias_code = getattr(h, "mbfc_bias", "")
            bias_label = getattr(h, "mbfc_bias_label", "")
            cred_code = getattr(h, "mbfc_cred", "")
            cred_label = getattr(h, "mbfc_cred_label", "")
            if fact_code:
                pills.append(
                    f'<span class="maro-mbfc fact-{fact_code}"><span class="maro-mbfc-key">Fact</span> {html.escape(fact_label)}</span>'
                )
            if bias_code:
                pills.append(
                    f'<span class="maro-mbfc bias-{bias_code}"><span class="maro-mbfc-key">Bias</span> {html.escape(bias_label)}</span>'
                )
            if cred_code:
                pills.append(
                    f'<span class="maro-mbfc cred-{cred_code}"><span class="maro-mbfc-key">Cred</span> {html.escape(cred_label)}</span>'
                )
            if not pills:
                # untrusted by pattern (social media), no Iffy data — still list the host
                pills.append(
                    '<span class="maro-mbfc" style="color:#6B7280">Social / free-blog host</span>'
                )
            rows.append(
                f'<div style="margin-top:6px;font-size:11.5px">'
                f'  <span style="color:#991B1B;font-weight:700">⚠️ {html.escape(host)}</span>&nbsp; {"".join(pills)}'
                f'</div>'
            )
        breakdown_html = "".join(rows)

    return (
        f'<div class="maro-trust-bar">{"".join(segs)}</div>'
        f'<div class="maro-trust-legend">'
        f'🟢 {trusted} trusted &nbsp;·&nbsp; ⚪ {neutral} neutral &nbsp;·&nbsp; 🔴 {untrusted} untrusted'
        f'</div>'
        f'{breakdown_html}'
    )


def _render_vote_row(verdicts: Mapping[str, str], judge_verdict: str) -> str:
    """Colored pills for each expert agent's implicit verdict + the Judge's final."""
    pills: list[str] = []
    for agent_key, v in verdicts.items():
        info = AGENT_META.get(agent_key, {"icon": "•", "label": agent_key.title()})
        pills.append(
            f'<span class="maro-vote-dot {v}">'
            f'  <span class="swatch"></span>{info["icon"]} {info["label"]}'
            f'</span>'
        )
    pills.append(
        f'<span class="maro-vote-dot judge {judge_verdict}">'
        f'  <span class="swatch"></span>⚖️ Judge (majority)'
        f'</span>'
    )
    return f'<div class="maro-vote-row">{"".join(pills)}</div>'


def _mbfc_pills(hit) -> str:
    """Three small colored pills showing MBFC Fact / Bias / Credibility for a hit."""
    fact_code = getattr(hit, "mbfc_fact", "")
    fact_label = getattr(hit, "mbfc_fact_label", "")
    bias_code = getattr(hit, "mbfc_bias", "")
    bias_label = getattr(hit, "mbfc_bias_label", "")
    cred_code = getattr(hit, "mbfc_cred", "")
    cred_label = getattr(hit, "mbfc_cred_label", "")
    if not (fact_code or bias_code or cred_code):
        return ""
    parts: list[str] = []
    if fact_code:
        parts.append(
            f'<span class="maro-mbfc fact-{fact_code}">'
            f'<span class="maro-mbfc-key">Fact</span> {html.escape(fact_label or fact_code)}</span>'
        )
    if bias_code:
        parts.append(
            f'<span class="maro-mbfc bias-{bias_code}">'
            f'<span class="maro-mbfc-key">Bias</span> {html.escape(bias_label or bias_code)}</span>'
        )
    if cred_code:
        parts.append(
            f'<span class="maro-mbfc cred-{cred_code}">'
            f'<span class="maro-mbfc-key">Cred</span> {html.escape(cred_label or cred_code)}</span>'
        )
    return f'<div style="margin-top:6px">{"".join(parts)}</div>'


def _render_evidence(container, report) -> None:
    if not report.serper_hits and not report.wiki_hits:
        container.info("No external evidence retrieved.")
        return
    cards: list[str] = []
    for w in report.wiki_hits:
        cards.append(
            f'<div class="maro-evidence-card">'
            f'<div><span class="maro-badge trusted">TRUSTED</span>'
            f'<span class="title"><a href="{html.escape(w.url)}" target="_blank">{html.escape(w.title)}</a></span></div>'
            f'<div class="snippet">{html.escape(w.summary)}</div>'
            f'<div class="host">wikipedia.org</div>'
            f"</div>"
        )
    for h in report.serper_hits:
        from urllib.parse import urlparse
        host = urlparse(h.url).netloc
        mbfc_pills = _mbfc_pills(h)
        cards.append(
            f'<div class="maro-evidence-card">'
            f'<div><span class="maro-badge {h.trust}">{h.trust.upper()}</span>'
            f'<span class="title"><a href="{html.escape(h.url)}" target="_blank">{html.escape(h.title)}</a></span></div>'
            f'<div class="snippet">{html.escape(h.snippet)}</div>'
            f'<div class="host">{html.escape(host)}</div>'
            f'{mbfc_pills}'
            f"</div>"
        )
    container.markdown("".join(cards), unsafe_allow_html=True)


def _render_raw_calls(expander, calls: list[AgentCall]) -> None:
    if not calls:
        expander.caption("No LLM calls recorded yet.")
        return
    for call in calls:
        with expander.expander(
            f"[{call.step}] {call.model} · {call.latency_ms:.0f} ms · "
            f"{call.tokens_in}→{call.tokens_out} tok",
            expanded=False,
        ):
            st.markdown("**System prompt**")
            st.code(call.system_prompt)
            st.markdown("**User prompt**")
            st.code(call.user_prompt)
            st.markdown("**Response**")
            st.markdown(call.response)
            if call.tool_calls:
                st.markdown("**Tool calls**")
                for tc in call.tool_calls:
                    st.write(f"- `{tc.tool}` · `{tc.query}` · {tc.latency_ms:.0f} ms")


# ------------------------------------------------------------------ Sidebar

with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in AVAILABLE_MODELS else 1,
    )
    use_reflection = st.toggle("Question-reflection", value=True)
    use_trust = st.toggle("Evidence trust weighting (improvement #1)", value=True)
    use_optimized_rules = st.toggle("Use optimized rules (if available)", value=True)

# ------------------------------------------------------------------ Input row

st.markdown("### Pick an example or paste your own")
example_labels = ["(custom)"] + [e[0] for e in DEMO_EXAMPLES]
chosen = st.selectbox("Example", example_labels, index=1, label_visibility="collapsed")

if chosen != "(custom)":
    sel = next(e for e in DEMO_EXAMPLES if e[0] == chosen)
    default_news = sel[2]
    default_comments = "\n".join(sel[3])
    truth_is_rumour: bool | None = sel[4]
    selected_event: str | None = sel[1]
    truth_hint = (
        "unverified / misinformation — model should predict FAKE"
        if sel[4] else
        "well-sourced factual news — model should predict REAL"
    )
    st.caption(
        f"Event: `{sel[1]}` · Ground truth: "
        f"**{'RUMOUR' if sel[4] else 'CONFIRMED'}** ({truth_hint})"
    )
else:
    default_news = ""
    default_comments = ""
    truth_is_rumour = None
    selected_event = None

# Load the decision rules that correspond to the selected event's held-out
# optimization run (rules optimized on the OTHER 4 events — no leak).
rules_src_label = "seed rule r_0 (no optimization run yet)"
rules = [INITIAL_DECISION_RULE]
if use_optimized_rules and selected_event is not None:
    rules_path = _RULES_BY_EVENT.get(selected_event)
    if rules_path is not None and rules_path.exists():
        rules = load_rules(rules_path)
        rules_src_label = f"{rules_path.name}  (optimized WITHOUT seeing {selected_event})"
st.caption(f"📏 Decision rules: **{len(rules)}** · source: {rules_src_label}")

with st.expander(f"📜 View the {len(rules)} decision rule(s) being applied", expanded=False):
    st.caption(
        "These are the exact rules the **Judge Agent** will apply to the news. "
        "The final verdict is the majority vote across them."
    )
    for i, rule in enumerate(rules, start=1):
        st.markdown(f"**Rule {i}**")
        st.code(rule.strip())

# When the dropdown selection changes, we have to explicitly overwrite the
# text_area's session-state value — Streamlit ignores `value=` once a keyed
# widget has been rendered once, and would otherwise keep the previous text.
if st.session_state.get("_last_chosen") != chosen:
    st.session_state["_last_chosen"] = chosen
    st.session_state["news"] = default_news
    st.session_state["comments"] = default_comments

col_news, col_com = st.columns([3, 2])
with col_news:
    news = st.text_area("News text", height=140, key="news")
with col_com:
    comments_raw = st.text_area(
        "Comments (one per line, optional)",
        height=140,
        key="comments",
    )
comments = [c.strip() for c in comments_raw.splitlines() if c.strip()]

go = st.button("▶ Run MARO", type="primary", disabled=not news.strip())

# ------------------------------------------------------------------ Run

if go:
    # ------------------------------------------------------------- Layout
    # Hero verdict slot goes FIRST so the answer lands at the top once ready.
    hero_slot = st.empty()
    hero_slot.info("⏳ Running — the verdict will appear here as soon as the Judge votes.")

    st.markdown("### Why each agent decided")
    cols = st.columns(4, gap="small")
    slots: dict[str, dict[str, Any]] = {}
    for i, key in enumerate(("linguistic", "comment", "fact_checking", "judge")):
        with cols[i]:
            slots[key] = {
                "card":     st.empty(),
                "reflect":  st.empty(),   # per-expert reflection follow-ups
                "evidence": st.empty(),   # fact_checking only
                "report":   st.empty(),   # full analysis expander
            }
            slots[key]["card"].markdown(
                _agent_card_html(agent_key=key, summary="waiting…", pending=True),
                unsafe_allow_html=True,
            )

    st.markdown("### Pipeline")
    pipeline_slot = st.empty()
    state: dict[str, str] = {}
    _render_pipeline(pipeline_slot, state)

    # Per-expert reflection questions (populated via progress callback).
    reflect_questions: dict[str, list[str]] = {
        "linguistic": [], "comment": [], "fact_checking": []
    }
    stage_to_expert = {
        "questioning_linguistic": "linguistic",
        "questioning_comment":    "comment",
        "questioning_fact":       "fact_checking",
    }

    def _render_reflect(expert_key: str) -> None:
        qs = reflect_questions[expert_key]
        slot = slots[expert_key]["reflect"]
        with slot.container():
            if not qs:
                st.caption("🔁 no reflection follow-ups — analysis was thorough")
            else:
                with st.expander(f"🔁 {len(qs)} reflection follow-up(s)", expanded=False):
                    for q in qs:
                        st.markdown(f"- {q}")

    def _render_full_report(expert_key: str, data) -> None:
        with slots[expert_key]["report"].container():
            with st.expander("📄 Full analysis text", expanded=False):
                st.markdown(data.text)

    def progress(stage: str, status: str, data: Any = None) -> None:
        state[stage] = status
        _render_pipeline(pipeline_slot, state)

        # Non-'complete' transitions worth surfacing.
        if stage == "comment" and status == "skipped":
            slots["comment"]["card"].markdown(
                _agent_card_html(
                    agent_key="comment",
                    summary="no comments provided — agent skipped",
                    pending=True,
                ),
                unsafe_allow_html=True,
            )
            return
        if stage == "fact_search" and status == "running":
            slots["fact_checking"]["card"].markdown(
                _agent_card_html(
                    agent_key="fact_checking",
                    summary="searching Google (Serper) + Wikipedia…",
                    pending=True,
                ),
                unsafe_allow_html=True,
            )
            return

        if status != "complete":
            return

        if stage in ("linguistic", "refine_linguistic"):
            slots["linguistic"]["card"].markdown(
                _agent_card_html(
                    agent_key="linguistic",
                    findings=parse_key_findings(data.text),
                    verdict=data.implicit_verdict,
                ),
                unsafe_allow_html=True,
            )
            _render_full_report("linguistic", data)
        elif stage in ("comment", "refine_comment"):
            slots["comment"]["card"].markdown(
                _agent_card_html(
                    agent_key="comment",
                    findings=parse_key_findings(data.text),
                    verdict=data.implicit_verdict,
                ),
                unsafe_allow_html=True,
            )
            _render_full_report("comment", data)
        elif stage in ("fact_checking", "refine_fact"):
            slots["fact_checking"]["card"].markdown(
                _agent_card_html(
                    agent_key="fact_checking",
                    findings=parse_key_findings(data.text),
                    verdict=data.implicit_verdict,
                ),
                unsafe_allow_html=True,
            )
            _render_full_report("fact_checking", data)
        elif stage == "fact_search":
            hits = data or {}
            n_serper = len(hits.get("serper", []))
            n_wiki = len(hits.get("wiki", []))
            slots["fact_checking"]["card"].markdown(
                _agent_card_html(
                    agent_key="fact_checking",
                    summary=f"retrieved {n_serper} web hits + {n_wiki} wiki pages — now analysing…",
                    pending=True,
                ),
                unsafe_allow_html=True,
            )
        elif stage in stage_to_expert:
            expert = stage_to_expert[stage]
            reflect_questions[expert] = data or []
            _render_reflect(expert)

    with trace_context() as trace:
        try:
            report = analyze(
                news=news,
                comments=comments,
                model=model,
                use_reflection=use_reflection,
                use_trust_weighting=use_trust,
                progress=progress,
            )
            state["judge"] = "running"
            _render_pipeline(pipeline_slot, state)
            slots["judge"]["card"].markdown(
                _agent_card_html(
                    agent_key="judge",
                    summary="voting with top-K decision rules…",
                    pending=True,
                ),
                unsafe_allow_html=True,
            )
            result = infer(report=report, rules=rules, demonstrations=[], model=model)
            state["judge"] = "complete"
            _render_pipeline(pipeline_slot, state)
            rule_findings = {
                f"Rule {i}": "FAKE" if jr.label == 1 else "REAL"
                for i, (_, jr) in enumerate(result.per_rule)
            }
            majority = "FAKE" if result.label == 1 else "REAL"
            slots["judge"]["card"].markdown(
                _agent_card_html(
                    agent_key="judge",
                    summary=f"Majority = {majority}  ·  confidence {result.confidence:.0%}",
                    findings=rule_findings,
                    verdict=majority,
                ),
                unsafe_allow_html=True,
            )
            with slots["judge"]["report"].container():
                with st.expander("⚖️ Per-rule reasoning", expanded=False):
                    for i, (rule, jr) in enumerate(result.per_rule):
                        emoji = "🔴 FAKE" if jr.label == 1 else "🟢 REAL"
                        st.markdown(f"**Rule {i} → {emoji}** · {jr.reasoning}")
                        st.code(rule)
        except Exception as exc:
            st.error(f"Run failed: {exc}")
            raise

    # Evidence moves INSIDE the fact-check column as a collapsed expander.
    with slots["fact_checking"]["evidence"].container():
        n_ev = len(report.serper_hits) + len(report.wiki_hits)
        with st.expander(f"🌐 Evidence used ({n_ev} source(s), trust-tagged)", expanded=False):
            _render_evidence(st, report)

    # ------------------------------------------------------------- Hero (final)
    _hero(
        hero_slot,
        label=result.label,
        confidence=result.confidence,
        verdicts=report.implicit_verdicts(),
        n_rules=len(result.per_rule),
        truth_is_rumour=truth_is_rumour,
    )

    # ------------------------------------------------------------- Extensions panel
    st.markdown("#### 🔬 Course-project extensions — at a glance")
    ext_left, ext_right = st.columns(2)
    judge_verdict = "FAKE" if result.label == 1 else "REAL"
    with ext_left:
        st.markdown(
            f'<div class="maro-ext-panel">'
            f'  <div class="maro-ext-title">Extension #1 · Evidence trust weighting</div>'
            f'  <div class="maro-ext-sub">Sources retrieved for the Fact-Check Agent, '
            f'  tagged by credibility tier before the LLM sees them.</div>'
            f'  {_render_trust_bar(report.serper_hits, report.wiki_hits)}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with ext_right:
        st.markdown(
            f'<div class="maro-ext-panel">'
            f'  <div class="maro-ext-title">Extension #2 · Agent agreement</div>'
            f'  <div class="maro-ext-sub">Each expert agent\'s implicit verdict; '
            f'  the Judge\'s majority vote sits on the right.</div>'
            f'  {_render_vote_row(report.implicit_verdicts(), judge_verdict)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------- Diagnostics
    st.markdown("### Diagnostics")
    with st.expander("Agent disagreement  (improvement #3)", expanded=False):
        verdicts = report.implicit_verdicts()
        majority_fake = result.label == 1
        rows = [
            {
                "Agent": agent,
                "Implicit verdict": v,
                "Matches majority": "✓" if (v == "FAKE") == majority_fake else "✗ (dissent)",
            }
            for agent, v in verdicts.items()
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)

    with st.expander("Raw LLM calls (all agents, all steps)", expanded=False):
        raw_tabs = st.tabs(
            ["Linguistic", "Comment", "FactQ", "FactCheck", "Questioning", "Judge"]
        )
        for tab, agent in zip(
            raw_tabs,
            ["linguistic", "comment", "fact_questioning", "fact_checking", "questioning", "judge"],
        ):
            with tab:
                _render_raw_calls(tab, trace.for_agent(agent))

    tin, tout = trace.total_tokens
    total_latency = sum(c.latency_ms for c in trace.calls)
    st.caption(
        f"{len(trace.calls)} LLM calls · "
        f"{tin:,} input / {tout:,} output tokens · "
        f"{total_latency / 1000:.1f}s total latency"
    )


# ------------------------------------------------------------------ Citation footer

st.divider()
st.caption(
    "This is a **faithful replication** of MARO (Li et al., EMNLP 2025). All "
    "credit for the framework — the multi-agent decomposition, the "
    "question-reflection mechanism, and Algorithm 1 (decision-rule "
    "optimization) — belongs to the original authors. "
    "Paper: [aclanthology.org/2025.emnlp-main.291](https://aclanthology.org/2025.emnlp-main.291/)"
)

with st.expander("📚 Cite the original paper (BibTeX)", expanded=False):
    st.code(
        """@inproceedings{li-etal-2025-multi-agent,
    title = "A Multi-Agent Framework with Automated Decision Rule Optimization for Cross-Domain Misinformation Detection",
    author = "Li, Hui  and
      Wang, Ante  and
      Li, Kunquan  and
      Wang, Zhihao  and
      Zhang, Liang  and
      Qiu, Delai  and
      Liu, Qingsong  and
      Su, Jinsong",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.291/",
    doi = "10.18653/v1/2025.emnlp-main.291",
    pages = "5709--5725",
    ISBN = "979-8-89176-332-6",
}""",
        language="bibtex",
    )
