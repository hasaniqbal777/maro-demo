"""Classify URLs into trusted / neutral / untrusted tiers (improvement #1).

Classification is driven by two external data sources, not a hand-curated list:

  1. Wikipedia → trusted (always — it's a curated reference).
  2. Iffy+ Mis/Disinformation Index (CC BY 4.0, ~2k domains) → untrusted, with
     full MBFC Fact / Bias / Credibility breakdown attached.
  3. Host matches an UNTRUSTED_SUBSTRINGS pattern (social media, free blog
     hosts) → untrusted.
  4. Everything else → neutral.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from ..config import (
    IFFY_DOMAINS,
    MBFC_BIAS_MAP,
    MBFC_CRED_MAP,
    MBFC_FACT_MAP,
    UNTRUSTED_SUBSTRINGS,
    format_mbfc,
)

Tier = Literal["trusted", "neutral", "untrusted"]


@dataclass
class TrustResult:
    tier: Tier
    note: str = ""  # Human-readable one-line summary (shown in prompt + UI).
    # Raw codes + expanded labels for the 3 MBFC dimensions, empty when unavailable.
    mbfc_fact: str = ""
    mbfc_fact_label: str = ""
    mbfc_bias: str = ""
    mbfc_bias_label: str = ""
    mbfc_cred: str = ""
    mbfc_cred_label: str = ""


def _host(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def _match_iffy(host: str) -> dict | None:
    """Exact or parent-domain match against the Iffy+ Index."""
    if host in IFFY_DOMAINS:
        return IFFY_DOMAINS[host]
    parts = host.split(".")
    for i in range(1, len(parts) - 1):
        sub = ".".join(parts[i:])
        if sub in IFFY_DOMAINS:
            return IFFY_DOMAINS[sub]
    return None


def classify(url: str) -> TrustResult:
    host = _host(url)
    if not host:
        return TrustResult("neutral")
    if "wikipedia.org" in host:
        return TrustResult("trusted", "Wikipedia")
    if (iffy := _match_iffy(host)) is not None:
        fact = (iffy.get("mbfc_fact") or "").upper()
        bias = (iffy.get("mbfc_bias") or "").upper()
        cred = (iffy.get("mbfc_cred") or "").upper()
        return TrustResult(
            tier="untrusted",
            note=f"Iffy+ Index · {format_mbfc(iffy)}",
            mbfc_fact=fact,
            mbfc_fact_label=MBFC_FACT_MAP.get(fact, ""),
            mbfc_bias=bias,
            mbfc_bias_label=MBFC_BIAS_MAP.get(bias, ""),
            mbfc_cred=cred,
            mbfc_cred_label=MBFC_CRED_MAP.get(cred, ""),
        )
    for bad in UNTRUSTED_SUBSTRINGS:
        if bad in host:
            return TrustResult("untrusted", "Social media / free-blog host")
    return TrustResult("neutral")


# Backwards-compatible tier-only helper for any existing callers / tests.
def classify_source(url: str) -> Tier:
    return classify(url).tier
