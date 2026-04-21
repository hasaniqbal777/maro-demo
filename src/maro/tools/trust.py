"""Classify URLs into trusted / neutral / untrusted tiers (improvement #1)."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from ..config import TRUSTED_DOMAINS, UNTRUSTED_SUBSTRINGS

Tier = Literal["trusted", "neutral", "untrusted"]


def _host(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def classify_source(url: str) -> Tier:
    host = _host(url)
    if not host:
        return "neutral"
    if "wikipedia.org" in host:
        return "trusted"
    for bad in UNTRUSTED_SUBSTRINGS:
        if bad in host:
            return "untrusted"
    for good in TRUSTED_DOMAINS:
        if host == good or host.endswith("." + good):
            return "trusted"
    return "neutral"
