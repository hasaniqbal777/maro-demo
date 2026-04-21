"""Google search via the Serper API."""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import SERPER_API_KEY
from ..trace import ToolCall, current_trace
from .trust import Tier, classify_source

SERPER_URL = "https://google.serper.dev/search"


@dataclass
class SearchHit:
    title: str
    snippet: str
    url: str
    trust: Tier


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _post(query: str, num: int) -> dict:
    if not SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY missing; check .env")
    resp = requests.post(
        SERPER_URL,
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        json={"q": query, "num": num},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def search(query: str, num: int = 5) -> list[SearchHit]:
    """Run a Serper search; return hits with trust tiers attached."""
    t0 = time.perf_counter()
    data = _post(query, num)
    hits: list[SearchHit] = []
    for item in data.get("organic", [])[:num]:
        url = item.get("link", "")
        hits.append(
            SearchHit(
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                url=url,
                trust=classify_source(url),
            )
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    trace = current_trace()
    if trace is not None:
        trace.add_tool(ToolCall(tool="serper", query=query, result=hits, latency_ms=latency_ms))
    return hits
