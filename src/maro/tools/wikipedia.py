"""Wikipedia lookup via the `wikipedia` package, with trace logging."""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass

import wikipedia as wp

from ..trace import ToolCall, current_trace

# The `wikipedia` package uses the stdlib default socket timeout (= no timeout).
# Set a global socket timeout so a slow DNS or TCP connection can't hang us.
socket.setdefaulttimeout(15.0)


@dataclass
class WikipediaHit:
    title: str
    summary: str
    url: str
    trust: str = "trusted"  # Wikipedia always trusted per config


def lookup(query: str, sentences: int = 3) -> WikipediaHit | None:
    """Return a short summary of the best-matching Wikipedia page, or None."""
    t0 = time.perf_counter()
    hit: WikipediaHit | None = None
    try:
        results = wp.search(query, results=3)
        for title in results:
            try:
                page = wp.page(title, auto_suggest=False)
                summary = wp.summary(title, sentences=sentences, auto_suggest=False)
                hit = WikipediaHit(title=page.title, summary=summary, url=page.url)
                break
            except (wp.DisambiguationError, wp.PageError):
                continue
    except Exception:
        hit = None

    latency_ms = (time.perf_counter() - t0) * 1000
    trace = current_trace()
    if trace is not None:
        trace.add_tool(ToolCall(tool="wikipedia", query=query, result=hit, latency_ms=latency_ms))
    return hit
