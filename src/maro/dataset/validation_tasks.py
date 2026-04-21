"""Cross-event validation task construction (paper Fig. 2)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .pheme_loader import NewsItem


@dataclass
class ValidationTask:
    query: NewsItem
    demos: list[NewsItem]  # labeled items from OTHER events


def build_tasks(items: list[NewsItem], n_tasks: int, n_demos: int = 3, seed: int = 7) -> list[ValidationTask]:
    """Sample `n_tasks` cross-event validation tasks.

    For each task: query news from one event; demonstrations from other events only.
    Cycle events to keep them balanced.
    """
    rng = random.Random(seed)
    by_event: dict[str, list[NewsItem]] = {}
    for it in items:
        by_event.setdefault(it.event, []).append(it)
    events = [e for e, lst in by_event.items() if lst]
    if len(events) < 2:
        raise ValueError("need items from >=2 events to build cross-event tasks")

    tasks: list[ValidationTask] = []
    for i in range(n_tasks):
        query_event = events[i % len(events)]
        query = rng.choice(by_event[query_event])
        other_pool = [it for e, lst in by_event.items() if e != query_event for it in lst]
        demos = rng.sample(other_pool, k=min(n_demos, len(other_pool)))
        tasks.append(ValidationTask(query=query, demos=demos))
    return tasks
