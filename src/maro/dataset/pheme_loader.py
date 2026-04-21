"""Load the PHEME rumour-detection dataset from disk.

Expected layout after `scripts/download_pheme.py`:
    data/pheme/<event>/<rumours|non-rumours>/<thread_id>/
        source-tweet/<thread_id>.json
        reactions/<reply_id>.json*
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from ..config import PHEME_EVENTS, PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data" / "pheme"


@dataclass
class NewsItem:
    id: str
    event: str
    text: str
    comments: list[str]
    label: int  # 0 real, 1 fake/rumour


def _read_text(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return ""
    return str(data.get("text", "")).strip()


def _load_thread(thread_dir: Path, event: str, label: int) -> NewsItem | None:
    source_dir = thread_dir / "source-tweet"
    if not source_dir.exists():
        source_dir = thread_dir / "source-tweets"
    if not source_dir.exists():
        return None
    source_files = list(source_dir.glob("*.json"))
    if not source_files:
        return None
    text = _read_text(source_files[0])
    if not text:
        return None

    comments: list[str] = []
    reactions = thread_dir / "reactions"
    if reactions.exists():
        for r in sorted(reactions.glob("*.json")):
            t = _read_text(r)
            if t:
                comments.append(t)

    return NewsItem(id=thread_dir.name, event=event, text=text, comments=comments, label=label)


def load_event(event: str, limit: int | None = None) -> list[NewsItem]:
    event_dir = DATA_DIR / event
    if not event_dir.exists():
        raise FileNotFoundError(
            f"{event_dir} not found — run `uv run python scripts/download_pheme.py` first."
        )
    items: list[NewsItem] = []
    for label, subdir in ((1, "rumours"), (0, "non-rumours")):
        sub = event_dir / subdir
        if not sub.exists():
            continue
        for thread in sorted(sub.iterdir()):
            if not thread.is_dir():
                continue
            item = _load_thread(thread, event, label)
            if item is not None:
                items.append(item)
    random.Random(event).shuffle(items)
    return items[:limit] if limit is not None else items


def load_all(limit_per_event: int | None = None) -> list[NewsItem]:
    out: list[NewsItem] = []
    for event in PHEME_EVENTS:
        try:
            out.extend(load_event(event, limit=limit_per_event))
        except FileNotFoundError:
            continue
    return out


def load_source(target_event: str, limit_per_event: int | None = None) -> list[NewsItem]:
    """Load items from all events EXCEPT `target_event` — the paper's 'source domains'
    for cross-event decision-rule optimization (Algorithm 1)."""
    if target_event not in PHEME_EVENTS:
        raise ValueError(
            f"unknown event {target_event!r}; expected one of {PHEME_EVENTS}"
        )
    out: list[NewsItem] = []
    for event in PHEME_EVENTS:
        if event == target_event:
            continue
        try:
            out.extend(load_event(event, limit=limit_per_event))
        except FileNotFoundError:
            continue
    return out


def load_target(target_event: str, limit: int | None = None) -> list[NewsItem]:
    """Load items from `target_event` only — the held-out evaluation set."""
    return load_event(target_event, limit=limit)
