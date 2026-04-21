from maro.dataset.pheme_loader import NewsItem
from maro.dataset.validation_tasks import build_tasks


def _items():
    out = []
    for event in ("a", "b", "c"):
        for i in range(10):
            out.append(
                NewsItem(
                    id=f"{event}-{i}",
                    event=event,
                    text=f"news {event}{i}",
                    comments=[],
                    label=i % 2,
                )
            )
    return out


def test_tasks_have_cross_event_demos():
    tasks = build_tasks(_items(), n_tasks=9, n_demos=3)
    assert len(tasks) == 9
    for t in tasks:
        assert all(d.event != t.query.event for d in t.demos)
        assert len(t.demos) == 3


def test_round_robin_query_events():
    tasks = build_tasks(_items(), n_tasks=6, n_demos=2)
    events = [t.query.event for t in tasks]
    # round-robin: each event appears twice in 6 tasks across 3 events
    assert sorted(events) == sorted(["a", "a", "b", "b", "c", "c"])
