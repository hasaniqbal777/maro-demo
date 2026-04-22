"""Download the Iffy+ Mis/Disinformation Index and write to disk as JSON.

Each entry captures the Iffy+ columns we use downstream:
  - Domain
  - MBFC Fact  (factual-reporting level: HIGH, MIXED, LOW, VERY LOW, …)
  - MBFC Bias  (political lean: LEFT, RIGHT, QUESTIONABLE, CONSPIRACY, …)
  - MBFC Cred  (credibility: HIGH, MEDIUM, LOW)
  - Score      (Iffy's composite score)

Source: https://iffy.news/index/ (Barrett Golding)
License: MIT + CC BY 4.0 — attribution included in the output file.

Run this occasionally to refresh the list. The output file
`data/iffy_untrusted.json` is committed so Spaces and CI don't need network
access to Google Sheets at build time.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import requests

from maro.config import PROJECT_ROOT

IFFY_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1ck1_FZC-97uDLIlvRJDTrGqBk0FuDe9yHkluROgpGS8/gviz/tq?tqx=out:csv"
)
OUT = PROJECT_ROOT / "data" / "iffy_untrusted.json"


def _clean(v: str | None) -> str:
    return (v or "").strip()


def _float_or_none(v: str | None) -> float | None:
    v = _clean(v)
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main() -> None:
    print("Fetching Iffy+ Index CSV …")
    resp = requests.get(IFFY_CSV_URL, timeout=60, allow_redirects=True)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))

    records: dict[str, dict] = {}
    for row in reader:
        d = _clean(row.get("Domain")).lower()
        if not d or "." not in d or " " in d or d.startswith("#"):
            continue
        if d.startswith("www."):
            d = d[4:]
        records[d] = {
            "mbfc_fact": _clean(row.get("MBFC Fact")),
            "mbfc_bias": _clean(row.get("MBFC Bias")),
            "mbfc_cred": _clean(row.get("MBFC cred")) or _clean(row.get("MBFC Cred")),
            "score": _float_or_none(row.get("Score")),
        }

    out = {
        "_meta": {
            "source": "https://iffy.news/index/",
            "license": "MIT + CC BY 4.0",
            "maintainer": "Barrett Golding",
            "entries": len(records),
            "refresh": "uv run python scripts/refresh_iffy_untrusted.py",
        },
        "domains": dict(sorted(records.items())),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(records)} domains → {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
