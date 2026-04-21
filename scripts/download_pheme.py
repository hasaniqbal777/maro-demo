"""Download the PHEME rumour dataset from Figshare and extract into data/pheme/.

Source: figshare article 6392078
  "PHEME dataset for Rumour Detection and Veracity Classification"
  by Kochkina, Liakata, Zubiaga.
File: PHEME_veracity.tar.bz2 (~46 MB compressed, expands to ~350 MB).
"""

from __future__ import annotations

import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import requests

from maro.config import PHEME_EVENTS, PROJECT_ROOT

URL = "https://ndownloader.figshare.com/files/11767817"
ARCHIVE = PROJECT_ROOT / "data" / "PHEME_veracity.tar.bz2"
DATA_DIR = PROJECT_ROOT / "data" / "pheme"

EVENT_ALIASES = {
    # Figshare archive uses the 5-event names with a suffix; map to our canonical ones.
    "charliehebdo-all-rnr-threads": "charliehebdo",
    "ferguson-all-rnr-threads":     "ferguson",
    "germanwings-crash-all-rnr-threads": "germanwings-crash",
    "ottawashooting-all-rnr-threads": "ottawashooting",
    "sydneysiege-all-rnr-threads":  "sydneysiege",
}


def download() -> None:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        print(f"[skip download] {ARCHIVE} already exists.")
        return
    print(f"Downloading PHEME from {URL} …")
    with requests.get(URL, stream=True, timeout=300, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(ARCHIVE, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.1f}%)", end="")
        print()


def _looks_like_event_dir(path: Path) -> bool:
    """An event dir contains either 'rumours' or 'non-rumours' (any casing)."""
    if not path.is_dir():
        return False
    names = {p.name.lower() for p in path.iterdir() if p.is_dir()}
    return ("rumours" in names) or ("non-rumours" in names) or ("nonrumours" in names)


def _canonical_event_name(raw: str) -> str:
    if raw in EVENT_ALIASES:
        return EVENT_ALIASES[raw]
    base = raw.lower()
    for alias, canon in EVENT_ALIASES.items():
        if base.startswith(canon) or canon in base:
            return canon
    return raw


def extract() -> None:
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print(f"[skip extract] {DATA_DIR} is non-empty.")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PROJECT_ROOT / "data" / "_pheme_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    # Figshare misnames the file — it's actually gzip-compressed despite the
    # .tar.bz2 extension. Use "r:*" so tarfile auto-detects the real compression.
    print(f"Extracting {ARCHIVE} → {tmp} …")
    with tarfile.open(ARCHIVE, "r:*") as t:
        # Python 3.12+ requires specifying a filter. "data" is safe & sufficient.
        try:
            t.extractall(tmp, filter="data")
        except TypeError:
            t.extractall(tmp)

    # Walk to find event directories anywhere in the tree.
    event_dirs: list[Path] = []
    for candidate in tmp.rglob("*"):
        if _looks_like_event_dir(candidate):
            event_dirs.append(candidate)
    if not event_dirs:
        sys.exit("Could not locate any PHEME event directories in the archive.")
    print(f"Found {len(event_dirs)} event directories.")

    copied = 0
    for ev in event_dirs:
        canon = _canonical_event_name(ev.name)
        if canon not in PHEME_EVENTS:
            print(f"  [skip] {ev.name} → {canon} (not in PHEME_EVENTS)")
            continue
        target = DATA_DIR / canon
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(ev, target)
        print(f"  [copy] {ev.name} → {target.relative_to(PROJECT_ROOT)}")
        copied += 1
    if copied == 0:
        sys.exit("No event directories matched the expected 5-event set.")

    shutil.rmtree(tmp)


if __name__ == "__main__":
    download()
    extract()
    print(f"Done. PHEME is at {DATA_DIR}.")
