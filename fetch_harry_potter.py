"""
fetch_harry_potter.py  -  download English subtitles for all 8 Harry Potter films (+ Fantastic Beasts)

Uses the same free providers and API keys as fetch_missing.py (see its header):
    WYZIE_API_KEY and/or SUBDL_API_KEY (optional: OPENSUBTITLES_* ).

    python fetch_harry_potter.py              # the 8 Harry Potter films -> subs_harry_potter/
    python fetch_harry_potter.py --beasts     # also the 3 Fantastic Beasts films
    python fetch_harry_potter.py --retry      # retry films that failed last time

Files are named like the Marvel ones (movie__harry_potter_and_the_philosopher_s_stone_2001.txt), so the
index can use them directly. They are NOT added to the Marvel index automatically. To add the franchise,
put this in config.json "sources" and run  python build_index.py build :
    {"franchise": "harry_potter", "folder": "subs_harry_potter"}
"""
import argparse, sys
from pathlib import Path

import subtitle_providers as sp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "subs_harry_potter"

HARRY_POTTER = [
    ("Harry Potter and the Philosopher's Stone", 2001),
    ("Harry Potter and the Chamber of Secrets", 2002),
    ("Harry Potter and the Prisoner of Azkaban", 2004),
    ("Harry Potter and the Goblet of Fire", 2005),
    ("Harry Potter and the Order of the Phoenix", 2007),
    ("Harry Potter and the Half-Blood Prince", 2009),
    ("Harry Potter and the Deathly Hallows - Part 1", 2010),
    ("Harry Potter and the Deathly Hallows - Part 2", 2011),
]
FANTASTIC_BEASTS = [
    ("Fantastic Beasts and Where to Find Them", 2016),
    ("Fantastic Beasts: The Crimes of Grindelwald", 2018),
    ("Fantastic Beasts: The Secrets of Dumbledore", 2022),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beasts", action="store_true", help="also the Fantastic Beasts films")
    ap.add_argument("--retry", action="store_true")
    ap.add_argument("--providers", help="comma list, e.g. wyzie,subdl")
    a = ap.parse_args()
    films = HARRY_POTTER + (FANTASTIC_BEASTS if a.beasts else [])
    targets = [sp.Target("movie", t, y, reason="new") for t, y in films]
    providers, skipped = sp.configured_providers(a.providers.split(",") if a.providers else None)
    for s in skipped:
        print(f"  provider off: {s}")
    if not providers:
        sys.exit("No provider configured. Set WYZIE_API_KEY and/or SUBDL_API_KEY (see fetch_missing.py).")
    counts = sp.Downloader(OUT, providers, retry=a.retry).run(targets)
    print(f"\n{dict(counts)} -> {OUT.name}/  (details in {OUT.name}/_progress.json)")
    print('To index them: add {"franchise": "harry_potter", "folder": "subs_harry_potter"} to '
          'config.json "sources", then  python build_index.py build')


if __name__ == "__main__":
    main()
