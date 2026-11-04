"""
missing_report.py  -  everything still missing or broken in the subtitle collection -> MISSING.md

Reads status.json (Fetch.py's memory), subs/ and the episode lists Fetch.py looked up. Reads only.
collect() returns the same information as data, for fetch_missing.py.
"""
import importlib.util, json, re
from collections import defaultdict
from pathlib import Path

import build_index as bi

ROOT = Path(__file__).resolve().parent

# Real episode counts per season where Fetch.py's lookup came back empty or matched the wrong show
KNOWN_SEASONS = {
    "daredevil_2015": {1: 13, 2: 13, 3: 13},
    "the_punisher_2017": {1: 13, 2: 13},
    "cloak_dagger_2018": {1: 10, 2: 10},
    "the_defenders_2017": {1: 8},
    "runaways_2017": {1: 10, 2: 13, 3: 10},
    "agents_of_s_h_i_e_l_d_2013": {1: 22, 2: 22, 3: 22, 4: 22, 5: 22, 6: 13, 7: 13},
    "agent_carter_2015": {1: 8, 2: 10},
    "iron_fist_2017": {1: 13, 2: 10},
    "legion_2017": {1: 8, 2: 11, 3: 8},
    "she_hulk_attorney_at_law_2022": {1: 9},
    "agatha_all_along_2024": {1: 9},
}
LOOKUP_NOTE = {
    "daredevil_2015": "episode lookup returned nothing, so impossible episodes were never filtered "
                      "(a 'Daredevil' search also matches 'Kick Buttowski: Suburban Daredevil')",
    "the_punisher_2017": "episode lookup returned nothing, so impossible episodes were never filtered",
    "cloak_dagger_2018": "episode lookup returned nothing, so impossible episodes were never filtered",
    "the_defenders_2017": "lookup matched the 2010 CBS legal drama 'The Defenders' (18 episodes), not Marvel's (8)",
    "spidey_super_stories_1974": "episode lookup returned nothing (they were segments of The Electric Company)",
}
# Wrong-show files found by data_check.py (first words checked by hand)
WRONG = {
    **{f"tv__daredevil_2015__s01e{e:02d}": "Young Indiana Jones / Speed Racer / Kick Buttowski" for e in range(14, 21)},
    **{f"tv__daredevil_2015__s02e{e:02d}": "Kick Buttowski (cartoon)" for e in range(14, 33)},
    **{f"tv__the_punisher_2017__s01e{e:02d}": "a Taiwanese Netflix series" for e in range(14, 21)},
    **{f"tv__the_defenders_2017__s01e{e:02d}": "The Defenders (2010, CBS legal drama)" for e in range(9, 19)},
    "tv__cloak_dagger_2018__s02e12": "All the Queen's Men (BET+)",
    "tv__legion_2017__s03e10": "Supergirl (Legion of Super-Heroes episode)",
    "tv__the_gifted_2017__s01e13": "Urusei Yatsura (anime) - a real episode, re-download it",
}
# Marvel titles never searched for by Fetch.py (from memory, not from an API: please verify)
NEW_TITLES = [  # (group, kind, title, year)
    ("Movies (live action)", "movie", "The Punisher", 1989),
    ("Movies (live action)", "movie", "Captain America", 1990),
    ("Movies (live action)", "movie", "The Fantastic Four", 1994),
    ("Movies (live action)", "movie", "Man-Thing", 2005),
    ("Movies (live action)", "movie", "Nick Fury: Agent of S.H.I.E.L.D.", 1998),
    ("Movies (live action)", "movie", "Generation X", 1996),
    ("Movies (live action)", "movie", "Dr. Strange", 1978),
    ("Movies (live action)", "movie", "Captain America", 1979),
    ("Movies (live action)", "movie", "Captain America II: Death Too Soon", 1979),
    ("Movies (live action)", "movie", "The Incredible Hulk Returns", 1988),
    ("Movies (live action)", "movie", "The Trial of the Incredible Hulk", 1989),
    ("Movies (live action)", "movie", "The Death of the Incredible Hulk", 1990),
    ("Movies (animated / specials)", "movie", "Spider-Man: Into the Spider-Verse", 2018),
    ("Movies (animated / specials)", "movie", "Spider-Man: Across the Spider-Verse", 2023),
    ("Movies (animated / specials)", "movie", "Big Hero 6", 2014),
    ("Movies (animated / specials)", "movie", "Werewolf by Night", 2022),
    ("Movies (animated / specials)", "movie", "The Guardians of the Galaxy Holiday Special", 2022),
    ("Series", "tv", "Daredevil: Born Again", 2025),
    ("Series", "tv", "Marvel Zombies", 2025),
    ("Series", "tv", "Eyes of Wakanda", 2025),
    ("Series", "tv", "I Am Groot", 2022),
    ("Series", "tv", "Hit-Monkey", 2021),
    ("Series", "tv", "M.O.D.O.K.", 2021),
    ("Series", "tv", "The Avengers: Earth's Mightiest Heroes", 2010),
    ("Series", "tv", "Ultimate Spider-Man", 2012),
    ("Series", "tv", "Spider-Man: The New Animated Series", 2003),
    ("Series", "tv", "Spider-Man", 1967),
]
# Movies whose file is a partial / forced subtitle: worth re-downloading
DAMAGED_MOVIE_MIN_WORDS = 4000
EP_RE = re.compile(r"^tv__(?P<show>.+?)__s(?P<s>\d+)e(?P<e>\d+)$")


def fetch_catalog():
    """Pretty titles of every show/movie Fetch.py knows (its MOVIES / SHOWS lists)."""
    spec = importlib.util.spec_from_file_location("fetch_catalog", ROOT / "subs" / "Fetch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                     # only defines constants and functions
    shows = {f"{mod.slug(t)}_{y}": (t, y) for t, y, *_ in mod.SHOWS}
    movies = {f"{mod.slug(t)}_{y}": (t, y) for t, y in mod.MOVIES}
    return shows, movies


def pretty(show):
    m = re.match(r"(.+)_(\d{4})$", show)
    return f"{m.group(1).replace('_', ' ').title()} ({m.group(2)})" if m else show


def ranges(nums):
    nums, out = sorted(nums), []
    for n in nums:
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return ", ".join(f"E{a:02d}" if a == b else f"E{a:02d}-E{b:02d}" for a, b in out)


def collect():
    status = json.loads((ROOT / "status.json").read_text())
    items, shows = status["items"], status["shows"]
    on_disk = {p.stem for p in (ROOT / "subs").glob("*.txt")}

    expected = defaultdict(set)          # Fetch.py's lookup, or the known counts above
    for show, v in shows.items():
        for s, eps in v.get("episodes", {}).items():
            expected[show] |= {(int(s), int(e)) for e in eps}
    for show, seasons in KNOWN_SEASONS.items():
        expected[show] = {(s, e) for s, n in seasons.items() for e in range(1, n + 1)}
    have = defaultdict(set)
    for stem in on_disk - set(WRONG):
        m = EP_RE.match(stem)
        if m:
            have[m.group("show")].add((int(m.group("s")), int(m.group("e"))))
    not_found = defaultdict(set)
    for k, v in items.items():
        m = EP_RE.match(k)
        if m and v == "not_found":
            not_found[m.group("show")].add((int(m.group("s")), int(m.group("e"))))
    needed = {}
    for show in sorted(set(expected) | set(not_found)):
        exp = expected[show] or not_found[show]
        if exp - have[show]:
            needed[show] = {"expected": exp, "known": bool(expected[show]), "have": have[show],
                            "need": exp - have[show]}

    damaged = []
    for p in sorted((ROOT / "subs").glob("*.txt")):
        if p.stem in WRONG:
            continue
        raw = p.read_text(encoding="utf-8", errors="ignore")
        a, b = len(raw.split()), len(bi.clean_text(raw).split())
        if a and b / a < 0.7:
            damaged.append((p.stem, f"cleaning keeps only {b / a:.0%} of {a} words "
                                    f"(MUSIC_RE '#' bug or heavy sound tags)", False))
        elif p.stem.startswith("movie__") and b < DAMAGED_MOVIE_MIN_WORDS:
            damaged.append((p.stem, f"only {b} words of dialogue for a feature film "
                                    "(probably a partial or forced-subtitles file)", True))
    return {"on_disk": on_disk, "items": items, "needed": needed, "wrong": WRONG,
            "damaged": damaged, "new": NEW_TITLES}


def main():
    c = collect()
    items, needed = c["items"], c["needed"]
    out = ["# Missing and broken subtitles\n",
           f"Collection: {len(c['on_disk'])} files on disk; Fetch.py status: "
           f"{sum(v == 'done' for v in items.values())} done, "
           f"{sum(v == 'not_found' for v in items.values())} not found on OpenSubtitles. "
           f"{len(WRONG)} of the 'done' files contain the wrong show.\n",
           "Fill the gaps with `python fetch_missing.py` (see its header for the free API keys).\n"]

    out.append("## 1. TV episodes still needed\n")
    out.append("Correct episode = on disk and not a wrong-show file. 'Needed' = expected minus correct.\n")
    out.append("| show | expected | correct on disk | needed | missing episodes |\n|---|---|---|---|---|")
    total_need = 0
    for show, v in needed.items():
        total_need += len(v["need"])
        by_season = defaultdict(list)
        for s, e in v["need"]:
            by_season[s].append(e)
        eps = "; ".join(f"S{s:02d} {ranges(x)}" for s, x in sorted(by_season.items()))
        tag = " *" if show in LOOKUP_NOTE else ""
        out.append(f"| {pretty(show)}{tag} | {len(v['expected']) if v['known'] else '?'} "
                   f"| {len(v['have'])} | **{len(v['need'])}** | {eps} |")
    out.append(f"\n**{total_need} episodes needed in total.** "
               "* = Fetch.py's episode lookup failed for this show:\n")
    out += [f"- {pretty(k)}: {v}" for k, v in LOOKUP_NOTE.items()]

    out.append(f"\n## 2. Files on disk that contain the wrong show ({len(WRONG)}) - re-download\n")
    out.append("These need replacing (or removing). They are still in subs/; the index option "
               "`exclude_files` can leave them out.\n")
    groups = defaultdict(list)
    for stem, what in WRONG.items():
        m = EP_RE.match(stem)
        groups[(m.group("show"), int(m.group("s")), what)].append(int(m.group("e")))
    out.append("| show | season | episodes | actually contains |\n|---|---|---|---|")
    for (show, s, what), eps in sorted(groups.items()):
        real = KNOWN_SEASONS.get(show, {}).get(s)
        note = f" (season has {real} episodes)" if real else ""
        out.append(f"| {pretty(show)} | {s} | {ranges(eps)}{note} | {what} |")

    out.append("\n## 3. Files that are present but damaged or incomplete\n")
    out.append("| file | problem |\n|---|---|")
    out += [f"| {stem}.txt | {why} |" for stem, why, _ in c["damaged"]]

    out.append("\n## 4. Marvel titles Fetch.py never searched for\n")
    out.append("This list is from memory, not from an API. Check it before downloading.\n")
    for group in dict.fromkeys(g for g, *_ in NEW_TITLES):
        out.append(f"**{group}**\n")
        out += [f"- {t} ({y})" for g, _, t, y in NEW_TITLES if g == group]
        out.append("")
    out.append("Not released yet (as of Nov 2026): Avengers: Doomsday (Dec 2026).")
    (ROOT / "MISSING.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:4]))
    print(f"total needed episodes: {total_need}")


if __name__ == "__main__":
    main()
