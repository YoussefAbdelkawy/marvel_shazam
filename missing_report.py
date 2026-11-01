"""
missing_report.py  -  everything still missing or broken in the subtitle collection -> MISSING.md

Reads status.json (Fetch.py's memory), subs/ and the episode lists Fetch.py looked up. Reads only.
"""
import json, re
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
    "daredevil_2015": "episode lookup returned nothing, so impossible episodes were never filtered",
    "the_punisher_2017": "episode lookup returned nothing, so impossible episodes were never filtered",
    "cloak_dagger_2018": "episode lookup returned nothing, so impossible episodes were never filtered",
    "the_defenders_2017": "lookup matched the 2010 CBS legal drama 'The Defenders' (18 episodes), not Marvel's (8)",
    "spidey_super_stories_1974": "episode lookup returned nothing",
}
# Wrong-show files found by data_check.py (first words checked by hand)
WRONG = {
    **{f"tv__daredevil_2015__s01e{e:02d}": "Young Indiana Jones / Speed Racer / Kick Buttowski" for e in range(14, 21)},
    **{f"tv__daredevil_2015__s02e{e:02d}": "Kick Buttowski (cartoon)" for e in range(14, 33)},
    **{f"tv__the_punisher_2017__s01e{e:02d}": "a Taiwanese Netflix series" for e in range(14, 21)},
    **{f"tv__the_defenders_2017__s01e{e:02d}": "The Defenders (2010, CBS legal drama)" for e in range(9, 19)},
    "tv__cloak_dagger_2018__s02e12": "All the Queen's Men (BET+)",
    "tv__legion_2017__s03e10": "Supergirl (Legion of Super-Heroes episode)",
}
# Marvel titles never searched for by Fetch.py (from memory, not from an API: please verify)
NOT_IN_CATALOG = {
    "Movies (live action)": [
        "The Punisher (1989)", "Captain America (1990)", "The Fantastic Four (1994, unreleased)",
        "Man-Thing (2005)", "Nick Fury: Agent of S.H.I.E.L.D. (1998, TV movie)",
        "Generation X (1996, TV movie)", "Doctor Strange (1978, TV movie)",
        "Captain America (1979) and Captain America II: Death Too Soon (1979, TV movies)",
        "The Incredible Hulk Returns (1988)", "The Trial of the Incredible Hulk (1989)",
        "The Death of the Incredible Hulk (1990)",
    ],
    "Movies (animated / specials)": [
        "Spider-Man: Into the Spider-Verse (2018)", "Spider-Man: Across the Spider-Verse (2023)",
        "Big Hero 6 (2014)", "Werewolf by Night (2022, special)",
        "The Guardians of the Galaxy Holiday Special (2022)",
    ],
    "Series": [
        "Daredevil: Born Again (2025)", "Marvel Zombies (2025)", "Eyes of Wakanda (2025)",
        "I Am Groot (2022)", "Hit-Monkey (2021)", "M.O.D.O.K. (2021)",
        "Agents of S.H.I.E.L.D.: Slingshot (2016, web series)",
        "The Avengers: Earth's Mightiest Heroes (2010)", "Ultimate Spider-Man (2012)",
        "Spider-Man: The New Animated Series (2003)", "Spider-Man (1967)",
    ],
}
EP_RE = re.compile(r"^tv__(?P<show>.+?)__s(?P<s>\d+)e(?P<e>\d+)$")


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


def main():
    status = json.loads((ROOT / "status.json").read_text())
    items, shows = status["items"], status["shows"]
    on_disk = {p.stem for p in (ROOT / "subs").glob("*.txt")}
    out = ["# Missing and broken subtitles\n",
           f"Collection: {len(on_disk)} files on disk; Fetch.py status: "
           f"{sum(v == 'done' for v in items.values())} done, "
           f"{sum(v == 'not_found' for v in items.values())} not found on OpenSubtitles. "
           f"{len(WRONG)} of the 'done' files contain the wrong show.\n"]

    # expected episodes per show: Fetch.py's lookup, or the known counts above
    expected = defaultdict(set)
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

    # 1. per-show table of what's missing
    out.append("## 1. TV episodes still needed\n")
    out.append("Correct episode = on disk and not a wrong-show file. 'Needed' = expected minus correct.\n")
    out.append("| show | expected | correct on disk | needed | missing episodes |\n|---|---|---|---|---|")
    total_need = 0
    for show in sorted(set(expected) | set(not_found)):
        exp = expected[show] or not_found[show]
        need = exp - have[show]
        total_need += len(need)
        if not need:
            continue
        by_season = defaultdict(list)
        for s, e in need:
            by_season[s].append(e)
        eps = "; ".join(f"S{s:02d} {ranges(v)}" for s, v in sorted(by_season.items()))
        tag = " *" if show in LOOKUP_NOTE else ""
        out.append(f"| {pretty(show)}{tag} | {len(exp) if expected[show] else '?'} "
                   f"| {len(have[show])} | **{len(need)}** | {eps} |")
    out.append(f"\n**{total_need} episodes needed in total.** "
               "* = Fetch.py's episode lookup failed for this show:\n")
    out += [f"- {pretty(k)}: {v}" for k, v in LOOKUP_NOTE.items()]

    # 2. wrong files on disk
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

    # 3. damaged / incomplete files
    cfg = bi.load_config()
    out.append("\n## 3. Files that are present but damaged or incomplete\n")
    out.append("| file | problem |\n|---|---|")
    for p in sorted((ROOT / "subs").glob("*.txt")):
        if p.stem in WRONG:
            continue
        raw = p.read_text(encoding="utf-8", errors="ignore")
        a, b = len(raw.split()), len(bi.clean_text(raw).split())
        kind = "movie" if p.stem.startswith("movie__") else "tv"
        if a and b / a < 0.7:
            out.append(f"| {p.name} | cleaning keeps only {b / a:.0%} of {a} words "
                       f"(MUSIC_RE '#' bug or heavy sound tags) |")
        elif kind == "movie" and b < 4000:
            out.append(f"| {p.name} | only {b} words of dialogue for a feature film "
                       "(probably a partial or forced-subtitles file) |")

    # 4. not in the catalog at all
    out.append("\n## 4. Marvel titles Fetch.py never searched for\n")
    out.append("This list is from memory, not from an API. Check it before downloading.\n")
    for group, titles in NOT_IN_CATALOG.items():
        out.append(f"**{group}**\n")
        out += [f"- {t}" for t in titles]
        out.append("")
    out.append("Not released yet (as of Nov 2026): Avengers: Doomsday (Dec 2026).")
    (ROOT / "MISSING.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:4]))
    print(f"total needed episodes: {total_need}")


if __name__ == "__main__":
    main()
