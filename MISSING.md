# Missing and broken subtitles

Collection: 1233 files on disk; Fetch.py status: 1232 done, 60 not found on OpenSubtitles. 46 of the 'done' files contain the wrong show.

Fill the gaps with `python fetch_missing.py` (see its header for the free API keys).

## 1. TV episodes still needed

Correct episode = on disk and not a wrong-show file. 'Needed' = expected minus correct.

| show | expected | correct on disk | needed | missing episodes |
|---|---|---|---|---|
| Spidey Super Stories (1974) * | ? | 0 | **29** | S01 E01-E10; S02 E01-E10; S03 E01-E09 |
| The Amazing Spider Man (1977) | 14 | 11 | **3** | S01 E03; S02 E06, E08 |
| The Gifted (2017) | 29 | 28 | **1** | S01 E13 |
| Wolverine And The X Men (2009) | 26 | 23 | **3** | S01 E16, E23, E26 |
| X Men Evolution (2000) | 52 | 48 | **4** | S02 E10; S03 E09; S04 E08-E09 |
| X Men The Animated Series (1992) | ? | 70 | **6** | S01 E14-E16; S02 E14-E16 |
| Your Friendly Neighborhood Spider Man (2025) | 11 | 10 | **1** | S02 E01 |

**47 episodes needed in total.** * = Fetch.py's episode lookup failed for this show:

- Daredevil (2015): episode lookup returned nothing, so impossible episodes were never filtered (a 'Daredevil' search also matches 'Kick Buttowski: Suburban Daredevil')
- The Punisher (2017): episode lookup returned nothing, so impossible episodes were never filtered
- Cloak Dagger (2018): episode lookup returned nothing, so impossible episodes were never filtered
- The Defenders (2017): lookup matched the 2010 CBS legal drama 'The Defenders' (18 episodes), not Marvel's (8)
- Spidey Super Stories (1974): episode lookup returned nothing (they were segments of The Electric Company)

## 2. Files on disk that contain the wrong show (46) - re-download

These need replacing (or removing). They are still in subs/; the index option `exclude_files` can leave them out.

| show | season | episodes | actually contains |
|---|---|---|---|
| Cloak Dagger (2018) | 2 | E12 (season has 10 episodes) | All the Queen's Men (BET+) |
| Daredevil (2015) | 1 | E14-E20 (season has 13 episodes) | Young Indiana Jones / Speed Racer / Kick Buttowski |
| Daredevil (2015) | 2 | E14-E32 (season has 13 episodes) | Kick Buttowski (cartoon) |
| Legion (2017) | 3 | E10 (season has 8 episodes) | Supergirl (Legion of Super-Heroes episode) |
| The Defenders (2017) | 1 | E09-E18 (season has 8 episodes) | The Defenders (2010, CBS legal drama) |
| The Gifted (2017) | 1 | E13 | Urusei Yatsura (anime) - a real episode, re-download it |
| The Punisher (2017) | 1 | E14-E20 (season has 13 episodes) | a Taiwanese Netflix series |

## 3. Files that are present but damaged or incomplete

| file | problem |
|---|---|
| movie__spider_man_homecoming_2017.txt | cleaning keeps only 61% of 12399 words (MUSIC_RE '#' bug or heavy sound tags) |

## 4. Marvel titles Fetch.py never searched for

This list is from memory, not from an API. Check it before downloading.

**Movies (live action)**

- The Punisher (1989)
- Captain America (1990)
- The Fantastic Four (1994)
- Man-Thing (2005)
- Nick Fury: Agent of S.H.I.E.L.D. (1998)
- Generation X (1996)
- Dr. Strange (1978)
- Captain America (1979)
- Captain America II: Death Too Soon (1979)
- The Incredible Hulk Returns (1988)
- The Trial of the Incredible Hulk (1989)
- The Death of the Incredible Hulk (1990)

**Movies (animated / specials)**

- Spider-Man: Into the Spider-Verse (2018)
- Spider-Man: Across the Spider-Verse (2023)
- Big Hero 6 (2014)
- Werewolf by Night (2022)
- The Guardians of the Galaxy Holiday Special (2022)

**Series**

- Daredevil: Born Again (2025)
- Marvel Zombies (2025)
- Eyes of Wakanda (2025)
- I Am Groot (2022)
- Hit-Monkey (2021)
- M.O.D.O.K. (2021)
- The Avengers: Earth's Mightiest Heroes (2010)
- Ultimate Spider-Man (2012)
- Spider-Man: The New Animated Series (2003)
- Spider-Man (1967)

Not released yet (as of Nov 2026): Avengers: Doomsday (Dec 2026).
