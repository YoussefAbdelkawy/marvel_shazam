# Missing and broken subtitles

Collection: 706 files on disk; Fetch.py status: 660 done, 339 not found on OpenSubtitles. 46 of the 'done' files contain the wrong show.

Fill the gaps with `python fetch_missing.py` (see its header for the free API keys).

## 1. TV episodes still needed

Correct episode = on disk and not a wrong-show file. 'Needed' = expected minus correct.

| show | expected | correct on disk | needed | missing episodes |
|---|---|---|---|---|
| Agatha All Along (2024) | 9 | 0 | **9** | S01 E01-E09 |
| Echo (2024) | 5 | 0 | **5** | S01 E01-E05 |
| Ironheart (2025) | 6 | 0 | **6** | S01 E01-E06 |
| Runaways (2017) | 33 | 4 | **29** | S01 E05-E10; S02 E01-E13; S03 E01-E10 |
| Secret Invasion (2023) | 6 | 0 | **6** | S01 E01-E06 |
| She Hulk Attorney At Law (2022) | 9 | 7 | **2** | S01 E08-E09 |
| Spider Man The Animated Series (1994) | 13 | 0 | **13** | S01 E01-E13 |
| Spider Noir (2026) | 8 | 0 | **8** | S01 E01-E08 |
| Spidey Super Stories (1974) * | ? | 0 | **29** | S01 E01-E10; S02 E01-E10; S03 E01-E09 |
| The Amazing Spider Man (1977) | 14 | 3 | **11** | S01 E03, E05-E06; S02 E01-E08 |
| The Gifted (2017) | 29 | 28 | **1** | S01 E13 |
| The Incredible Hulk (1977) | 82 | 64 | **18** | S04 E01-E18 |
| What If (2021) | 26 | 0 | **26** | S01 E01-E09; S02 E01-E09; S03 E01-E08 |
| Wolverine And The X Men (2009) | 26 | 0 | **26** | S01 E01-E26 |
| Wonder Man (2026) | 8 | 0 | **8** | S01 E01-E08 |
| X Men 97 (2024) | 19 | 0 | **19** | S01 E01-E10; S02 E01-E09 |
| X Men Evolution (2000) | 52 | 0 | **52** | S01 E01-E13; S02 E01-E17; S03 E01-E13; S04 E01-E09 |
| X Men The Animated Series (1992) | ? | 0 | **76** | S01 E01-E16; S02 E01-E16; S03 E01-E16; S04 E01-E16; S05 E01-E12 |
| Your Friendly Neighborhood Spider Man (2025) | 11 | 0 | **11** | S01 E01-E10; S02 E01 |

**355 episodes needed in total.** * = Fetch.py's episode lookup failed for this show:

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
| movie__daredevil_2003.txt | only 3332 words of dialogue for a feature film (probably a partial or forced-subtitles file) |
| movie__elektra_2005.txt | only 3989 words of dialogue for a feature film (probably a partial or forced-subtitles file) |
| movie__spider_man_3_2007.txt | only 2770 words of dialogue for a feature film (probably a partial or forced-subtitles file) |
| movie__spider_man_homecoming_2017.txt | cleaning keeps only 61% of 12399 words (MUSIC_RE '#' bug or heavy sound tags) |
| movie__the_punisher_2004.txt | cleaning keeps only 58% of 6543 words (MUSIC_RE '#' bug or heavy sound tags) |
| tv__agents_of_s_h_i_e_l_d_2013__s03e18.txt | cleaning keeps only 12% of 4673 words (MUSIC_RE '#' bug or heavy sound tags) |

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
