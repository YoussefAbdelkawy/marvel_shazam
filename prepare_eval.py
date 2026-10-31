"""
prepare_eval.py  -  frozen evaluation data for marvel_shazam (run once, then never edit)

    python prepare_eval.py            # writes eval/splits.json, eval/dev.parquet, eval/test.parquet
    python prepare_eval.py --force    # overwrite (only if you really mean to reset the benchmark)

Splits (per title = per subtitle file, by word position in the cleaned dialogue):
    one contiguous 10% block -> dev, another contiguous 10% block -> test, the rest -> train.
    The index still contains ALL dialogue; splits only decide where queries / training pairs
    come from.
Queries: random windows of 10/20/40 words (never starting on a chunk boundary), each as a
    clean and a Whisper-style noisy version, plus out-of-domain negatives from the Cornell
    Movie-Dialogs corpus (expected answer: "no match").
"""
import argparse, json, random, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

import build_index as bi

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "eval"
CORNELL = ROOT / "external" / "cornell" / "cornell movie-dialogs corpus"
SEED = 1234
REGION_FRAC = 0.10            # dev block and test block, each this share of a title
LENGTHS = (10, 20, 40)
N_PER_LENGTH = 300            # positive windows per length per split
N_NEG_PER_LENGTH = 100        # negative windows per length per split (300 per split)
CHUNK_ALIGN = 15              # every chunk in every scale starts at a multiple of 15 words

# ------------------------------------------------------------------ Whisper-style noise
HOMOPHONE_GROUPS = [
    "their there they're", "to too two", "your you're", "its it's", "know no", "knew new",
    "right write", "here hear", "then than", "we're were where", "whether weather",
    "by buy bye", "for four", "one won", "see sea", "would wood", "our hour are",
    "wait weight", "peace piece", "die dye", "i'll aisle", "whole hole", "who's whose",
    "break brake", "meet meat", "sun son", "night knight", "week weak", "made maid",
    "allowed aloud", "mind mine", "and an in", "a the", "this these", "that than",
    "he's his", "i'm im", "don't dont", "can't cant", "won't want", "didn't did",
    "gonna going", "wanna want", "gotta got", "kinda kind", "okay ok", "yeah yes",
    "him them", "been being", "of have", "could good", "tell tale", "dead dad",
    "stark stuck", "thor tour", "loki low-key", "peter pizza", "tony toni", "steve steven",
    "shield shielded", "hydra hider", "avengers avenges", "wakanda wakonda",
]
SOUND_ALIKE = {}
for g in HOMOPHONE_GROUPS:
    ws = g.split()
    for w in ws:
        SOUND_ALIKE.setdefault(w, []).extend(x for x in ws if x != w)
PHONETIC_EDITS = [("ph", "f"), ("ck", "k"), ("tion", "shun"), ("ight", "ite"), ("ou", "ow"),
                  ("ee", "ea"), ("er", "a"), ("ing", "in"), ("th", "d"), ("ss", "s"), ("ll", "l")]


def _sound_alike(w, rng):
    if w in SOUND_ALIKE:
        return rng.choice(SOUND_ALIKE[w])
    for a, b in rng.sample(PHONETIC_EDITS, len(PHONETIC_EDITS)):
        if a in w:
            return w.replace(a, b, 1)
    return None


def whisperify(text, rng):
    """Corrupt a clean subtitle window the way speech-to-text output tends to look:
    lowercase, no punctuation, sentence cut short, 1-3 words dropped, a few sound-alike swaps."""
    words = re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()
    n = len(words)
    if n > 4 and rng.random() < 0.5:                         # cut short
        words = words[:n - max(1, int(n * rng.uniform(0.10, 0.25)))]
    n_drop = rng.randint(1, max(1, min(3, len(words) // 4)))  # drop 1-3 words
    for _ in range(n_drop):
        if len(words) > 3:
            words.pop(rng.randrange(len(words)))
    n_swap = max(1, round(len(words) * rng.uniform(0.05, 0.15)))
    swappable = [i for i, w in enumerate(words) if w in SOUND_ALIKE]
    others = [i for i in range(len(words)) if i not in swappable]
    rng.shuffle(swappable)
    rng.shuffle(others)
    for i in (swappable + others)[:n_swap]:                 # prefer real homophones
        alt = _sound_alike(words[i], rng)
        if alt:
            words[i] = alt
    return " ".join(words)


# ------------------------------------------------------------------ splits
def make_splits(docs, rng):
    files = {}
    for d in sorted(docs, key=lambda d: d["source_file"]):
        n = len(d["text"].split())
        L = max(1, round(n * REGION_FRAC))
        x, y = sorted(rng.randint(0, max(0, n - 2 * L)) for _ in range(2))
        first, second = [x, x + L], [y + L, y + 2 * L]
        dev, test = (first, second) if rng.random() < 0.5 else (second, first)
        files[d["source_file"]] = {
            "title": bi.title_label(d["title"], d["season"], d["episode"]),
            "n_words": n, "dev": dev, "test": test}
    return files


def sample_windows(words, region, length, rng, tries=50):
    a, b = region
    if b - a < length:
        return None
    for _ in range(tries):
        s = rng.randint(a, b - length)
        if s % CHUNK_ALIGN:
            return s
    return None


def positive_queries(docs, splits, split, rng):
    by_file = {d["source_file"]: d for d in docs}
    rows = []
    for length in LENGTHS:
        files = sorted(by_file)
        rng.shuffle(files)
        got = 0
        for f in files:
            if got >= N_PER_LENGTH:
                break
            words = by_file[f]["text"].split()
            s = sample_windows(words, splits[f][split], length, rng)
            if s is None:
                continue
            got += 1
            clean = " ".join(words[s:s + length])
            base = {"true_title": splits[f]["title"], "length": length, "is_negative": False,
                    "source_file": f, "word_start": s}
            rows.append({"query": clean, "noisy": False, **base})
            rows.append({"query": whisperify(clean, rng), "noisy": True, **base})
    return rows


# ------------------------------------------------------------------ Cornell negatives
MARVEL_RE = re.compile(
    r"spider|x-?men|x men|hulk|iron man|blade|daredevil|fantastic four|ghost rider|punisher|"
    r"elektra|howard the duck|captain america|thor\b|avengers|wolverine|deadpool|marvel|"
    r"ant-?man|guardians of the galaxy|black panther|doctor strange|venom|morbius|man-thing|"
    r"nick fury|generation x|kick-ass|men in black|blade runner", re.I)
SEP = " +++$+++ "


def cornell_movies():
    read = lambda name: (CORNELL / name).read_text(encoding="iso-8859-1").splitlines()
    titles = {p[0]: p[1] for p in (l.split(SEP) for l in read("movie_titles_metadata.txt"))}
    lines = {p[0]: p[4] for p in (l.split(SEP) for l in read("movie_lines.txt")) if len(p) == 5}
    dropped = sorted({t for t in titles.values() if MARVEL_RE.search(t)})
    texts = {}
    for l in read("movie_conversations.txt"):
        p = l.split(SEP)
        if MARVEL_RE.search(titles[p[2]]):
            continue
        ids = re.findall(r"L\d+", p[3])
        texts.setdefault(p[2], []).extend(lines[i] for i in ids if i in lines)
    movies = {m: bi.clean_text(" ".join(ls)) for m, ls in texts.items()}
    return movies, titles, dropped


def negative_queries(movie_ids, movies, titles, rng):
    rows = []
    for length in LENGTHS:
        pool = list(movie_ids)
        rng.shuffle(pool)
        got = 0
        for m in pool:
            if got >= N_NEG_PER_LENGTH:
                break
            words = movies[m].split()
            if len(words) < length + 1:
                continue
            s = rng.randint(0, len(words) - length)
            clean = " ".join(words[s:s + length])
            got += 1
            base = {"true_title": bi.NO_MATCH, "length": length, "is_negative": True,
                    "source_file": f"cornell:{m}:{titles[m]}", "word_start": s}
            rows.append({"query": clean, "noisy": False, **base})
            rows.append({"query": whisperify(clean, rng), "noisy": True, **base})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if (EVAL / "dev.parquet").exists() and not a.force:
        sys.exit("eval/ already exists and is frozen. Use --force only to reset the benchmark.")
    EVAL.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    np.random.seed(SEED)

    cfg = bi.load_config()
    docs, _, _ = bi.load_docs(cfg)
    labels = [bi.title_label(d["title"], d["season"], d["episode"]) for d in docs]
    assert len(set(labels)) == len(labels), "two files map to the same title label"
    splits = make_splits(docs, rng)
    (EVAL / "splits.json").write_text(json.dumps(
        {"seed": SEED, "region_frac": REGION_FRAC, "files": splits}, indent=1))

    movies, titles, dropped = cornell_movies()
    print(f"Cornell: {len(movies)} movies kept, dropped as Marvel/look-alike: {dropped}")
    mids = sorted(movies)
    rng.shuffle(mids)
    half = len(mids) // 2
    neg_pool = {"dev": mids[:half], "test": mids[half:]}

    for split in ("dev", "test"):
        rows = positive_queries(docs, splits, split, rng)
        rows += negative_queries(neg_pool[split], movies, titles, rng)
        df = pd.DataFrame(rows)
        df.insert(0, "qid", [f"{split}-{i:05d}" for i in range(len(df))])
        df.to_parquet(EVAL / f"{split}.parquet", index=False)
        print(f"{split}: {len(df)} queries\n"
              f"{df.groupby(['is_negative', 'length', 'noisy']).size().to_string()}")
    for _, r in pd.read_parquet(EVAL / "dev.parquet").sample(6, random_state=SEED).iterrows():
        print(f"  [{r.length} {'noisy' if r.noisy else 'clean'}] {r.true_title}: {r.query}")


if __name__ == "__main__":
    main()
