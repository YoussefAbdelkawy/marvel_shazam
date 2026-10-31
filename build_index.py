"""
build_index.py  -  Shazam-for-TV/movies: subtitles -> clean -> chunks -> embeddings -> FAISS

Usage (from the marvel_shazam folder):
    python build_index.py build                 # clean, chunk, embed only what's new, build index
    python build_index.py search "some text"    # quick test: top titles for a piece of dialogue
    python build_index.py stats                 # what's in the index

Everything is driven by config.json (created on first run). Add a franchise by adding a
{"franchise": ..., "folder": ...} entry to "sources" and running build again: only the
new chunks get embedded, everything already embedded is reused from the cache.
"""
import argparse, hashlib, json, re, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"

DEFAULT_CONFIG = {
    "sources": [{"franchise": "marvel", "folder": "subs"}],
    "model": "BAAI/bge-small-en-v1.5",
    # each scale = one chunk size; several scales = multi-scale index
    # [[60, 30]] is the default; try [[30, 15], [60, 30], [80, 40]] later
    "scales": [[60, 30]],          # [chunk_size_words, stride_words]
    "min_chunk_words": 12,         # drop tiny tail chunks
    "recap_window_words": 400,     # look for "previously on" recaps in the first N words of an episode
    "recap_min_sentence_words": 4, # recap sentences shorter than this are kept (too generic to match)
    "shingle_words": 5,            # n-gram size used to measure how common a chunk is across titles
    "flat_index_max": 1_000_000,   # above this many chunks switch IndexFlatIP -> IndexHNSWFlat
    "embed_batch_size": 64,
    "work_dir": "index_data",
    "titles_csv": "titles.csv",    # optional: file_stem,title  to restore real titles ("S.H.I.E.L.D.")
}


# ------------------------------------------------------------------ config
def load_config():
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
        print(f"Created {CONFIG_PATH.name} with defaults.")
    cfg = {**DEFAULT_CONFIG, **json.loads(CONFIG_PATH.read_text())}
    return cfg


def model_slug(name):
    return re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()


def index_dir(cfg):
    scales = "_".join(f"{c}-{s}" for c, s in cfg["scales"])
    return ROOT / cfg["work_dir"] / "indexes" / f"{model_slug(cfg['model'])}__{scales}"


# ------------------------------------------------------------------ filename -> metadata
NAME_RE = re.compile(
    r"(?:^|_)(movie|tv)__(?P<slug>.+?)_(?P<year>\d{4})(?:__s(?P<s>\d+)e(?P<e>\d+))?$", re.I)


def parse_name(stem, title_map):
    m = NAME_RE.search(stem)
    if not m:
        return None
    kind, slug, year = m.group(1).lower(), m.group("slug"), m.group("year")
    title_key = f"{kind}__{slug}_{year}"          # stable id for a movie / a whole show
    pretty = title_map.get(title_key) or f"{slug.replace('_', ' ').title()} ({year})"
    return {
        "title_key": title_key,
        "title": pretty,
        "type": kind,
        "season": int(m.group("s")) if m.group("s") else None,
        "episode": int(m.group("e")) if m.group("e") else None,
    }


def load_title_map(cfg):
    p = ROOT / cfg["titles_csv"]
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))


# ------------------------------------------------------------------ cleaning
CREDIT_PATTERNS = [
    r"sync(?:ed)?\s*(?:&|and)\s*correct(?:ed|ions)\s+by\s+\S+",
    r"(?:web\s+)?sync\s+by\s+\S+(?:\s+\S+)?",
    r"subtitles?\s+(?:by|ripped by|downloaded from)\s+\S+",
    r"(?:encoded|ripped|transcript|captions?)\s+by\s+\S+",
    r"www\.\S+|https?://\S+|\S+\.(?:com|org|net|tv)\b",
    r"advertise your product or brand here.*?(?=[.!?]|$)",
    r"support us and become vip member.*?(?=[.!?]|$)",
]
CREDIT_RE = re.compile("|".join(CREDIT_PATTERNS), re.I)
SOUND_TAG_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|<[^>]+>")
MUSIC_RE = re.compile(r"[♪♫#][^♪♫#]*[♪♫#]|[♪♫]")
SPEAKER_RE = re.compile(r"(?:(?<=^)|(?<=[\s.!?\"]))[A-Z][A-Z0-9 .'\-]{1,24}:(?=\s)")
EP_CODE_RE = re.compile(r"\bS\d{1,2}E\d{1,3}\b")
PREVIOUSLY_RE = re.compile(
    r"previously on\s+(?:[\"“][^\"”]*[\"”]|[^.!?\"]*?)\s*(?:\.\.\.|…|[.!?:])?", re.I)


def clean_text(raw):
    t = raw.replace("﻿", " ")
    t = CREDIT_RE.sub(" ", t)
    t = MUSIC_RE.sub(" ", t)
    t = SOUND_TAG_RE.sub(" ", t)
    t = SPEAKER_RE.sub(" ", t)
    t = EP_CODE_RE.sub(" ", t)
    t = PREVIOUSLY_RE.sub(" ", t)
    t = re.sub(r"\s+-+\s+|\s-(?=\w)", " ", t)  # dialogue dashes "- Hi" / dangling "--"
    t = re.sub(r"\s+", " ", t).strip()
    return t


SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def norm(s):
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


def remove_recaps(docs, cfg):
    """For TV episodes: drop sentences near the start that appear verbatim in an EARLIER
    episode of the same show ("Previously on..." recaps, with or without the header)."""
    by_show = defaultdict(list)
    for d in docs:
        if d["type"] == "tv":
            by_show[d["title_key"]].append(d)
    def grams(words, n=3):
        return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

    removed = 0
    for eps in by_show.values():
        eps.sort(key=lambda d: (d["season"] or 0, d["episode"] or 0))
        seen = set()                                   # 3-grams of all earlier episodes
        for d in eps:
            sents = SENT_SPLIT_RE.split(d["text"])
            sizes = [len(s.split()) for s in sents]
            # which sentences in the opening window are (mostly) copied from earlier episodes?
            matched, words_so_far = [], 0
            for s, n in zip(sents, sizes):
                if words_so_far >= cfg["recap_window_words"]:
                    break
                w = norm(s)
                g = grams(w)
                hit = (len(w) >= cfg["recap_min_sentence_words"] and g
                       and len(g & seen) / len(g) >= 0.6)
                matched.append(bool(hit))
                words_so_far += n
            drop = set()
            if any(matched):
                last = max(i for i, m in enumerate(matched) if m)
                span = sum(sizes[:last + 1])
                hit_words = sum(n for n, m in zip(sizes, matched) if m)
                if hit_words >= 0.5 * span:            # a recap block: drop the whole opening
                    drop = set(range(last + 1))
                else:                                  # scattered matches: drop only those
                    drop = {i for i, m in enumerate(matched) if m}
            removed += sum(sizes[i] for i in drop)
            for s in sents:
                seen |= grams(norm(s))
            d["text"] = " ".join(s for i, s in enumerate(sents) if i not in drop)
    return removed


# ------------------------------------------------------------------ chunking
def chunk_words(words, size, stride, min_words):
    n = len(words)
    if n == 0:
        return []
    if n <= size:
        return [(0, n)] if n >= min_words else []
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] + size < n:                      # make sure the tail is covered
        starts.append(n - size)
    return [(s, s + size) for s in starts]


def make_chunks(docs, cfg):
    rows = []
    for d in docs:
        words = d["text"].split()
        seen_in_title = set()
        for size, stride in cfg["scales"]:
            for a, b in chunk_words(words, size, stride, cfg["min_chunk_words"]):
                text = " ".join(words[a:b])
                key = " ".join(norm(text))
                if key in seen_in_title:          # exact duplicate inside the same title
                    continue
                seen_in_title.add(key)
                rows.append({
                    "franchise": d["franchise"],
                    "title_key": d["title_key"],
                    "title": d["title"],
                    "type": d["type"],
                    "season": d["season"],
                    "episode": d["episode"],
                    "scale": size,
                    "position": round(a / max(len(words), 1), 4),
                    "n_words": b - a,
                    "source_file": d["source_file"],
                    "file_hash": d["file_hash"],
                    "text_hash": hashlib.sha1(text.encode()).hexdigest(),
                    "text": text,
                })
    df = pd.DataFrame(rows)
    df.insert(0, "chunk_id", np.arange(len(df), dtype=np.int64))
    return df


def add_vote_weights(df, cfg):
    """How many different titles share this chunk's wording? Catchphrases, theme songs and
    recaps across shows get df>1 and a smaller vote weight (same idea as IDF)."""
    k = cfg["shingle_words"]
    title_ids = {t: i for i, t in enumerate(df["title_key"].unique())}
    hashes, owners, chunk_of = [], [], []
    for cid, (tk, text) in enumerate(zip(df["title_key"], df["text"])):
        w = norm(text)
        tid = title_ids[tk]
        for i in range(max(len(w) - k + 1, 0)):
            hashes.append(hash(" ".join(w[i:i + k])))
            owners.append(tid)
            chunk_of.append(cid)
    if not hashes:
        df["title_df"] = 1.0
        df["vote_weight"] = 1.0
        return df
    hashes = np.array(hashes, dtype=np.int64)
    owners = np.array(owners, dtype=np.int64)
    chunk_of = np.array(chunk_of, dtype=np.int64)
    pairs = np.unique(np.stack([hashes, owners], 1), axis=0)          # distinct (shingle, title)
    uh, counts = np.unique(pairs[:, 0], return_counts=True)            # titles per shingle
    shingle_df = counts[np.searchsorted(uh, hashes)]
    # per chunk: median number of titles its shingles appear in
    order = np.argsort(chunk_of, kind="stable")
    sc, sdf = chunk_of[order], shingle_df[order]
    bounds = np.flatnonzero(np.diff(sc)) + 1
    med = np.ones(len(df))
    for grp_c, grp_df in zip(np.split(sc, bounds), np.split(sdf, bounds)):
        med[grp_c[0]] = np.median(grp_df)
    df["title_df"] = med
    df["vote_weight"] = 1.0 / med
    return df


# ------------------------------------------------------------------ embedding (with cache)
class Embedder:
    def __init__(self, name):
        self.name = name
        if name == "hash":            # offline test mode: no download, meaningless vectors
            self.dim = 256
            self.model = None
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(name)
            self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size=64):
        if self.model is None:
            out = np.zeros((len(texts), self.dim), np.float32)
            for i, t in enumerate(texts):
                for w in norm(t):
                    out[i, hash(w) % self.dim] += 1
            return normalize(out)
        v = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)
        return v.astype(np.float32)


def normalize(v):
    v = np.asarray(v, np.float32)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def cache_paths(cfg):
    d = ROOT / cfg["work_dir"] / "embeddings" / model_slug(cfg["model"])
    d.mkdir(parents=True, exist_ok=True)
    return d / "vectors.npy", d / "keys.npy"


def embed_with_cache(df, cfg):
    vec_p, key_p = cache_paths(cfg)
    if vec_p.exists():
        vecs = np.load(vec_p)
        keys = np.load(key_p, allow_pickle=False).astype(str)
    else:
        vecs, keys = None, np.array([], dtype=str)
    pos = {k: i for i, k in enumerate(keys)}
    todo = sorted(set(df["text_hash"]) - pos.keys())
    print(f"Embeddings: {len(pos)} cached, {len(todo)} new to embed with {cfg['model']}")
    if todo:
        emb = Embedder(cfg["model"])
        text_of = dict(zip(df["text_hash"], df["text"]))
        t0 = time.time()
        new = emb.encode([text_of[h] for h in todo], cfg["embed_batch_size"])
        print(f"  embedded {len(todo)} chunks in {time.time() - t0:.1f}s")
        vecs = new if vecs is None else np.vstack([vecs, new])
        keys = np.concatenate([keys, np.array(todo)])
        np.save(vec_p, vecs)
        np.save(key_p, keys)
        pos = {k: i for i, k in enumerate(keys)}
    return normalize(vecs[[pos[h] for h in df["text_hash"]]])


# ------------------------------------------------------------------ FAISS
def build_faiss(vectors, cfg):
    import faiss
    n, dim = vectors.shape
    if n <= cfg["flat_index_max"]:
        index = faiss.IndexFlatIP(dim)                       # exact cosine (vectors normalized)
        kind = "IndexFlatIP"
    else:
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        kind = "IndexHNSWFlat"
    index.add(vectors)
    return index, kind


# ------------------------------------------------------------------ build
def file_sha1(p):
    return hashlib.sha1(p.read_bytes()).hexdigest()


def build(cfg):
    title_map = load_title_map(cfg)
    manifest_p = ROOT / cfg["work_dir"] / "manifest.json"
    manifest_p.parent.mkdir(parents=True, exist_ok=True)
    old_manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {}

    docs, skipped, manifest = [], [], {}
    for src in cfg["sources"]:
        folder = (ROOT / src["folder"]).resolve()
        for p in sorted(folder.glob("*.txt")):
            meta = parse_name(p.stem, title_map)
            if meta is None:
                skipped.append(p.name)
                continue
            h = file_sha1(p)
            manifest[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)] = h
            raw = p.read_text(encoding="utf-8", errors="ignore")
            docs.append({**meta, "franchise": src["franchise"], "source_file": p.name,
                         "file_hash": h, "text": clean_text(raw)})
    new_files = [f for f, h in manifest.items() if old_manifest.get(f) != h]
    gone = [f for f in old_manifest if f not in manifest]
    print(f"Files: {len(docs)} subtitle files, {len(new_files)} new/changed, {len(gone)} removed"
          + (f", {len(skipped)} ignored (name not recognised)" if skipped else ""))

    removed = remove_recaps(docs, cfg)
    print(f"Recaps: removed {removed} words of repeated 'previously on' dialogue")

    df = make_chunks(docs, cfg)
    df = add_vote_weights(df, cfg)
    common = (df["title_df"] >= 2).sum()
    print(f"Chunks: {len(df)} (scales {cfg['scales']}), {common} shared with other titles "
          f"(down-weighted when voting)")

    vectors = embed_with_cache(df, cfg)
    index, kind = build_faiss(vectors, cfg)

    import faiss
    out = index_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "index.faiss"))
    df.to_parquet(out / "chunks.parquet", index=False)
    (out / "config.json").write_text(json.dumps(
        {**cfg, "index_type": kind, "n_chunks": int(len(df)), "dim": int(vectors.shape[1]),
         "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    manifest_p.write_text(json.dumps(manifest, indent=1))
    print(f"Index: {kind}, {len(df)} vectors x {vectors.shape[1]} dims -> {out}")


# ------------------------------------------------------------------ search (voting)
def search(cfg, query, k=20, top=5, franchise=None):
    import faiss
    out = index_dir(cfg)
    index = faiss.read_index(str(out / "index.faiss"))
    df = pd.read_parquet(out / "chunks.parquet")
    q = Embedder(cfg["model"]).encode([query])
    sims, ids = index.search(q, k if franchise is None else min(len(df), k * 10))
    scores, best = defaultdict(float), {}
    hits = 0
    for s, i in zip(sims[0], ids[0]):
        if i < 0:
            continue
        r = df.iloc[i]
        if franchise and r["franchise"] != franchise:
            continue
        hits += 1
        key = (r["title"], r["season"], r["episode"])
        scores[key] += float(s) * float(r["vote_weight"])
        if key not in best or s > best[key][0]:
            best[key] = (float(s), r["text"], r["position"])
        if hits >= k:
            break
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top]
    print(f'\nQuery: "{query}"\n')
    for rank, ((title, s, e), score) in enumerate(ranked, 1):
        ep = f" S{int(s):02d}E{int(e):02d}" if pd.notna(s) else ""
        sim, text, pos = best[(title, s, e)]
        print(f"{rank}. {title}{ep}   vote={score:.3f}  best_sim={sim:.3f}  ~{pos:.0%} in")
        print(f"     ...{text[:110]}...")


def stats(cfg):
    df = pd.read_parquet(index_dir(cfg) / "chunks.parquet")
    print(df.groupby(["franchise", "type"]).agg(titles=("title_key", "nunique"),
                                                files=("source_file", "nunique"),
                                                chunks=("chunk_id", "count")))
    print("\nMost shared chunks (lowest vote weight):")
    for _, r in df.nsmallest(5, "vote_weight").iterrows():
        print(f"  df={r.title_df:.0f}  {r.title}: {r.text[:80]}...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "search", "stats"])
    ap.add_argument("query", nargs="?")
    ap.add_argument("--franchise")
    ap.add_argument("-k", type=int, default=20)
    a = ap.parse_args()
    cfg = load_config()
    if a.cmd == "build":
        build(cfg)
    elif a.cmd == "search":
        if not a.query:
            sys.exit('give some dialogue: python build_index.py search "where is my son"')
        search(cfg, a.query, k=a.k, franchise=a.franchise)
    else:
        stats(cfg)
