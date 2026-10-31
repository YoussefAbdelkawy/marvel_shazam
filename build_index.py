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

# how search turns retrieved chunks into an answer; every option can be switched in config.json
DEFAULT_SEARCH = {
    "k": 20,                       # chunks that get a vote
    "scoring": "linear",           # linear: sum(sim * w)   exp: sum(exp(sim / T) * w)
    "temperature": 0.05,           # T for exp scoring
    "confidence": "top_score",     # top_score (best chunk score of the winner) | top_sim | share
    "threshold": None,             # confidence below this -> "no match"
    "hybrid": False,               # BM25 + dense, fused with Reciprocal Rank Fusion
    "dense_depth": 50,             # dense candidates fed to fusion / reranking
    "bm25_depth": 50,
    "rrf_k": 60,
    "bm25_k1": 1.5, "bm25_b": 0.75, "bm25_epsilon": 0.25,   # rank_bm25.BM25Okapi defaults
    "rerank": False,               # rescore candidates with a cross-encoder
    "rerank_model": "BAAI/bge-reranker-base",
    "rerank_depth": 50,
}
NO_MATCH = "no match"


# ------------------------------------------------------------------ config
def load_config(path=CONFIG_PATH):
    path = Path(path)
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
        print(f"Created {path.name} with defaults.")
    cfg = {**DEFAULT_CONFIG, **json.loads(path.read_text())}
    cfg["search"] = {**DEFAULT_SEARCH, **cfg.get("search", {})}
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


def title_label(title, season, episode):
    """The answer string search returns: 'Iron Man (2008)' or 'Loki (2021) S01E03'."""
    if season is not None and pd.notna(season):
        return f"{title} S{int(season):02d}E{int(episode):02d}"
    return title


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
            local = ROOT / name                       # fine-tuned models live inside the project
            self.model = SentenceTransformer(str(local) if local.exists() else name)
            self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size=64, progress=True):
        if self.model is None:
            out = np.zeros((len(texts), self.dim), np.float32)
            for i, t in enumerate(texts):
                for w in norm(t):
                    out[i, hash(w) % self.dim] += 1
            return normalize(out)
        v = self.model.encode(texts, batch_size=batch_size, show_progress_bar=progress,
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


def load_docs(cfg, verbose=True):
    """Every subtitle file, cleaned and with recaps removed: exactly the text that gets chunked."""
    title_map = load_title_map(cfg)
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
    removed = remove_recaps(docs, cfg)
    if verbose:
        print(f"Recaps: removed {removed} words of repeated 'previously on' dialogue")
    return docs, manifest, skipped


def build(cfg):
    manifest_p = ROOT / cfg["work_dir"] / "manifest.json"
    manifest_p.parent.mkdir(parents=True, exist_ok=True)
    old_manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {}

    docs, manifest, skipped = load_docs(cfg)
    new_files = [f for f, h in manifest.items() if old_manifest.get(f) != h]
    gone = [f for f in old_manifest if f not in manifest]
    print(f"Files: {len(docs)} subtitle files, {len(new_files)} new/changed, {len(gone)} removed"
          + (f", {len(skipped)} ignored (name not recognised)" if skipped else ""))

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


# ------------------------------------------------------------------ BM25 (sparse)
class BM25Index:
    """rank_bm25.BM25Okapi (same formula, same defaults) stored as a sparse docs x terms matrix,
    so scoring 100k+ chunks is one sparse product instead of a Python loop per query term."""

    def __init__(self, texts, k1=1.5, b=0.75, epsilon=0.25):
        from sklearn.feature_extraction.text import CountVectorizer
        self.vec = CountVectorizer(tokenizer=norm, lowercase=False, token_pattern=None)
        tf = self.vec.fit_transform(texts).tocsr().astype(np.float32)
        n_docs = tf.shape[0]
        dl = np.asarray(tf.sum(1)).ravel()
        n_t = np.bincount(tf.indices, minlength=tf.shape[1])          # docs containing each term
        idf = np.log(n_docs - n_t + 0.5) - np.log(n_t + 0.5)
        idf[idf < 0] = epsilon * idf.mean()
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        f = tf.data
        tf.data = (idf[tf.indices] * f * (k1 + 1)
                   / (f + k1 * (1 - b + b * dl[rows] / dl.mean()))).astype(np.float32)
        self.weights_t = tf.T.tocsr()                                  # terms x docs

    def scores(self, queries):
        return self.vec.transform(queries).astype(np.float32) @ self.weights_t   # sparse q x docs

    def top(self, queries, depth, batch=256):
        out = []
        for a in range(0, len(queries), batch):
            s = self.scores(queries[a:a + batch]).tocsr()
            for r in range(s.shape[0]):
                lo, hi = s.indptr[r], s.indptr[r + 1]
                ids, vals = s.indices[lo:hi], s.data[lo:hi]
                if len(vals) > depth:
                    p = np.argpartition(-vals, depth)[:depth]
                    ids, vals = ids[p], vals[p]
                out.append(ids[np.argsort(-vals, kind="stable")].astype(np.int64))
        return out


# ------------------------------------------------------------------ search (voting)
class Searcher:
    """Loads one index once and answers batches of queries (used by the CLI and evaluate.py).

    candidates(): dense FAISS top-k, optionally fused with BM25 (RRF) and/or reranked
    decide():     votes chunks into titles, then applies the confidence threshold
    """

    def __init__(self, cfg):
        import faiss
        self.cfg, self.s = cfg, cfg["search"]
        out = index_dir(cfg)
        self.index = faiss.read_index(str(out / "index.faiss"))
        self.df = pd.read_parquet(out / "chunks.parquet")
        self.labels = np.array([title_label(t, s, e) for t, s, e in
                                zip(self.df["title"], self.df["season"], self.df["episode"])])
        self.weights = self.df["vote_weight"].to_numpy(np.float64)
        self.embedder = Embedder(cfg["model"])
        self._bm25 = self._reranker = None
        self._rerank_cache = None

    # ---- retrieval
    def candidates(self, queries, franchise=None):
        """Per query: (chunk_ids, chunk_scores, dense_sims), best first, at most k long."""
        s, queries = self.s, list(queries)
        q = self.embedder.encode(queries, self.cfg["embed_batch_size"], progress=len(queries) > 500)
        depth = s["k"]
        if s["hybrid"] or s["rerank"]:
            depth = max(depth, s["dense_depth"], s["rerank_depth"] if s["rerank"] else 0)
        if franchise:
            depth = min(len(self.df), depth * 10)
        sims, ids = self.index.search(q, depth)
        bm = self._bm25_index().top(queries, s["bm25_depth"]) if s["hybrid"] else None
        fr = self.df["franchise"].to_numpy() if franchise else None
        out = []
        for qi in range(len(queries)):
            keep = ids[qi] >= 0
            if franchise:
                keep &= fr[np.clip(ids[qi], 0, None)] == franchise
            cid, sim = ids[qi][keep], sims[qi][keep].astype(np.float64)
            if s["hybrid"]:
                fused = defaultdict(float)
                for ranked in (cid[:s["dense_depth"]], bm[qi]):
                    for r, i in enumerate(ranked):
                        fused[int(i)] += 1.0 / (s["rrf_k"] + r + 1)
                order = sorted(fused, key=lambda i: -fused[i])
                if franchise:
                    order = [i for i in order if fr[i] == franchise]
                cid = np.array(order, dtype=np.int64)
                score = np.array([fused[i] for i in order])
                sim = self._dense_sims(q[qi], cid)
            else:
                score = sim
            out.append((cid, score, sim))
        if s["rerank"]:
            out = self._rerank(queries, out)
        return [(c[:s["k"]], sc[:s["k"]], d[:s["k"]]) for c, sc, d in out]

    def _dense_sims(self, qv, cid):
        if len(cid) == 0:
            return np.zeros(0)
        return (self.index.reconstruct_batch(cid) @ qv).astype(np.float64)

    def _bm25_index(self):
        if self._bm25 is None:
            s = self.s
            self._bm25 = BM25Index(self.df["text"].tolist(), s["bm25_k1"], s["bm25_b"],
                                   s["bm25_epsilon"])
        return self._bm25

    def _rerank(self, queries, cands):
        import pickle
        from sentence_transformers import CrossEncoder
        s = self.s
        cache_p = ROOT / self.cfg["work_dir"] / "rerank_cache" / f"{model_slug(s['rerank_model'])}.pkl"
        if self._rerank_cache is None:
            self._rerank_cache = pickle.loads(cache_p.read_bytes()) if cache_p.exists() else {}
        cache, hashes, texts = self._rerank_cache, self.df["text_hash"].to_numpy(), self.df["text"]
        cut = [(c[:s["rerank_depth"]], d[:s["rerank_depth"]]) for c, _, d in cands]
        todo = {(q, hashes[i]): (q, texts.iat[i]) for q, (c, _) in zip(queries, cut)
                for i in c if (q, hashes[i]) not in cache}
        if todo:
            if self._reranker is None:
                self._reranker = CrossEncoder(s["rerank_model"], max_length=256)
            keys = list(todo)
            logits = self._reranker.predict([todo[k] for k in keys], batch_size=64,
                                            show_progress_bar=len(keys) > 2000)
            cache.update(zip(keys, map(float, logits)))
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            cache_p.write_bytes(pickle.dumps(cache))
        out = []
        for q, (c, d) in zip(queries, cut):
            sc = np.array([cache[(q, hashes[i])] for i in c])
            o = np.argsort(-sc, kind="stable")
            out.append((c[o], sc[o], d[o]))
        return out

    # ---- voting + threshold
    def decide(self, cands, **override):
        """Per query: {"answer", "ranking": [(label, vote), ...], "confidence", "conf": {...}}."""
        s = {**self.s, **override}
        res = []
        for cid, score, sim in cands:
            if len(cid) == 0:
                res.append({"answer": NO_MATCH, "ranking": [], "confidence": -np.inf,
                            "conf": {}, "best": {}})
                continue
            w = self.weights[cid]
            if s["scoring"] == "exp":
                v = np.exp((score - score.max()) / s["temperature"]) * w
            else:
                v = score * w
            votes, best, best_sim = defaultdict(float), {}, {}
            for c, vv, sc, d in zip(cid, v, score, sim):
                lab = self.labels[c]
                votes[lab] += vv
                if lab not in best or sc > best[lab][0]:
                    best[lab] = (float(sc), int(c))
                best_sim[lab] = max(best_sim.get(lab, -np.inf), float(d))
            ranking = sorted(votes.items(), key=lambda x: -x[1])
            top = ranking[0][0]
            total = sum(votes.values())
            conf = {"top_score": best[top][0], "top_sim": best_sim[top],
                    "share": ranking[0][1] / total if total > 0 else 0.0}
            c = conf[s["confidence"]]
            answer = NO_MATCH if s["threshold"] is not None and c < s["threshold"] else top
            res.append({"answer": answer, "ranking": ranking, "confidence": c, "conf": conf,
                        "best": best})
        return res

    def search_batch(self, queries, franchise=None):
        return self.decide(self.candidates(queries, franchise))


def search(cfg, query, top=5, franchise=None):
    sr = Searcher(cfg)
    r = sr.search_batch([query], franchise)[0]
    print(f'\nQuery: "{query}"\n')
    if r["answer"] == NO_MATCH:
        print(f"-> no match (confidence {r['confidence']:.3f} < threshold "
              f"{cfg['search']['threshold']})\n")
    for rank, (label, vote) in enumerate(r["ranking"][:top], 1):
        score, cid = r["best"][label]
        row = sr.df.iloc[cid]
        print(f"{rank}. {label}   vote={vote:.3f}  best={score:.3f}  ~{row['position']:.0%} in")
        print(f"     ...{row['text'][:110]}...")


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
    ap.add_argument("-k", type=int, help="chunks that vote (default: config search.k)")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.k:
        cfg["search"]["k"] = a.k
    if a.cmd == "build":
        build(cfg)
    elif a.cmd == "search":
        if not a.query:
            sys.exit('give some dialogue: python build_index.py search "where is my son"')
        search(cfg, a.query, franchise=a.franchise)
    else:
        stats(cfg)
