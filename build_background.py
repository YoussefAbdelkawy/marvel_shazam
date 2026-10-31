"""
build_background.py  -  a non-Marvel "background" index used to say "no match" (search.background)

    python build_background.py [--config config.json] [--n 60000]

Source: sedthh/tv_dialogue TV transcripts only (Star Trek, Doctor Who, Friends, The Office,
Knight Rider). The movie-script parts (imsdb, dailyscript) are skipped because they overlap the
Cornell movies used as evaluation negatives, and the Marvel scripts are skipped for obvious
reasons. Chunked like the main index, embedded with the same model (shared embedding cache).
"""
import argparse, hashlib, random
from pathlib import Path

import numpy as np
import pandas as pd

import build_index as bi

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "external" / "tv_dialogue" / "data" / "train-00000-of-00001-b60de8ba0700f755.parquet"
TV_SOURCES = {"startrek/chakoteya", "drwho/jeanmidev", "friends/emorynlp",
              "office/nasirkhalid24", "knightriderarchives"}
SEED = 2024


def background_chunks(cfg, n):
    raw = pd.read_parquet(SRC)
    raw = raw[raw.SOURCE.isin(TV_SOURCES)]
    rows = []
    for i, (text, src) in enumerate(zip(raw.TEXT, raw.SOURCE)):
        text = bi.SPEAKER_RE.sub(" ", text.replace("[", "\n[").replace("]", "]\n"))
        words = bi.clean_text(text).split()
        for size, stride in cfg["scales"]:
            for a, b in bi.chunk_words(words, size, stride, cfg["min_chunk_words"]):
                rows.append((src, i, size, " ".join(words[a:b])))
    random.Random(SEED).shuffle(rows)
    df = pd.DataFrame(rows[:n], columns=["source", "doc", "scale", "text"])
    df["text_hash"] = [hashlib.sha1(t.encode()).hexdigest() for t in df.text]
    return df.drop_duplicates("text_hash").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(bi.CONFIG_PATH))
    ap.add_argument("--n", type=int, default=60000)
    a = ap.parse_args()
    import faiss
    cfg = bi.load_config(a.config)
    df = background_chunks(cfg, a.n)
    print(f"background: {len(df)} chunks\n{df.source.value_counts().to_string()}")
    vecs = bi.embed_with_cache(df, cfg)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    out = bi.background_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "index.faiss"))
    df.to_parquet(out / "chunks.parquet", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
