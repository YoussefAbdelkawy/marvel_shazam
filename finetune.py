"""
finetune.py  -  fine-tune the embedding model on (noisy window, source chunk) pairs from TRAIN only

    python finetune.py --config config.json --out models/bge-small-marvel-ft [--pairs-per-title 40]

Anchors are 10/20/40-word windows (mostly Whisper-noised with prepare_eval.whisperify, a different
seed than the benchmark) taken from the train part of each title; the positive is the index chunk
they overlap, and that chunk lies entirely inside train (no dev/test words). Loss:
MultipleNegativesRankingLoss with in-batch negatives. The result is a separate model folder; build
an index with it by pointing a config's "model" at that folder.
"""
import argparse, json, random, time
from pathlib import Path

import build_index as bi
from prepare_eval import whisperify

ROOT = Path(__file__).resolve().parent
LENGTHS = (10, 20, 40)


def train_spans(n, dev, test):
    """[a, b) word ranges of a title that are neither dev nor test."""
    cuts = sorted([tuple(dev), tuple(test)])
    spans, pos = [], 0
    for a, b in cuts:
        if a > pos:
            spans.append((pos, a))
        pos = max(pos, b)
    if pos < n:
        spans.append((pos, n))
    return spans


def make_pairs(cfg, per_title, noisy_frac, seed):
    rng = random.Random(seed)
    docs, _, _ = bi.load_docs(cfg, verbose=False)
    splits = json.loads((ROOT / "eval" / "splits.json").read_text())["files"]
    pairs = []
    for d in docs:
        words = d["text"].split()
        sp = splits.get(d["source_file"])
        # titles added after the benchmark was frozen have no dev/test blocks: all of it is train
        spans = train_spans(len(words), sp["dev"], sp["test"]) if sp else [(0, len(words))]
        chunks = []                                          # (a, b, span) fully inside train
        for size, stride in cfg["scales"]:
            for a, b in bi.chunk_words(words, size, stride, cfg["min_chunk_words"]):
                for s0, s1 in spans:
                    if s0 <= a and b <= s1:
                        chunks.append((a, b, s0, s1))
        if not chunks:
            continue
        for _ in range(per_title):
            a, b, s0, s1 = rng.choice(chunks)
            L = rng.choice([l for l in LENGTHS if l <= b - a] or [b - a])
            lo = max(s0, a - L // 2)                         # window overlaps the chunk >= half
            hi = min(s1 - L, b - L + L // 2)
            if hi < lo:
                continue
            st = rng.randint(lo, hi)
            anchor = " ".join(words[st:st + L])
            if rng.random() < noisy_frac:
                anchor = whisperify(anchor, rng)
            pairs.append((anchor, " ".join(words[a:b])))
    rng.shuffle(pairs)
    return pairs


def add_background_negatives(cfg, pairs, seed, top=5):
    """Third column: a non-Marvel background chunk close to the anchor (one of its top-5),
    so the model learns to keep generic dialogue away from Marvel chunks."""
    import faiss
    import pandas as pd
    rng = random.Random(seed)
    bg_dir = bi.background_dir(cfg)
    index = faiss.read_index(str(bg_dir / "index.faiss"))
    texts = pd.read_parquet(bg_dir / "chunks.parquet")["text"].tolist()
    q = bi.Embedder(cfg["model"]).encode([p[0] for p in pairs], cfg["embed_batch_size"])
    _, ids = index.search(q, top)
    return [(a, p, texts[rng.choice(list(ids[i]))]) for i, (a, p) in enumerate(pairs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(bi.CONFIG_PATH))
    ap.add_argument("--out", default="models/bge-small-marvel-ft")
    ap.add_argument("--pairs-per-title", type=int, default=40)
    ap.add_argument("--noisy-frac", type=float, default=0.7)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bg-negatives", action="store_true",
                    help="add a hard non-Marvel negative per pair (needs build_background.py)")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers
    torch.manual_seed(a.seed)

    cfg = bi.load_config(a.config)
    pairs = make_pairs(cfg, a.pairs_per_title, a.noisy_frac, a.seed)
    print(f"{len(pairs)} train pairs, e.g. {pairs[0][0]!r} -> {pairs[0][1][:60]!r}...")
    cols = {"anchor": [p[0] for p in pairs], "positive": [p[1] for p in pairs]}
    if a.bg_negatives:
        trip = add_background_negatives(cfg, pairs, a.seed)
        cols["negative"] = [t[2] for t in trip]
        print(f"  + background negative, e.g. {trip[0][2][:60]!r}...")
    model = SentenceTransformer(cfg["model"])
    model.max_seq_length = 128
    ds = Dataset.from_dict(cols)
    out = ROOT / a.out
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "_checkpoints"), num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.batch, learning_rate=a.lr, warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES, logging_steps=20, save_strategy="no",
        seed=a.seed, report_to="none", dataloader_drop_last=True)
    t0 = time.time()
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                               loss=losses.MultipleNegativesRankingLoss(model)).train()
    model.save(str(out))
    print(f"saved {out} after {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
