"""
evaluate.py  -  frozen evaluator for marvel_shazam (do not edit after Phase 1)

    python evaluate.py --run baseline                 # dev split, config.json
    python evaluate.py --run x --config other.json    # any config
    python evaluate.py --split test --run final       # test: only for the final report

Uses build_index.Searcher, i.e. exactly the search the pipeline runs.
top-1 counts "no match" answers as wrong; top-5 and MRR are measured on the ranking before the
threshold. The last printed line is always  SCORE=<value>  where
    SCORE = top-1 on noisy positive queries - 0.5 * false-accept rate on negatives.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd

import build_index as bi

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "eval"


def run(cfg, split):
    q = pd.read_parquet(EVAL / f"{split}.parquet")
    t0 = time.time()
    searcher = bi.Searcher(cfg)
    res = searcher.search_batch(q["query"].tolist())
    secs = time.time() - t0
    rows = []
    for (_, r), out in zip(q.iterrows(), res):
        labels = [lab for lab, _ in out["ranking"]]
        rank = labels.index(r.true_title) + 1 if r.true_title in labels else 0
        rows.append({**r.to_dict(), "answer": out["answer"],
                     "top5": " | ".join(labels[:5]), "true_rank": rank,
                     "confidence": out["confidence"],
                     "correct": out["answer"] == r.true_title})
    return pd.DataFrame(rows), secs


def metrics(df):
    pos, neg = df[~df.is_negative], df[df.is_negative]
    rank = pos.true_rank

    def block(p):
        r = p.true_rank
        return {"n": len(p), "top1": p.correct.mean(), "top5": ((r > 0) & (r <= 5)).mean(),
                "mrr": np.where(r > 0, 1.0 / r.where(r > 0, 1), 0.0).mean()}

    m = {"overall": block(pos),
         "by_length_noise": {f"{l}w_{'noisy' if n else 'clean'}": block(g)
                             for (l, n), g in pos.groupby(["length", "noisy"])},
         "by_length": {f"{l}w": block(g) for l, g in pos.groupby("length")},
         "clean": block(pos[~pos.noisy]), "noisy": block(pos[pos.noisy]),
         "neg_rejection": (neg.answer == bi.NO_MATCH).mean() if len(neg) else float("nan"),
         "pos_false_reject": (pos.answer == bi.NO_MATCH).mean(),
         "top1_raw": (rank == 1).mean()}
    m["neg_false_accept"] = 1.0 - m["neg_rejection"]
    m["score"] = m["noisy"]["top1"] - 0.5 * m["neg_false_accept"]
    worst = (pos.groupby("true_title")
             .agg(n=("correct", "size"), top1=("correct", "mean"))
             .sort_values(["top1", "n"], ascending=[True, False]).head(20))
    return m, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--config", default=str(bi.CONFIG_PATH))
    ap.add_argument("--run", default="latest")
    a = ap.parse_args()
    cfg = bi.load_config(a.config)
    if a.split == "test":
        print("!! TEST SPLIT: use only for the final report row.")
    df, secs = run(cfg, a.split)
    m, worst = metrics(df)
    df.to_csv(EVAL / f"results_{a.run}.csv", index=False)
    (EVAL / f"summary_{a.run}.json").write_text(json.dumps(
        {"split": a.split, "config": a.config, "index": str(bi.index_dir(cfg).name),
         "search": cfg["search"], "seconds": secs, "metrics": m,
         "worst_titles": worst.reset_index().to_dict("records")}, indent=1, default=float))

    f = lambda b: f"top1={b['top1']:.3f}  top5={b['top5']:.3f}  mrr={b['mrr']:.3f}  (n={b['n']})"
    print(f"\n== {a.split} | index {bi.index_dir(cfg).name} | {secs:.0f}s for {len(df)} queries")
    print(f"overall   {f(m['overall'])}")
    print(f"clean     {f(m['clean'])}")
    print(f"noisy     {f(m['noisy'])}")
    for k, b in m["by_length_noise"].items():
        print(f"  {k:<10}{f(b)}")
    print(f"top-1 before threshold: {m['top1_raw']:.3f}")
    print(f"negatives: rejection={m['neg_rejection']:.3f}  false-accept={m['neg_false_accept']:.3f}"
          f"  | positives false-reject={m['pos_false_reject']:.3f}"
          f"  (threshold={cfg['search']['threshold']}, confidence={cfg['search']['confidence']})")
    print("\n20 worst titles (top-1 over their positive queries):")
    for t, r in worst.iterrows():
        print(f"  {r.top1:.2f}  n={r.n:<3} {t}")
    print(f"\nresults -> eval/results_{a.run}.csv")
    print(f"SCORE={m['score']:.4f}")


if __name__ == "__main__":
    main()
