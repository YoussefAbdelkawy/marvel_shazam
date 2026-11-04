"""
evaluate_stream.py  -  "keep listening" policy: answer, say "no match", or wait for more audio

    python evaluate_stream.py                     # dev: tune the early-accept bar, report
    python evaluate_stream.py --split test --accept 0.9   # final report only

A query arrives as consecutive Whisper windows of WINDOW words (each noised separately). After
each window the transcript so far is searched. If the configured confidence clears the stricter
early-accept bar, answer; after the last window fall back to the normal threshold. Compared with
deciding on the first window only, and with always waiting for all windows.
Positives come from the split's dev/test blocks; negatives from the Cornell movies already
assigned to that split in eval/<split>.parquet. Built deterministically into eval_stream/.
"""
import argparse, json, random
from pathlib import Path

import numpy as np
import pandas as pd

import build_index as bi
import prepare_eval as pe

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eval_stream"
WINDOW, N_WINDOWS, N_SEQ = 10, 3, 300
SEED = 4321


def make_sequences(split):
    p = OUT / f"{split}.json"
    if p.exists():
        return json.loads(p.read_text())
    rng = random.Random(SEED + (split == "test"))
    cfg = bi.load_config()
    docs, _, _ = bi.load_docs(cfg, verbose=False)
    splits = json.loads((ROOT / "eval" / "splits.json").read_text())["files"]
    span = WINDOW * N_WINDOWS
    seqs = []
    files = [d for d in docs if d["source_file"] in splits and splits[d["source_file"]][split][1]
             - splits[d["source_file"]][split][0] >= span]
    rng.shuffle(files)
    for d in files[:N_SEQ]:
        words = d["text"].split()
        a, b = splits[d["source_file"]][split]
        s = rng.randint(a, b - span)
        seqs.append({"true": splits[d["source_file"]]["title"], "neg": False, "windows": [
            pe.whisperify(" ".join(words[s + i * WINDOW:s + (i + 1) * WINDOW]), rng)
            for i in range(N_WINDOWS)]})
    q = pd.read_parquet(ROOT / "eval" / f"{split}.parquet")
    mids = sorted({f.split(":")[1] for f in q[q.is_negative].source_file})
    movies, _, _ = pe.cornell_movies()
    rng.shuffle(mids)
    for m in (mids * 2)[:N_SEQ]:
        words = movies[m].split()
        s = rng.randint(0, len(words) - span)
        seqs.append({"true": bi.NO_MATCH, "neg": True, "windows": [
            pe.whisperify(" ".join(words[s + i * WINDOW:s + (i + 1) * WINDOW]), rng)
            for i in range(N_WINDOWS)]})
    OUT.mkdir(exist_ok=True)
    p.write_text(json.dumps(seqs, indent=0))
    return seqs


def run_policy(steps, seqs, accept):
    """steps[i][j] = decide() result for sequence j after i+1 windows."""
    correct, fa, used = [], [], []
    for j, sq in enumerate(seqs):
        ans, n = bi.NO_MATCH, N_WINDOWS
        for i in range(N_WINDOWS):
            r = steps[i][j]
            top = r["ranking"][0][0] if r["ranking"] else bi.NO_MATCH
            if i < N_WINDOWS - 1 and r["confidence"] >= accept:
                ans, n = top, i + 1
                break
            if i == N_WINDOWS - 1:
                ans = r["answer"]
        used.append(n)
        (fa if sq["neg"] else correct).append(ans != bi.NO_MATCH if sq["neg"] else ans == sq["true"])
    acc, far = float(np.mean(correct)), float(np.mean(fa))
    return {"top1": acc, "false_accept": far, "score": acc - 0.5 * far,
            "avg_windows": float(np.mean(used))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--config", default=str(bi.CONFIG_PATH))
    ap.add_argument("--accept", type=float, help="early-accept bar (default: tuned on dev)")
    a = ap.parse_args()
    cfg = bi.load_config(a.config)
    seqs = make_sequences(a.split)
    sr = bi.Searcher(cfg)
    steps = [sr.decide(sr.candidates([" ".join(s["windows"][:i + 1]) for s in seqs]))
             for i in range(N_WINDOWS)]
    out = {}
    for i in range(N_WINDOWS):                      # fixed-length baselines
        pos = [r["answer"] == s["true"] for r, s in zip(steps[i], seqs) if not s["neg"]]
        neg = [r["answer"] != bi.NO_MATCH for r, s in zip(steps[i], seqs) if s["neg"]]
        out[f"always {i + 1} window(s)"] = {"top1": np.mean(pos), "false_accept": np.mean(neg),
                                            "score": np.mean(pos) - 0.5 * np.mean(neg),
                                            "avg_windows": i + 1}
    if a.accept is None:
        grid = sorted({round(float(x), 4) for r in steps[0] for x in [r["confidence"]]
                       if np.isfinite(x)})
        grid = np.quantile(grid, np.linspace(0, 1, 101)) if grid else [np.inf]
        accept = max(grid, key=lambda t: run_policy(steps, seqs, t)["score"])
    else:
        accept = a.accept
    out[f"keep listening (accept >= {accept:.4g})"] = run_policy(steps, seqs, accept)
    print(f"\n== stream {a.split}: {sum(not s['neg'] for s in seqs)} Marvel / "
          f"{sum(s['neg'] for s in seqs)} other sequences of {N_WINDOWS} x {WINDOW} noisy words")
    for k, m in out.items():
        print(f"  {k:<34} top1={m['top1']:.3f}  false-accept={m['false_accept']:.3f}  "
              f"score={m['score']:.4f}  windows={m['avg_windows']:.2f}")
    (OUT / f"summary_{a.split}.json").write_text(json.dumps(
        {"accept": float(accept), "results": out}, indent=1, default=float))
    print(f"STREAM_SCORE={out[f'keep listening (accept >= {accept:.4g})']['score']:.4f}")


if __name__ == "__main__":
    main()
