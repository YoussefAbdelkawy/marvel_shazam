"""
tune.py  -  tune voting, confidence and the "no match" threshold on the DEV split

    python tune.py --config config.json            # print the best settings
    python tune.py --config config.json --write    # ...and store them in that config's "search"
    python tune.py ... --simple                    # single-feature thresholds only (no calibrator)

1. scoring / T: every combination re-voted from candidates retrieved once
2. single-feature confidence + one threshold (the plain step-3 option)
3. calibrator: logistic regression over all confidence features ("is the top answer right?"),
   5-fold cross-validated, grouped so the clean and noisy copy of a window share a fold;
   then one threshold, or one per query-length bucket, chosen on the out-of-fold probabilities.
Objective everywhere = evaluate.py SCORE (top-1 noisy positives - 0.5 * negative false-accept).
"""
import argparse, json

import numpy as np
import pandas as pd

import build_index as bi
from evaluate import EVAL

TEMPS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
SINGLE = ["top_score", "top_sim", "share", "margin", "sim_z", "bg_gap", "rerank_top"]


def sweep(conf, correct, noisy_pos, neg, n_noisy=None, n_neg=None):
    """Best threshold for SCORE; n_* let a bucket be scored as its share of the whole split."""
    n_noisy = n_noisy or max(noisy_pos.sum(), 1)
    n_neg = n_neg or max(neg.sum(), 1)
    vals = np.unique(np.concatenate([conf[noisy_pos], conf[neg]]))
    cands = np.concatenate([[-np.inf], (vals[:-1] + vals[1:]) / 2, [np.inf]])
    acc_v, neg_v = np.sort(conf[noisy_pos & correct]), np.sort(conf[neg])
    acc = (len(acc_v) - np.searchsorted(acc_v, cands, "left")) / n_noisy
    fa = (len(neg_v) - np.searchsorted(neg_v, cands, "left")) / n_neg
    score = acc - 0.5 * fa
    i = int(np.argmax(score))
    t = None if np.isneginf(cands[i]) else float(cands[i])
    return t, float(score[i]), float(acc[i]), float(fa[i])


def fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    mean, scale = X.mean(0), X.std(0) + 1e-9
    lr = LogisticRegression(C=1.0, max_iter=2000).fit((X - mean) / scale, y)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coef": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0])}


def predict(cal, X):
    z = ((X - np.array(cal["mean"])) / np.array(cal["scale"])) @ np.array(cal["coef"])
    return 1 / (1 + np.exp(-(z + cal["intercept"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(bi.CONFIG_PATH))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--simple", action="store_true")
    ap.add_argument("--scorings", default="linear,exp")
    a = ap.parse_args()
    cfg = bi.load_config(a.config)
    q = pd.read_parquet(EVAL / "dev.parquet")
    sr = bi.Searcher(cfg)
    cands = sr.candidates(q["query"].tolist())
    noisy_pos = (~q.is_negative & q.noisy).to_numpy()
    neg = q.is_negative.to_numpy()
    truth = q.true_title.to_numpy()
    buckets = np.array([bi.length_bucket(len(bi.norm(t))) for t in q["query"]])
    groups = pd.factorize(q.source_file + ":" + q.word_start.astype(str))[0]

    # 1 + 2: scoring, T, single-feature threshold
    rows, kept = [], {}
    for scoring in a.scorings.split(","):
        for T in (TEMPS if scoring == "exp" else [cfg["search"]["temperature"]]):
            res = sr.decide(cands, scoring=scoring, temperature=T, threshold=None,
                            threshold_by_length=None, calibrator=None)
            top = np.array([r["ranking"][0][0] if r["ranking"] else "" for r in res])
            correct = top == truth
            kept[(scoring, T)] = (res, correct)
            for c in SINGLE:
                if c not in res[0]["conf"]:
                    continue
                conf = np.array([r["conf"].get(c, -np.inf) for r in res])
                t, score, acc, fa = sweep(conf, correct, noisy_pos, neg)
                rows.append({"scoring": scoring, "temperature": T, "confidence": c,
                             "threshold": t, "SCORE": score, "noisy_top1": acc,
                             "false_accept": fa, "noisy_top1_raw": correct[noisy_pos].mean()})
    df = pd.DataFrame(rows).sort_values("SCORE", ascending=False)
    pd.set_option("display.width", 170)
    print(df.head(10).to_string(index=False))
    b = df.iloc[0]
    best = {"scoring": b.scoring, "temperature": float(b.temperature), "confidence": b.confidence,
            "threshold": b.threshold, "threshold_by_length": None, "calibrator": None}
    print(f"\nsingle-feature best: SCORE={b.SCORE:.4f}  {best}")
    best_score = b.SCORE

    # 3: calibrator over all features, cross-validated
    if not a.simple:
        from sklearn.model_selection import GroupKFold
        res, correct = kept[(b.scoring, float(b.temperature))]
        names = sorted(res[0]["conf"])
        X = np.array([[r["conf"][n] for n in names] for r in res])
        y = correct.astype(int)
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, groups):
            oof[te] = predict(fit_lr(X[tr], y[tr]), X[te])
        t, s_glob, acc, fa = sweep(oof, correct, noisy_pos, neg)
        print(f"calibrator, one threshold (out-of-fold): SCORE={s_glob:.4f} "
              f"top1={acc:.3f} FA={fa:.3f} t={t}")
        by_len, s_len = {}, 0.0
        for bk in ("short", "mid", "long"):
            m = buckets == bk
            tb, sb, _, _ = sweep(oof, correct, noisy_pos & m, neg & m,
                                 n_noisy=noisy_pos.sum(), n_neg=neg.sum())
            by_len[bk], s_len = (tb if tb is not None else -1.0), s_len + sb
        print(f"calibrator, per-length thresholds (out-of-fold): SCORE={s_len:.4f} {by_len}")
        cal = {"features": names, **fit_lr(X, y)}
        print("calibrator weights:", {n: round(w, 2) for n, w in zip(names, cal["coef"])})
        if max(s_glob, s_len) > best_score:
            best.update(confidence="calibrated", calibrator=cal, threshold=t,
                        threshold_by_length=by_len if s_len > s_glob else None)
            best_score = max(s_glob, s_len)
    print(f"\nchosen: confidence={best['confidence']} threshold={best['threshold']} "
          f"by_length={best['threshold_by_length']}  (dev SCORE estimate {best_score:.4f})")
    if a.write:
        raw = json.loads(open(a.config).read())
        raw["search"] = {**cfg["search"], **best}
        open(a.config, "w").write(json.dumps(raw, indent=2, default=float))
        print(f"written to {a.config}")


if __name__ == "__main__":
    main()
