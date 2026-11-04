"""
data_check.py  -  report suspicious subtitle files (reads only; never deletes or downloads)

    python data_check.py [--results eval/results_step0_baseline.csv]   -> DATA_CHECK.md
"""
import argparse, hashlib, json, random
from collections import Counter
from pathlib import Path

import pandas as pd

import build_index as bi

ROOT = Path(__file__).resolve().parent
MIN_WORDS = {"movie": 3000, "tv": 1500}
PROBES_PER_TITLE, PROBE_LEN = 5, 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval/results_step0_baseline.csv")
    a = ap.parse_args()
    cfg = bi.load_config()
    docs, _, _ = bi.load_docs(cfg, verbose=False)
    splits = json.loads((ROOT / "eval" / "splits.json").read_text())["files"]
    out = ["# Data check\n"]

    # 1. too few words
    short = []
    for d in docs:
        raw = (ROOT / "subs" / d["source_file"]).read_text(encoding="utf-8", errors="ignore")
        n = len(d["text"].split())
        if n < MIN_WORDS[d["type"]]:
            short.append((d["type"], n, len(raw.split()), d["source_file"]))
    short.sort()
    out.append(f"## Files with suspiciously few words ({len(short)})\n")
    out.append(f"Thresholds on cleaned dialogue: movies < {MIN_WORDS['movie']}, "
               f"episodes < {MIN_WORDS['tv']}.\n")
    out.append("| type | cleaned words | raw words | file |\n|---|---|---|---|")
    out += [f"| {t} | {n} | {r} | {f} |" for t, n, r, f in short]

    # 2. identical files
    by_hash = {}
    for d in docs:
        by_hash.setdefault(d["file_hash"], []).append(d["source_file"])
    dups = [v for v in by_hash.values() if len(v) > 1]
    out.append(f"\n## Byte-identical files ({len(dups)} groups)\n")
    out += [f"- {', '.join(v)}" for v in dups] or ["- none"]

    # 3. zero top-1 on the dev queries
    res = pd.read_csv(ROOT / a.results)
    pos = res[~res.is_negative]
    per = pos.groupby("true_title").correct.agg(["size", "mean"])
    zero = per[per["mean"] == 0].sort_values("size", ascending=False)
    out.append(f"\n## Titles with 0% top-1 on dev queries ({len(zero)} of {len(per)})\n")
    out.append("Only ~2-4 dev queries per title (clean+noisy), so this list is noisy on its own; "
               "see the probe below.\n")
    out.append(", ".join(f"{t} (n={int(r['size'])})" for t, r in zero.iterrows()))

    # 4. probe: 5 clean 20-word windows from every title's dev block
    rng = random.Random(99)
    queries, truth = [], []
    for d in docs:
        if d["source_file"] not in splits:          # added after the benchmark was frozen
            continue
        words = d["text"].split()
        a0, b0 = splits[d["source_file"]]["dev"]
        if b0 - a0 < PROBE_LEN:
            continue
        for _ in range(PROBES_PER_TITLE):
            s = rng.randint(a0, b0 - PROBE_LEN)
            queries.append(" ".join(words[s:s + PROBE_LEN]))
            truth.append(splits[d["source_file"]]["title"])
    sr = bi.Searcher(cfg)
    answers = [r["ranking"][0][0] for r in sr.search_batch(queries)]
    probe = pd.DataFrame({"title": truth, "pred": answers})
    probe["ok"] = probe.title == probe.pred
    agg = probe.groupby("title").ok.mean()
    bad = agg[agg == 0].index
    out.append(f"\n## Probe: {PROBES_PER_TITLE} clean {PROBE_LEN}-word dev windows per title\n")
    out.append(f"Overall probe top-1: {probe.ok.mean():.3f}. "
               f"Titles at 0/{PROBES_PER_TITLE}: {len(bad)}; at <=1/{PROBES_PER_TITLE}: "
               f"{(agg <= 1 / PROBES_PER_TITLE + 1e-9).sum()}.\n")
    out.append("| title | most common wrong answer (count) |\n|---|---|")
    for t in bad:
        wrong = Counter(probe[(probe.title == t)].pred).most_common(1)[0]
        out.append(f"| {t} | {wrong[0]} ({wrong[1]}) |")

    # 5. confusion pairs: a title answered as one specific other title >= 3 of 5 times
    out.append("\n## Strong confusions (>= 3 of 5 probes go to the same wrong title)\n")
    out.append("Usually a same-show neighbour episode; could also mean a mislabeled or "
               "duplicated subtitle file.\n")
    rows = []
    for t, g in probe[~probe.ok].groupby("title"):
        p, c = Counter(g.pred).most_common(1)[0]
        if c >= 3:
            rows.append(f"- {t} -> {p} ({c}/5)")
    out += rows or ["- none"]

    # 6. words lost in cleaning (a regex eating real dialogue)
    out.append("\n## Files losing > 30% of their words in clean_text\n")
    out.append("| kept | raw words | cleaned words | file |\n|---|---|---|---|")
    for d in docs:
        raw = (ROOT / "subs" / d["source_file"]).read_text(encoding="utf-8", errors="ignore")
        a0, b0 = len(raw.split()), len(bi.clean_text(raw).split())
        if a0 and b0 / a0 < 0.7:
            out.append(f"| {b0 / a0:.0%} | {a0} | {b0} | {d['source_file']} |")

    # 7. episodes whose dialogue doesn't look like the rest of their show (wrong show downloaded)
    import numpy as np
    vec = sr.index.reconstruct_n(0, sr.index.ntotal)
    df = sr.df
    cent = {f: bi.normalize(vec[g.index].mean(0, keepdims=True))[0]
            for f, g in df.groupby("source_file")}
    meta = df.drop_duplicates("source_file").set_index("source_file")
    rows = []
    for show, g in meta[meta.type == "tv"].groupby("title_key"):
        files = list(g.index)
        if len(files) < 4:
            continue
        C = np.stack([cent[f] for f in files])
        sims = C @ bi.normalize(np.median(C, 0, keepdims=True))[0]
        med, mad = np.median(sims), np.median(np.abs(sims - np.median(sims))) + 1e-6
        for f, s in zip(files, sims):
            if (s - med) / mad < -6:
                rows.append((show, round(float(s), 3), round(float(med), 3), f))
    out.append(f"\n## Episodes that don't match their show ({len(rows)})\n")
    out.append("Cosine of the episode's mean embedding to the show's median, vs the show's "
               "typical value. Very low = probably a different show's subtitles.\n")
    out.append("| show | episode sim | show median | file | first words |\n|---|---|---|---|---|")
    for show, s, med, f in sorted(rows):
        head = " ".join((ROOT / "subs" / f).read_text(encoding="utf-8", errors="ignore")
                        .split()[:10]).replace("|", "/")
        out.append(f"| {show} | {s} | {med} | {f} | {head} |")
    print(f"cleaning-loss and show-outlier sections: {len(rows)} outlier episodes")

    # 8. episodes per season, to compare against the real episode counts by eye
    eps = meta[meta.type == "tv"].groupby(["title", "season"]).episode.agg(["count", "max"])
    out.append("\n## Episodes per season (files present / highest episode number)\n")
    out.append("| show | season | files | max ep |\n|---|---|---|---|")
    out += [f"| {t} | {int(s)} | {r['count']} | {int(r['max'])} |" for (t, s), r in eps.iterrows()]
    (ROOT / "DATA_CHECK.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:3]))
    print(f"short files: {len(short)}, dup groups: {len(dups)}, zero-dev titles: {len(zero)}, "
          f"probe top1: {probe.ok.mean():.3f}, probe-zero titles: {len(bad)}, "
          f"strong confusions: {len(rows)}")


if __name__ == "__main__":
    main()
