"""
make_results.py  -  write RESULTS.md from eval/summary_<run>.json and eval_stream/summary_*.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [  # (run, label, kept, note)
    ("step0_baseline", "0. Baseline (60-word chunks, linear voting, no threshold)", "-", ""),
    ("step0_baseline", "1. Data check (report only, see DATA_CHECK.md)", "-", "no pipeline change"),
    ("step2_multiscale", "2. Multi-scale chunks 30+60 words", "yes", ""),
    ("step3a_simple", "3. Sharpened voting exp(sim/T) + share threshold", "yes", "T=0.02"),
    ("step3b_calibrated", "3+. Calibrated confidence (11 features) + per-length thresholds [#1 #2 #3 #5]", "yes", ""),
    ("step7_exclude", "+. Exclude 44 wrong-show subtitle files [#7]", "no", "see note 2"),
    ("step4_background", "+. Background non-Marvel index as feature [#4]", "no", "OOF SCORE lower"),
    ("step4_hybrid", "4. Hybrid BM25 + dense (RRF), recalibrated", "yes", ""),
    ("step6_finetune", "6. Fine-tuned bge-small, background hard negatives [#8]", "yes", "9.9k train triplets"),
    ("step5_rerank", "5. Cross-encoder reranker bge-reranker-base, on top of 6 [#6]", "yes", "depth 20, 128 tokens"),
]


def load(run):
    p = ROOT / "eval" / f"summary_{run}.json"
    return json.loads(p.read_text()) if p.exists() else None


def row(label, s, kept, note):
    m = s["metrics"]
    return (f"| {label} | **{m['score']:.4f}** | {m['noisy']['top1']:.3f} | {m['clean']['top1']:.3f} "
            f"| {m['overall']['top5']:.3f} | {m['overall']['mrr']:.3f} | {m['neg_rejection']:.3f} "
            f"| {m['pos_false_reject']:.3f} | {kept} | {note} |")


def main(final_run="final_test", kept_runs=None):
    kept_runs = kept_runs or {}
    out = ["# RESULTS\n",
           "SCORE = top-1 on noisy positive dev queries - 0.5 x false-accept rate on "
           "non-Marvel negatives. top-1 counts \"no match\" as wrong; top-5/MRR are before the "
           "threshold. Each step builds on the last kept one.\n",
           "| step | dev SCORE | noisy top-1 | clean top-1 | top-5 | MRR | neg. rejected "
           "| pos. false-reject | kept | note |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for run, label, kept, note in STEPS:
        s = load(run)
        if s:
            out.append(row(label, s, kept if kept else kept_runs.get(run, "?"), note))
    t = load(final_run)
    if t:
        out += ["\n## Test split (final configuration, run once)\n",
                "| | SCORE | noisy top-1 | clean top-1 | top-5 | MRR | neg. rejected "
                "| pos. false-reject | | |", "|---|---|---|---|---|---|---|---|---|---|",
                row("final config on TEST", t, "", "")]
        b = load("final_test_baseline")
        if b:
            out.append(row("baseline on TEST (for comparison, untuned)", b, "", ""))
    for split in ("dev", "test"):
        p = ROOT / "eval_stream" / f"summary_{split}.json"
        if p.exists():
            s = json.loads(p.read_text())
            out += [f"\n## Keep-listening policy [#9], stream {split} (300 Marvel + 300 other "
                    "sequences of 3 x 10 noisy words)\n",
                    "| policy | top-1 | false-accept | score | avg windows |", "|---|---|---|---|---|"]
            out += [f"| {k} | {m['top1']:.3f} | {m['false_accept']:.3f} | {m['score']:.4f} "
                    f"| {m['avg_windows']:.2f} |" for k, m in s["results"].items()]
    notes = ROOT / "RESULTS_notes.md"
    if notes.exists():
        out += ["", notes.read_text(encoding="utf-8")]
    (ROOT / "RESULTS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    import sys
    main(kept_runs=json.loads(sys.argv[1]) if len(sys.argv) > 1 else None)
