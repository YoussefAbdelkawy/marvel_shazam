# RESULTS

SCORE = top-1 on noisy positive dev queries - 0.5 x false-accept rate on non-Marvel negatives. top-1 counts "no match" as wrong; top-5/MRR are before the threshold. Each step builds on the last kept one.

| step | dev SCORE | noisy top-1 | clean top-1 | top-5 | MRR | neg. rejected | pos. false-reject | kept | note |
|---|---|---|---|---|---|---|---|---|---|
| 0. Baseline (60-word chunks, linear voting, no threshold) | **-0.0333** | 0.467 | 0.561 | 0.722 | 0.606 | 0.000 | 0.000 | - |  |
| 1. Data check (report only, see DATA_CHECK.md) | **-0.0333** | 0.467 | 0.561 | 0.722 | 0.606 | 0.000 | 0.000 | - | no pipeline change |
| 2. Multi-scale chunks 30+60 words | **0.1378** | 0.638 | 0.754 | 0.859 | 0.766 | 0.000 | 0.000 | yes |  |
| 3. Sharpened voting exp(sim/T) + share threshold | **0.6203** | 0.658 | 0.809 | 0.861 | 0.824 | 0.925 | 0.236 | yes | T=0.02 |
| 3+. Calibrated confidence (11 features) + per-length thresholds [#1 #2 #3 #5] | **0.7003** | 0.704 | 0.859 | 0.861 | 0.824 | 0.992 | 0.207 | yes |  |
| +. Exclude 44 wrong-show subtitle files [#7] | **0.6489** | 0.659 | 0.798 | 0.805 | 0.767 | 0.980 | 0.258 | no | see note 2 |
| +. Background non-Marvel index as feature [#4] | **0.7006** | 0.706 | 0.859 | 0.861 | 0.824 | 0.990 | 0.206 | no | OOF SCORE lower |
| 4. Hybrid BM25 + dense (RRF), recalibrated | **0.8906** | 0.902 | 0.962 | 0.990 | 0.960 | 0.977 | 0.050 | yes |  |
| 6. Fine-tuned bge-small, background hard negatives [#8] | **0.9261** | 0.934 | 0.982 | 0.991 | 0.976 | 0.983 | 0.031 | yes | 9.9k train triplets |
| 5. Cross-encoder reranker bge-reranker-base, on top of 6 [#6] | **0.9378** | 0.948 | 0.987 | 0.994 | 0.983 | 0.980 | 0.022 | yes | depth 20, 128 tokens |

## Test split (final configuration, run once)

| | SCORE | noisy top-1 | clean top-1 | top-5 | MRR | neg. rejected | pos. false-reject | | |
|---|---|---|---|---|---|---|---|---|---|
| final config on TEST | **0.9361** | 0.948 | 0.987 | 0.994 | 0.981 | 0.977 | 0.023 |  |  |
| baseline on TEST (for comparison, untuned) | **-0.0333** | 0.467 | 0.557 | 0.742 | 0.612 | 0.000 | 0.000 |  |  |

## Keep-listening policy [#9], stream dev (300 Marvel + 300 other sequences of 3 x 10 noisy words)

| policy | top-1 | false-accept | score | avg windows |
|---|---|---|---|---|
| always 1 window(s) | 0.927 | 0.023 | 0.9150 | 1.00 |
| always 2 window(s) | 0.977 | 0.000 | 0.9767 | 2.00 |
| always 3 window(s) | 0.997 | 0.003 | 0.9950 | 3.00 |
| keep listening (accept >= 0.7459) | 0.997 | 0.003 | 0.9950 | 2.08 |

## Keep-listening policy [#9], stream test (300 Marvel + 300 other sequences of 3 x 10 noisy words)

| policy | top-1 | false-accept | score | avg windows |
|---|---|---|---|---|
| always 1 window(s) | 0.923 | 0.047 | 0.9000 | 1.00 |
| always 2 window(s) | 0.983 | 0.003 | 0.9817 | 2.00 |
| always 3 window(s) | 0.983 | 0.000 | 0.9833 | 3.00 |
| keep listening (accept >= 0.7459) | 0.980 | 0.017 | 0.9717 | 2.05 |

## Final configuration (`config.json`)

- Index: fine-tuned `models/bge-small-marvel-ft`, chunks of 30 words (stride 15) + 60 words (stride 30), 291,310 chunks.
- Retrieval: dense top-50 + BM25 top-50 fused by Reciprocal Rank Fusion, then `BAAI/bge-reranker-base` rescores the top 20.
- Voting: the 20 best chunks vote with exp(score / T) x vote_weight, T tuned on dev.
- "No match": a logistic-regression calibrator turns 13 confidence features into a probability. Strongest features: bigram overlap with the best chunk, vote margin, BM25 score per word, sim z-score, query length. There is one threshold per query-length bucket (short <= 11 words, mid <= 25, long).

## Notes

1. **Tuned on dev, scored on dev.** T, thresholds and the calibrator are fitted on dev, so the dev SCOREs are optimistic. The calibrator rows also report a 5-fold out-of-fold estimate (folds grouped so the clean and noisy copies of a window stay together); it matched the in-sample number within 0.003 every time. The test row is the honest number.
2. **#7, excluding the wrong-show files.** The frozen evaluator still asks queries whose "answer" is one of the 44 excluded files, so those queries become unanswerable and the frozen SCORE drops to 0.649. On queries that don't involve those files there is still no gain (0.6951 vs 0.6962). The option `exclude_files` stays in the code, off. The real fix is to re-download those subtitles and then regenerate `eval/`, which needs your OK.
3. **#4, background index.** 60k chunks of Star Trek / Doctor Who / Friends / The Office / Knight Rider transcripts (sedthh/tv_dialogue). The movie-script parts were skipped because they overlap the Cornell negatives. As a calibrator feature it added nothing; phrase overlap already carries that signal. It is still used as the hard-negative pool for fine-tuning (#8).
4. **CPU compromises.** The reranker uses depth 20 and 128 tokens instead of depth 50; depth 50 was about 7 h per evaluation on this laptop. Fine-tuning used 9.9k triplets (14 per title, 1 epoch, 43 min) instead of 28k (about 3.4 h). Re-embedding the index with the fine-tuned model took 51 min.
5. **Fine-tuning leakage check.** Anchors and positive chunks come only from the train part of each title; a positive chunk must lie entirely outside both the dev and the test block. Negatives are background TV transcripts, never Cornell.
6. **Latency.** The reranker adds about 20 cross-encoder pairs per query (~1 s on this CPU). Without it the fine-tuned hybrid setup scores dev 0.926 and needs no cross-encoder.
7. **Cleaning bug not fixed.** `MUSIC_RE` treats `#` as a music marker, so "Agent #2:" deletes text up to the next `#` (S.H.I.E.L.D. S03E18 keeps 12% of its words). Fixing it changes the cleaned text, so the frozen splits would have to be regenerated.
8. **#9 is a separate benchmark.** The keep-listening policy is evaluated on its own sequences in `eval_stream/` (3 consecutive, separately noised 10-word windows), because the frozen query set has no consecutive windows. The early-accept bar is tuned on stream-dev and applied unchanged to stream-test.

