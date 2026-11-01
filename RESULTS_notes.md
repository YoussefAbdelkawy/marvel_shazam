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
