# Marvel Shazam for Android: full spec

Status: draft, 2026-11-01. Owner: Youssef. Source of truth for the search behaviour: `build_index.py`
(`Searcher`) and `configs/step6_finetune.json` in this repo.

---

## 1. Goal

One installable Android APK that identifies which Marvel movie or episode is playing from about
5-15 seconds of dialogue, **fully offline**: speech recognition, the embedding model, the
vector/keyword index and the confidence model all ship inside the app. It looks and behaves like
the desktop app (Captain America shield button, spinning while listening, top-3 reveal).

### Success criteria

| # | Criterion | Target |
|---|---|---|
| S1 | Accuracy on the frozen **dev** set, measured in Python with the exact mobile models and data (section 9) | SCORE >= 0.90 (desktop without reranker: 0.926) |
| S2 | Kotlin result identical to the Python mobile simulator on 500 golden queries | top-1 identical for >= 99.5%; top-3 identical >= 98% |
| S3 | Accuracy on the **test** set, run once at the end | reported next to desktop (0.936) |
| S4 | Time from tap to answer on the reference phone (section 3) | <= 12 s typical (2 windows of 5 s + <= 2 s processing) |
| S5 | Processing per window (ASR + search) | <= 2.5 s |
| S6 | Cold start until the shield is ready | <= 8 s |
| S7 | Peak memory (PSS) | <= 1.2 GB |
| S8 | APK size | <= 450 MB (sideload) |
| S9 | Works with flight mode on | yes |

### Non-goals (v1)

- Play Store release (see 11.3 for what it would need).
- iOS.
- Adding franchises, or fixing the subtitle data (MISSING.md); the phone ships whatever index is exported.
- The cross-encoder reranker (too big and slow for a phone; costs about 0.012 SCORE).
- Training on the phone. All learning happens on the desktop and is exported.

---

## 2. What exists today (desktop)

| Stage | Desktop implementation | Notes |
|---|---|---|
| Audio | `sounddevice`, 16 kHz mono float32 | `app.py` |
| ASR | faster-whisper `small.en`, int8, VAD on | ~1.3 s per 10 s clip on the laptop |
| Query embedding | fine-tuned `models/bge-small-marvel-ft` (33M params, 384-dim, CLS pooling, L2-normalised) | 128 MB fp32 |
| Dense search | FAISS `IndexFlatIP`, 291,310 vectors x 384, exact | 427 MB fp32 |
| Keyword search | BM25Okapi (k1 1.5, b 0.75, eps 0.25) as a sparse matrix, 41,610 terms, 9.69M non-zeros | built at startup |
| Fusion | RRF (k=60) of dense top-50 and BM25 top-50 | |
| Reranker | bge-reranker-base, top 20 | **dropped on mobile** |
| Voting | 20 chunks, vote = exp((s - s_max)/T) x vote_weight | |
| Confidence | logistic regression over 12 features (13 with the reranker), per-length thresholds | `search.calibrator` |
| Policy | 5 s windows; answer at 2 windows, after 1 if p >= 0.95, else at 3 | `app.py` |
| UI | `app/static/index.html` in pywebview | |

Measured quality (RESULTS.md): dev SCORE 0.926 without reranker, 0.938 with; test 0.936 with.

---

## 3. Target devices

- **ABI**: arm64-v8a only (no 32-bit, no x86; the emulator image must be arm64 or use x86_64 only for UI work).
- **Min SDK**: 28 (Android 9). **Target SDK**: 35 or newer, whatever the Play policy requires at build time (SDK 34 and 36 are installed).
- **RAM**: 4 GB minimum; 6 GB+ recommended.
- **Reference phone for budgets**: a mid-range 2022+ device (e.g. Snapdragon 7-series / Tensor G1). Record the actual model used in the test report.
- Storage: ~1 GB free (APK + extracted assets if any are copied out; see 7.6).

---

## 4. Architecture

```
 ┌──────────────────────────── Android app (Kotlin) ─────────────────────────────┐
 │  WebView  ── index.html (same UI as desktop) ── JS bridge (window.Shazam) ──┐  │
 │                                                                             │  │
 │  ListenController (coroutines): windows, keep-listening policy, state ──────┘  │
 │     │                                                                          │
 │     ├─ AudioRecorder      AudioRecord 16 kHz PCM16 mono → float ring buffer    │
 │     ├─ Asr                whisper.cpp via JNI (libwhisper.so), base.en q8_0    │
 │     └─ Identifier         (port of Searcher, CPU only)                         │
 │          ├─ Tokenizer       BERT WordPiece (vocab.txt), Kotlin                 │
 │          ├─ Embedder        ONNX Runtime Android, bge-small-ft int8            │
 │          ├─ DenseIndex      int8 vectors, mmap, brute-force top-50 (C++ NEON)  │
 │          ├─ Bm25Index       postings (mmap), top-50                            │
 │          ├─ Fusion+Voting   RRF → top-20 → exp voting with vote_weight         │
 │          ├─ Features        12 features, same formulas as Python               │
 │          └─ Calibrator      logistic regression + per-length thresholds        │
 │                                                                                │
 │  Assets (stored uncompressed, memory-mapped): see section 6                     │
 └────────────────────────────────────────────────────────────────────────────────┘
```

Single process, no network permission. The native parts (whisper.cpp, dense dot products) are
built with the NDK; everything else is Kotlin.

---

## 5. Component specs

### 5.1 Audio capture

- `AudioRecord`, `MediaRecorder.AudioSource.VOICE_RECOGNITION` (falls back to `MIC`), 16,000 Hz,
  mono, `ENCODING_PCM_16BIT`, read in 100 ms blocks into a growing float buffer (max 3 windows = 48,000 x 3 samples).
- RMS level of each block is pushed to the UI for the ripple animation (same as desktop `level`).
- Permission: `RECORD_AUDIO` at runtime. The first tap asks; if denied, show the "Microphone is off"
  state with a button to app settings. No foreground service: recording only while the app is visible;
  `onStop` cancels the session.

### 5.2 Speech recognition

- **Engine**: whisper.cpp (pin a release tag at build time), compiled for arm64-v8a with NEON/dotprod,
  4 threads by default (`min(4, cores)`).
- **Model**: `ggml-base.en-q8_0.bin` (~80 MB). Alternatives evaluated in milestone M1:
  `tiny.en-q8_0` (~45 MB, faster, worse), `small.en-q5_1` (~180 MB, better, slower).
  Pick the smallest that meets S1 **and** S5, measured with noisy audio (section 9.3).
- **Decoding**: language `en`, greedy (beam 1), no timestamps, `no_context = true`,
  `single_segment = false`, suppress non-speech tokens. Transcribe the **whole audio heard so far**
  at each window (same as desktop), not just the new window.
- **VAD**: whisper.cpp's built-in VAD if available in the pinned version, otherwise an energy gate:
  if RMS of the whole buffer < 0.002, skip ASR and treat it as silence.
- **Output cleaning**: strip bracketed non-speech (`[MUSIC]`, `(laughs)`), collapse whitespace.
  An empty result means "silence".

### 5.3 Tokenizer (for the embedding model)

- BERT uncased WordPiece using the model's `vocab.txt` (30,522 tokens), `[CLS] … [SEP]`, lowercase,
  strip accents, basic punctuation split, max length 128 (queries are short; truncate beyond).
- Implemented in Kotlin, no extra library.
- **Golden test**: 2,000 strings (dev queries, clean + noisy, plus edge cases with apostrophes, numbers,
  "S.H.I.E.L.D.", unicode quotes) tokenised by HuggingFace `tokenizers` in Python, stored as JSON;
  the Kotlin tokenizer must match **100%** of token ids.

### 5.4 Query embedding

- Export `models/bge-small-marvel-ft` to ONNX (opset 17, inputs `input_ids`, `attention_mask`,
  `token_type_ids`, dynamic batch/sequence), output = CLS vector, then L2-normalise.
- Dynamic int8 quantisation (`onnxruntime.quantization.quantize_dynamic`, weights int8) → ~35 MB.
  Keep the fp32 ONNX as a fallback if int8 fails S1.
- Run with ONNX Runtime Android (pin version; CPU EP, 4 intra-op threads; try XNNPACK EP and keep it
  if it is faster and the parity test still passes).
- **Parity**: cosine(Kotlin vector, Python ONNX vector) >= 0.999 on the 500 golden queries.

### 5.5 Dense index

- Vectors of all 291,310 chunks re-embedded **with the same int8 ONNX model** used on the phone
  (so query and index come from one model), then quantised to int8 with one scale per dimension:
  `q = round(v / s_d)`, `s_d = max|v_d| / 127`. Store `int8[N][384]` + `float32[384]` scales.
  Size: 291,310 x 384 = **112 MB**.
- Search: score_i = Σ_d (q_query_d · s_d) · q_i,d, i.e. pre-scale the query once, then int8×float dot products;
  in C++ with NEON over the memory-mapped file, 4 threads, keep top-100 (the dense-spread features use 100).
- Exact brute force (no ANN): 291k × 384 ≈ 112M multiply-adds, target <= 60 ms on the reference phone.
- **Parity**: the Python mobile simulator (section 9) uses the identical int8 data and formula.

### 5.6 BM25 index

- Export from `BM25Index` exactly as built in Python: vocabulary (term → id, 41,610 terms; the
  tokenizer is `norm()`: lowercase, keep `[a-z0-9']`, split on everything else) and, per term,
  a posting list of `(chunk_id: uint32, weight: float16)` sorted by chunk_id, where weight already includes
  idf, tf saturation and length normalisation.
  Size: 9.69M postings x 6 bytes ≈ **58 MB**, plus offsets.
- Query: tokenise with the same `norm()` (port the regex exactly; golden test on 2,000 strings),
  count term frequency in the query, accumulate `count x weight` into a float array of length N
  (reused buffer), take top-50 with a partial sort.
- Report `bm25_top` (best score) for the `bm25_per_word` feature.
- Ties: break by lower chunk_id, same as a stable sort in Python. The Python simulator must use the same rule.

### 5.7 Fusion and voting (port of `Searcher.candidates` / `decide`)

1. Dense top-100 → keep first 50 for RRF; BM25 top-50.
2. RRF score per chunk = Σ 1 / (60 + rank + 1) over both lists (rank from 0).
3. Sort by fused score (ties → lower chunk_id), keep the first k = 20.
4. For those 20: `sim` = dense similarity (from the dense scores; for BM25-only chunks compute the
   dot product directly).
5. vote = exp((score - max_score) / T) x vote_weight[chunk], with T from the exported config.
6. Sum votes per title label; rank labels by total vote.

All constants (k, depths, rrf_k, T) come from `search_config.json`, never hard-coded.

### 5.8 Confidence features and calibrator

Port `Searcher.features` exactly (names as in Python):

| feature | definition |
|---|---|
| share | top label vote / total vote |
| margin | (vote #1 - vote #2) / total vote |
| n_titles | distinct labels among the 20 chunks / 20 |
| top_score | best fused score among the winner's chunks |
| top_sim | best dense sim among the winner's chunks |
| sim_gap | top_sim - mean of the dense top-100 sims |
| sim_z | (top_sim - mean) / (std + 1e-6) over the dense top-100 |
| overlap2 | share of the query's word bigrams (norm tokens) found in the winner's best chunk |
| bm25_per_word | bm25_top / number of query words |
| log_words | ln(max(words, 1)) |
| len_short, len_mid | one-hot length bucket: short <= 11 words, mid <= 25, long otherwise |

p = sigmoid(Σ coef_i x (f_i - mean_i) / scale_i + intercept), with the arrays exported in feature order.
Answer = winner if p >= threshold[bucket], else "no match".

The calibrator **must be refitted** for the mobile configuration (no reranker, int8 models) with
`tune.py` on the Python simulator. The desktop calibrator includes `rerank_top` and cannot be reused.

### 5.9 Listening policy (same as desktop `app.py`)

| constant | value |
|---|---|
| window length | 5.0 s |
| max windows | 3 |
| normal decision | after 2 windows, if status = match |
| early answer | after 1 window if p >= 0.95 |
| "no match" instead of guesses | rejected and p < 0.05 |
| candidate confidence #2/#3 | p x vote_i / vote_1 |

Processing of window n overlaps recording of window n+1 (recording never pauses). Cancel by tapping the shield.

### 5.10 Result model (Kotlin → UI)

Same JSON as the desktop `/api/session` response (so the UI code is shared):
`{ state, window, max_windows, transcript, level, result: { status: match|unsure|none|silence|error,
confidence, transcript, candidates: [{label, title, year, episode, kind, confidence, quote, position}] } }`.
`quote` = 26 words of the matched chunk centred on the first heard bigram (port `quote_around`).

### 5.11 UI

- Reuse `app/static/index.html` unchanged in look. Load it from `assets/web/` in a `WebView`
  (JavaScript on, file access only to the app's own assets, no network).
- Add `app/static/api.js`: an adapter that calls `fetch('/api/...')` on the desktop and the
  `window.Shazam` bridge on Android (`@JavascriptInterface` methods `status()`, `listen()`, `session(id)`,
  `cancel(id)`, `identifyText(text)`, all returning JSON strings). The desktop keeps working unchanged.
- Android extras: edge-to-edge with safe-area padding (`env(safe-area-inset-*)`), haptic tick on tap and on result,
  keep the screen on while listening, back button cancels listening, respects system dark mode
  (the UI is dark-only; that's fine) and the "remove animations" setting (already handled by
  `prefers-reduced-motion`).
- Loading state while assets are mapped and models load (same dimmed shield as desktop).

---

## 6. Assets (exported by `export_android.py`)

All files go to `android/app/src/main/assets/` and are listed in `manifest.json` with size and SHA-256.

| file | content | approx size |
|---|---|---|
| `asr/ggml-base.en-q8_0.bin` | whisper.cpp model | 80 MB |
| `embed/model.int8.onnx` | fine-tuned bge-small, int8 | 35 MB |
| `embed/vocab.txt` | WordPiece vocabulary | 0.2 MB |
| `dense/vectors.i8` | int8[291,310][384], row = chunk_id | 112 MB |
| `dense/scales.f32` | float32[384] | 1.5 KB |
| `bm25/vocab.json` | term → term_id | 0.6 MB |
| `bm25/offsets.u32` | uint32[V+1] start of each term's postings | 0.2 MB |
| `bm25/postings.bin` | (uint32 chunk_id, float16 weight)[9.69M] | 58 MB |
| `chunks/meta.bin` | per chunk: label_id u16, vote_weight f32, type u8, position f16, title_id u16, word_start u32, n_words u16 | 4 MB |
| `chunks/labels.json` | label_id → {label, title, year, episode, kind} (706 labels) | 0.1 MB |
| `chunks/text.bin` + `text_index.bin` | each title's cleaned text once (UTF-8, zstd per title); chunk text = words[word_start : word_start+n_words] | ~15 MB compressed |
| `search_config.json` | k, depths, rrf_k, T, calibrator (features, mean, scale, coef, intercept), thresholds, policy constants, buckets | 4 KB |
| `golden/*.json` | parity test data (only in the test APK flavour) | 5 MB |
| `manifest.json` | versions, export git commit, SHA-256 of every file | — |

Total ≈ **305 MB**, of which ~250 MB memory-mapped.

Notes:
- Chunk text is needed only for the quote and the `overlap2` feature (one chunk per query), so it is
  stored compactly, not per chunk. `word_start` is recomputed in export with `chunk_words()`, because
  `chunks.parquet` only has a rounded position.
- Files that are memory-mapped (`vectors.i8`, `postings.bin`, `meta.bin`) must be stored **uncompressed** in the APK
  (`androidResources { noCompress += listOf("i8", "bin", "f32", "u32", "onnx") }`) so they can be
  opened with `AssetFileDescriptor` + `mmap` without copying.

---

## 7. Build

### 7.1 Toolchain

- Android Studio (installed), JDK 17 (the one bundled with Android Studio; the `java` on PATH is 1.8 and is too old).
- **To install** via SDK Manager: NDK (pin one version, e.g. r27 LTS), CMake 3.22+.
- Gradle Kotlin DSL, AGP and Kotlin versions pinned when the project is created.

### 7.2 Project layout

```
android/
  settings.gradle.kts
  app/
    build.gradle.kts
    src/main/
      AndroidManifest.xml          # RECORD_AUDIO only, no INTERNET
      assets/                      # filled by export_android.py (git-ignored, large)
      cpp/
        CMakeLists.txt
        whisper/                   # git submodule, pinned tag
        dense_search.cpp           # int8 NEON top-k
        jni_bridge.cpp
      java/com/yusef/marvelshazam/
        MainActivity.kt            # WebView + permission
        Bridge.kt                  # @JavascriptInterface
        ListenController.kt        # windows, policy, state
        AudioRecorder.kt
        Asr.kt                     # JNI wrapper for whisper.cpp
        search/Tokenizer.kt  Embedder.kt  DenseIndex.kt  Bm25Index.kt
        search/Identifier.kt       # fusion, voting, features, calibrator
        data/Assets.kt             # manifest check, mmap helpers
    src/test/                      # JVM unit tests (tokenizer, norm, voting, calibrator)
    src/androidTest/               # on-device parity + performance tests
export_android.py                  # repo root, Python
mobile_sim.py                      # repo root, Python simulator of the phone pipeline
```

### 7.3 Build variants

- `release`: assets without `golden/`, R8 minify on, signed with a local keystore (not committed).
- `parity` (debug-based): includes `golden/` and the instrumentation tests.

### 7.4 Signing and install

Local keystore in `~/.android/marvel_shazam.jks` (never committed). Install with `adb install -r app-release.apk`
or by copying the APK to the phone. Unknown-sources install must be allowed on the phone.

---

## 8. Python side (in this repo)

### 8.1 `export_android.py`

Inputs: `configs/mobile.json` (= `step6_finetune.json` with `rerank: false` and a recalibrated calibrator).
Steps:
1. Export the fine-tuned model to ONNX, quantise to int8, check cosine >= 0.999 vs PyTorch on 500 queries.
2. Re-embed all chunks with the int8 ONNX model (≈ 50 min on the laptop; cache by text hash like `embed_with_cache`).
3. Quantise vectors to int8 with per-dimension scales.
4. Build BM25 with `BM25Index` and write vocab, offsets and postings (float16 weights).
5. Write chunk metadata, labels, compact text.
6. Write `search_config.json` from the mobile config.
7. Download/convert the whisper.cpp model (official ggml file, check its SHA-256).
8. Generate golden files (section 9.2).
9. Write `manifest.json` (git commit, file hashes) into `android/app/src/main/assets/`.

### 8.2 `mobile_sim.py`

A pure-Python/numpy re-implementation of the phone pipeline that reads **only the exported asset
files** (ONNX int8 via onnxruntime, int8 vectors, float16 BM25 postings, same tie rules). It exposes the
same `candidates()` / `decide()` interface as `build_index.Searcher`, so `tune.py` and the frozen
`evaluate.py` logic can run on it (via a `--engine mobile` switch in a thin wrapper; `evaluate.py` itself stays frozen).

---

## 9. Testing and evaluation

### 9.1 Accuracy (Python, before any Android code)

| run | engine | split | purpose |
|---|---|---|---|
| A | desktop step-6 config (fp32, FAISS, no reranker) | dev | reference: 0.926 |
| B | `mobile_sim` with fp32 ONNX + fp32 vectors | dev | isolates ONNX export |
| C | `mobile_sim` with int8 ONNX + int8 vectors + fp16 BM25 | dev | the shipped configuration |
| D | C + recalibrated (`tune.py --write configs/mobile.json`) | dev | must meet S1 |
| E | D | test | once, at the end (S3) |

If D < 0.90: try fp32 ONNX (B-style) with int8 vectors, then fp16 vectors (224 MB); keep the
smallest combination that meets S1.

### 9.2 Kotlin ↔ Python parity (golden files)

- `golden/tokenizer.json`: 2,000 strings → WordPiece ids. Must match 100%.
- `golden/norm.json`: 2,000 strings → `norm()` tokens. Must match 100%.
- `golden/queries.json`: 500 dev queries (clean, noisy, negatives) with expected query vector (first 8 dims + norm),
  dense top-10 ids, BM25 top-10 ids, fused top-20, all 12 feature values, p, answer, top-3 labels.
- Instrumentation test on the phone compares everything: ids exact (allow swaps of exact ties),
  features |Δ| <= 1e-3, p |Δ| <= 1e-3; top-1 equal >= 99.5% (S2).

### 9.3 End-to-end audio

- 60 audio clips: 40 Marvel (TTS + 20 recordings of real playback from a TV/laptop speaker at 1-2 m),
  20 non-Marvel (other shows, podcasts, silence, music).
- Run on the phone through the full app (`ListenController` with a file-backed `AudioRecorder` in the
  parity flavour) and on the desktop app (`--fake-mic`) → compare answers and record time per stage.
- Report: correct / wrong / no-match for each set, time to answer, per-stage latency.

### 9.4 Performance

On the reference phone, 20 runs each (`androidTest` with `System.nanoTime`):
cold start, ASR per window (5 s, 10 s, 15 s audio), embedding, dense search, BM25, decide, peak PSS (`dumpsys meminfo`).
All must meet S4-S7.

### 9.5 Manual QA checklist

Permission denied → message + settings button; permission revoked mid-session; incoming phone call
while listening; app backgrounded while listening (session cancelled); rotation; airplane mode; very loud
music with no speech; tap shield repeatedly; low-storage device; TalkBack reads the result.

---

## 10. Milestones

| # | Milestone | Deliverables | Exit criteria |
|---|---|---|---|
| M0 | Toolchain | NDK + CMake installed, empty Compose/WebView project builds, runs on the phone | "hello" APK installs |
| M1 | Mobile accuracy in Python | `export_android.py` steps 1-6, `mobile_sim.py`, runs A-D, Whisper model choice | S1 met on dev; numbers in `RESULTS_MOBILE.md` |
| M2 | Assets + golden files | full export, manifest, golden JSON | sizes within section 6 |
| M3 | Search engine in Kotlin | tokenizer, embedder, dense (C++), BM25, identifier | JVM unit tests + on-device parity S2 |
| M4 | ASR + audio | whisper.cpp JNI, recorder, ListenController with policy | 60-clip test runs; S5 measured |
| M5 | UI | WebView + bridge + `api.js`; desktop still works | all result states shown on the phone |
| M6 | Hardening | performance pass, QA checklist, release build, signing | S4-S9 met |
| M7 | Final report | test-split run E, 60-clip results, APK | `RESULTS_MOBILE.md` complete |

Rough effort: M1 1-2 days (mostly re-embedding time), M3 2-3 days, M4 1-2 days, M5 1 day, M6 1-2 days.

---

## 11. Risks and decisions

### 11.1 Risks

| risk | impact | mitigation |
|---|---|---|
| int8 embedding or int8 vectors lose accuracy | S1 missed | fall back step by step (9.1); fp16 vectors double the size but are near-lossless |
| whisper base.en worse than small.en on real TV audio | more wrong / unsure answers | M1 compares tiny/base/small on recorded clips; the keep-listening policy compensates |
| ASR too slow on older phones | S5 missed | tiny.en fallback chosen at startup from a 1-second benchmark |
| Kotlin and Python disagree on ties / float rounding | parity failures | exact tie rules (lower id first), float16 rounding done in export, not on device |
| Large assets on low-storage phones | install fails | clear message; v1 needs ~1 GB free |
| WebView quirks (old Android System WebView) | broken animations | min SDK 28 + "update WebView" hint; UI uses only standard CSS |
| mmap of APK assets fails on some devices | slow start | fall back to copying to app storage on first run (one-time, ~300 MB) |

### 11.2 Decisions taken

- **No reranker on mobile** (size 280 MB+ and ~1 s per query on desktop CPU). Revisit with a distilled
  MiniLM reranker (~25 MB) as a v2 experiment.
- **WebView UI** instead of a Compose rewrite: one UI codebase for desktop and phone.
- **Brute-force int8 search** instead of an ANN library: exact, simple, fast enough at 291k vectors.
  Revisit (HNSW / IVF-PQ) above ~1M chunks, i.e. when more franchises are added.
- **Fully offline**, no network permission.

### 11.3 If it ever goes on the Play Store

Base APK limit 200 MB: move the model/index files into Play Asset Delivery install-time packs
(check the current size limits when doing this). Distributing subtitle text and derived indexes
publicly raises copyright questions; v1 is for personal sideloading only.

---

## 12. Open questions

1. Which phone is the reference device for the budgets?
2. Should a fixed "decide after 2 windows" rule replace the early answer at p >= 0.95? (On the test stream,
   always-2-windows scored 0.982 vs 0.972 for the adaptive policy.)
3. Do the 45 wrong-show files get fixed before the first export (MISSING.md)? They don't break the app,
   but they are ~6% of the episodes.
4. Keep "Type a line instead" on the phone?
