"""
app.py  -  the desktop app: microphone -> faster-whisper -> marvel_shazam search -> shield UI

    python app.py               # native window (pywebview / Edge WebView2)
    python app.py --browser     # open in your default browser instead
    python app.py --no-window   # only the server, on http://127.0.0.1:8765

Listening follows the keep-listening policy measured in RESULTS.md: record in WINDOW_S-second
steps, transcribe everything heard so far, and search. Answer after 2 windows (the safest rule on
the test split); answer after 1 only when the calibrated confidence is very high; if still unsure
after 2, listen to a 3rd window and decide then.
"""
import argparse, re, threading, time, uuid, webbrowser
from pathlib import Path

import numpy as np

import build_index as bi

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "app" / "static"
SAMPLE_RATE = 16000
WINDOW_S = 5.0          # one "window" of audio
MAX_WINDOWS = 3
DECIDE_AT = 2           # normal decision point
EARLY_ACCEPT = 0.95     # confidence needed to answer after only 1 window
NONE_BELOW = 0.05       # rejected and this unsure -> "no Marvel match" instead of guesses
WHISPER_MODEL = "small.en"
PORT = 8765
FAKE_MIC = None         # path to a WAV played as if it came from the microphone (--fake-mic)
LABEL_RE =re.compile(r"^(?P<title>.+?) \((?P<year>\d{4})\)(?: (?P<ep>S\d{2}E\d{2}))?$")


# ------------------------------------------------------------------ engine (loaded once)
class Engine:
    def __init__(self):
        self.stage, self.ready, self.error = "starting", False, None
        self.searcher = self.whisper = None
        self.lock = threading.Lock()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            self.stage = "loading speech model"
            from faster_whisper import WhisperModel
            self.whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            self.stage = "loading the Marvel index"
            self.searcher = bi.Searcher(bi.load_config())
            self.stage = "warming up"
            self.identify("I can do this all day")          # builds BM25, loads the reranker
            self.stage, self.ready = "ready", True
        except Exception as e:                               # shown in the UI
            self.error, self.stage = f"{type(e).__name__}: {e}", "error"

    def transcribe(self, audio):
        segs, _ = self.whisper.transcribe(audio, language="en", beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip()

    def identify(self, text):
        with self.lock:
            r = self.searcher.search_batch([text])[0]
        return format_result(self.searcher, r, text)


def split_label(label):
    m = LABEL_RE.match(label)
    if not m:
        return {"title": label, "year": "", "episode": ""}
    return {"title": m["title"], "year": m["year"], "episode": m["ep"] or ""}


def quote_around(chunk, heard, words=26):
    """The part of the matched subtitle chunk where the heard words are, not just its start."""
    cw = chunk.split()
    nc = [" ".join(bi.norm(w)) for w in cw]
    hb = set(zip(bi.norm(heard), bi.norm(heard)[1:]))
    hits = [i for i in range(len(nc) - 1) if (nc[i], nc[i + 1]) in hb]
    start = max(0, (hits[0] if hits else 0) - 4)
    start = min(start, max(0, len(cw) - words))
    q = " ".join(cw[start:start + words])
    return ("…" if start > 0 else "") + q + ("…" if start + words < len(cw) else "")


def format_result(sr, r, text):
    """Top 3 titles with a confidence each: the calibrated probability for #1, scaled by vote
    relative to #1 for #2 and #3 (so they never outrank it). Below NONE_BELOW it's 'no match'."""
    if not text or not r["ranking"]:
        return {"status": "silence" if not text else "none", "transcript": text, "candidates": []}
    p = float(np.clip(r["confidence"], 0, 1))
    if r["answer"] == bi.NO_MATCH and p < NONE_BELOW:
        return {"status": "none", "transcript": text, "confidence": p, "candidates": []}
    top = r["ranking"][:3]
    first = top[0][1] or 1.0
    cands = []
    for i, (label, vote) in enumerate(top):
        score, cid = r["best"][label]
        row = sr.df.iloc[cid]
        conf = p * vote / first
        cands.append({**split_label(label), "label": label, "confidence": round(conf, 4),
                      "kind": row["type"], "quote": quote_around(row["text"], text),
                      "position": float(row["position"])})
    status = "match" if r["answer"] != bi.NO_MATCH else "unsure"
    return {"status": status, "transcript": text, "confidence": p, "candidates": cands}


# ------------------------------------------------------------------ one listening session
class Session:
    def __init__(self, engine):
        self.id = uuid.uuid4().hex[:8]
        self.engine = engine
        self.state, self.window, self.transcript, self.result = "listening", 0, "", None
        self.cancelled = False
        self.level = 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def view(self):
        return {"id": self.id, "state": self.state, "window": self.window,
                "max_windows": MAX_WINDOWS, "transcript": self.transcript,
                "level": round(self.level, 3), "result": self.result}

    def _run(self):
        import sounddevice as sd
        chunks = []

        def on_audio(indata, frames, t, status):
            chunks.append(indata[:, 0].copy())
            self.level = float(np.sqrt(np.mean(indata[:, 0] ** 2)))

        def fake_stream():                       # feeds the WAV in real time, 0.1 s blocks
            import soundfile as sf
            audio, sr = sf.read(FAKE_MIC, dtype="float32", always_2d=True)
            audio = audio[:, 0]
            if sr != SAMPLE_RATE:
                audio = np.interp(np.arange(0, len(audio), sr / SAMPLE_RATE),
                                  np.arange(len(audio)), audio).astype(np.float32)
            step = SAMPLE_RATE // 10
            for i in range(0, len(audio), step):
                if self.cancelled or self.state == "done":
                    return
                on_audio(audio[i:i + step, None], step, None, None)
                time.sleep(0.1)

        if FAKE_MIC:
            from contextlib import nullcontext
            threading.Thread(target=fake_stream, daemon=True).start()
            stream = nullcontext()
        else:
            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                    callback=on_audio)
        try:
            with stream:
                for w in range(1, MAX_WINDOWS + 1):
                    end = time.time() + WINDOW_S
                    while time.time() < end:
                        if self.cancelled:
                            self.state = "cancelled"
                            return
                        time.sleep(0.05)
                    self.window = w
                    audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
                    self.state = "thinking"
                    self.transcript = self.engine.transcribe(audio)
                    res = self.engine.identify(self.transcript)
                    confident = res["status"] == "match" and res["confidence"] >= EARLY_ACCEPT
                    if (w >= DECIDE_AT and res["status"] == "match") or confident \
                            or w == MAX_WINDOWS:
                        self.result, self.state = res, "done"
                        return
                    self.state = "listening"                 # not sure yet: keep listening
        except Exception as e:
            self.result = {"status": "error", "error": f"{type(e).__name__}: {e}",
                           "transcript": self.transcript, "candidates": []}
            self.state = "done"


# ------------------------------------------------------------------ web server
def make_app(engine):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    app = FastAPI()
    sessions = {}

    class Text(BaseModel):
        text: str

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        return {"ready": engine.ready, "stage": engine.stage, "error": engine.error}

    @app.post("/api/listen")
    def listen():
        if not engine.ready:
            raise HTTPException(503, engine.stage)
        for s in sessions.values():
            s.cancelled = True
        s = Session(engine)
        sessions.clear()
        sessions[s.id] = s
        return s.view()

    @app.get("/api/session/{sid}")
    def session(sid: str):
        if sid not in sessions:
            raise HTTPException(404)
        return sessions[sid].view()

    @app.post("/api/session/{sid}/cancel")
    def cancel(sid: str):
        if sid in sessions:
            sessions[sid].cancelled = True
        return {"ok": True}

    @app.post("/api/identify_text")        # for testing the UI without a microphone
    def identify_text(t: Text):
        if not engine.ready:
            raise HTTPException(503, engine.stage)
        return engine.identify(t.text)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--fake-mic", help="WAV file to use instead of the microphone (testing)")
    a = ap.parse_args()
    global FAKE_MIC
    FAKE_MIC = a.fake_mic
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(make_app(Engine()), host="127.0.0.1", port=a.port,
                                           log_level="warning"))
    url = f"http://127.0.0.1:{a.port}/"
    if a.no_window:
        server.run()
        return
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    if a.browser:
        webbrowser.open(url)
        threading.Event().wait()
    try:
        import webview
        webview.create_window("Marvel Shazam", url, width=460, height=820, min_size=(380, 640),
                              background_color="#0b1020")
        webview.start()
    except Exception as e:
        print(f"Native window unavailable ({e}); opening the browser instead.")
        webbrowser.open(url)
        threading.Event().wait()


if __name__ == "__main__":
    main()
