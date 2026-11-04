"""Generates docs/diagrams/*.html following the diagram-design skill (default skin)."""
from pathlib import Path

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

PAPER, INK, MUTED, SOFT, ACCENT, LINK = "#f5f5f5", "#2d3142", "#4f5d75", "#5a6580", "#bf4520", "#2e5aa8"
SANS = "'Geist', system-ui, sans-serif"
MONO = "'Geist Mono', ui-monospace, monospace"
FONTS = ("https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1"
         "&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap")
TREAT = {  # node type -> (fill, stroke, dash)
    "input": ("rgba(79,93,117,0.10)", SOFT, None),
    "backend": ("#ffffff", INK, None),
    "store": ("rgba(45,49,66,0.05)", MUTED, None),
    "focal": ("rgba(191,69,32,0.08)", ACCENT, None),
}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def node(x, y, w, h, kind, tag, name, subs):
    fill, stroke, _ = TREAT[kind]
    tag_color = ACCENT if kind == "focal" else stroke
    tw = max(28, int(len(tag) * 7 * 0.62 + 12) // 4 * 4 + 4)
    cx = x + w / 2
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{PAPER}"/>',
           f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"/>',
           f'<rect x="{x + 8}" y="{y + 8}" width="{tw}" height="12" rx="2" fill="transparent" stroke="{tag_color}" '
           f'stroke-opacity="0.45" stroke-width="0.8"/>',
           f'<text x="{x + 8 + tw / 2}" y="{y + 17}" fill="{tag_color}" font-size="7" font-family="{MONO}" '
           f'text-anchor="middle" letter-spacing="0.08em">{esc(tag)}</text>',
           f'<text x="{cx}" y="{y + 40}" fill="{INK}" font-size="12" font-weight="600" font-family="{SANS}" '
           f'text-anchor="middle">{esc(name)}</text>']
    for i, s in enumerate(subs):
        out.append(f'<text x="{cx}" y="{y + 56 + 12 * i}" fill="{MUTED}" font-size="9" font-family="{MONO}" '
                   f'text-anchor="middle">{esc(s)}</text>')
    return "\n".join(out)


def label(cx, top, text, color=SOFT):
    """Arrow label: opaque mask with its top at `top` (12px tall)."""
    w = int(len(text) * 8 * 0.62 + 8 + 3) // 4 * 4
    return (f'<rect x="{cx - w / 2}" y="{top}" width="{w}" height="12" rx="2" fill="{PAPER}"/>\n'
            f'<text x="{cx}" y="{top + 9}" fill="{color}" font-size="8" font-family="{MONO}" '
            f'text-anchor="middle" letter-spacing="0.06em">{esc(text)}</text>')


def path(d, kind="muted", dashed=False):
    stroke, marker = {"muted": (MUTED, "arrow"), "accent": (ACCENT, "arrow-accent"),
                      "link": (LINK, "arrow-link")}[kind]
    dash = ' stroke-dasharray="4,3"' if dashed else ""
    sw = "1" if dashed else "1.2"
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"{dash} marker-end="url(#{marker})"/>'


def zone(x, y, w, h, text, label_dx=12):
    lw = int(len(text) * 7 * 0.62 + 16) // 4 * 4
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="rgba(45,49,66,0.02)" '
            f'stroke="rgba(45,49,66,0.10)" stroke-width="0.8"/>\n'
            f'<rect x="{x + label_dx}" y="{y + 4}" width="{lw}" height="12" rx="2" fill="{PAPER}"/>\n'
            f'<text x="{x + label_dx + lw / 2}" y="{y + 13}" fill="{SOFT}" font-size="7" font-family="{MONO}" '
            f'text-anchor="middle" letter-spacing="0.14em">{esc(text)}</text>')


def legend(y, items, width):
    out = [f'<line x1="30" y1="{y - 8}" x2="{width - 30}" y2="{y - 8}" stroke="rgba(45,49,66,0.10)" stroke-width="0.8"/>',
           f'<text x="30" y="{y + 12}" fill="{MUTED}" font-size="8" font-family="{MONO}" letter-spacing="0.14em">LEGEND</text>']
    x = 110
    for kind, text in items:
        if kind == "dashed":
            out.append(f'<line x1="{x}" y1="{y + 9}" x2="{x + 24}" y2="{y + 9}" stroke="{MUTED}" stroke-width="1" '
                       f'stroke-dasharray="4,3"/>')
        elif kind == "accent-arrow":
            out.append(f'<line x1="{x}" y1="{y + 9}" x2="{x + 24}" y2="{y + 9}" stroke="{ACCENT}" stroke-width="1.2"/>')
        else:
            fill, stroke, _ = TREAT[kind]
            out.append(f'<rect x="{x}" y="{y + 2}" width="24" height="14" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
        out.append(f'<text x="{x + 32}" y="{y + 12}" fill="{INK}" font-size="9" font-family="{SANS}">{esc(text)}</text>')
        x += 40 + int(len(text) * 9 * 0.6) + 24
    return "\n".join(out)


DEFS = f"""<defs>
  <marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{MUTED}"/></marker>
  <marker id="arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{ACCENT}"/></marker>
  <marker id="arrow-link" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{LINK}"/></marker>
</defs>"""


def page(slug, eyebrow, h1, title, desc, w, h, body, cards="", note=""):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{esc(title)}</title>
  <link href="{FONTS}" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --color-paper: {PAPER}; --color-ink: {INK}; --color-muted: {MUTED}; --color-accent: {ACCENT};
      --color-rule: rgba(45,49,66,0.12);
      --font-sans: 'Geist', system-ui, sans-serif;
      --font-serif: 'Instrument Serif', serif;
      --font-mono: 'Geist Mono', ui-monospace, monospace;
    }}
    body {{ font-family: var(--font-sans); background: var(--color-paper); color: var(--color-ink);
           min-height: 100vh; display: flex; justify-content: center; padding: 3rem 2rem; }}
    .frame {{ max-width: {w + 40}px; width: 100%; }}
    .diagram-container {{ width: 100%; overflow-x: auto; }}
    .eyebrow {{ font-family: var(--font-mono); font-size: 0.66rem; font-weight: 500; letter-spacing: 0.18em;
               text-transform: uppercase; color: var(--color-muted); margin-bottom: 0.5rem; }}
    h1 {{ font-family: var(--font-serif); font-size: clamp(1.5rem, 2.4vw + 0.75rem, 2rem); font-weight: 400;
         letter-spacing: -0.02em; line-height: 1.15; margin-bottom: 0.5rem; }}
    .sub {{ color: var(--color-muted); font-size: 0.95rem; line-height: 1.5; margin-bottom: 1.5rem; max-width: 760px; }}
    svg {{ width: 100%; min-width: {w}px; display: block; }}
    .cards {{ display: grid; grid-template-columns: 1.1fr 1fr 0.9fr; gap: 1rem; margin-top: 1.75rem; }}
    .card {{ background: #ffffff; border: 1px solid var(--color-rule); border-radius: 6px; padding: 1.25rem; }}
    .card h3 {{ font-size: 0.95rem; font-weight: 600; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.5rem; }}
    .card p, .card li {{ font-size: 0.86rem; line-height: 1.55; color: var(--color-ink); }}
    .card ul {{ padding-left: 1.1rem; }}
    .card .eyebrow {{ margin-bottom: 0.35rem; }}
    .dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; background: var(--color-muted); }}
    .dot.coral {{ background: var(--color-accent); }} .dot.ink {{ background: var(--color-ink); }}
    code {{ font-family: var(--font-mono); font-size: 0.8em; }}
    footer {{ margin-top: 1.75rem; padding-top: 0.75rem; border-top: 1px solid var(--color-rule);
             font-family: var(--font-mono); font-size: 0.7rem; color: var(--color-muted); }}
    @media (max-width: 760px) {{ .cards {{ grid-template-columns: 1fr; }} }}
    @media print {{ .diagram-container {{ overflow-x: visible; }} svg {{ min-width: 0; }} }}
  </style>
</head>
<body>
  <div class="frame">
    <p class="eyebrow">{esc(eyebrow)}</p>
    <h1>{esc(h1)}</h1>
    <p class="sub">{note}</p>
    <div class="diagram-container">
      <svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{slug}-title {slug}-desc">
        <title id="{slug}-title">{esc(title)}</title>
        <desc id="{slug}-desc">{esc(desc)}</desc>
        {DEFS}
        <rect width="100%" height="100%" fill="{PAPER}"/>
{body}
      </svg>
    </div>
    {cards}
    <footer>marvel_shazam · numbers from RESULTS.md (dev split unless marked test) · diagram-design skill, default skin</footer>
  </div>
</body>
</html>
"""


# =================================================================== 1. query path
def query_path():
    W, H = 1280, 500                         # drawn on a 640-tall grid, shifted up 140 (empty top band)
    b = ['<g transform="translate(0,-140)">']
    b.append(zone(24, 268, 320, 128, "SPEECH TO TEXT"))
    b.append(zone(360, 168, 512, 336, "SEARCH · 291,310 SUBTITLE CHUNKS"))
    b.append(zone(880, 268, 376, 128, "DECIDE"))
    # arrows (before nodes)
    b.append(path("M168,340 H200"))                                        # mic -> whisper
    b.append(path("M328,324 H340 Q348,324 348,316 V248 Q348,240 356,240 H368"))   # whisper -> dense
    b.append(path("M328,356 H340 Q348,356 348,364 V432 Q348,440 356,440 H368"))   # whisper -> bm25
    b.append(path("M528,240 H540 Q548,240 548,248 V316 Q548,324 556,324 H568"))   # dense -> rrf
    b.append(path("M528,440 H540 Q548,440 548,432 V364 Q548,356 556,356 H568"))   # bm25 -> rrf
    b.append(path("M696,340 H728"))                                        # rrf -> rerank
    b.append(path("M856,340 H888"))                                        # rerank -> decide
    b.append(path("M1048,340 H1080", "accent"))                            # decide -> result
    b.append(path("M968,380 V552 Q968,560 960,560 H112 Q104,560 104,552 V384", dashed=True))  # keep listening
    b.append(label(536, 540, "NOT SURE: LISTEN"))
    # nodes
    b.append(node(40, 300, 128, 80, "input", "MIC", "Microphone", ["16 kHz mono", "5 s windows"]))
    b.append(node(200, 300, 128, 80, "backend", "ASR", "Whisper", ["faster-whisper", "small.en · int8"]))
    b.append(node(368, 200, 160, 80, "store", "VEC", "Meaning search", ["bge-small (fine-tuned)", "FAISS · top 50"]))
    b.append(node(368, 400, 160, 80, "store", "WORD", "Keyword search", ["BM25 · 41.6k words", "top 50"]))
    b.append(node(568, 300, 128, 80, "backend", "FUSE", "Fusion", ["reciprocal rank", "keep top 20"]))
    b.append(node(728, 300, 128, 80, "backend", "RANK", "Reranker", ["bge-reranker-base", "rescore top 20"]))
    b.append(node(888, 300, 160, 80, "focal", "DECIDE", "Vote + confidence", ["exp(score/T) votes", "logistic regression"]))
    b.append(node(1080, 300, 160, 80, "backend", "OUT", "Top-3 titles", ["or \u201cno match\u201d", "if p < threshold"]))
    b.append(legend(612, [("input", "input"), ("backend", "model / step"), ("store", "index search"),
                          ("focal", "focal: the decision"), ("dashed", "keep listening loop")], W))
    b.append("</g>")
    cards = """<div class="cards">
      <div class="card"><p class="eyebrow">Open weights</p><h3><span class="dot coral"></span>Every model is open and runs offline</h3>
        <ul><li><b>Whisper small.en</b> (OpenAI, MIT) via faster-whisper</li>
        <li><b>bge-small-en-v1.5</b> (BAAI, MIT), fine-tuned here on our own train split</li>
        <li><b>bge-reranker-base</b> (BAAI, MIT)</li>
        <li>No LLM and no API: nothing leaves the laptop</li></ul></div>
      <div class="card"><p class="eyebrow">Learned on our data</p><h3><span class="dot ink"></span>Two small things are trained</h3>
        <ul><li>The fine-tuned embedder (contrastive loss, 9.9k pairs)</li>
        <li>A <b>logistic regression</b> that turns 13 features into p("the top answer is right")</li>
        <li>Thresholds per query length, tuned on dev</li></ul></div>
      <div class="card"><p class="eyebrow">Not RAG</p><h3><span class="dot"></span>Retrieval, no generation</h3>
        <p>It is the "R" of RAG only: it finds the chunks most like what was heard and votes. Nothing is generated, so it can't make up a title.</p></div>
    </div>"""
    note = ("What happens after you tap the shield: speech becomes text, two searches run side by side over every "
            "subtitle chunk, their rankings are fused and reranked, and a vote plus a confidence model decides between "
            "an answer and \u201cno match\u201d.")
    (OUT / "query-path.html").write_text(page(
        "query-path", "Architecture · query path", "How Marvel Shazam identifies a scene",
        "How Marvel Shazam identifies a scene",
        "Pipeline from microphone audio through Whisper, parallel meaning and keyword search, fusion, reranking, "
        "and a vote plus confidence model that returns the top three titles or no match.",
        W, H, "\n".join(b), cards, note), encoding="utf-8")


# =================================================================== 2. waterfall
def waterfall():
    W, H = 1000, 560
    k = round(3.8 * 46.7) / 46.7             # px per point, fitted to the start anchor's whole-pixel height
    yv = lambda v: round(420 - k * v)
    steps = [  # name, value (total or signed delta), role, SCORE after this step
        ("Baseline", 46.7, "total", "-0.033"),
        ("Multi-scale", 17.1, "delta", "0.138"),
        ("Voting + thr.", 2.0, "delta", "0.620"),
        ("Calibrator", 4.6, "delta", "0.700"),
        ("Hybrid BM25", 19.8, "delta", "0.891"),
        ("Fine-tune", 3.2, "delta", "0.926"),
        ("Reranker", 1.4, "delta", "0.938"),
        ("Final", 94.8, "total", "0.938"),
    ]
    b = []
    for g in (0, 20, 40, 60, 80, 100):
        y = yv(g)
        b.append(f'<line x1="80" y1="{y}" x2="960" y2="{y}" stroke="rgba(45,49,66,0.08)" stroke-width="0.8"/>')
        b.append(f'<text x="72" y="{y + 3}" fill="{MUTED}" font-size="8" font-family="{MONO}" text-anchor="end">{g}</text>')
    b.append(f'<line x1="80" y1="40" x2="80" y2="420" stroke="rgba(45,49,66,0.25)" stroke-width="1"/>')
    b.append(f'<line x1="80" y1="420" x2="960" y2="420" stroke="rgba(45,49,66,0.25)" stroke-width="1"/>')
    b.append(f'<text x="28" y="230" fill="{SOFT}" font-size="7" font-family="{MONO}" letter-spacing="0.14em" '
             f'text-anchor="middle" transform="rotate(-90 28 230)">NOISY TOP-1 (%)</text>')
    b.append(f'<text x="80" y="476" fill="{SOFT}" font-size="7" font-family="{MONO}" letter-spacing="0.14em">SCORE</text>')
    running = 0.0
    for i, (name, v, role, score) in enumerate(steps):
        x = 104 + 104 * i
        cx = x + 32
        before = running
        if i > 0:                                         # carry the running total across the gap
            b.append(f'<line x1="{x - 40}" y1="{yv(before)}" x2="{x}" y2="{yv(before)}" '
                     f'stroke="rgba(45,49,66,0.55)" stroke-width="1" data-carry="{before:.1f}"/>')
        if role == "total":
            top, bot, val = yv(v), 420, f"{v:.1f}"
            fill, stroke = "rgba(45,49,66,0.08)", INK
            running = v
        else:
            new = round(running + v, 1)
            top, bot, val = yv(new), yv(running), f"+{v:.1f}"
            focal = name == "Hybrid BM25"
            fill, stroke = ("rgba(191,69,32,0.12)", ACCENT) if focal else ("rgba(79,93,117,0.15)", MUTED)
            running = new
        b.append(f'<rect x="{x}" y="{top}" width="64" height="{bot - top}" fill="{PAPER}"/>')
        b.append(f'<rect x="{x}" y="{top}" width="64" height="{bot - top}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="1" data-role="{role}" data-value="{val}" data-name="{esc(name)}"/>')
        vc = ACCENT if stroke == ACCENT else MUTED
        b.append(f'<text x="{cx}" y="{top - 8}" fill="{vc}" font-size="8" font-family="{MONO}" text-anchor="middle">{val}</text>')
        nc = ACCENT if stroke == ACCENT else INK
        b.append(f'<text x="{cx}" y="440" fill="{nc}" font-size="11" font-weight="600" font-family="{SANS}" '
                 f'text-anchor="middle">{esc(name)}</text>')
        b.append(f'<text x="{cx}" y="476" fill="{MUTED}" font-size="9" font-family="{MONO}" '
                 f'text-anchor="middle">{score.replace("-", "\u2212")}</text>')
    b.append(legend(512, [("backend", "total (start, final)"), ("store", "increase"),
                          ("focal", "focal: biggest single gain")], W).replace(
        'fill="#ffffff" stroke="#2d3142"', 'fill="rgba(45,49,66,0.08)" stroke="#2d3142"').replace(
        'fill="rgba(45,49,66,0.05)" stroke="#4f5d75"', 'fill="rgba(79,93,117,0.15)" stroke="#4f5d75"').replace(
        'fill="rgba(191,69,32,0.08)"', 'fill="rgba(191,69,32,0.12)"'))
    cards = """<div class="cards">
      <div class="card"><p class="eyebrow">Why SCORE jumps at step 3</p><h3><span class="dot coral"></span>Saying "no match" is half the job</h3>
        <p>SCORE = noisy top-1 &minus; 0.5 &times; how often non-Marvel dialogue gets a Marvel answer. The baseline never said "no match", so it lost the full 0.5. The threshold (step 3) took rejection of non-Marvel lines from 0% to 92.5%, and the calibrator to 99.2%. That is why SCORE rises far more than top-1 there.</p></div>
      <div class="card"><p class="eyebrow">Tried, not kept</p><h3><span class="dot"></span>Two ideas that didn't help</h3>
        <ul><li><b>Leave the 44 wrong-show files out:</b> 0.6951 vs 0.6962 on fair queries</li>
        <li><b>Non-Marvel "background" index:</b> 0.7006 vs 0.7003, lower when cross-validated</li></ul></div>
      <div class="card"><p class="eyebrow">Test split, run once</p><h3><span class="dot ink"></span>It holds up</h3>
        <ul><li>SCORE <b>0.936</b> (baseline &minus;0.033)</li><li>noisy top-1 <b>94.8%</b> (46.7%)</li>
        <li>clean top-1 <b>98.7%</b> (55.7%)</li><li>non-Marvel rejected <b>97.7%</b> (0%)</li></ul></div>
    </div>"""
    note = ("Each bar adds what one change contributed to top-1 accuracy on Whisper-style noisy queries (dev split, 900 queries). "
            "SCORE, the metric every change had to improve, is printed under each step.")
    (OUT / "baseline-to-final.html").write_text(page(
        "baseline-to-final", "Waterfall · dev split", "From baseline to final, one change at a time",
        "From baseline to final, one change at a time",
        "Noisy-query top-1 accuracy rises from 46.7% to 94.8% across six kept changes, with hybrid BM25 search the "
        "biggest gain; SCORE goes from -0.033 to 0.938.",
        W, H, "\n".join(b), cards, note), encoding="utf-8")


# =================================================================== 3. build + eval
def build_eval():
    W, H = 1200, 528                         # drawn on a 600-tall grid, shifted up 72
    b = ['<g transform="translate(0,-72)">']
    b.append(zone(24, 88, 1152, 128, "BUILD THE INDEX"))
    b.append(zone(24, 328, 1152, 196, "FROZEN BENCHMARK · TRAINING · TUNING", label_dx=160))  # clear of x=120 arrow
    # arrows
    b.append(path("M200,160 H280")); b.append(label(240, 140, "706 FILES"))
    b.append(path("M440,160 H520")); b.append(label(480, 140, "CHUNKS"))
    b.append(path("M680,160 H760")); b.append(label(720, 140, "VECTORS"))
    b.append(path("M120,200 V360")); b.append(label(164, 274, "BY POSITION"))
    b.append(path("M200,400 H280")); b.append(label(240, 380, "DEV/TEST"))
    b.append(path("M120,440 V472 Q120,480 128,480 H352 a 8,8 0 0,1 16,0 H592 Q600,480 600,472 V444"))
    b.append(label(476, 460, "TRAIN 80%"))
    b.append(path("M600,360 V204", "accent")); b.append(label(644, 274, "NEW WEIGHTS", ACCENT))
    b.append(path("M840,200 V360")); b.append(label(876, 274, "SEARCH"))
    b.append(path("M360,440 V496 Q360,504 368,504 H856 Q864,504 864,496 V444"))
    b.append(label(712, 484, "DEV QUERIES"))
    b.append(path("M920,400 H1000")); b.append(label(960, 380, "SCORE"))
    b.append(path("M1080,360 V168 Q1080,160 1072,160 H924", dashed=True)); b.append(label(996, 140, "CONFIG.JSON"))
    # nodes
    b.append(node(40, 120, 160, 80, "store", "DATA", "Subtitle files", ["subs/*.txt", "76 movies · 630 eps"]))
    b.append(node(280, 120, 160, 80, "backend", "PREP", "Clean + chunk", ["recaps removed", "30 & 60-word chunks"]))
    b.append(node(520, 120, 160, 80, "backend", "MODEL", "Embedding model", ["bge-small-en-v1.5", "fine-tuned"]))
    b.append(node(760, 120, 160, 80, "store", "INDEX", "Search index", ["FAISS 291k vectors", "BM25 · chunks.parquet"]))
    b.append(node(40, 360, 160, 80, "store", "SPLIT", "Splits per title", ["80 train / 10 dev", "10 test (locked)"]))
    b.append(node(280, 360, 160, 80, "store", "QUERY", "Query sets", ["10/20/40 words", "clean · noisy · other"]))
    b.append(node(520, 360, 160, 80, "backend", "TRAIN", "Fine-tune", ["contrastive (MNRL)", "train split only"]))
    b.append(node(760, 360, 160, 80, "backend", "EVAL", "evaluate.py", ["frozen", "prints SCORE"]))
    b.append(node(1000, 360, 160, 80, "focal", "TUNE", "tune.py", ["T · calibrator (LR)", "length thresholds"]))
    b.append(legend(548, [("store", "data / index"), ("backend", "step"), ("focal", "focal: tuning"),
                          ("dashed", "writes settings")], W))
    b.append("</g>")
    cards = """<div class="cards">
      <div class="card"><p class="eyebrow">No data leaks</p><h3><span class="dot coral"></span>Three walls between train and test</h3>
        <ul><li>Each title is cut by position: 80% train, 10% dev, 10% test</li>
        <li>Fine-tuning pairs come only from train; tuning uses only dev</li>
        <li>Test ran once, at the very end</li></ul></div>
      <div class="card"><p class="eyebrow">The benchmark</p><h3><span class="dot ink"></span>2,400 dev + 2,400 test queries</h3>
        <ul><li>Random windows of 10/20/40 words, never chunk-aligned</li>
        <li>Each in a clean and a Whisper-style noisy version</li>
        <li>600 non-Marvel lines (Cornell movies) per split</li></ul></div>
      <div class="card"><p class="eyebrow">No notebooks</p><h3><span class="dot"></span>Plain scripts</h3>
        <p>Every step is a <code>.py</code> file you can re-run; results land in <code>eval/</code> and <code>RESULTS.md</code>.</p></div>
    </div>"""
    note = ("The offline side: how the index is built from subtitle files, and how the frozen benchmark measures every "
            "change and feeds both the fine-tune and the tuned settings back into search.")
    (OUT / "build-and-evaluation.html").write_text(page(
        "build-and-evaluation", "Architecture · build + evaluation", "Building the index, and keeping score honestly",
        "Building the index, and keeping score honestly",
        "Subtitle files are cleaned, chunked, embedded and indexed; per-title splits feed a frozen query benchmark, "
        "a train-only fine-tune of the embedder, and tune.py, which writes the search settings.",
        W, H, "\n".join(b), cards, note), encoding="utf-8")


query_path()
waterfall()
build_eval()
print("written:", sorted(p.name for p in OUT.glob("*.html")))
