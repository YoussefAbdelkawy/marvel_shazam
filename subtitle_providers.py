"""
subtitle_providers.py  -  free subtitle sources + a resumable downloader with a progress bar

Used by fetch_missing.py (fills MISSING.md) and fetch_harry_potter.py.

Providers (tried in this order, each only if its key is set as an environment variable):
    wyzie          WYZIE_API_KEY          free key: https://store.wyzie.io/redeem  (1,000 requests/day)
    subdl          SUBDL_API_KEY          free key: subdl.com -> Profile -> API key (2,000/day)
    opensubtitles  OPENSUBTITLES_API_KEY + OPENSUBTITLES_USERNAME + OPENSUBTITLES_PASSWORD (small daily quota)

IDs: TV shows are looked up on TVmaze (IMDb id + the real episode list, no key needed), movies on
Wikidata (IMDb id, no key). Results are cached in subtitle_ids.json; edit that file to fix a
wrong match by hand.

Every download is checked before it is kept: right season/episode in the release name, not a
"forced" (signs-only) subtitle, enough words, and for shows that already have episodes in subs/,
enough shared character names with them (this is what would have caught the Kick Buttowski files
saved as Daredevil).
"""
import io, json, os, re, sys, time, zipfile
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
IDS_CACHE = ROOT / "subtitle_ids.json"
UA = "marvel-shazam-subtitles/1.0 (personal research project)"
TIMEOUT = 30
MIN_WORDS = {"movie": 4000, "tv": 1000}
NAME_OVERLAP_MIN = 0.15         # share of a show's usual character names a new episode must contain
NAME_CHECK_MIN_NAMES = 6        # only judge shows with at least this many known names
# Calibrated on subs/: all 46 wrong-show files score <= 0.14; 506 of 508 real episodes score >= 0.18.


# ------------------------------------------------------------------ text
TAGS = re.compile(r"<[^>]+>|\{[^}]*\}|\[[^\]]*\]|\([A-Z ,'\-]{3,}\)|[♪♫]+")
# provider ads / credits injected as subtitle lines (Wyzie's free plan adds one to every file)
AD_RE = re.compile(r"wyzie|subdl|opensubtitles|free plan|ad-free|www\.|https?://|\.(?:com|org|net|io)\b", re.I)


def srt_to_text(srt):
    """Same conversion as subs/Fetch.py's clean(), so new files match the existing ones."""
    out = []
    for line in srt.splitlines():
        line = line.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        line = TAGS.sub("", line).replace("�", "")      # U+FFFD: broken chars in the upload
        line = re.sub(r"^[A-Z][A-Z .'\-]{1,20}:\s*", "", line)
        line = line.lstrip("- ").strip()
        if line and not AD_RE.search(line):
            out.append(line)
    return " ".join(out)


def decode(raw):
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def ep_code_re(season, episode):
    return re.compile(rf"(?<!\d)(?:s0*{season}[ ._-]*e0*{episode}(?!\d)|0*{season}x0*{episode}(?!\d))", re.I)


# ------------------------------------------------------------------ targets
@dataclass
class Target:
    kind: str                    # "movie" | "tv"
    title: str                   # pretty title, e.g. "X-Men: The Animated Series"
    year: int
    season: int = None
    episode: int = None
    reason: str = "missing"      # missing | wrong | damaged | new
    out_name: str = ""           # file stem, e.g. tv__x_men_the_animated_series_1992__s01e01

    def __post_init__(self):
        if not self.out_name:
            self.out_name = (f"tv__{slug(self.title)}_{self.year}__s{self.season:02d}e{self.episode:02d}"
                             if self.kind == "tv" else f"movie__{slug(self.title)}_{self.year}")

    @property
    def label(self):
        return (f"{self.title} ({self.year}) S{self.season:02d}E{self.episode:02d}"
                if self.kind == "tv" else f"{self.title} ({self.year})")


@dataclass
class Candidate:
    provider: str
    release: str
    url: str
    hi: bool = False
    downloads: int = 0
    season: int = None
    episode: int = None
    extra: dict = field(default_factory=dict)


class RateLimited(Exception):
    pass


# ------------------------------------------------------------------ http
SESSION = requests.Session()
SESSION.headers["User-Agent"] = UA


def get(url, wait_on_429=False, **kw):
    """wait_on_429: for free lookup services (TVmaze, Wikidata) slow down and retry instead of
    treating 429 as a daily limit."""
    for attempt in range(5 if wait_on_429 else 3):
        try:
            r = SESSION.get(url, timeout=TIMEOUT, **kw)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 429 and wait_on_429:
            time.sleep(float(r.headers.get("Retry-After") or 5 * (attempt + 1)))
            continue
        if r.status_code in (429, 402):
            raise RateLimited(f"{url.split('/')[2]} returned {r.status_code}")
        if r.status_code >= 500:
            time.sleep(2 * (attempt + 1))
            continue
        return r
    raise requests.RequestException(f"giving up on {url}")


# ------------------------------------------------------------------ id lookup
class Ids:
    """IMDb ids (and episode lists for shows), cached in subtitle_ids.json."""

    def __init__(self):
        self.cache = json.loads(IDS_CACHE.read_text()) if IDS_CACHE.exists() else {}

    def save(self):
        IDS_CACHE.write_text(json.dumps(self.cache, indent=1, sort_keys=True))

    @staticmethod
    def _close(a, b):
        f = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower().replace("marvel's", "")).strip()
        return SequenceMatcher(None, f(a), f(b)).ratio()

    def _lookup(self, key, fn, *args):
        if key not in self.cache:
            try:
                self.cache[key] = fn(*args)
            except requests.RequestException as e:       # network trouble: don't cache, retry next run
                return {"error": f"lookup failed ({e})"}
            self.save()
        return self.cache[key]

    def show(self, title, year):
        return self._lookup(f"tv:{title}:{year}", self._tvmaze, title, year)

    def _tvmaze(self, title, year):
        variants = [title, title.split(":")[0], re.sub(r":? the animated series", "", title, flags=re.I)]
        best = None
        for q in dict.fromkeys(v.strip() for v in variants if v.strip()):
            time.sleep(0.6)                                   # TVmaze: 20 calls / 10 s
            for hit in get("https://api.tvmaze.com/search/shows", wait_on_429=True, params={"q": q}).json():
                s = hit["show"]
                prem = int((s.get("premiered") or "0")[:4] or 0)
                if abs(prem - year) > 1:
                    continue
                score = max(self._close(s["name"], title), self._close(s["name"], q))
                if score >= 0.6 and (best is None or score > best[0]):
                    best = (score, s)
            if best:
                break
        if not best:
            return {"error": "not found on TVmaze"}
        s = best[1]
        time.sleep(0.6)
        eps = get(f"https://api.tvmaze.com/shows/{s['id']}/episodes", wait_on_429=True).json()
        return {"imdb": s["externals"].get("imdb"), "tvmaze": s["id"], "name": s["name"],
                "premiered": s.get("premiered"),
                "episodes": sorted({(e["season"], e["number"]) for e in eps if e.get("number")})}

    def movie(self, title, year):
        return self._lookup(f"movie:{title}:{year}", self._wikidata, title, year)

    def _wikidata(self, title, year):
        api = "https://www.wikidata.org/w/api.php"
        time.sleep(1.0)                                     # be polite: Wikidata rate-limits bursts
        hits = get(api, wait_on_429=True, params={"action": "wbsearchentities", "search": title, "language": "en",
                                "type": "item", "limit": 10, "format": "json"}).json().get("search", [])
        if not hits:
            return {"error": "not found on Wikidata"}
        time.sleep(1.0)
        ents = get(api, wait_on_429=True, params={"action": "wbgetentities", "ids": "|".join(h["id"] for h in hits),
                                "props": "claims|labels", "languages": "en",
                                "format": "json"}).json().get("entities", {})
        best = None
        for h in hits:
            c = ents.get(h["id"], {}).get("claims", {})
            kinds = {x["mainsnak"].get("datavalue", {}).get("value", {}).get("id") for x in c.get("P31", [])}
            if not kinds & FILM_TYPES:          # same title + year can be a video game or a book
                continue
            imdb = [x["mainsnak"].get("datavalue", {}).get("value") for x in c.get("P345", [])]
            imdb = [i for i in imdb if i and i.startswith("tt")]
            years = {int(x["mainsnak"]["datavalue"]["value"]["time"][1:5])
                     for x in c.get("P577", []) if x["mainsnak"].get("datavalue")}
            if imdb and any(abs(y - year) <= 1 for y in years):
                score = self._close(h.get("label", ""), title)
                if best is None or score > best[0]:
                    best = (score, imdb[0], h.get("label"))
        if not best:
            return {"error": "no Wikidata film with an IMDb id and a matching year"}
        return {"imdb": best[1], "name": best[2]}


# Wikidata "instance of" values that count as a film (film, feature film, animated feature, TV film,
# television special, short film, live-action/animated film, 3D film, superhero film)
FILM_TYPES = {"Q11424", "Q24869", "Q29168811", "Q202866", "Q506240", "Q1261214", "Q24862",
              "Q20650540", "Q229390", "Q1257444", "Q18011172", "Q1535153"}


# ------------------------------------------------------------------ providers
class Wyzie:
    name, env = "wyzie", ("WYZIE_API_KEY",)

    def __init__(self):
        self.key = os.environ["WYZIE_API_KEY"]

    def search(self, t, imdb):
        p = {"id": imdb, "language": "en", "format": "srt", "source": "charlie,lima", "key": self.key}
        if t.kind == "tv":
            p.update(season=t.season, episode=t.episode)
        r = get("https://sub.wyzie.io/search", params=p)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        data = r.json()
        return [Candidate(self.name, " ".join(d.get("releases") or [d.get("release") or d.get("fileName") or ""]),
                          d["url"], bool(d.get("isHearingImpaired")), int(d.get("downloadCount") or 0),
                          t.season, t.episode)
                for d in (data if isinstance(data, list) else [])]

    def download(self, c):
        r = get(c.url)
        r.raise_for_status()
        return decode(r.content)


class SubDL:
    name, env = "subdl", ("SUBDL_API_KEY",)

    def __init__(self):
        self.key = os.environ["SUBDL_API_KEY"]

    def search(self, t, imdb):
        p = {"api_key": self.key, "languages": "EN", "subs_per_page": 30, "releases": 1,
             "comment": 1, "hi": 1, "imdb_id": imdb}
        if t.kind == "tv":
            p.update(type="tv", season_number=t.season, episode_number=t.episode)
        else:
            p["type"] = "movie"
        data = get("https://api.subdl.com/api/v1/subtitles", params=p).json()
        if not data.get("status"):
            return []
        out = []
        for s in data.get("subtitles", []):
            if "forced" in (s.get("comment") or "").lower():
                continue
            out.append(Candidate(self.name, " ".join(s.get("releases") or [s.get("release_name") or s.get("name", "")]),
                                 s["url"], bool(s.get("hi")), 0, s.get("season"), s.get("episode"),
                                 {"full_season": bool(s.get("full_season"))}))
        return out

    def download(self, c, target=None):
        r = get("https://dl.subdl.com" + c.url)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        srts = [n for n in z.namelist() if n.lower().endswith(".srt")]
        if target is not None and target.kind == "tv" and len(srts) > 1:   # season pack
            rx = ep_code_re(target.season, target.episode)
            srts = [n for n in srts if rx.search(n)] or []
        if not srts:
            raise ValueError("no matching .srt in the zip")
        return decode(z.read(srts[0]))


class OpenSubtitles:
    name = "opensubtitles"
    env = ("OPENSUBTITLES_API_KEY", "OPENSUBTITLES_USERNAME", "OPENSUBTITLES_PASSWORD")
    api = "https://api.opensubtitles.com/api/v1"

    def __init__(self):
        self.h = {"Api-Key": os.environ["OPENSUBTITLES_API_KEY"], "User-Agent": UA,
                  "Content-Type": "application/json", "Accept": "application/json"}
        r = SESSION.post(f"{self.api}/login", headers=self.h, timeout=TIMEOUT, json={
            "username": os.environ["OPENSUBTITLES_USERNAME"],
            "password": os.environ["OPENSUBTITLES_PASSWORD"]})
        r.raise_for_status()
        self.h["Authorization"] = "Bearer " + r.json()["token"]

    def search(self, t, imdb):
        n = imdb[2:].lstrip("0")
        p = {"languages": "en", "order_by": "download_count"}
        if t.kind == "tv":
            p.update(parent_imdb_id=n, season_number=t.season, episode_number=t.episode)
        else:
            p["imdb_id"] = n
        r = SESSION.get(f"{self.api}/subtitles", headers=self.h, params=p, timeout=TIMEOUT)
        if r.status_code == 429:
            raise RateLimited("opensubtitles 429")
        out = []
        for d in r.json().get("data", []):
            a = d["attributes"]
            if a.get("foreign_parts_only") or not a.get("files"):
                continue
            fd = a.get("feature_details", {})
            out.append(Candidate(self.name, a.get("release", ""), str(a["files"][0]["file_id"]),
                                 bool(a.get("hearing_impaired")), int(a.get("download_count") or 0),
                                 fd.get("season_number"), fd.get("episode_number")))
        return out

    def download(self, c):
        r = SESSION.post(f"{self.api}/download", headers=self.h, json={"file_id": int(c.url)},
                         timeout=TIMEOUT)
        if r.status_code in (406, 429):
            raise RateLimited("opensubtitles download quota used up")
        r.raise_for_status()
        return decode(get(r.json()["link"]).content)


ALL_PROVIDERS = [Wyzie, SubDL, OpenSubtitles]


def load_saved_keys():
    """`setx` only reaches windows opened afterwards. On Windows, also read the keys it saved
    (HKCU\\Environment), so the scripts work in the same window right after setx."""
    if os.name != "nt":
        return
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            for cls in ALL_PROVIDERS:
                for name in cls.env:
                    if not os.environ.get(name):
                        try:
                            os.environ[name] = winreg.QueryValueEx(k, name)[0]
                        except FileNotFoundError:
                            pass
    except OSError:
        pass


def configured_providers(only=None):
    load_saved_keys()
    out, skipped = [], []
    for cls in ALL_PROVIDERS:
        if only and cls.name not in only:
            continue
        missing = [e for e in cls.env if not os.environ.get(e)]
        if missing:
            skipped.append(f"{cls.name} (set {', '.join(missing)})")
            continue
        try:
            out.append(cls())
        except Exception as e:
            skipped.append(f"{cls.name} (login failed: {e})")
    return out, skipped


# ------------------------------------------------------------------ checks
NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
COMMON = set("""The This That There They Then What When Where Why How Who You Your Yes Yeah
Okay Well Now Here Come Look Just Don Let And But Not Are Was Were Can Get Got Thank Thanks Please
Sorry Hey Good Right Sir Mister Miss Mrs Man God Oh All One Two Okay Maybe Because Never Ever Wait
Stop Go Hello Listen Tell Know Think Want Need Like Love Time Day Night Today Tomorrow Mom Dad""".split())


def names(text):
    return Counter(w for w in NAME_RE.findall(text) if w not in COMMON)


class ShowVocab:
    """Character names that appear in most episodes of a show already on disk."""

    def __init__(self, subs_dir, exclude=()):
        self.subs, self.exclude, self.cache = Path(subs_dir), set(exclude), {}

    def for_show(self, show_prefix, leave_out=None):
        key = (show_prefix, leave_out)
        if key not in self.cache:
            files = [p for p in self.subs.glob(f"{show_prefix}__s*.txt")
                     if p.stem not in self.exclude and p.stem != leave_out]
            df, lower = Counter(), Counter()
            for p in files:
                text = p.read_text(encoding="utf-8", errors="ignore")
                df.update(set(names(text)))
                lower.update(set(re.findall(r"\b[a-z]{3,}\b", text)))
            # a name: in most episodes, and (almost) never written in lowercase ("After", "Could" are)
            usual = {w for w, c in df.items()
                     if c >= max(2, 0.4 * len(files)) and lower[w.lower()] <= 0.1 * c} \
                if len(files) >= 3 else set()
            self.cache[key] = usual
        return self.cache[key]


def check(target, cand, text, vocab):
    """Return None if the download is acceptable, else the reason it was rejected."""
    if target.kind == "tv":
        if cand.season is not None and cand.episode is not None and cand.provider != "wyzie":
            if (int(cand.season), int(cand.episode)) != (target.season, target.episode) \
                    and not cand.extra.get("full_season"):
                return f"provider says S{cand.season}E{cand.episode}"
        if cand.release and re.search(r"s\d+\s*e\d+|\d+x\d+", cand.release, re.I) \
                and not ep_code_re(target.season, target.episode).search(cand.release):
            return f"release name is another episode ({cand.release[:60]})"
    if re.search(r"\bforced\b", cand.release, re.I):
        return "forced (signs-only) subtitle"
    n = len(text.split())
    if n < MIN_WORDS[target.kind]:
        return f"only {n} words"
    if target.kind == "tv" and vocab is not None:
        usual = vocab.for_show(target.out_name.split("__s")[0])
        if len(usual) >= NAME_CHECK_MIN_NAMES:
            share = len(usual & set(names(text))) / len(usual)
            if share < NAME_OVERLAP_MIN:
                return f"only {share:.0%} of the show's usual character names - probably another show"
    return None


# ------------------------------------------------------------------ progress
class Progress:
    def __init__(self, total, counts=None):
        self.total, self.t0 = total, time.time()
        self.counts = Counter(counts or {})
        self.start = self.finished

    def bump(self, outcome):
        self.counts[outcome] += 1

    @property
    def finished(self):
        return sum(self.counts.values())

    def line(self, current=""):
        f = self.finished
        pct = 100 * f / self.total if self.total else 100.0
        bar = "#" * int(pct / 4) + "-" * (25 - int(pct / 4))
        rate = (f - self.start) / max(time.time() - self.t0, 1e-6)
        eta = (self.total - f) / rate if rate > 0 else 0
        eta_s = f"{int(eta // 60)}m{int(eta % 60):02d}s" if rate > 0 else "--"
        c = self.counts
        msg = (f"\r[{bar}] {pct:5.1f}%  {f}/{self.total}  ok {c['done']}  not found {c['not_found']}"
               f"  rejected {c['rejected']}  no id {c['no_id']}  ETA {eta_s}  {current[:40]:<40}")
        sys.stdout.write(msg)
        sys.stdout.flush()


# ------------------------------------------------------------------ downloader
class Downloader:
    """Search every configured provider for each target, check the download, save it to out_dir.
    Progress is kept in out_dir/_progress.json, so a stopped run resumes where it left off."""

    def __init__(self, out_dir, providers, vocab=None, retry=False, delay=1.0):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_p = self.out / "_progress.json"
        self.state = json.loads(self.state_p.read_text()) if self.state_p.exists() else {}
        self.providers, self.vocab, self.retry, self.delay = providers, vocab, retry, delay
        self.ids = Ids()

    def save(self):
        self.state_p.write_text(json.dumps(self.state, indent=1, sort_keys=True))

    def _imdb(self, t):
        info = self.ids.show(t.title, t.year) if t.kind == "tv" else self.ids.movie(t.title, t.year)
        return info.get("imdb"), info

    def _one(self, t):
        imdb, info = self._imdb(t)
        if not imdb:
            return "no_id", {"why": info.get("error", "no IMDb id")}
        rejected, limited = [], False
        for p in list(self.providers):
            try:
                time.sleep(self.delay)
                cands = p.search(t, imdb)
            except RateLimited as e:
                print(f"\n  {p.name}: daily limit reached ({e}); skipping it for the rest of this run")
                self.providers.remove(p)
                limited = True
                continue
            except Exception as e:
                rejected.append(f"{p.name} search error: {e}")
                continue
            cands.sort(key=lambda c: (c.hi, -c.downloads))      # non-HI, most downloaded first
            for c in cands[:3]:
                try:
                    time.sleep(self.delay)
                    srt = p.download(c, t) if isinstance(p, SubDL) else p.download(c)
                except RateLimited as e:
                    print(f"\n  {p.name}: download limit reached ({e})")
                    self.providers.remove(p)
                    limited = True
                    break
                except Exception as e:
                    rejected.append(f"{p.name} download error: {e}")
                    continue
                text = srt_to_text(srt)
                why = check(t, c, text, self.vocab)
                if why:
                    rejected.append(f"{p.name}: {why}")
                    continue
                (self.out / f"{t.out_name}.txt").write_text(text, encoding="utf-8")
                return "done", {"provider": p.name, "release": c.release, "words": len(text.split()),
                                "imdb": imdb}
        if limited and not self.providers:          # never fully searched: keep it for the next run
            raise RateLimited("all providers hit their limit")
        return ("rejected" if rejected else "not_found"), {"why": rejected[:6]}

    def run(self, targets):
        """Already-finished targets count towards the %; with retry, only successes do."""
        todo, before = [], Counter()
        for t in targets:
            outcome = self.state.get(t.out_name, {}).get("outcome")
            if outcome == "done" or (outcome and not self.retry):
                before[outcome] += 1
            else:
                todo.append(t)
        prog = Progress(len(targets), before)
        print(f"{len(targets)} targets: {sum(before.values())} already handled, {len(todo)} to go. "
              f"Providers: {', '.join(p.name for p in self.providers) or 'none'}")
        prog.line()
        for t in todo:
            if not self.providers:
                print("\nNo providers left for today (limits reached). Run again later to continue.")
                break
            prog.line(t.label)
            try:
                outcome, info = self._one(t)
            except RateLimited:
                print("\nEvery provider hit its daily limit. Run the same command again tomorrow; "
                      "it continues where it stopped.")
                break
            except Exception as e:
                outcome, info = "not_found", {"why": [f"error: {e}"]}
            self.state[t.out_name] = {"outcome": outcome, "label": t.label, "reason": t.reason,
                                      "kind": t.kind, **info, "at": time.strftime("%Y-%m-%d %H:%M")}
            self.save()
            prog.bump(outcome)
            prog.line(t.label)
        print()
        return prog.counts
