"""
fetch_missing.py  -  download what MISSING.md says is missing, from free subtitle providers

Setup (once): get free API keys and set them as environment variables (cmd):
    setx WYZIE_API_KEY  your-key        <- https://store.wyzie.io/redeem      (1,000 requests/day)
    setx SUBDL_API_KEY  your-key        <- subdl.com -> Profile -> API key   (2,000 requests/day)
  optional, your OpenSubtitles account (small daily quota):
    setx OPENSUBTITLES_API_KEY ...   setx OPENSUBTITLES_USERNAME ...   setx OPENSUBTITLES_PASSWORD ...
  then open a NEW cmd window so the variables are visible.

Usage:
    python fetch_missing.py --dry-run          # what would be downloaded, no requests to providers
    python fetch_missing.py                    # download into subs_incoming/ (resumes if stopped)
    python fetch_missing.py --only "X-Men"     # just titles containing this text
    python fetch_missing.py --retry            # also retry things that failed last time
    python fetch_missing.py --install          # move checked downloads into subs/
    python fetch_missing.py --quarantine-wrong # move the 45 wrong-show files to subs_wrong/

Nothing in subs/ is deleted: replaced files go to subs_replaced/, wrong-show files to subs_wrong/.
After --install, rebuild the index:  python build_index.py build
"""
import argparse, json, shutil, sys, time
from collections import Counter
from pathlib import Path

import missing_report as mr
import subtitle_providers as sp

ROOT = Path(__file__).resolve().parent
SUBS = ROOT / "subs"
INCOMING = ROOT / "subs_incoming"


def build_targets(new=True, full_seasons=False, ids=None):
    c = mr.collect()
    shows, _ = mr.fetch_catalog()
    targets = []
    for show, v in c["needed"].items():                       # 1. missing episodes
        title, year = shows.get(show, (mr.pretty(show).rsplit(" (", 1)[0], int(show[-4:])))
        for s, e in sorted(v["need"]):
            targets.append(sp.Target("tv", title, year, s, e, "missing"))
    if full_seasons:                                          # 1b. episodes TVmaze knows but we don't
        have = {t.out_name for t in targets} | c["on_disk"]
        for show, (title, year) in shows.items():
            info = ids.show(title, year)
            for s, e in info.get("episodes", []):
                t = sp.Target("tv", title, year, s, e, "missing")
                if t.out_name not in have:
                    targets.append(t)
    for stem, why, redownload in c["damaged"]:                # 2. partial movie files
        if redownload and stem.startswith("movie__"):
            _, movies = mr.fetch_catalog()
            title, year = movies.get(stem[len("movie__"):], (None, None))
            if title:
                targets.append(sp.Target("movie", title, year, reason="damaged", out_name=stem))
    if new:                                                   # 3. titles never searched for
        for _, kind, title, year in c["new"]:
            if kind == "movie":
                targets.append(sp.Target("movie", title, year, reason="new"))
            else:
                info = ids.show(title, year) if ids else {}
                for s, e in info.get("episodes", []):
                    targets.append(sp.Target("tv", title, year, s, e, "new"))
                if not info.get("episodes"):
                    targets.append(sp.Target("tv", title, year, 1, 1, "new"))   # shows up as "no id"
    seen, out = set(), []
    for t in targets:
        if t.out_name not in seen:
            seen.add(t.out_name)
            out.append(t)
    return out


def install():
    state_p = INCOMING / "_progress.json"
    if not state_p.exists():
        print("Nothing to install yet: run  python fetch_missing.py  first (downloads go to subs_incoming/).")
        return
    state = json.loads(state_p.read_text())
    status_p = ROOT / "status.json"
    status = json.loads(status_p.read_text())
    moved = Counter()
    for name, st in sorted(state.items()):
        src = INCOMING / f"{name}.txt"
        if st.get("outcome") != "done" or not src.exists() or st.get("installed"):
            continue
        dst = SUBS / src.name
        if dst.exists():                                      # keep the old copy
            (ROOT / "subs_replaced").mkdir(exist_ok=True)
            shutil.move(str(dst), ROOT / "subs_replaced" / src.name)
            moved["replaced"] += 1
        shutil.move(str(src), dst)
        st["installed"] = time.strftime("%Y-%m-%d %H:%M")
        status["items"][name] = "done"
        moved["installed"] += 1
    state_p.write_text(json.dumps(state, indent=1, sort_keys=True))
    status_p.write_text(json.dumps(status, indent=1))
    print(f"installed {moved['installed']} files into subs/ ({moved['replaced']} older copies kept in "
          f"subs_replaced/). Next: python build_index.py build   then   python missing_report.py")


def quarantine_wrong():
    (ROOT / "subs_wrong").mkdir(exist_ok=True)
    n = 0
    for stem in mr.WRONG:
        p = SUBS / f"{stem}.txt"
        if p.exists():
            shutil.move(str(p), ROOT / "subs_wrong" / p.name)
            n += 1
    print(f"moved {n} wrong-show files to subs_wrong/ (nothing deleted). "
          "Next: python build_index.py build")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="only targets whose title contains this text (case-insensitive)")
    ap.add_argument("--limit", type=int, help="at most this many targets this run")
    ap.add_argument("--retry", action="store_true", help="retry targets that failed before")
    ap.add_argument("--providers", help="comma list, e.g. wyzie,subdl (default: every configured one)")
    ap.add_argument("--no-new", action="store_true", help="skip the never-searched titles (MISSING.md part 4)")
    ap.add_argument("--full-seasons", action="store_true",
                    help="also fetch episodes TVmaze lists that MISSING.md doesn't (e.g. Spider-Man TAS)")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--quarantine-wrong", action="store_true")
    a = ap.parse_args()
    if a.install:
        return install()
    if a.quarantine_wrong:
        return quarantine_wrong()

    ids = sp.Ids()
    print("Building the target list (looks up episode lists on TVmaze the first time)...")
    targets = build_targets(new=not a.no_new, full_seasons=a.full_seasons, ids=ids)
    if a.only:
        targets = [t for t in targets if a.only.lower() in t.title.lower()]
    if a.limit:
        targets = targets[:a.limit]
    by = Counter((t.reason, t.title) for t in targets)
    print(f"\n{len(targets)} targets:")
    for reason in ("missing", "damaged", "new"):
        rows = [(n, title) for (r, title), n in by.items() if r == reason]
        if rows:
            print(f"  {reason}: {sum(n for n, _ in rows)}")
            for n, title in sorted(rows, key=lambda x: -x[0]):
                print(f"     {n:>4}  {title}")
    if a.dry_run:
        return
    providers, skipped = sp.configured_providers(a.providers.split(",") if a.providers else None)
    for s in skipped:
        print(f"  provider off: {s}")
    if not providers:
        sys.exit("\nNo provider configured. Set WYZIE_API_KEY and/or SUBDL_API_KEY (see the top of this file).")
    vocab = sp.ShowVocab(SUBS, exclude=mr.WRONG)
    counts = sp.Downloader(INCOMING, providers, vocab, retry=a.retry).run(targets)
    print(f"\nThis run: {dict(counts)}.  Downloads are in subs_incoming/  "
          f"(details per title in subs_incoming/_progress.json).")
    print("When you're happy with them:  python fetch_missing.py --install")


if __name__ == "__main__":
    main()
