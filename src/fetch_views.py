"""Probe: re-fetch public view counts for mention URLs (feasibility, not a data source yet).

Re-fetches the view count that a logged-out visitor can see today, for a sample of mention URLs, and writes numbers
only (no author names, no URLs). Used to answer: which platforms expose views at all, and how far today's count is
from the snapshot Brand24 delivered (`view_count`).

    .venv/bin/python src/fetch_views.py --limit 8 --out data/processed/fetched_views_probe.csv

Platform routes (all public pages, no account):
- X: FxTwitter JSON (`views`) - the public x.com page needs a login, this mirror does not.
- Facebook: the public watch page (`video_view_count`, `reaction_count`, `total_comment_count`, `play_count`), no
  cookies; the page that yt-dlp reads also carries these counts, yt-dlp just does not expose them.
- YouTube, TikTok, Instagram: yt-dlp metadata (`view_count`). Instagram answers only with a logged-in session
  (`--cookies-from-browser firefox`), otherwise `needs_auth`.

Terms: automated access is outside the platform terms even for public pages; the GDPR and terms decision for any use
of these numbers belongs to the team (backlog B14, `docs/en/fbig_manual_check.md`). Do not run at full scale before
that decision. Fetching is rate-limited and jittered; deleted or protected posts come back as `gone`.
"""
import argparse
import json
import random
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
YTDLP = str(ROOT / ".venv" / "bin" / "yt-dlp")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
COLS = ["mention_id", "platform", "status", "views", "likes", "reactions", "comments", "shares", "plays",
        "post_views", "posted_at", "fetched_at", "source", "views_dataset", "note"]
LOCK = threading.Lock()


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _classify(msg):
    m = msg.lower()
    if any(k in m for k in ("login", "cookies", "empty media response", "not available in your country")):
        return "needs_auth"
    if any(k in m for k in ("404", "not found", "unavailable", "removed", "does not exist", "private")):
        return "gone"
    if "429" in m or "rate" in m:
        return "rate_limited"
    if "timed out" in m or "timeout" in m:
        return "timeout"
    return "error"


def _ytdlp(url, cookies, cookiefile=None):
    cmd = [YTDLP, "--no-warnings", "--no-playlist", "--skip-download", "-J", "--socket-timeout", "25"]
    if cookiefile:
        cmd += ["--cookies", cookiefile]
    if cookies:
        cmd += ["--cookies-from-browser", cookies]
    cmd.append(url)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        return None, (p.stderr.strip().splitlines() or ["yt-dlp failed"])[-1]
    return json.loads(p.stdout), None


def _x(url):
    m = re.search(r"/status(?:es)?/(\d+)", url)
    if not m:
        return {"status": "error", "note": "no status id"}
    req = urllib.request.Request(f"https://api.fxtwitter.com/i/status/{m.group(1)}",
                                 headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return {"status": "gone" if e.code == 404 else ("rate_limited" if e.code == 429 else "error"),
                "note": f"http {e.code}"}
    t = d.get("tweet") or {}
    if not t.get("views") and not t.get("id"):
        return {"status": "gone", "note": str(d.get("message"))[:80]}
    return {"status": "ok", "views": t.get("views"), "likes": t.get("likes"), "comments": t.get("replies"),
            "shares": t.get("retweets"), "posted_at": t.get("created_at"), "source": "fxtwitter"}


def _facebook(url):
    """Public watch page of a Reel or video: views, reactions, comments (shares are not exposed)."""
    m = re.search(r"/(?:reel|videos)/(\d+)", url) or re.search(r"[?&]v=(\d+)", url)
    if not m:
        return {"status": "error", "note": "no video id"}
    vid = m.group(1)
    req = urllib.request.Request(f"https://www.facebook.com/watch/?v={vid}",
                                 headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml",
                                          "Accept-Language": "en-US,en;q=0.9"})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            h = r.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        return {"status": "gone" if e.code == 404 else "error", "note": f"http {e.code}"}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "note": f"{type(e).__name__}: {e}"[:120]}
    # anchor on the view-count renderer of *this* video; related videos carry their own counters
    mm = re.search(r'"associated_video":\{[^}]*?"id":"%s"\},"video_view_count":(\d+)' % vid, h)
    if not mm:
        return {"status": "gone", "note": "watch page without this video"}
    head, block = h[:mm.start()], h[mm.start():mm.start() + 600]
    def last(pat):
        x = re.findall(pat, head)
        return int(x[-1]) if x else None
    play = re.search(r'"play_count":(\d+)', block)
    post = re.search(r'"video_post_view_count":(\d+)', block)
    # Brand24 `likes` counts the Like reaction, not all reactions: keep both
    like = last(r'"localized_name":"Like"\},"i18n_reaction_count":"\d+","reaction_count":(\d+)')
    return {"status": "ok", "views": int(mm.group(1)), "plays": int(play.group(1)) if play else None,
            "post_views": int(post.group(1)) if post else None,
            "likes": like, "reactions": last(r'"reaction_count":\{"count":(\d+)'),
            "comments": last(r'"total_comment_count":(\d+)'), "source": "fb:watch"}


def fetch_one(row, cookies, cookiefile=None):
    url, plat = row["url"], row["platform"]
    base = {"mention_id": str(row["mention_id"]), "platform": plat, "fetched_at": _now(),
            "views_dataset": row.get("view_count") if pd.notna(row.get("view_count")) else None}
    if not isinstance(url, str) or not url.startswith("http"):
        return {**base, "status": "error", "note": "no url"}
    if plat == "twitter":
        return {**base, **_x(url)}
    if plat == "facebook":
        return {**base, **_facebook(url)}
    try:
        info, err = _ytdlp(url, cookies, cookiefile)
    except subprocess.TimeoutExpired:
        return {**base, "status": "timeout", "note": "yt-dlp 120 s"}
    except Exception as e:  # noqa: BLE001 - probe: record, do not crash the run
        return {**base, "status": "error", "note": f"{type(e).__name__}: {e}"[:120]}
    if info is None:
        return {**base, "status": _classify(err), "note": err[:160]}
    ts = info.get("timestamp")
    return {**base, "status": "ok" if info.get("view_count") is not None else "no_views",
            "views": info.get("view_count"), "likes": info.get("like_count"),
            "comments": info.get("comment_count"), "shares": info.get("repost_count"),
            "posted_at": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d") if ts else None,
            "source": f"yt-dlp:{info.get('extractor_key', '')}"}


def sample(df, platforms, limit, seed):
    out = []
    for p in platforms:
        q = df[df.platform == p]
        if limit and len(q) > limit:
            q = q.sample(limit, random_state=seed)
        out.append(q)
    return pd.concat(out).sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default=str(ROOT / "data" / "raw" / "reach_samples_students.csv"), help="raw sample CSV (needs mention_id, platform, url)")
    ap.add_argument("--platforms", default="twitter,youtube,tiktok,facebook,instagram", help="comma-separated platforms to fetch")
    ap.add_argument("--limit", type=int, default=0, help="posts per platform, 0 = all")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "fetched_views_probe.csv"), help="output CSV (local only; not committed)")
    ap.add_argument("--workers", type=int, default=2, help="parallel fetches")
    ap.add_argument("--cookies-from-browser", default=None, help="e.g. firefox; needed for Instagram")
    ap.add_argument("--cookies", default=None, help="Netscape cookies.txt exported from a logged-in browser")
    ap.add_argument("--refresh", action="store_true", help="ignore rows already fetched with status ok")
    ap.add_argument("--url", default=None, help="check a single post URL, print the parsed fields and stop")
    args = ap.parse_args()

    if args.url:
        host = urllib.parse.urlparse(args.url).netloc.lower()
        plat = ("facebook" if "facebook" in host else "instagram" if "instagram" in host else
                "tiktok" if "tiktok" in host else "youtube" if "youtu" in host else
                "twitter" if ("twitter" in host or host.endswith("x.com")) else "unknown")
        row = pd.Series({"mention_id": "-", "platform": plat, "url": args.url})
        print(json.dumps(fetch_one(row, args.cookies_from_browser, args.cookies), indent=1, ensure_ascii=False))
        return

    p = Path(args.input)
    if not p.exists():
        raise SystemExit(f"Missing input file: {p}\nUse --input <raw sample CSV>.")
    df = pd.read_csv(p, dtype={"mention_id": str})
    df = sample(df, [p.strip() for p in args.platforms.split(",") if p.strip()], args.limit, args.seed)
    out = Path(args.out)
    if out.exists() and out.open().readline().strip() != ",".join(COLS):
        out.rename(out.with_suffix(".old.csv"))  # columns changed since the last run: never append across schemas
    done = set()
    if out.exists() and not args.refresh:
        old = pd.read_csv(out, dtype={"mention_id": str})
        done = set(old.loc[old.status == "ok", "mention_id"])
    rows = [r for _, r in df.iterrows() if str(r["mention_id"]) not in done]
    print(f"{len(df)} sampled, {len(rows)} to fetch, {len(done)} cached", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        pd.DataFrame(columns=COLS).to_csv(out, index=False)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for i, r in enumerate(rows):
            futs[ex.submit(fetch_one, r, args.cookies_from_browser, args.cookies)] = r
            time.sleep(random.uniform(0.3, 0.9))
        for n, f in enumerate(as_completed(futs), 1):
            res = f.result()
            time.sleep(random.uniform(0.4, 1.0))
            with LOCK:
                pd.DataFrame([{c: res.get(c) for c in COLS}]).to_csv(out, mode="a", header=False, index=False)
            print(f"{n}/{len(rows)} {res['platform']:9s} {res['status']:10s} views={res.get('views')}", flush=True)
    print(f"done in {time.time() - t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
