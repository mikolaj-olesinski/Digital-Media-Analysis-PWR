"""Full same-moment re-read of X views/likes/replies/reposts (B24/E201).

Looks up every X mention by id (100 ids per call) with public metrics, and appends to
data/processed/x_full_fetched.csv. Paid: $0.005 per post read, deduplicated within 24 h; the script prints the estimate
first and refuses to run above --max-usd. Resumable.

    .venv/bin/python src/full_reread_x.py --dry-run
    .venv/bin/python src/full_reread_x.py --max-usd 5 --yes
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "x_full_fetched.csv"
COLS = ["mention_id", "platform", "views", "likes", "comments", "shares", "fetched_at", "status", "source_field"]
API = "https://api.x.com/2/tweets"
POST_COST = 0.005


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def token():
    p = ROOT / ".tools" / "x_token.txt"
    if p.exists():
        return p.read_text().strip()
    import os
    t = os.environ.get("X_BEARER_TOKEN", "").strip()
    if not t:
        raise SystemExit("missing X token")
    return t


def ids():
    d = eu.load(ROOT)
    x = d[(d.platform == "twitter") & d.url.notna()][["mention_id", "url"]].astype({"mention_id": str})
    x["tid"] = x.url.str.extract(r"/status(?:es)?/(\d+)")
    return x.dropna(subset=["tid"])


def done_ids():
    if not OUT.exists():
        return set()
    d = pd.read_csv(OUT, dtype={"mention_id": str})
    return set(d[d.status.isin(["ok", "gone", "no_views"])].mention_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--free", action="store_true", help="use the free FxTwitter API instead of the paid X API")
    a = ap.parse_args()
    x = ids()
    todo = x[~x.mention_id.isin(done_ids())]
    est = len(todo) * POST_COST
    print(json.dumps({"ids": len(x), "todo": len(todo), "estimated_usd": round(est, 2), "max_usd": a.max_usd}, indent=1))
    if a.dry_run:
        return
    if a.free:
        free_run(todo)
        return
    if est > a.max_usd and not a.yes:
        raise SystemExit(f"estimate ${est:.2f} > ${a.max_usd:.2f}; rerun with --yes after approval")
    tok = token()
    buf, spent = [], 0
    for start in range(0, len(todo), 100):
        chunk = todo.iloc[start:start + 100]
        import urllib.parse, urllib.request
        q = urllib.parse.urlencode({"ids": ",".join(chunk.tid), "tweet.fields": "public_metrics,created_at"})
        req = urllib.request.Request(f"{API}?{q}", headers={"Authorization": f"Bearer {tok}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
        except Exception as e:  # noqa: BLE001
            print("error batch", e, flush=True)
            time.sleep(5)
            continue
        got = {t["id"]: t for t in data.get("data", [])}
        for r in chunk.itertuples():
            t = got.get(r.tid)
            m = (t or {}).get("public_metrics") or {}
            st = "ok" if t and m.get("impression_count") is not None else ("gone" if not t else "no_views")
            buf.append({"mention_id": r.mention_id, "platform": "twitter", "views": m.get("impression_count"),
                        "likes": m.get("like_count"), "comments": m.get("reply_count"),
                        "shares": m.get("retweet_count"), "fetched_at": now(), "status": st,
                        "source_field": "public_metrics.impression_count"})
        spent += len(chunk) * POST_COST
        print("batch", start // 100 + 1, "spent $", round(spent, 2), flush=True)
        pd.DataFrame(buf, columns=COLS).to_csv(OUT, mode="a", header=not OUT.exists() or OUT.stat().st_size == 0, index=False)
        buf = []
        time.sleep(1.0)
    print("finished, estimated spend $", round(spent, 2), flush=True)


def free_run(todo):
    import random, urllib.request, urllib.error
    buf, consec = [], 0
    for i, r in enumerate(todo.itertuples(), 1):
        try:
            req = urllib.request.Request(f"https://api.fxtwitter.com/i/status/{r.tid}",
                                         headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                j = json.load(resp)
            t = j.get("tweet") or {}
            st = "ok" if t.get("views") is not None else "no_views"
            buf.append({"mention_id": r.mention_id, "platform": "twitter", "views": t.get("views"),
                        "likes": t.get("likes"), "comments": t.get("replies"), "shares": t.get("retweets"),
                        "fetched_at": now(), "status": st, "source_field": "fxtwitter.views"})
        except urllib.error.HTTPError as e:
            st = "gone" if e.code == 404 else ("blocked" if e.code == 429 else "error")
            if st == "gone":
                consec = 0
            buf.append({"mention_id": r.mention_id, "platform": "twitter", "views": None, "likes": None,
                        "comments": None, "shares": None, "fetched_at": now(), "status": st,
                        "source_field": "fxtwitter.views"})
            if e.code == 429:
                pd.DataFrame(buf, columns=COLS).to_csv(OUT, mode="a", header=not OUT.exists() or OUT.stat().st_size == 0, index=False)
                print("stopping: 429", flush=True)
                return
            consec += 1
        except Exception:  # noqa: BLE001
            buf.append({"mention_id": r.mention_id, "platform": "twitter", "views": None, "likes": None,
                        "comments": None, "shares": None, "fetched_at": now(), "status": "error",
                        "source_field": "fxtwitter.views"})
            consec += 1
        if i % 50 == 0:
            pd.DataFrame(buf, columns=COLS).to_csv(OUT, mode="a", header=not OUT.exists() or OUT.stat().st_size == 0, index=False)
            buf = []
            print("done", i, flush=True)
        if consec >= 10:
            print("stopping: repeated errors", flush=True)
            break
        time.sleep(1.0 + random.random() * 0.5)
    if buf:
        pd.DataFrame(buf, columns=COLS).to_csv(OUT, mode="a", header=not OUT.exists() or OUT.stat().st_size == 0, index=False)
    print("finished free run", flush=True)


if __name__ == "__main__":
    main()
