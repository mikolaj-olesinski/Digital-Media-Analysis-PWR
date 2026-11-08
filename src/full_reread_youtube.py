"""Full same-moment re-read of YouTube views/likes/comments (B24/E201).

Iterates every YouTube mention in the Brand24 export, reads the public counters with yt-dlp in one request, and appends
to data/processed/youtube_full_fetched.csv. Resumable, polite (>=1 s per request), stops on repeated failures.

    .venv/bin/python src/full_reread_youtube.py
"""
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "youtube_full_fetched.csv"
COLS = ["mention_id", "platform", "views", "likes", "comments", "shares", "fetched_at", "status", "source_field"]
YTDLP = [sys.executable, "-m", "yt_dlp"]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def done_ids():
    if not OUT.exists():
        return set()
    d = pd.read_csv(OUT, dtype={"mention_id": str})
    return set(d[d.status.isin(["ok", "gone", "private", "no_views"])].mention_id)


def fetch(url):
    try:
        r = subprocess.run([*YTDLP, url, "--dump-json", "--no-warnings", "--no-download"],
                           capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {"status": "error", "note": "timeout"}
    if r.returncode != 0 or not r.stdout.strip():
        err = (r.stderr or "").lower()
        if "private" in err or "members-only" in err:
            return {"status": "private"}
        if "unavailable" in err or "removed" in err or "not exist" in err:
            return {"status": "gone"}
        return {"status": "error", "note": err[:120]}
    j = json.loads(r.stdout.splitlines()[0])
    if j.get("view_count") is None:
        return {"status": "no_views"}
    return {"status": "ok", "views": j.get("view_count"), "likes": j.get("like_count"),
            "comments": j.get("comment_count"), "field": "view_count"}


def main():
    d = eu.load(ROOT)
    yt = d[(d.platform == "youtube") & d.url.notna()][["mention_id", "url"]].astype({"mention_id": str})
    skip = done_ids()
    todo = yt[~yt.mention_id.isin(skip)]
    print("todo", len(todo), "done", len(skip), flush=True)
    buf = []
    consec = 0
    for i, r in enumerate(todo.itertuples(), 1):
        res = fetch(r.url)
        row = {"mention_id": r.mention_id, "platform": "youtube", "views": res.get("views"),
               "likes": res.get("likes"), "comments": res.get("comments"), "shares": None,
               "fetched_at": now(), "status": res["status"],
               "source_field": res.get("field") or "yt-dlp.view_count"}
        buf.append(row)
        consec = consec + 1 if res["status"] == "error" else 0
        if res["status"] == "error" and consec >= 8:
            print("stopping: repeated errors", flush=True)
            pd.DataFrame(buf, columns=COLS).to_csv(OUT, mode="a", header=not OUT.exists() or OUT.stat().st_size == 0, index=False)
            return
        if i % 25 == 0:
            pd.DataFrame(buf, columns=COLS).to_csv(OUT, mode="a", header=not OUT.exists() or OUT.stat().st_size == 0, index=False)
            buf = []
            print("done", i, flush=True)
        time.sleep(1.0 + random.random() * 0.6)
    if buf:
        pd.DataFrame(buf, columns=COLS).to_csv(OUT, mode="a", header=not OUT.exists() or OUT.stat().st_size == 0, index=False)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
