"""Classify why YouTube likes/comments are missing (E204 fix): hidden likes vs disabled comments vs tool gap.

For every ok YouTube row with a missing like or comment count, read the public watch page and look for the signals:
"comments are turned off/disabled" and a visible numeric like count. Writes a flags CSV; resumable, 1 req/s.

    python src/yt_reaction_flags.py
"""
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "youtube_reaction_flags.csv"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"
COLS = ["mention_id", "likes_hidden", "comments_off", "like_visible", "comments_off_page", "checked_at"]


def main():
    y = pd.read_csv(ROOT / "data" / "processed" / "youtube_full_fetched.csv", dtype={"mention_id": str})
    todo = y[(y.status == "ok") & (y.likes.isna() | y.comments.isna())][["mention_id"]].drop_duplicates()
    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT, dtype={"mention_id": str}).mention_id)
    todo = todo[~todo.mention_id.isin(done)]
    raw = eu.load(ROOT)[["mention_id", "url"]].astype({"mention_id": str}).set_index("mention_id").url
    print("todo", len(todo), flush=True)
    for i, r in enumerate(todo.itertuples(), 1):
        url = raw.get(r.mention_id)
        html = ""
        if isinstance(url, str):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    html = resp.read().decode("utf-8", "ignore")
            except Exception:  # noqa: BLE001
                pass
        off = bool(re.search(r"comments? (are|is) turned off|comments? (are|is) disabled", html, re.I))
        like_visible = bool(re.search(r'"likeCount"\s*:\s*"?\d', html) or re.search(r"like this video along with [\d,.]+", html)
                            or re.search(r"[\d,.]+[KM]? likes", html))
        row = {"mention_id": r.mention_id, "likes_hidden": "unknown", "comments_off": "unknown" if not html else str(off).lower(),
               "like_visible": str(like_visible).lower() if html else "unknown",
               "comments_off_page": str(off).lower() if html else "unknown",
               "checked_at": pd.Timestamp.utcnow().isoformat(timespec="seconds")}
        pd.DataFrame([row], columns=COLS).to_csv(OUT, mode="a", header=not OUT.exists() or OUT.stat().st_size == 0, index=False)
        if i % 50 == 0:
            print("done", i, flush=True)
        time.sleep(1.0 + random.random() * 0.5)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
