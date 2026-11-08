"""Diverse external YouTube sample (E190): popular, recent, small and zero-view brand videos.

For each top brand: search YouTube, take the brand's most frequent channel among the results, list that channel's
uploads, and draw four strata - most viewed from search, most recent from search, least viewed from the channel, newest
from the channel. This deliberately includes small and zero-view videos, which a plain search never returns.

    .venv/bin/python src/external_youtube_diverse.py [n_brands]
"""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
YTDLP = str(Path(sys.executable).with_name("yt-dlp"))
MIN_DATE = "20250101"


def flat(target, limit):
    cmd = [YTDLP, target, "--flat-playlist", "--dump-json", "--no-warnings", "--ignore-errors",
           "--playlist-end", str(limit), "--no-download"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
    rows = []
    for line in out.splitlines():
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append({"id": j.get("id"), "view_count": j.get("view_count"), "channel_id": j.get("channel_id"),
                     "upload_date": j.get("upload_date"), "title": j.get("title")})
    return pd.DataFrame(rows)


def full(vid):
    cmd = [YTDLP, f"https://www.youtube.com/watch?v={vid}", "--dump-json", "--no-warnings", "--ignore-errors",
           "--no-download"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
    if not out:
        return None
    j = json.loads(out.splitlines()[0])
    return {"video_id": j.get("id"), "channel_id": j.get("channel_id"), "view_count": j.get("view_count"),
            "like_count": j.get("like_count"), "comment_count": j.get("comment_count"),
            "channel_follower_count": j.get("channel_follower_count"),
            "upload_date": j.get("upload_date"), "duration_s": j.get("duration"), "title": j.get("title")}


def main():
    n_brands = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    brands = list(eu.load(ROOT).brand.value_counts().head(n_brands).index)
    rows = []
    for b in brands:
        try:
            s = flat(f"ytsearch20:{b}", 20)
        except subprocess.TimeoutExpired:
            continue
        if s.empty:
            continue
        s["upload_date"] = s.upload_date.fillna("")
        s = s[(s.view_count.notna()) & (s.view_count >= 0)]
        ch = Counter(s.channel_id.dropna()).most_common(1)
        strata = []
        if ch:
            try:
                up = flat(f"https://www.youtube.com/channel/{ch[0][0]}/videos", 30)
            except subprocess.TimeoutExpired:
                up = pd.DataFrame()
            if not up.empty:
                up["upload_date"] = up.upload_date.fillna("")
                up = up[up.view_count.notna()]
                strata.append(("small", list(up.sort_values("view_count").head(3).id)))
                strata.append(("newest", list(up.sort_values("upload_date", ascending=False).head(3).id)))
        strata.append(("popular", list(s.sort_values("view_count", ascending=False).head(3).id)))
        strata.append(("recent", list(s.sort_values("upload_date", ascending=False).head(3).id)))
        seen = set()
        for name, ids in strata:
            for vid in ids:
                if vid in seen:
                    continue
                seen.add(vid)
                try:
                    r = full(vid)
                except subprocess.TimeoutExpired:
                    continue
                if r:
                    r["query_brand"] = b
                    r["stratum"] = name
                    rows.append(r)
        print(b, len(seen), flush=True)
    if not rows:
        raise SystemExit("no rows")
    d = pd.DataFrame(rows).drop_duplicates("video_id")
    d.to_csv(ROOT / "data" / "processed" / "external_youtube_diverse_fetched.csv", index=False)
    v = d.view_count.astype(float)
    out = {"rows": int(len(d)), "brands": int(d.query_brand.nunique()),
           "zero_views": int((v == 0).sum()), "below_100_views": int((v < 100).sum()),
           "median_views": float(np.median(v)), "p10_views": float(np.quantile(v, .1)),
           "p90_views": float(np.quantile(v, .9)),
           "by_stratum": {k: {"n": int(len(g)), "median_views": float(np.median(g.view_count.astype(float))),
                              "zero": int((g.view_count == 0).sum())}
                          for k, g in d.groupby("stratum")},
           "share_likes": round(float(d.like_count.notna().mean()), 3),
           "share_comments": round(float(d.comment_count.notna().mean()), 3),
           "share_followers": round(float(d.channel_follower_count.notna().mean()), 3),
           "date_min": str(d.upload_date.min()), "date_max": str(d.upload_date.max())}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "external_youtube_diverse_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
