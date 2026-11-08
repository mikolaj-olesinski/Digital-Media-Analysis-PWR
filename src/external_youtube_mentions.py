"""Third-party YouTube mentions (E191): videos about brands, not the brands' own channels.

The earlier external samples searched the brand name and also pulled the brand's channel, so they measured brand-owned
videos. This collector searches several mention-like queries, removes videos from the brand's own channel (heuristic:
channel name contains the brand, or the channel dominates the results), and draws top / middle / low-view third-party
videos. That is the population closest to the Brand24 mentions.

    .venv/bin/python src/external_youtube_mentions.py [n_brands]
"""
import json
import re
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
QUERIES = ["{b} review", "{b} unboxing", "{b} haul", "{b} honest review"]


def flat(target, limit=30):
    cmd = [YTDLP, target, "--flat-playlist", "--dump-json", "--no-warnings", "--ignore-errors",
           "--playlist-end", str(limit), "--no-download"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
    rows = []
    for line in out.splitlines():
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append({"id": j.get("id"), "channel_id": j.get("channel_id"), "channel": j.get("channel"),
                     "view_count": j.get("view_count"), "upload_date": j.get("upload_date"),
                     "duration": j.get("duration"), "title": j.get("title")})
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


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    n_brands = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    brands = list(eu.load(ROOT).brand.value_counts().head(n_brands).index)
    rows = []
    for b in brands:
        frames = []
        for q in QUERIES:
            try:
                f = flat(f"ytsearch30:{q.format(b=b)}", 30)
            except subprocess.TimeoutExpired:
                continue
            if not f.empty:
                f["query"] = q.format(b=b)
                frames.append(f)
        if not frames:
            continue
        s = pd.concat(frames, ignore_index=True).drop_duplicates("id")
        s["upload_date"] = s.upload_date.fillna("")
        s["duration"] = pd.to_numeric(s.duration, errors="coerce")
        s["view_count"] = pd.to_numeric(s.view_count, errors="coerce")
        s = s[(s.duration.fillna(0) <= 900) & (s.view_count.notna())]  # dates come only with the full metadata
        if s.empty:
            continue
        # brand-owned: channel name contains the brand, or dominates the results
        top_ch = Counter(s.channel_id.dropna()).most_common(1)[0][0]
        own = s.channel.map(lambda c: norm(b).replace(" ", "") in norm(c)) | (s.channel_id == top_ch)
        third = s[~own].copy()
        if third.empty:
            continue
        third["brand_followers"] = None
        pos = third.view_count.rank(method="first")
        picks = list(third.nlargest(2, "view_count").id)
        picks += list(third.sort_values("view_count").head(2).id)          # the lowest views we can see
        mid = third[(pos > pos.quantile(.4)) & (pos < pos.quantile(.6))]
        picks += list(mid.head(2).id)
        for vid in dict.fromkeys(picks):
            try:
                r = full(vid)
            except subprocess.TimeoutExpired:
                continue
            if r and str(r.get("upload_date") or "") >= MIN_DATE and (r.get("duration_s") or 0) <= 900:
                r["query_brand"] = b
                r["stratum"] = "top" if vid in set(third.nlargest(2, "view_count").id) else ("low" if vid in set(third.sort_values("view_count").head(2).id) else "mid")
                rows.append(r)
        print(b, len(picks), flush=True)
    if not rows:
        raise SystemExit("no rows")
    d = pd.DataFrame(rows).drop_duplicates("video_id")
    d.to_csv(ROOT / "data" / "processed" / "external_youtube_mentions_fetched.csv", index=False)
    v = d.view_count.astype(float)
    out = {"rows": int(len(d)), "brands": int(d.query_brand.nunique()),
           "zero_views": int((v == 0).sum()), "below_100_views": int((v < 100).sum()),
           "below_1000_views": int((v < 1000).sum()),
           "median_views": float(np.median(v)), "p10_views": float(np.quantile(v, .1)),
           "p90_views": float(np.quantile(v, .9)),
           "by_stratum": {k: {"n": int(len(g)), "median_views": float(np.median(g.view_count.astype(float)))}
                          for k, g in d.groupby("stratum")},
           "share_likes": round(float(d.like_count.notna().mean()), 3),
           "share_comments": round(float(d.comment_count.notna().mean()), 3),
           "share_followers": round(float(d.channel_follower_count.notna().mean()), 3),
           "date_min": str(d.upload_date.min()), "date_max": str(d.upload_date.max())}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "external_youtube_mentions_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
