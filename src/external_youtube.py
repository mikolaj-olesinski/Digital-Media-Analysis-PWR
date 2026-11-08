"""External YouTube sample (E170): brand-related videos fetched independently of the Brand24 export.

For each of the top brands by mention count, search YouTube with yt-dlp, keep the most recent videos in the Brand24
period (upload >= 2025-01-01, duration <= 15 min, views > 0), extract views/likes/comments/followers and summarise.

Fetched rows go to data/processed/external_youtube_fetched.csv (git-ignored, contains no author names, only channel ids).
The committed summary is data/processed/external_youtube_numbers.json.

    .venv/bin/python src/external_youtube.py [n_brands] [per_brand]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MIN_DATE = "20250101"


def top_brands(n):
    d = eu.load(ROOT)
    return list(d.brand.value_counts().head(n).index)


def search(brand, per_brand):
    cmd = [str(Path(sys.executable).with_name("yt-dlp")), f"ytsearch{per_brand * 3}:{brand}", "--dump-json", "--no-warnings", "--ignore-errors",
           "--playlist-end", str(per_brand * 3), "--no-download"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
    rows = []
    for line in out.splitlines():
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append({
            "platform": "youtube", "query_brand": brand, "video_id": j.get("id"),
            "channel_id": j.get("channel_id"), "view_count": j.get("view_count"),
            "like_count": j.get("like_count"), "comment_count": j.get("comment_count"),
            "channel_follower_count": j.get("channel_follower_count"),
            "upload_date": j.get("upload_date"), "duration_s": j.get("duration"),
            "title": j.get("title"),
        })
    return pd.DataFrame(rows)


def main():
    n_brands = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    per_brand = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    frames = []
    for b in top_brands(n_brands):
        try:
            f = search(b, per_brand)
        except subprocess.TimeoutExpired:
            continue
        if f.empty:
            continue
        f["upload_date"] = f.upload_date.fillna("")
        f = f[(f.view_count > 0) & (f.upload_date >= MIN_DATE) & (f.duration_s.fillna(0) <= 900)]
        f = f.sort_values("upload_date", ascending=False).head(per_brand)  # recent, not top-viewed: avoids picking only hits
        frames.append(f)
        print(b, len(f), flush=True)
    if not frames:
        raise SystemExit("no rows fetched")
    d = pd.concat(frames, ignore_index=True)
    d.to_csv(ROOT / "data" / "processed" / "external_youtube_fetched.csv", index=False)
    v = d.view_count.astype(float)
    lk = d.like_count.astype(float)
    ok = (v > 0) & (lk > 0)
    from scipy.stats import spearmanr
    rho = spearmanr(np.log10(v[ok]), np.log10(lk[ok])).statistic if ok.sum() > 5 else None
    out = {
        "rows": int(len(d)), "brands": int(d.query_brand.nunique()),
        "median_views": float(np.median(v)), "p10_views": float(np.quantile(v, .1)),
        "p90_views": float(np.quantile(v, .9)),
        "share_likes": round(float(d.like_count.notna().mean()), 3),
        "share_comments": round(float(d.comment_count.notna().mean()), 3),
        "share_followers": round(float(d.channel_follower_count.notna().mean()), 3),
        "date_min": str(d.upload_date.min()), "date_max": str(d.upload_date.max()),
        "spearman_log_views_likes": None if rho is None else round(float(rho), 3),
        "median_views_per_like": float(np.median(v[ok] / lk[ok])) if ok.sum() else None,
        "per_brand": d.query_brand.value_counts().to_dict(),
    }
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "external_youtube_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
