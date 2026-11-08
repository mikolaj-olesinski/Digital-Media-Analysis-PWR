"""YouTube re-fetch (E194): compare the export's view counts with the public counts today.

Samples 60 YouTube mentions with views > 0, re-reads each with yt-dlp (view_count, like_count, comment_count) and reports
the ratio, the rank correlation and the share within 5%/20%.

    .venv/bin/python src/youtube_refetch.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
YTDLP = str(Path(sys.executable).with_name("yt-dlp"))


def main():
    d = eu.load(ROOT)
    d = d[(d.platform == "youtube") & d.url.notna() & (d.view_count > 0)].sample(60, random_state=42)
    rows = []
    for r in d.itertuples():
        try:
            out = subprocess.run([YTDLP, r.url, "--dump-json", "--no-warnings", "--ignore-errors", "--no-download"],
                                 capture_output=True, text=True, timeout=60).stdout.strip()
            j = json.loads(out.splitlines()[0]) if out else {}
        except Exception:
            j = {}
        rows.append({"old_views": float(r.view_count), "new_views": j.get("view_count"),
                     "new_likes": j.get("like_count"), "new_comments": j.get("comment_count")})
    ok = [x for x in rows if x["new_views"]]
    old = np.array([x["old_views"] for x in ok]); new = np.array([x["new_views"] for x in ok])
    ratio = new / old
    out = {"rows": len(rows), "ok": len(ok), "failed": len(rows) - len(ok),
           "median_ratio": round(float(np.median(ratio)), 3),
           "p10_ratio": round(float(np.quantile(ratio, .1)), 3),
           "p90_ratio": round(float(np.quantile(ratio, .9)), 3),
           "share_within_5pct": round(float((np.abs(ratio - 1) <= .05).mean()), 3),
           "share_within_20pct": round(float((np.abs(ratio - 1) <= .2).mean()), 3),
           "spearman_log_views": round(float(spearmanr(np.log10(old), np.log10(new)).statistic), 3),
           "median_old_views": float(np.median(old)), "median_new_views": float(np.median(new))}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "youtube_refetch_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
