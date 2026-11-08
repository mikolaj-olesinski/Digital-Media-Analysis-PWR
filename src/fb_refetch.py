"""Facebook re-fetch (E196): how much did the public video_view_count move since 2026-11-05?

Samples 60 rows from the fetched Facebook sample with views > 0, re-reads the public watch page and compares the
counters with the stored ones.

    .venv/bin/python src/fb_refetch.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
from fetch_views import _facebook  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    old = pd.read_csv(ROOT / "data" / "processed" / "fb_all_fetched.csv", dtype={"mention_id": str})
    old = old[(old.status == "ok") & (old.views > 0)][["mention_id", "views", "fetched_at"]]
    raw = eu.load(ROOT)[["mention_id", "url", "platform"]].astype({"mention_id": str})
    d = old.merge(raw, on="mention_id").sample(min(60, len(old)), random_state=42)
    rows = []
    for r in d.itertuples():
        res = _facebook(r.url)
        rows.append({"old_views": float(r.views), "new_views": res.get("views"), "status": res.get("status")})
    ok = [x for x in rows if x["new_views"]]
    old_v = np.array([x["old_views"] for x in ok]); new_v = np.array([x["new_views"] for x in ok])
    ratio = new_v / old_v
    out = {"rows": len(rows), "ok": len(ok), "failed": len(rows) - len(ok),
           "median_ratio": round(float(np.median(ratio)), 3),
           "p10_ratio": round(float(np.quantile(ratio, .1)), 3),
           "p90_ratio": round(float(np.quantile(ratio, .9)), 3),
           "share_within_5pct": round(float((np.abs(ratio - 1) <= .05).mean()), 3),
           "share_within_20pct": round(float((np.abs(ratio - 1) <= .2).mean()), 3),
           "spearman_log_views": round(float(spearmanr(np.log10(old_v), np.log10(new_v)).statistic), 3),
           "median_old_views": float(np.median(old_v)), "median_new_views": float(np.median(new_v))}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "fb_refetch_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
