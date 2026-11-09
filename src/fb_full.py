"""Fetch public Facebook views for the whole Facebook slice (E199), resumable.

The export has 2,500 Facebook rows and no view counts; the earlier sample was 470 attempts (402 readable), stratified by
reactions. This script walks the remaining rows with a small delay, resumes from the previous output and stores the
counters locally. Run:

    .venv/bin/python src/fb_full.py
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
from fetch_views import _facebook  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "fb_full_fetched.csv"


def main():
    raw = eu.load(ROOT)
    fb = raw[(raw.platform == "facebook") & raw.url.notna()][["mention_id", "url"]].astype({"mention_id": str})
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT, dtype={"mention_id": str})
        done = set(prev[prev.status == "ok"].mention_id)
    todo = fb[~fb.mention_id.isin(done)]
    print("todo", len(todo), "done", len(done), flush=True)

    def one(r):
        time.sleep(0.4)
        res = _facebook(r.url)
        return {"mention_id": r.mention_id, "status": res.get("status"), "views": res.get("views"),
                "likes": res.get("likes"), "reactions": res.get("reactions"),
                "comments": res.get("comments"), "plays": res.get("plays")}

    buf = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        for i, r in enumerate(ex.map(one, list(todo.itertuples())), 1):
            buf.append(r)
            if i % 100 == 0:
                df = pd.DataFrame(buf)
                df.to_csv(OUT, mode="a", header=not OUT.exists() or OUT.stat().st_size == 0, index=False)
                buf = []
                print("done", i, flush=True)
    if buf:
        pd.DataFrame(buf).to_csv(OUT, mode="a", header=not OUT.exists() or OUT.stat().st_size == 0, index=False)
    d = pd.read_csv(OUT, dtype={"mention_id": str})
    ok = d[(d.status == "ok") & (d.views > 0)]
    out = {"attempted": int(len(d)), "ok": int(len(ok)), "sum_views": int(ok.views.sum()),
           "median_views": float(ok.views.median()) if len(ok) else None}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "fb_full_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
