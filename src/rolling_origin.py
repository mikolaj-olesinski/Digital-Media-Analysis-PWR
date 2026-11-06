"""Rolling-origin validation (E136): for the last five months with enough posts, train on everything before the month and
test on the month. Is the brand-CV MAE representative of the future?

    .venv/bin/python src/rolling_origin.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import extra_models as em  # noqa: E402
import reaction_tiers as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = eu.load(ROOT)
    f, meta = em.load(ROOT)
    dates = raw.created_at.loc[f.index].reset_index(drop=True)
    month = dates.dt.to_period("M")
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True)
        mo = month[m].reset_index(drop=True)
        counts = mo.value_counts().sort_index()
        months = [k for k in counts.index if counts[k] >= 150][-5:]
        rows = []
        for k in months:
            tr = (mo < k).values; te = (mo == k).values
            if tr.sum() < 800: continue
            pred = rt.FINAL().fit(X[tr], y[tr]).predict(X[te])
            rows.append({"month": str(k), "train": int(tr.sum()), "test": int(te.sum()),
                         "MAE": round(float(np.abs(y[te].values - pred).mean()), 3)})
        maes = [r["MAE"] for r in rows]
        out[p] = {"windows": rows, "mean": round(float(np.mean(maes)), 3), "sd": round(float(np.std(maes)), 3)}
        print(p, out[p])
    (ROOT / "data" / "processed" / "rolling_origin_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
