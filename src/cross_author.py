"""Cross-platform author history (E120): an author's posts on *other* platforms may carry their level even when the
target platform has no history. Add the shrunk mean log views of the same author on the other platforms as a feature.

    .venv/bin/python src/cross_author.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em  # noqa: E402
import reaction_tiers as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
K = 5.0


def main():
    f, meta = em.load(ROOT)
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X = f[m][em.ENG].reset_index(drop=True); y = meta.y[m].reset_index(drop=True)
        g = meta.brand[m].reset_index(drop=True); a = meta.author[m].fillna("NA").reset_index(drop=True)
        other = ~m
        base = np.full(len(y), np.nan); withx = np.full(len(y), np.nan); seen = np.zeros(len(y), bool)
        for tr, te in GroupKFold(5).split(X, y, g):
            test_brands = set(g.iloc[te])
            # training rows on all platforms, excluding the test brands
            tr_all = ~meta.brand.isin(test_brands).values
            tr_other = tr_all & other
            stats = pd.DataFrame({"a": meta.author[tr_other].fillna("NA"), "y": meta.y[tr_other]}).groupby("a").y.agg(["sum", "count"])
            prior = float(meta.y[tr_all].mean())
            mp = (stats["sum"] + K * prior) / (stats["count"] + K)
            seen[te] = a.iloc[te].isin(mp.index).values
            f_other = a.iloc[te].map(mp).fillna(prior).values
            base[te] = rt.FINAL().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
            Xtr = X.iloc[tr].assign(author_other=np.nan)
            Xte = X.iloc[te].assign(author_other=f_other)
            withx[te] = rt.FINAL().fit(Xtr, y.iloc[tr]).predict(Xte)
        out[p] = {"posts": int(len(y)), "seen_share": round(float(seen.mean()), 3),
                  "base_MAE": round(float(np.abs(y.values - base).mean()), 3),
                  "cross_author_MAE": round(float(np.abs(y.values - withx).mean()), 3),
                  "base_seen": round(float(np.abs(y.values[seen] - base[seen]).mean()), 3) if seen.any() else None,
                  "cross_seen": round(float(np.abs(y.values[seen] - withx[seen]).mean()), 3) if seen.any() else None}
        print(p, out[p])
    (ROOT / "data" / "processed" / "cross_author_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
