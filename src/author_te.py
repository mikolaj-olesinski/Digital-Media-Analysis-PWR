"""Author target encoding as a model feature (E119): the author's past views are the largest known signal (E68), but it
was used as a post-hoc correction. Here the author's shrunk mean log views is a feature inside CatBoost, computed
out-of-fold inside each training fold (no leakage), and compared with the base model.

    .venv/bin/python src/author_te.py
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


def te_map(author, y, prior, k=K):
    g = pd.DataFrame({"a": author, "y": y}).groupby("a").y.agg(["sum", "count"])
    return (g["sum"] + k * prior) / (g["count"] + k)


def main():
    f, meta = em.load(ROOT)
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        author = meta.author[m].fillna("NA").reset_index(drop=True)
        base = np.full(len(y), np.nan); withte = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, g):
            prior = float(y.iloc[tr].mean())
            # train rows: inner OOF encoding
            enc = np.full(len(tr), np.nan)
            for itr, ite in GroupKFold(5).split(X.iloc[tr], y.iloc[tr], g.iloc[tr]):
                mp = te_map(author.iloc[tr].iloc[itr], y.iloc[tr].iloc[itr], float(y.iloc[tr].iloc[itr].mean()))
                enc[ite] = author.iloc[tr].iloc[ite].map(mp).fillna(prior).values
            # test rows: encoding from all training rows
            mp = te_map(author.iloc[tr], y.iloc[tr], prior)
            enc_te = author.iloc[te].map(mp).fillna(prior).values
            seen = author.iloc[te].isin(set(author.iloc[tr])).values
            base[te] = rt.FINAL().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
            Xtr = X.iloc[tr].assign(author_te=enc); Xte = X.iloc[te].assign(author_te=enc_te)
            withte[te] = rt.FINAL().fit(Xtr, y.iloc[tr]).predict(Xte)
        ok = ~np.isnan(base)
        seen = np.zeros(len(y), bool)
        # recompute seen mask per fold for the subset metric
        for tr, te in GroupKFold(5).split(X, y, g):
            seen[te] = author.iloc[te].isin(set(author.iloc[tr])).values
        out[p] = {"posts": int(len(y)),
                  "base_MAE": round(float(np.abs(y.values - base).mean()), 3),
                  "author_te_MAE": round(float(np.abs(y.values - withte).mean()), 3),
                  "base_MAE_seen": round(float(np.abs(y.values[seen] - base[seen]).mean()), 3),
                  "author_te_MAE_seen": round(float(np.abs(y.values[seen] - withte[seen]).mean()), 3),
                  "seen_share": round(float(seen.mean()), 3)}
        print(p, out[p])
    (ROOT / "data" / "processed" / "author_te_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
