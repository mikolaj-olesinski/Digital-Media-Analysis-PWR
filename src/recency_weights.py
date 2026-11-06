"""Recency weighting against temporal drift (E133): train on older posts, test on the newest 20%; do exponential sample
weights (half-life 90/180/365 days) help the model adapt?

    .venv/bin/python src/recency_weights.py
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
HALF = [90, 180, 365]


def main():
    raw = eu.load(ROOT)
    f, meta = em.load(ROOT)
    dates = raw.created_at.loc[f.index].reset_index(drop=True)
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        d = dates[m].reset_index(drop=True)
        order = np.argsort(d.values); n_test = max(150, int(.2 * len(y)))
        tr, te = order[:-n_test], order[-n_test:]
        out[p] = {"train": int(len(tr)), "test": int(len(te)), "test_from": str(d.iloc[te].min().date())}
        base = rt.FINAL().fit(X.iloc[tr], y.iloc[tr]); out[p]["base"] = round(float(np.abs(y.iloc[te].values - base.predict(X.iloc[te])).mean()), 3)
        ref = d.iloc[te].min()
        for h in HALF:
            age = (ref - d.iloc[tr]).dt.days.clip(lower=0).values
            w = 0.5 ** (age / h)
            mw = rt.FINAL().fit(X.iloc[tr], y.iloc[tr], sample_weight=w)
            out[p][f"half_{h}"] = round(float(np.abs(y.iloc[te].values - mw.predict(X.iloc[te])).mean()), 3)
        print(p, out[p])
    (ROOT / "data" / "processed" / "recency_weights_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
