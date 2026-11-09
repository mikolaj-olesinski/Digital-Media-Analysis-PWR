"""Follower-tier recalibration (E135): the model over-estimates small posts and under-estimates large ones (E18). Fit a
residual median per follower tier on calibration brands and apply it to held-out brands; does MAE fall?

    .venv/bin/python src/tier_recal.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em  # noqa: E402
import reaction_tiers as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def main():
    f, meta = em.load(ROOT)
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        fol = meta.author_followers[m].reset_index(drop=True)
        tier = pd.cut(fol, [0, 1e3, 1e4, 1e5, 1e6, 1e9], labels=["<1k", "1k-10k", "10k-100k", "100k-1M", ">1M"], right=False)
        base = np.full(len(y), np.nan); adj = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, g):
            inner = next(iter(GroupShuffleSplit(1, test_size=.25, random_state=SEED).split(X.iloc[tr], y.iloc[tr], g.iloc[tr])))
            fit_idx, cal_idx = np.array(tr)[inner[0]], np.array(tr)[inner[1]]
            mod = rt.FINAL().fit(X.iloc[fit_idx], y.iloc[fit_idx])
            base[te] = mod.predict(X.iloc[te])
            r_cal = y.iloc[cal_idx].values - mod.predict(X.iloc[cal_idx])
            med = pd.Series(r_cal).groupby(tier.iloc[cal_idx].values).median()
            corr = pd.Series(tier.iloc[te].values).map(med).astype(float).fillna(0).values
            adj[te] = base[te] + corr
        out[p] = {"posts": int(len(y)), "base_MAE": round(float(np.abs(y.values - base).mean()), 3),
                  "recal_MAE": round(float(np.abs(y.values - adj).mean()), 3),
                  "tier_counts": {str(k): int(v) for k, v in tier.value_counts().items()}}
        print(p, out[p])
    (ROOT / "data" / "processed" / "tier_recal_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
