"""Partial missingness crossover (E164): at what share of missing reactions (or followers) does retraining without the
feature beat the full model? Levels 0/25/50/75/100%.

    .venv/bin/python src/partial_missing.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em  # noqa: E402
import reaction_tiers as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
FOL = ["log_followers", "na_followers"]
REA = ["log_likes", "log_comments", "log_shares", "na_likes", "na_comments", "na_shares"]


def main():
    f, meta = em.load(ROOT)
    rng = np.random.default_rng(SEED)
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        res_r = {l: {"full": [], "spec": []} for l in LEVELS}
        res_f = {l: {"full": [], "spec": []} for l in LEVELS}
        for tr, te in GroupKFold(5).split(X, y, g):
            full = rt.FINAL().fit(X.iloc[tr], y.iloc[tr])
            only_fol = rt.FINAL().fit(X.iloc[tr][FOL], y.iloc[tr])
            only_rea = rt.FINAL().fit(X.iloc[tr][REA], y.iloc[tr])
            base = X.iloc[te].copy(); yt = y.iloc[te].values
            for lvl in LEVELS:
                # missing reactions
                Xr = base.copy()
                if lvl > 0:
                    mask = rng.random(len(Xr)) < lvl
                    Xr.loc[mask, REA] = [np.nan, np.nan, np.nan, 1, 1, 1]
                res_r[lvl]["full"].append(np.abs(yt - full.predict(Xr)).mean())
                res_r[lvl]["spec"].append(np.abs(yt - only_fol.predict(Xr[FOL])).mean())
                # missing followers
                Xf2 = base.copy()
                if lvl > 0:
                    mask = rng.random(len(Xf2)) < lvl
                    Xf2.loc[mask, FOL] = [np.nan, 1]
                res_f[lvl]["full"].append(np.abs(yt - full.predict(Xf2)).mean())
                res_f[lvl]["spec"].append(np.abs(yt - only_rea.predict(Xf2[REA])).mean())
        out[p] = {"missing_reactions": {str(l): {"full": round(float(np.mean(res_r[l]["full"])), 3),
                                                 "followers_only": round(float(np.mean(res_r[l]["spec"])), 3)} for l in LEVELS},
                  "missing_followers": {str(l): {"full": round(float(np.mean(res_f[l]["full"])), 3),
                                                 "reactions_only": round(float(np.mean(res_f[l]["spec"])), 3)} for l in LEVELS}}
        print(p, json.dumps(out[p]))
    (ROOT / "data" / "processed" / "partial_missing_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
