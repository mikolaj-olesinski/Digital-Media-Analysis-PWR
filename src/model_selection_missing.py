"""Which model to use when data is missing (E163): under a missing-reactions or missing-followers test set, does a
specialised model (followers-only / reactions-only) beat the full model?

    .venv/bin/python src/model_selection_missing.py
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
FOL = ["log_followers", "na_followers"]
REA = ["log_likes", "log_comments", "log_shares", "na_likes", "na_comments", "na_shares"]


def main():
    f, meta = em.load(ROOT)
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        full_r, spec_r = [], []
        full_f, spec_f = [], []
        for tr, te in GroupKFold(5).split(X, y, g):
            full = rt.FINAL().fit(X.iloc[tr], y.iloc[tr])
            only_fol = rt.FINAL().fit(X.iloc[tr][FOL], y.iloc[tr])
            only_rea = rt.FINAL().fit(X.iloc[tr][REA], y.iloc[tr])
            # missing reactions
            Xm = X.iloc[te].copy(); Xm[["log_likes", "log_comments", "log_shares", "na_likes", "na_comments", "na_shares"]] = [np.nan, np.nan, np.nan, 1, 1, 1]
            full_r.append(np.abs(y.iloc[te].values - full.predict(Xm)).mean())
            spec_r.append(np.abs(y.iloc[te].values - only_fol.predict(Xm[FOL])).mean())
            # missing followers
            Xf = X.iloc[te].copy(); Xf[["log_followers", "na_followers"]] = [np.nan, 1]
            full_f.append(np.abs(y.iloc[te].values - full.predict(Xf)).mean())
            spec_f.append(np.abs(y.iloc[te].values - only_rea.predict(Xf[REA])).mean())
        out[p] = {"no_reactions": {"full": round(float(np.mean(full_r)), 3), "followers_only_model": round(float(np.mean(spec_r)), 3)},
                  "no_followers": {"full": round(float(np.mean(full_f)), 3), "reactions_only_model": round(float(np.mean(spec_f)), 3)}}
        print(p, out[p])
    (ROOT / "data" / "processed" / "model_selection_missing_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
