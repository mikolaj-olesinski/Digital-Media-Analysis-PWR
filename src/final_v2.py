"""Final estimator v2 (E124): author target encoding + CQR intervals on top of the final model. Does the combination
keep the point accuracy and the interval coverage, and where does it help?

    .venv/bin/python src/final_v2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em  # noqa: E402
import reaction_tiers as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
K = 5.0


def qm(a):
    return LGBMRegressor(objective="quantile", alpha=a, n_estimators=300, learning_rate=.05, num_leaves=31,
                         subsample=.9, colsample_bytree=.9, random_state=SEED, verbose=-1)


def te_fold(author_tr, y_tr, author_other, prior, inner_groups):
    """OOF encoding for the training rows, direct encoding for the others."""
    enc = np.full(len(y_tr), np.nan)
    for itr, ite in GroupKFold(5).split(np.zeros(len(y_tr)), y_tr, inner_groups):
        g = pd.DataFrame({"a": author_tr[itr], "y": y_tr[itr]}).groupby("a").y.agg(["sum", "count"])
        mp = (g["sum"] + K * prior) / (g["count"] + K)
        enc[ite] = pd.Series(author_tr[ite]).map(mp).fillna(prior).values
    g = pd.DataFrame({"a": author_tr, "y": y_tr}).groupby("a").y.agg(["sum", "count"])
    mp = (g["sum"] + K * prior) / (g["count"] + K)
    return enc, pd.Series(author_other).map(mp).fillna(prior).values


def main():
    f, meta = em.load(ROOT)
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        a = meta.author[m].fillna("NA").reset_index(drop=True)
        res = {k: [] for k in ["mae_base", "mae_te", "cov_base", "cov_te", "w_base", "w_te"]}
        for tr, te in GroupKFold(5).split(X, y, g):
            inner = next(iter(GroupShuffleSplit(1, test_size=.25, random_state=SEED).split(X.iloc[tr], y.iloc[tr], g.iloc[tr])))
            fit_idx, cal_idx = np.array(tr)[inner[0]], np.array(tr)[inner[1]]
            prior = float(y.iloc[fit_idx].mean())
            enc_fit, enc_cal = te_fold(a.iloc[fit_idx].values, y.iloc[fit_idx].values, a.iloc[cal_idx].values, prior, g.iloc[fit_idx].values)
            _, enc_te = te_fold(a.iloc[fit_idx].values, y.iloc[fit_idx].values, a.iloc[te].values, prior, g.iloc[fit_idx].values)
            # point models
            b = rt.FINAL().fit(X.iloc[fit_idx], y.iloc[fit_idx]); pb = b.predict(X.iloc[te])
            t = rt.FINAL().fit(X.iloc[fit_idx].assign(author_te=enc_fit), y.iloc[fit_idx]); pt = t.predict(X.iloc[te].assign(author_te=enc_te))
            res["mae_base"].append(np.abs(y.iloc[te].values - pb).mean()); res["mae_te"].append(np.abs(y.iloc[te].values - pt).mean())
            # CQR on base features
            for tag, cols, cal, test in [("base", list(em.ENG), cal_idx, te), ("te", list(em.ENG) + ["author_te"], cal_idx, te)]:
                Xf = X.iloc[fit_idx][list(em.ENG)].copy(); Xc = X.iloc[cal][list(em.ENG)].copy(); Xt = X.iloc[test][list(em.ENG)].copy()
                if tag == "te":
                    Xf["author_te"] = enc_fit; Xc["author_te"] = enc_cal
                    Xt["author_te"] = enc_te
                lo = qm(.05).fit(Xf, y.iloc[fit_idx]); hi = qm(.95).fit(Xf, y.iloc[fit_idx])
                E = np.maximum(lo.predict(Xc) - y.iloc[cal].values, y.iloc[cal].values - hi.predict(Xc)); q = float(np.quantile(E, .9))
                L, H = lo.predict(Xt) - q, hi.predict(Xt) + q
                yt = y.iloc[test].values
                res["cov_" + tag].append(((yt >= L) & (yt <= H)).mean()); res["w_" + tag].append(np.median(H - L))
        out[p] = {k: round(float(np.mean(v)), 3) for k, v in res.items()}
        print(p, out[p])
    (ROOT / "data" / "processed" / "final_v2_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
