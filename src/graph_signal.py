"""Which graph edges actually carry signal (E128). Instead of another embedding, measure whether the residual of a post
can be predicted from the residuals of training posts that share its author, brand or hashtags. This separates the
graph question into edge types.

    .venv/bin/python src/graph_signal.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import extra_models as em  # noqa: E402
import reaction_tiers as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
K = 0.0


def tags(s):
    return frozenset(x.lower() for x in re.findall(r"#\w+", str(s)))


def neighbor_mean(test_idx, train_idx, key_train, key_test, resid, min_n=2):
    g = pd.DataFrame({"k": key_train, "r": resid}).groupby("k").r.agg(["sum", "count"])
    m = (g["sum"] + K * 0) / (g["count"] + K)
    v = pd.Series(key_test).map(m)
    return v.values, pd.Series(key_test).map(g["count"]).fillna(0).values


def main():
    raw = eu.load(ROOT)
    f, meta = em.load(ROOT)
    hashtags = raw.content.fillna("").map(tags).values
    authors = raw.author_name.fillna("NA").astype(str).values
    brands = raw.brand.fillna("NA").astype(str).values
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        idx = np.where(m)[0]  # positions in raw
        rows = {e: {"rho": [], "mae_gain": []} for e in ["author", "brand", "hashtag"]}
        for tr, te in GroupKFold(5).split(X, y, g):
            # out-of-fold residuals on the training part
            r_tr = np.full(len(tr), np.nan)
            for itr, ite in GroupKFold(5).split(X.iloc[tr], y.iloc[tr], g.iloc[tr]):
                pv = rt.FINAL().fit(X.iloc[tr].iloc[itr], y.iloc[tr].iloc[itr]).predict(X.iloc[tr].iloc[ite])
                r_tr[ite] = y.iloc[tr].iloc[ite].values - pv
            base = rt.FINAL().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
            resid_te = y.iloc[te].values - base
            a_tr, a_te = authors[idx[tr]], authors[idx[te]]
            b_tr, b_te = brands[idx[tr]], brands[idx[te]]
            h_tr = [hashtags[i] for i in idx[tr]]; h_te = [hashtags[i] for i in idx[te]]
            for e, ktr, kte in [("author", a_tr, a_te), ("brand", b_tr, b_te)]:
                v, cnt = neighbor_mean(te, tr, ktr, kte, r_tr)
                ok = cnt >= 2
                if ok.sum() > 10:
                    rows[e]["rho"].append(spearmanr(v[ok], resid_te[ok])[0])
                    rows[e]["mae_gain"].append(float(np.abs(resid_te[ok]).mean() - np.abs(resid_te[ok] - v[ok]).mean()))
            # hashtag: mean residual of training posts sharing at least one tag
            inv = {}
            for j, tg in enumerate(h_tr):
                for t in tg: inv.setdefault(t, []).append(j)
            v = np.full(len(te), np.nan)
            for j, tg in enumerate(h_te):
                js = sorted({i for t in tg for i in inv.get(t, [])})
                if js: v[j] = np.mean(r_tr[js])
            ok = ~np.isnan(v)
            if ok.sum() > 10:
                rows["hashtag"]["rho"].append(spearmanr(v[ok], resid_te[ok])[0])
                rows["hashtag"]["mae_gain"].append(float(np.abs(resid_te[ok]).mean() - np.abs(resid_te[ok] - v[ok]).mean()))
        out[p] = {e: {"spearman": round(float(np.mean(v["rho"])), 3) if v["rho"] else None,
                       "mean_abs_error_reduction": round(float(np.mean(v["mae_gain"])), 3) if v["mae_gain"] else None,
                       "folds": len(v["rho"])} for e, v in rows.items()}
        print(p, out[p])
    (ROOT / "data" / "processed" / "graph_signal_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
