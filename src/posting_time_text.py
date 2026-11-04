"""Estimate at posting time with sentence embeddings and context from other posts (notebooks/models/22_posting_time_text.ipynb, E52).
Extends history.posting_time_data (E43) with the e5 embeddings and unlabelled context of context_text (E49-E50)."""
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
import history as H, context_text as ct

PLATS = ["tiktok", "youtube", "twitter"]
CONTEXT = ["author_posts", "author_brands", "author_mean_likes", "hashtag_max_freq", "hashtag_mean_likes",
           "brand_platform_posts", "brand_mean_likes", "brand_mean_followers", "brand_likes_other_platforms", "copies"]
BASE = "E43 posting-time columns"


def data(root, cache):
    groups, meta = H.posting_time_data(root)
    C = ct.context(root); E = ct.embeddings(root, cache); pe, _ = ct.reduce(E, root)
    pt = groups["posting time"]; tfidf = [c for c in pt.columns if str(c).startswith("svd") or "tfidf" in str(c).lower()]
    groups["posting time, no TF-IDF"] = pt.drop(columns=tfidf)
    groups["embeddings"] = pe.loc[meta.index]
    groups["context"] = C.loc[meta.index, CONTEXT]
    return groups, meta, tfidf


SETS = {BASE: ["followers", "posting time"],
        "+ embeddings": ["followers", "posting time", "embeddings"],
        "TF-IDF replaced by embeddings": ["followers", "posting time, no TF-IDF", "embeddings"],
        "+ context": ["followers", "posting time", "context"],
        "+ embeddings + context": ["followers", "posting time", "embeddings", "context"]}


def run(groups, meta):
    out = []
    for p in PLATS:
        m = (meta.platform == p).values; M = meta[m].reset_index(drop=True); y, g = M.y, M.brand
        G = {k: v[m].reset_index(drop=True) for k, v in groups.items()}
        for k, parts in SETS.items():
            X = pd.concat([G[q] for q in parts], axis=1); X.columns = [str(c) for c in X.columns]
            pred = np.full(len(y), np.nan)
            for tr, te in GroupKFold(5).split(X, y, g):
                pred[te] = H.catboost().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
            out.append(pd.DataFrame({"platform": p, "model": k, "brand": g, "y": y, "pred": pred}))
    return pd.concat(out, ignore_index=True)
