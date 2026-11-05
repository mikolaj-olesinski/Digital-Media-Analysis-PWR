"""Stack of every model family on the same brand-grouped folds (notebooks/models/24_stack_all.ipynb, E54).
Uses the folds and cached fine-tuned predictions of finetune_text (E53)."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from catboost import CatBoostRegressor
import extra_models as em, reaction_tiers as rt, check_age as ca, context_text as ct, tabular_foundation as tf

PLATS = ["tiktok", "youtube", "twitter"]


def members(f, meta, root, store, emb_cache):
    """Out-of-fold predictions of every member on the E53 folds (columns) for all posts (rows, meta order)."""
    fid = store["fid"]; age = ca.load(root)[0].log_age.loc[f.index]
    E = ct.embeddings(root, emb_cache); pe, _ = ct.reduce(E, root)
    F = f.assign(log_age=age).join(pe)
    sets = {"CatBoost + age of check": (em.ENG + ["log_age"], tf.catboost),
            "CatBoost + embeddings": (em.ENG + list(pe.columns), tf.catboost),
            "TabPFN v2": (em.ENG, tf.tabpfn), "TabICL": (em.ENG, tf.tabicl)}
    out = pd.DataFrame({"CatBoost (final model)": store["catboost"]}, index=range(len(meta)))
    for k, (cols, mk) in sets.items():
        pred = np.full(len(meta), np.nan); seeds = tf.SEEDS if k.startswith("Tab") else (42,)
        for p in PLATS:
            m = np.where((meta.platform == p).values)[0]; X = F.iloc[m][cols].fillna(0).reset_index(drop=True); y = meta.y.values[m]
            for fold in range(5):
                tr, te = np.where(fid[m] != fold)[0], np.where(fid[m] == fold)[0]
                pred[m[te]] = np.mean([mk(s)().fit(X.iloc[tr], y[tr]).predict(X.iloc[te]) for s in seeds], axis=0)
        out[k] = pred
    import gnn
    mlp = np.full(len(meta), np.nan)
    for p in PLATS:
        m = np.where((meta.platform == p).values)[0]; X = np.hstack([f.iloc[m][em.ENG].fillna(0).values, pe.loc[f.index[m]].values]).astype(np.float32); y = meta.y.values[m]
        for fold in range(5):
            tr, te = np.where(fid[m] != fold)[0], np.where(fid[m] == fold)[0]
            mlp[m[te]] = gnn.fit_predict(X, y, None, tr, te, False)
    out["MLP (ENG + embeddings)"] = mlp
    out["fine-tuned row as text"] = store["row as text"]
    out["CatBoost + fine-tuned text correction"] = store["catboost"] + store["text correction"]
    return out


def _weights(Z, y):
    k = Z.shape[1]
    def loss(a):
        w = np.exp(a - a.max()); w /= w.sum(); return np.abs(Z @ w - y).mean()
    a = minimize(loss, np.zeros(k), method="Powell", options={"maxiter": 4000, "xtol": 1e-4}).x
    w = np.exp(a - a.max()); return w / w.sum()


def stack(M, meta, fid):
    """Nested convex stack per platform; returns predictions and the weights of each outer fold."""
    pred = np.full(len(meta), np.nan); W = []
    for p in PLATS:
        m = np.where((meta.platform == p).values)[0]; Z, y = M.iloc[m].values, meta.y.values[m]
        for fold in range(5):
            tr, te = fid[m] != fold, fid[m] == fold
            w = _weights(Z[tr], y[tr]); pred[m[te]] = Z[te] @ w
            W.append({"platform": p, "fold": fold, **dict(zip(M.columns, w))})
    return pred, pd.DataFrame(W)


def long(M, meta, extra):
    cols = {**{c: M[c].values for c in M.columns}, **extra}
    return pd.concat([pd.DataFrame({"platform": meta.platform.values, "model": k, "brand": meta.brand.values, "y": meta.y.values, "pred": v}) for k, v in cols.items()], ignore_index=True)
