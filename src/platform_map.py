"""Mapping reaction scales between platforms before an out-of-platform model (notebooks/models/34_platform_map.ipynb, E87).
Leave one platform out on TikTok / YouTube / X: train on the other two, test on the third. The target's views are never used;
the mappings use only the target's own inputs (likes, comments, shares, followers), so they apply to Facebook / Instagram too.
  none       raw inputs (E13 / E61);
  z-score    each column standardised within its platform (sources each by their own statistics, target by its own);
  quantile   each target column replaced by the pooled-source value at the same percentile (distribution matching);
  ratios     scale-free columns: comments per like, shares per like, likes per follower, and each column's percentile within its platform.
Then optionally a shift fitted on n target posts with views (E15). Outputs carry statistics only."""
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import extra_models as em, reaction_tiers as rt
from validation import SEED

PLATS = ["tiktok", "youtube", "twitter"]
COLS = ["log_likes", "log_comments", "log_shares", "log_followers"]


def _pct(x):
    return pd.Series(x).rank(pct=True, method="average").values


def mapped(X, plat, target, how):
    """X: inputs of all posts (COLS), plat: platform per row. Returns the mapped frame (same rows)."""
    X = X.copy(); src = plat != target
    if how == "none": return X
    if how == "z-score":
        for p in np.unique(plat):
            m = plat == p
            for c in COLS:
                v = X.loc[m, c]; X.loc[m, c] = (v - v.mean()) / (v.std() + 1e-9)
        return X
    if how == "quantile":
        t = ~src
        for c in COLS:
            pool = np.sort(X.loc[src, c].dropna().values); v = X.loc[t, c]; ok = v.notna().values
            if len(pool) and ok.any():
                q = _pct(v[ok].values); X.loc[v.index[ok], c] = np.quantile(pool, q)
        return X
    if how == "ratios":
        out = pd.DataFrame(index=X.index)
        out["comments_per_like"] = X.log_comments - X.log_likes; out["shares_per_like"] = X.log_shares - X.log_likes
        out["likes_per_follower"] = X.log_likes - X.log_followers * np.log(10)
        for c in COLS:
            out[f"pct_{c}"] = np.nan
            for p in np.unique(plat):
                m = (plat == p) & X[c].notna().values; out.loc[m, f"pct_{c}"] = _pct(X.loc[m, c].values)
        return out
    raise ValueError(how)


def cb():
    return CatBoostRegressor(loss_function="MAE", iterations=300, verbose=0, random_seed=SEED, allow_writing_files=False, thread_count=4)


def lopo(f, meta, hows=("none", "z-score", "quantile", "ratios")):
    """Out-of-platform predictions on each target platform; target-fold brands are removed from training (as E13)."""
    M = meta.reset_index(drop=True); plat = M.platform.values; y = M.y.values
    X0 = f[COLS].reset_index(drop=True)
    out = []
    for t in PLATS:
        te_all = plat == t; tb = M.brand.values
        fid = np.full(len(M), -1)
        for k, (_, te) in enumerate(GroupKFold(5).split(np.zeros(te_all.sum()), y[te_all], tb[te_all])): fid[np.where(te_all)[0][te]] = k
        for how in hows:
            X = mapped(X0, plat, t, how).fillna(-1)
            twocols = [c for c in X.columns if "likes" in c or "comments" in c] if how != "ratios" else [c for c in X.columns if c in ("comments_per_like", "pct_log_likes", "pct_log_comments")]
            preds = {f"ridge, {how}": np.full(len(M), np.nan), f"CatBoost, {how}": np.full(len(M), np.nan)}
            for k in range(5):
                te = fid == k; tr = (~te_all) & ~M.brand.isin(set(M.brand[te])).values
                preds[f"ridge, {how}"][te] = rt.LOPO().fit(X.loc[tr, twocols], y[tr]).predict(X.loc[te, twocols])
                preds[f"CatBoost, {how}"][te] = cb().fit(X[tr], y[tr]).predict(X[te])
            for name, v in preds.items():
                out.append(pd.DataFrame({"platform": t, "model": name, "brand": M.brand.values[te_all], "y": y[te_all], "pred": v[te_all]}))
    return pd.concat(out, ignore_index=True)


def with_shift(P, ns=(10, 25, 50), n_draw=300):
    """MAE after a shift (median error) fitted on n random target posts, scored on the rest."""
    rng = np.random.default_rng(SEED); rows = []
    for (p, k), q in P.groupby(["platform", "model"], sort=False):
        e = (q.y - q.pred).values
        rows.append({"platform": p, "model": k, "n": 0, "MAE": float(np.abs(e).mean()), "bias (true - pred)": float(e.mean())})
        for n in ns:
            r = [np.abs(e[i[n:]] - np.median(e[i[:n]])).mean() for i in (rng.permutation(len(e)) for _ in range(n_draw))]
            rows.append({"platform": p, "model": k, "n": n, "MAE": float(np.mean(r)), "bias (true - pred)": np.nan})
    return pd.DataFrame(rows)
