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


# ---------- E88: adapting the model to the target's reaction structure (still no target views) ----------
from catboost import CatBoostClassifier


def _struct(X):
    """Inputs + scale-free structure columns (comments per like, shares per like, likes per follower)."""
    Z = X[COLS].copy(); Z["cpl"] = X.log_comments - X.log_likes; Z["spl"] = X.log_shares - X.log_likes; Z["lpf"] = X.log_likes - X.log_followers * np.log(10)
    return Z.fillna(-1)


def _clf():
    return CatBoostClassifier(iterations=200, depth=4, verbose=0, random_seed=SEED, allow_writing_files=False, thread_count=4)


def adapt(f, meta, clip=(0.05, 20)):
    """Leave one platform out. Variants (ridge on likes + comments and CatBoost on all inputs where it applies):
    importance weights  source posts weighted by p(target | x) / p(source | x) from a classifier on inputs and structure;
    source mixture      one model per source platform, mixed per target post by p(source k | x) among the two sources;
    views per like      target y - log10(likes + 1), added back after prediction; views per reaction likewise;
    raw + structure     raw inputs plus structure columns and within-platform percentiles."""
    M = meta.reset_index(drop=True); plat = M.platform.values; y = M.y.values; X0 = f[COLS].reset_index(drop=True)
    Z = _struct(X0); R = mapped(X0, plat, "none", "ratios").fillna(-1); ZR = pd.concat([Z, R.filter(like="pct_")], axis=1)
    ll = np.log10(np.expm1(X0.log_likes.fillna(0).values) + 1)
    lr = np.log10(np.expm1(X0.log_likes.fillna(0)) + np.expm1(X0.log_comments.fillna(0)) + np.expm1(X0.log_shares.fillna(0)) + 1).values
    C2 = ["log_likes", "log_comments"]; out = []
    for t in PLATS:
        te_all = plat == t; srcs = [p for p in PLATS if p != t]
        fid = np.full(len(M), -1)
        for k, (_, te) in enumerate(GroupKFold(5).split(np.zeros(te_all.sum()), y[te_all], M.brand.values[te_all])): fid[np.where(te_all)[0][te]] = k
        # domain classifiers use inputs only (no views), fitted once per target
        c = _clf().fit(Z, te_all.astype(int)); pt = c.predict_proba(Z)[:, 1]
        w = np.clip(pt / (1 - pt) * ((~te_all).sum() / te_all.sum()), *clip)
        s = ~te_all; cs = _clf().fit(Z[s], (plat[s] == srcs[0]).astype(int)); p0 = cs.predict_proba(Z)[:, 1]
        names = ["ridge, raw", "CatBoost, raw", "ridge, importance weights", "CatBoost, importance weights", "ridge, source mixture", "CatBoost, source mixture",
                 "ridge, views per like", "CatBoost, views per like", "CatBoost, views per reaction", "CatBoost, raw + structure"]
        P = {k: np.full(len(M), np.nan) for k in names}
        for k in range(5):
            te = fid == k; tr = (~te_all) & ~M.brand.isin(set(M.brand[te])).values
            P["ridge, raw"][te] = rt.LOPO().fit(X0.loc[tr, C2].fillna(-1), y[tr]).predict(X0.loc[te, C2].fillna(-1))
            P["CatBoost, raw"][te] = cb().fit(Z.loc[tr, COLS], y[tr]).predict(Z.loc[te, COLS])
            P["ridge, importance weights"][te] = rt.LOPO().fit(X0.loc[tr, C2].fillna(-1), y[tr], ridge__sample_weight=w[tr]).predict(X0.loc[te, C2].fillna(-1))
            P["CatBoost, importance weights"][te] = cb().fit(Z.loc[tr, COLS], y[tr], sample_weight=w[tr]).predict(Z.loc[te, COLS])
            mr, mc = [], []
            for sp in srcs:
                a = tr & (plat == sp)
                mr.append(rt.LOPO().fit(X0.loc[a, C2].fillna(-1), y[a]).predict(X0.loc[te, C2].fillna(-1))); mc.append(cb().fit(Z.loc[a, COLS], y[a]).predict(Z.loc[te, COLS]))
            P["ridge, source mixture"][te] = p0[te] * mr[0] + (1 - p0[te]) * mr[1]; P["CatBoost, source mixture"][te] = p0[te] * mc[0] + (1 - p0[te]) * mc[1]
            P["ridge, views per like"][te] = ll[te] + rt.LOPO().fit(X0.loc[tr, C2].fillna(-1), y[tr] - ll[tr]).predict(X0.loc[te, C2].fillna(-1))
            P["CatBoost, views per like"][te] = ll[te] + cb().fit(Z[tr], y[tr] - ll[tr]).predict(Z[te])
            P["CatBoost, views per reaction"][te] = lr[te] + cb().fit(Z[tr], y[tr] - lr[tr]).predict(Z[te])
            P["CatBoost, raw + structure"][te] = cb().fit(ZR[tr], y[tr]).predict(ZR[te])
        for name, v in P.items():
            out.append(pd.DataFrame({"platform": t, "model": name, "brand": M.brand.values[te_all], "y": y[te_all], "pred": v[te_all]}))
        out[-1].attrs = {}
    return pd.concat(out, ignore_index=True)


def weight_summary(f, meta):
    """Effective sample size share of the importance weights and the mixture weight per target."""
    M = meta.reset_index(drop=True); plat = M.platform.values; Z = _struct(f[COLS].reset_index(drop=True)); rows = []
    for t in PLATS:
        te = plat == t; pt = _clf().fit(Z, te.astype(int)).predict_proba(Z)[:, 1]; w = np.clip(pt / (1 - pt) * ((~te).sum() / te.sum()), .05, 20)[~te]
        srcs = [p for p in PLATS if p != t]; s = ~te; p0 = _clf().fit(Z[s], (plat[s] == srcs[0]).astype(int)).predict_proba(Z[te])[:, 1]
        rows.append({"target": t, "effective sample share": float(w.sum() ** 2 / (w ** 2).sum() / len(w)),
                     **{f"weight share from {p}": float(w[plat[~te] == p].sum() / w.sum()) for p in srcs}, f"mean mixture weight of {srcs[0]}": float(p0.mean())})
    return pd.DataFrame(rows)
