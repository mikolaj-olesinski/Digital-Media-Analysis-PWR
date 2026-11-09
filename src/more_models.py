"""Pooled monotone model, YouTube drift, copy-paste campaigns, posts without reactions (notebooks/models/27_more_models.ipynb, E61-E64)."""
import numpy as np, pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
import eda_utils as eu, extra_models as em, reaction_tiers as rt, check_age as ca, context_text as ct, posting_time_text as ptt, history as H
from validation import SEED

PLATS = ["tiktok", "youtube", "twitter"]
MONO = {"log_followers": 1, "log_likes": 1, "log_comments": 1, "log_shares": 1}


def cb(mono=False, cat=None, cols=None):
    kw = dict(loss_function="MAE", random_seed=SEED, iterations=300, verbose=0, allow_writing_files=False, thread_count=4)
    if mono: kw["monotone_constraints"] = [MONO.get(c, 0) for c in cols]
    if cat: kw["cat_features"] = cat
    return CatBoostRegressor(**kw)


def pooled(f, meta):
    """E61. Joint brand folds over the three platforms. Per-platform vs pooled (platform as a category), with and without
    monotone constraints on followers and reactions; and leave-one-platform-out transfer of pooled models without the platform column."""
    X = f[em.ENG].fillna(0).reset_index(drop=True); M = meta.reset_index(drop=True); y = M.y.values; plat = M.platform.values
    fid = np.zeros(len(M), int)
    for k, (_, te) in enumerate(GroupKFold(5).split(X, y, M.brand)): fid[te] = k
    cols = list(X.columns); Xp = X.assign(platform=plat); pc = cols + ["platform"]
    pred = {k: np.full(len(M), np.nan) for k in ["per platform (final model)", "per platform, monotone", "pooled with platform column", "pooled, monotone"]}
    for k in range(5):
        tr, te = fid != k, fid == k
        for p in PLATS:
            a, b = tr & (plat == p), te & (plat == p)
            pred["per platform (final model)"][b] = cb().fit(X[a], y[a]).predict(X[b])
            pred["per platform, monotone"][b] = cb(True, cols=cols).fit(X[a], y[a]).predict(X[b])
        pred["pooled with platform column"][te] = cb(cat=["platform"]).fit(Xp[tr], y[tr]).predict(Xp[te])
        pred["pooled, monotone"][te] = cb(True, ["platform"], pc).fit(Xp[tr], y[tr]).predict(Xp[te])
    P = pd.concat([pd.DataFrame({"platform": plat, "model": k, "brand": M.brand.values, "y": y, "pred": v}) for k, v in pred.items()], ignore_index=True)
    # leave one platform out: train on the other two (brands of the target fold removed), no platform column
    lo = {k: np.full(len(M), np.nan) for k in ["LOPO ridge on likes + comments (E13)", "LOPO CatBoost", "LOPO CatBoost, monotone"]}
    for p in PLATS:
        for k in range(5):
            te = (plat == p) & (fid == k); tr = (plat != p) & ~M.brand.isin(set(M.brand[te])).values
            lo["LOPO CatBoost"][te] = cb().fit(X[tr], y[tr]).predict(X[te])
            lo["LOPO CatBoost, monotone"][te] = cb(True, cols=cols).fit(X[tr], y[tr]).predict(X[te])
            C = ["log_likes", "log_comments"]; lo["LOPO ridge on likes + comments (E13)"][te] = rt.LOPO().fit(X[tr][C], y[tr]).predict(X[te][C])
    T = pd.concat([pd.DataFrame({"platform": plat, "model": k, "brand": M.brand.values, "y": y, "pred": v}) for k, v in lo.items()], ignore_index=True)
    return P, T


def youtube_drift(root):
    """E62. Month by month on YouTube: median age at the check, views per like, and the signed error of the final model with and without the age column."""
    f, meta, g, m2 = ca.load(root); P = ca.run(f, meta, g, m2)
    raw = eu.load(root).loc[f.index]; month = raw.created_at.dt.strftime("%Y-%m").values
    out = []
    for p in PLATS:
        m = (meta.platform == p).values; q = P[P.platform == p]
        for mod in ["final model", "final model + check age"]:
            z = q[q.model == mod].assign(month=month[m], age=meta.age_h.values[m], vpl=(meta.y.values[m] - np.log10(np.expm1(f.log_likes.fillna(0).values[m]) + 1)))
            z = z[z.month.between("2026-04", "2026-08")]
            out.append(z.groupby("month").agg(posts=("y", "size"), median_age_days=("age", lambda a: float(np.median(a) / 24)), median_views_per_like=("vpl", "median"), bias=("pred", lambda s: 0)).assign(platform=p, model=mod).reset_index())
            out[-1]["bias"] = z.groupby("month").apply(lambda w: float((w.pred - w.y).mean())).values
    return pd.concat(out, ignore_index=True)


def copies(root):
    """E63. Posts whose normalised text (> 20 characters) appears more than once in the data, by platform."""
    d = eu.load(root); norm = (d.title.fillna("") + " " + d.content.fillna("")).str.lower().str.replace(r"https?://\S+|[^\w ]", "", regex=True).str.split().str.join(" ")
    n = norm.map(norm.value_counts()).where(norm.str.len() > 20, 1)
    d = d.assign(is_copy=n > 1, group_size=n, n_brands=norm.map(d.groupby(norm).brand.nunique()).where(n > 1, 1),
                 vpl=np.log10(d.view_count.where(d.view_count > 0)) - np.log10(d.likes.fillna(0) + 1))
    t = d.groupby(["platform", "is_copy"]).agg(posts=("mention_id", "size"), median_views=("view_count", "median"), median_log_views_per_like=("vpl", "median"),
                                           median_reactions=("likes", "median"), median_followers=("author_followers", "median"), reach_share=("b24_estimated_reach", "sum"))
    t["reach_share"] = t.reach_share / t.groupby(level=0).reach_share.transform("sum")
    by_brand = d[d.is_copy].groupby(["platform", "brand"]).b24_estimated_reach.sum() / d.groupby(["platform", "brand"]).b24_estimated_reach.sum()
    top = by_brand.dropna().groupby(level=0).apply(lambda s: pd.Series({"brands with copies": int((s > 0).sum()), "brands with > 50% reach from copies": int((s > .5).sum())}))
    return t.reset_index(), top.unstack().reset_index(), d[["is_copy", "group_size", "n_brands"]]


def copy_errors(f, meta, d_copy):
    """Final model error on copied vs unique posts (TikTok / YouTube / X)."""
    rows = []
    for p in PLATS:
        m = (meta.platform == p).values; M = meta[m].reset_index(drop=True); X = f[m][em.ENG].fillna(0).reset_index(drop=True)
        pred = em.cv_predict(rt.FINAL, X, M.y, M.brand); cp = d_copy.loc[f.index[m], "is_copy"].values
        for k, s in [("copied text", cp), ("unique text", ~cp)]:
            e = pred[s] - M.y.values[s]; rows.append({"platform": p, "posts": k, "n": int(s.sum()), "MAE": float(np.abs(e).mean()), "bias": float(e.mean())})
    return pd.DataFrame(rows)


def zero_reactions(root, f, meta, cache):
    """E64. Posts with 0 likes and 0 comments: final model vs models trained only on such posts with posting-time columns,
    embeddings and context (E52 groups), vs the all-post model with those columns. Evaluated on the zero-reaction posts."""
    groups, gmeta, _ = ptt.data(root, cache)
    zero = ((f.log_likes.fillna(0) == 0) & (f.log_comments.fillna(0) == 0)).values
    out = []
    for p in PLATS:
        m = (meta.platform == p).values; M = meta[m].reset_index(drop=True); z = zero[m]
        if z.sum() < 100: continue
        Xf = f[m][em.ENG].fillna(0).reset_index(drop=True)
        Xr = pd.concat([groups[k][m].reset_index(drop=True) for k in ["followers", "posting time", "embeddings", "context"]], axis=1); Xr.columns = [str(c) for c in Xr.columns]
        Xa = pd.concat([Xf, Xr.drop(columns=["log_followers", "na_followers"])], axis=1)
        pr = {k: np.full(len(M), np.nan) for k in ["final model (all posts)", "all posts, + text and context", "zero-reaction posts only, text and context"]}
        for tr, te in GroupKFold(5).split(Xf, M.y, M.brand):
            pr["final model (all posts)"][te] = rt.FINAL().fit(Xf.iloc[tr], M.y[tr]).predict(Xf.iloc[te])
            pr["all posts, + text and context"][te] = H.catboost().fit(Xa.iloc[tr], M.y[tr]).predict(Xa.iloc[te])
            tz = tr[z[tr]]; pr["zero-reaction posts only, text and context"][te] = H.catboost().fit(Xr.iloc[tz], M.y[tz]).predict(Xr.iloc[te])
        for k, v in pr.items():
            out.append(pd.DataFrame({"platform": p, "model": k, "brand": M.brand.values[z], "y": M.y.values[z], "pred": v[z]}))
    return pd.concat(out, ignore_index=True)
