"""More from the history file (notebooks/models/26_history_more.ipynb, E59-E60).
E59: forecast horizon curve, and training on every (earlier, later) pair of checks instead of only first-last.
E60: early-reaction forecast without views (the FB/IG situation): later likes from early likes, then chained to views."""
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
import history as H, extra_models as em, reaction_tiers as rt

lg = H.lg


def all_pairs(s, min_gap_h=1.0):
    """Every (earlier check, later check) pair of a mention with gap >= min_gap_h."""
    out = []
    for mid, q in s.groupby("mention_id"):
        q = q.reset_index(drop=True)
        for i in range(len(q) - 1):
            for j in range(i + 1, len(q)):
                if q.age_h[j] - q.age_h[i] >= min_gap_h:
                    a = q.iloc[i]; b = q.iloc[j]
                    out.append({**a.to_dict(), **{f"{m}_end": b[m] for m in ["age_h"] + H.M}, "first_last": i == 0 and j == len(q) - 1})
    return pd.DataFrame(out)


def horizon(pp, bins=(0, 12, 48, 24 * 7, 24 * 30, 1e6), labels=("<12 h", "12-48 h", "2-7 d", "7-30 d", ">30 d")):
    """Out-of-fold forecasts (folds by brand). Evaluated on first-last pairs: model trained on first-last only vs on all pairs."""
    out = []
    for plat, q in pp.groupby("platform"):
        q = q.reset_index(drop=True); X, y = H.features(q); g = q.brand.values; fl = q.first_last.values
        pr = {k: np.full(len(q), np.nan) for k in ["persistence", "CatBoost, first-last pairs", "CatBoost, all pairs"]}
        for tr, te in GroupKFold(5).split(X, y, g):
            pr["persistence"][te] = X.log_views0.values[te]
            for k, sel in [("CatBoost, first-last pairs", tr[fl[tr]]), ("CatBoost, all pairs", tr)]:
                m = H.catboost().fit(X.iloc[sel], y.iloc[sel] - X.log_views0.iloc[sel]); pr[k][te] = X.log_views0.values[te] + m.predict(X.iloc[te])
        gap = pd.cut(q.age_h_end - q.age_h, bins, labels=labels).astype(str); a0 = pd.cut(q.age_h, bins, labels=labels).astype(str)
        for k, v in pr.items():
            out.append(pd.DataFrame({"platform": plat, "model": k, "brand": g, "y": y.values, "pred": v, "first_last": fl, "gap": gap, "age0": a0}))
    return pd.concat(out, ignore_index=True)


def by_bin(P, col):
    q = P[P.first_last]
    return q.groupby(["platform", col, "model"], observed=True).apply(lambda z: pd.Series({"pairs": len(z), "MAE": float(np.abs(z.y - z.pred).mean())})).reset_index()


def early_reactions(p, f, meta):
    """First-last pairs on TikTok / X. (a) forecast log likes at the last check from likes, comments, followers and ages at the
    first check (no views); (b) chain: forecast likes and comments -> final now-cast model trained on the main file
    (excluding the pair's brands) -> views at the last check; compared with E41 (CatBoost on reactions and ages, no views)."""
    out = []; L = []
    for plat, q in p.groupby("platform"):
        if plat not in ("tiktok", "twitter"): continue
        q = q[q.likes_end.notna() & q.comments_end.notna() & q.likes.notna() & q.comments.notna() & q.views_end.notna()].reset_index(drop=True); X, yv = H.features(q); g = q.brand.values
        yl, yc = lg(q.likes_end), lg(q.comments_end)
        R = X[H.REACT]; pred = {k: np.full(len(q), np.nan) for k in ["likes: persistence", "likes: CatBoost"]}
        views = {k: np.full(len(q), np.nan) for k in ["E41: CatBoost reactions + ages", "chain: forecast reactions -> now-cast", "chain with true later reactions (upper bound)"]}
        mm = (meta.platform == plat).values; Mf = f[mm][em.ENG].fillna(0); My = meta.y[mm]; Mb = meta.brand[mm]
        def nowcast_input(likes, comments, shares):   # same transforms as modeling.load_features (natural log1p)
            Z = pd.DataFrame(0.0, index=range(len(q)), columns=em.ENG)
            Z["log_followers"] = X.log_followers.fillna(0).values; Z["log_likes"] = np.log1p(likes); Z["log_comments"] = np.log1p(comments); Z["log_shares"] = np.log1p(shares)
            Z["na_followers"] = X.log_followers.isna().astype(int).values; return Z
        for tr, te in GroupKFold(5).split(X, yv, g):
            pred["likes: persistence"][te] = R.log_likes0.values[te]
            ml = H.catboost().fit(R.iloc[tr], yl[tr] - R.log_likes0.values[tr]); fl_ = R.log_likes0.values[te] + ml.predict(R.iloc[te]); pred["likes: CatBoost"][te] = fl_
            mc = H.catboost().fit(R.iloc[tr], yc[tr] - R.log_comments0.values[tr]); fc_ = R.log_comments0.values[te] + mc.predict(R.iloc[te])
            views["E41: CatBoost reactions + ages"][te] = H.catboost().fit(R.iloc[tr], yv.iloc[tr]).predict(R.iloc[te])
            keep = ~Mb.isin(set(g[te])).values; now = rt.FINAL().fit(Mf[keep], My[keep])
            sh = q.shares_end.fillna(0).values[te]
            Zf = nowcast_input(np.zeros(len(q)), np.zeros(len(q)), np.zeros(len(q))).iloc[te].assign(
                log_likes=np.log1p(np.clip(10 ** fl_ - 1, 0, None)), log_comments=np.log1p(np.clip(10 ** fc_ - 1, 0, None)), log_shares=np.log1p(sh))
            views["chain: forecast reactions -> now-cast"][te] = now.predict(Zf)
            views["chain with true later reactions (upper bound)"][te] = now.predict(nowcast_input(q.likes_end.fillna(0).values, q.comments_end.fillna(0).values, q.shares_end.fillna(0).values).iloc[te])
        for k, v in pred.items(): L.append(pd.DataFrame({"platform": plat, "model": k, "brand": g, "y": yl, "pred": v}))
        for k, v in views.items(): out.append(pd.DataFrame({"platform": plat, "model": k, "brand": g, "y": yv.values, "pred": v}))
    return pd.concat(L, ignore_index=True), pd.concat(out, ignore_index=True)
