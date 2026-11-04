"""Classification views of the task (notebooks/models/20_classification.ipynb, E48).
1. Viral: is a post in the top 10% of views of its platform? Scores from followers, the Brand24 reach column, the final
   regression model and a CatBoost classifier, on held-out brands.
2. Missing or zero views: which TikTok / YouTube / X posts have no view count, and what does the final model predict
   for the posts whose count is 0 (they are dropped from training)?"""
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
import eda_utils as eu, extra_models as em, reaction_tiers as rt
from validation import SEED

PLATS = ["tiktok", "youtube", "twitter"]
TOP = .10
CLF = lambda: CatBoostClassifier(random_seed=SEED, iterations=300, verbose=0, allow_writing_files=False, thread_count=4)


def raw_features(d):
    """Same transforms as modeling.load_features, on any rows (including posts without views)."""
    f = pd.DataFrame(index=d.index)
    f["log_followers"] = np.log10(d.author_followers.where(d.author_followers > 0))
    for c in ["likes", "comments", "shares"]:
        f[f"log_{c}"] = np.log1p(d[c]); f[f"na_{c}"] = d[c].isna().astype(int)
    f["na_followers"] = d.author_followers.isna().astype(int)
    return f


def viral(f, meta, top=TOP):
    """Out-of-fold scores per post. The top-10% threshold is set on the training fold."""
    out = []
    for p in PLATS:
        m = (meta.platform == p).values; M = meta[m].reset_index(drop=True)
        X = f[m][em.ENG].fillna(0).reset_index(drop=True); y = M.y.values
        reg = em.cv_predict(rt.FINAL, X, M.y, M.brand); clf = np.full(len(y), np.nan); lab = np.zeros(len(y), int)
        for tr, te in GroupKFold(5).split(X, y, M.brand):
            thr = np.quantile(y[tr], 1 - top); lab[te] = y[te] >= thr
            clf[te] = CLF().fit(X.iloc[tr], y[tr] >= thr).predict_proba(X.iloc[te])[:, 1]
        out.append(pd.DataFrame({"platform": p, "brand": M.brand.values, "viral": lab,
                                 "followers": X.log_followers.values, "Brand24 reach": np.log10(M.b24_estimated_reach.clip(lower=1)).values if p == "twitter" else np.nan,  # on TikTok / YouTube the column equals the views
                                 "regression (final model)": reg, "classifier (CatBoost)": clf}))
    return pd.concat(out, ignore_index=True)


SCORES = ["followers", "Brand24 reach", "regression (final model)", "classifier (CatBoost)"]


def _metrics(y, s, top=TOP):
    k = max(1, int(round(len(y) * top))); idx = np.argsort(-s)[:k]
    return roc_auc_score(y, s), average_precision_score(y, s), y[idx].mean()


def viral_table(V, n_boot=400, seed=SEED):
    """ROC AUC, average precision and precision in the top 10% of scores, with brand bootstrap CIs."""
    rows = []; rng = np.random.default_rng(seed)
    for p in PLATS:
        q = V[V.platform == p].reset_index(drop=True); brands = q.brand.unique(); idx = {b: np.where(q.brand == b)[0] for b in brands}
        boots = [np.concatenate([idx[b] for b in rng.choice(brands, len(brands))]) for _ in range(n_boot)]
        for s in SCORES:
            if q[s].isna().all(): continue
            y, sc = q.viral.values, q[s].fillna(q[s].min()).values
            a, ap, pk = _metrics(y, sc)
            B = np.array([_metrics(y[i], sc[i]) for i in boots if 0 < y[i].sum() < len(i)])
            lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
            rows.append({"platform": p, "score": s, "posts": len(q), "viral": int(y.sum()), "ROC AUC": a, "AUC low": lo[0], "AUC high": hi[0],
                         "avg precision": ap, "AP low": lo[1], "AP high": hi[1], "precision top 10%": pk, "P@10 low": lo[2], "P@10 high": hi[2]})
    return pd.DataFrame(rows)


def no_views(root):
    """TikTok / YouTube / X posts: view count present (> 0), zero, or missing, with the features known without views."""
    d = eu.load(root); d = d[d.platform.isin(PLATS)].copy()
    d["views"] = np.where(d.view_count.isna(), "missing", np.where(d.view_count == 0, "zero", "present"))
    f = raw_features(d).assign(post_year=d.created_at.dt.year + (d.created_at.dt.month - 1) / 12,
                               age_days=(d.metrics_updated_at - d.created_at).dt.total_seconds() / 86400,
                               text_len=np.log10((d.title.fillna("") + " " + d.content.fillna("")).str.len() + 1))
    return d[["platform", "brand", "views", "created_at"]].join(f)


def no_views_profile(N):
    """Median of each feature by view status; share of posts per status."""
    cols = ["post_year", "age_days", "log_followers", "log_likes", "log_comments", "na_likes", "na_comments", "text_len"]
    t = N.groupby(["platform", "views"])[cols].median()
    t.insert(0, "posts", N.groupby(["platform", "views"]).size()); return t


def no_views_auc(N, n_boot=400, seed=SEED):
    """Can the missing / zero view count be told apart from a present one, on held-out brands?"""
    cols = ["post_year", "age_days", "log_followers", "log_likes", "log_comments", "log_shares", "na_likes", "na_comments", "na_shares", "na_followers", "text_len"]
    rows = []; rng = np.random.default_rng(seed)
    for p in PLATS:
        q = N[N.platform == p].reset_index(drop=True); y = (q.views != "present").values.astype(int)
        if y.sum() < 30: rows.append({"platform": p, "posts": len(q), "no views": int(y.sum())}); continue
        s = np.zeros(len(q)); X = q[cols].fillna(-1)
        imp = np.zeros(len(cols))
        for tr, te in GroupKFold(5).split(X, y, q.brand):
            c = CLF().fit(X.iloc[tr], y[tr]); s[te] = c.predict_proba(X.iloc[te])[:, 1]; imp += c.get_feature_importance()
        brands = q.brand.unique(); idx = {b: np.where(q.brand == b)[0] for b in brands}
        B = [roc_auc_score(y[i], s[i]) for i in (np.concatenate([idx[b] for b in rng.choice(brands, len(brands))]) for _ in range(n_boot)) if 0 < y[i].sum() < len(i)]
        top = pd.Series(imp / 5, cols).sort_values(ascending=False)
        rows.append({"platform": p, "posts": len(q), "no views": int(y.sum()), "ROC AUC": roc_auc_score(y, s), "AUC low": np.percentile(B, 2.5), "AUC high": np.percentile(B, 97.5),
                     "top features": ", ".join(f"{k} {v:.0f}" for k, v in top.head(3).items())})
    return pd.DataFrame(rows)


def zero_view_predictions(f, meta, N):
    """Train the final model on posts with views > 0, predict the posts whose count is 0."""
    rows = []
    for p in PLATS:
        m = (meta.platform == p).values; X = f[m][em.ENG].fillna(0); z = N[(N.platform == p) & (N.views == "zero")]
        if len(z) == 0: continue
        pred = rt.FINAL().fit(X, meta.y[m]).predict(z[em.ENG].fillna(0))
        rows.append({"platform": p, "zero-view posts": len(z), "median predicted views": float(10 ** np.median(pred)),
                     "share predicted < 10 views": float((pred < 1).mean()), "share with 0 likes and comments": float(((z.log_likes.fillna(0) == 0) & (z.log_comments.fillna(0) == 0)).mean())})
    return pd.DataFrame(rows)


def plot_viral(T, path=None, lang="en"):
    import matplotlib.pyplot as plt
    pl = lang == "pl"
    nm = {"followers": "obserwujący" if pl else "followers", "Brand24 reach": "zasięg Brand24" if pl else "Brand24 reach",
          "regression (final model)": "regresja (model końcowy)" if pl else "regression (final model)", "classifier (CatBoost)": "klasyfikator (CatBoost)" if pl else "classifier (CatBoost)"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6), sharex=True)
    for x, p in zip(ax, PLATS):
        r = T[T.platform == p].reset_index(drop=True); i = np.arange(len(r))[::-1]
        x.errorbar(r["precision top 10%"], i, xerr=[r["precision top 10%"] - r["P@10 low"], r["P@10 high"] - r["precision top 10%"]], fmt="o", color=eu.PCOL[p], capsize=3)
        for j, v, hi in zip(i, r["precision top 10%"], r["P@10 high"]): x.text(hi + .01, j, f"{v:.0%}", va="center", fontsize=9)
        x.axvline(TOP, color="grey", ls=":", lw=.8); x.set_yticks(i); x.set_yticklabels([nm[k] for k in r.score]); x.set_xlim(0, 1)
        x.set_title(eu.NAMES[p], fontweight="bold"); x.set_xlabel("precyzja w 10% najwyżej ocenionych" if pl else "precision among the 10% top-scored")
    fig.suptitle("10% najczęściej oglądanych: model końcowy trafia w 63-83%, same obserwujące w 36-44%, losowo 10%" if pl else "Top 10% of views: the final model finds 63-83%, followers alone 36-44%, chance 10%", fontweight="bold")
    plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()
