"""Where the remaining error sits, how much author history is needed, author level without views, live forecast with
author history, intervals for brand totals (notebooks/models/31_author_next.ipynb, E76-E80).
E76 error breakdown by post and author traits (replies, retweets, links, author activity), and those traits as columns.
E77 history curve: correction from the author's last k earlier posts, k = 1, 2, 3, 5, 10.
E78 author level from the author's other posts in the sample (reactions only, no views), as columns of the final model.
E79 live forecast (first check -> later check, E40) plus the author's history.
E80 intervals for brand totals: brand-level empirical quantiles vs summing independent post residuals.
Author names and URLs are used only inside the functions as keys; outputs carry counts and statistics only."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import extra_models as em, reaction_tiers as rt, history as H, account_clusters as ac, author_more as am
from validation import SEED

PLATS = am.PLATS
lg = lambda x: np.log10(np.asarray(x, float) + 1)


# ---------- E76 ----------
def post_traits(raw_all, raw):
    """Columns known without views. Author activity is counted over every post of the platform in the sample (with or without views)."""
    c = raw.content.fillna("").astype(str); key = raw_all.author_url.fillna(raw_all.author_name)
    n_all = key.groupby([raw_all.platform, key]).transform("size")
    T = pd.DataFrame({"is_reply": c.str.match(r"^\s*@").astype(int), "is_retweet": c.str.match(r"^\s*RT @").astype(int), "has_link": c.str.contains("http").astype(int),
                      "n_mentions": c.str.count(r"@\w+"), "n_hashtags": c.str.count(r"#\w+"), "log_author_posts": np.log10(n_all.loc[raw.index].fillna(1).values)}, index=raw.index)
    return T


def error_breakdown(d, T):
    """Per platform and trait group: share of posts, share of the summed absolute error, MAE and bias of the final model."""
    q = d.assign(ae=np.abs(d.r), zero=((d.likes.fillna(0) == 0) & (d.comments.fillna(0) == 0)).map({True: "0 reactions", False: "some reactions"}),
                 reply=T.is_reply.map({1: "reply (starts with @)", 0: "not a reply"}), link=T.has_link.map({1: "has a link", 0: "no link"}),
                 activity=pd.cut(10 ** T.log_author_posts, [0, 1, 4, 19, 1e9], labels=["author: 1 post", "author: 2-4", "author: 5-19", "author: 20+"]).astype(str),
                 tier=ac.tier(d.followers))
    rows = []
    for p in PLATS:
        z = q[q.platform == p]; tot = z.ae.sum()
        for col in ["reply", "link", "zero", "activity", "tier"]:
            for k, s in z.groupby(col):
                rows.append({"platform": p, "trait": col, "group": k, "posts": len(s), "share of posts": len(s) / len(z), "share of error": s.ae.sum() / tot,
                             "MAE": float(s.ae.mean()), "bias": float(-s.r.mean())})
    return pd.DataFrame(rows)


def top_author_share(d, top=(1, 5, 10)):
    """Share of posts and of summed absolute error from the most active authors (by posts with views)."""
    rows = []
    for p in PLATS:
        z = d[d.platform == p]; n = z.groupby("author").size().sort_values(ascending=False)
        for k in top:
            s = z[z.author.isin(n.index[:k])]
            rows.append({"platform": p, "top authors": k, "posts": len(s), "share of posts": len(s) / len(z), "share of error": float(np.abs(s.r).sum() / np.abs(z.r).sum()), "MAE": float(np.abs(s.r).mean())})
    return pd.DataFrame(rows)


def with_columns(f, meta, extra, name):
    """Final model vs final model + extra columns, out of fold (same brand folds)."""
    out = []
    for p in PLATS:
        m = (meta.platform == p).values; M = meta[m].reset_index(drop=True); X = f[m][em.ENG].fillna(0).reset_index(drop=True)
        Xe = pd.concat([X, extra[m].reset_index(drop=True).fillna(-1)], axis=1)
        for k, XX in [("final model", X), (name, Xe)]:
            out.append(pd.DataFrame({"platform": p, "model": k, "brand": M.brand.values, "y": M.y.values, "pred": em.cv_predict(rt.FINAL, XX, M.y, M.brand.values), "idx": f.index[m]}))
    return pd.concat(out, ignore_index=True)


# ---------- E77 ----------
def history_lists(d):
    """For each post: list of over-performances of the author's earlier non-repeat posts, oldest first (same filter as E70)."""
    out = {}
    for p in PLATS:
        q = d[d.platform == p].sort_values("t")
        for _, z in q.groupby("author"):
            for i in range(1, len(z)):
                b = z.iloc[i]; e = z.iloc[:i]; e = e[(e.text != b.text) & (e.url != b.url) & (e.y != b.y)]
                if len(e): out[z.index[i]] = e.r.values
    return pd.Series(out, dtype=object)


GRID = np.array([0, .25, .5, 1, 2, 3, 5])


def history_curve(d, L, ks=(1, 2, 3, 5, 10)):
    """Two views. 'posts with >= k earlier': correction sum(last k) / (k + a) on every post with at least k earlier posts.
    'same posts (>= 5 earlier)': posts with at least 5 earlier posts, using only the last k (k <= 5), so the posts stay fixed."""
    out = []
    for p in PLATS:
        s = d[(d.platform == p) & d.index.isin(L.index)]; LL = L.loc[s.index]; n = LL.map(len).values
        for view, kk in [("posts with >= k earlier", ks), ("same posts (>= 5 earlier)", [k for k in ks if k <= 5])]:
            for k in kk:
                sel = n >= (k if view.startswith("posts") else 5)
                if sel.sum() < 30: continue
                z = s[sel]; y, b = z.y.values, z.pred.values; sm = np.array([v[-k:].sum() for v in LL[sel]]); kf = am._folds(z.brand)
                fit = am._fit(y, b, lambda a: sm / (k + a), kf, GRID)
                for name, v in [("final model", b), ("+ last k earlier, shrunk", fit)]:
                    out.append(pd.DataFrame({"platform": p, "view": view, "k": k, "model": name, "brand": z.brand.values, "y": y, "pred": v}))
    return pd.concat(out, ignore_index=True)


def curve_table(P):
    rows = []
    for (p, view, k), q in P.groupby(["platform", "view", "k"]):
        g, lo, hi = H.gain_ci(q, p, "final model", "+ last k earlier, shrunk")
        base = q[q.model == "final model"]
        rows.append({"platform": p, "view": view, "k": k, "posts": len(base), "MAE final": float(np.abs(base.y - base.pred).mean()), "gain": g, "CI low": lo, "CI high": hi})
    return pd.DataFrame(rows)


def plot_curve(C, path=None, lang="en"):
    t = {"en": ("Author history: gain from the last k earlier posts", "earlier posts used (k)", "fall in MAE (log10)", {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X"}, "posts with ≥ k earlier", "same posts (≥ 5 earlier)"),
         "pl": ("Historia autora: zysk z k ostatnich wcześniejszych postów", "liczba użytych wcześniejszych postów (k)", "spadek MAE (log10)", {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X"}, "posty z ≥ k wcześniejszymi", "te same posty (≥ 5 wcześniejszych)")}[lang]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for a, view, lab in [(ax[0], "posts with >= k earlier", t[4]), (ax[1], "same posts (>= 5 earlier)", t[5])]:
        for p, col in zip(PLATS, ["#2a6fdb", "#d1495b", "#3a3a3a"]):
            q = C[(C.platform == p) & (C.view == view)]
            if not len(q): continue
            a.plot(q.k, q.gain, "o-", color=col, label=t[3][p]); a.fill_between(q.k, q["CI low"], q["CI high"], color=col, alpha=.12)
        a.axhline(0, color="grey", lw=.8); a.set_title(lab, fontsize=10); a.set_xlabel(t[1]); a.set_xticks([1, 2, 3, 5, 10] if view.startswith("posts") else [1, 2, 3, 5])
    ax[0].set_ylabel(t[2]); ax[0].legend(frameon=False); fig.suptitle(t[0], y=1.02); fig.tight_layout()
    if path: fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


# ---------- E78 ----------
def author_proxy(raw_all, raw):
    """From the author's OTHER posts on the same platform in the sample (any time, with or without views; reactions only):
    count, mean log likes, mean log comments, mean log likes per follower, share with zero reactions, number of brands,
    and whether the same author name appears on another platform."""
    a = raw_all.assign(key=raw_all.author_url.fillna(raw_all.author_name), nm=raw_all.author_name.fillna("").str.lower().str.strip(),
                       ll=lg(raw_all.likes.fillna(0)), lc=lg(raw_all.comments.fillna(0)),
                       lpf=lg(raw_all.likes.fillna(0)) - lg(raw_all.author_followers.fillna(0)), z=((raw_all.likes.fillna(0) == 0) & (raw_all.comments.fillna(0) == 0)).astype(float))
    g = a.groupby(["platform", "key"]); n = g.ll.transform("size")
    def loo(col): return (g[col].transform("sum") - a[col]) / (n - 1).replace(0, np.nan)
    nb = g.brand.transform("nunique") - 1
    plats = a[a.nm != ""].groupby("nm").platform.nunique()
    P = pd.DataFrame({"other_posts": n - 1, "other_mean_log_likes": loo("ll"), "other_mean_log_comments": loo("lc"), "other_likes_per_follower": loo("lpf"),
                      "other_zero_share": loo("z"), "other_brands": nb, "on_other_platform": (a.nm.map(plats).fillna(1) > 1).astype(int)}, index=a.index)
    return P.loc[raw.index]


# ---------- E79 ----------
def live_frame(root, raw_all):
    """E40 out-of-fold forecasts (CatBoost on the first check + reactions + ages) joined with authors and texts by mention id."""
    hd = H.load(root); s = H.snapshots(hd); p = H.pairs(s); p = p[p.platform.isin(PLATS)]
    P = H.forecast_cv(p); P = P[P.model == "CatBoost: first check + reactions + ages"].reset_index(drop=True)
    r = raw_all.assign(mention_id=raw_all.mention_id.astype(str)).set_index("mention_id")
    j = r.reindex(P.mention_id.astype(str))
    norm = ac._clean(j.title.fillna("") + " " + j.content.fillna("")).str.split().str.join(" ").values
    return pd.DataFrame({"platform": P.platform, "author": j.author_url.fillna(j.author_name).values, "t": j.created_at.values, "brand": P.brand, "text": norm,
                         "url": j.url.fillna("").str.lower().str.rstrip("/").values, "y": P.y.values, "pred": P.pred.values, "r": P.y.values - P.pred.values,
                         "mention_id": P.mention_id.astype(str).values, "gap_h": P.gap_h.values})


def live_compare(dl, hl, hnow_by_mid):
    out = []
    for p in PLATS:
        s = dl[(dl.platform == p) & hl.hn.notna()]
        if len(s) < 30: continue
        hh = hl.loc[s.index]; k = am._folds(s.brand); y, b = s.y.values, s.pred.values
        hn = hnow_by_mid.reindex(s.mention_id.values); okn = hn.hn.notna().values
        V = {"live forecast (E40)": b, "+ author history (forecast residuals)": am._fit(y, b, lambda a: hh.hsum.values / (hh.hn.values + a), k, GRID)}
        smn, nn = hn.hsum.fillna(0).values, hn.hn.fillna(0).values
        V["+ author history (now-cast residuals)"] = am._fit(y, b, lambda a: np.where(okn, smn / (nn + a + 1e-9), 0), k, GRID)
        for name, v in V.items():
            out.append(pd.DataFrame({"platform": p, "model": name, "brand": s.brand.values, "y": y, "pred": v}))
    return pd.concat(out, ignore_index=True)


# ---------- E80 ----------
def brand_intervals(meta, pred, min_posts=5, alphas=(.8, .9), n_sim=500):
    """Brand totals (calibrated as in E73). Interval A: calibrated total × 10^quantiles of the brand-level log ratio on the
    other folds (pooled); A by size: quantiles from other-fold brands of a similar size (5-9, 10-29, 30+ posts).
    Interval B: each post's prediction plus residuals drawn independently from the other folds' posts, summed (ignores
    that errors of one brand's posts are correlated). Coverage and median width (× factor, half-width) on held-out brands."""
    rng = np.random.default_rng(SEED); rows = []
    for p in PLATS:
        z = pd.DataFrame({"brand": meta.brand, "y": meta.y, "pred": pred})[meta.platform == p].reset_index(drop=True)
        z["fold"] = am._folds(z.brand); z["tv"], z["mv"] = 10 ** z.y, 10 ** z.pred
        bt = z.groupby("brand").agg(tv=("tv", "sum"), mv=("mv", "sum"), n=("y", "size"), fold=("fold", "first")); bt = bt[bt.n >= min_posts].copy()
        bt["e"] = np.log10(bt.tv / bt.mv); bt["size"] = pd.cut(bt.n, [0, 9, 29, 1e9], labels=["5-9", "10-29", "30+"]).astype(str)
        res = {}
        for al in alphas:
            lo_q, hi_q = (1 - al) / 2, 1 - (1 - al) / 2
            for meth in ["A: brand-level quantiles", "A: by brand size", "B: independent post residuals"]:
                cov, wid = [], []
                for j in range(5):
                    tr, te = bt[bt.fold != j], bt[bt.fold == j]
                    for b, row in te.iterrows():
                        if meth.startswith("A: brand-level"):
                            lo, hi = np.quantile(tr.e, lo_q), np.quantile(tr.e, hi_q)
                        elif meth == "A: by brand size":
                            tt = tr[tr["size"] == row["size"]]; tt = tt if len(tt) >= 15 else tr; lo, hi = np.quantile(tt.e, lo_q), np.quantile(tt.e, hi_q)
                        else:
                            pool = (z.y - z.pred).values[(z.fold != j).values]; mv = z.mv.values[(z.brand == b).values]
                            sims = np.log10((mv[None, :] * 10 ** rng.choice(pool, (n_sim, len(mv)))).sum(1) / mv.sum()); lo, hi = np.quantile(sims, lo_q), np.quantile(sims, hi_q)
                        cov.append(lo <= row.e <= hi); wid.append((hi - lo) / 2)
                rows.append({"platform": p, "method": meth, "nominal": al, "brands": len(bt), "coverage": float(np.mean(cov)), "median half-width (×)": float(10 ** np.median(wid))})
        for sz, q in bt.groupby("size"):
            rows.append({"platform": p, "method": "spread of brand log ratio by size", "nominal": np.nan, "brands": len(q), "size": sz,
                         "coverage": np.nan, "median half-width (×)": float(10 ** ((np.quantile(q.e, .95) - np.quantile(q.e, .05)) / 2))})
    return pd.DataFrame(rows)
