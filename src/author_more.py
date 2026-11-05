"""Author history in depth, variance components, brand totals, anomalies, views per like (notebooks/models/30_author_more.ipynb, E70-E75).
E70 author history: number of earlier posts, shrinkage, last post vs mean, and at posting time (no reactions).
E71 can the author's level be predicted from what is known about the account (no history)?
E72 variance components of the final model's error: same author, same brand (other author), unrelated pairs.
E73 brand totals: error of a brand's summed views (what a client sees) against per-post error and Brand24's reach on X.
E74 anomalies: posts with ×10 more or fewer views than predicted; who they are.
E75 views per like: what the final model implies by follower and reaction tier, against Brand24's constants.
Author names and URLs are used only inside the functions as keys; outputs carry counts and statistics only."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
import eda_utils as eu, extra_models as em, reaction_tiers as rt, history as H, account_clusters as ac
from validation import SEED

PLATS = ["tiktok", "youtube", "twitter"]


def frame(raw, meta, pred):
    d = ac._author_frame(raw, meta, pred)
    d["followers"] = raw.author_followers.values; d["likes"] = raw.likes.values; d["comments"] = raw.comments.values; d["language"] = raw.language.values
    return d


# ---------- E70 ----------
def history_features(d):
    """For each post: over-performance of the author's earlier non-repeat posts (sum, count, last value, days since last)."""
    out = pd.DataFrame(index=d.index, columns=["hsum", "hn", "hlast", "hdays"], dtype=float)
    for p in PLATS:
        q = d[d.platform == p].sort_values("t")
        for _, z in q.groupby("author"):
            for i in range(1, len(z)):
                b = z.iloc[i]; e = z.iloc[:i]; e = e[(e.text != b.text) & (e.url != b.url) & (e.y != b.y)]
                if len(e): out.loc[z.index[i]] = [e.r.sum(), len(e), e.r.iloc[-1], (b.t - e.t.iloc[-1]).total_seconds() / 86400]
    return out


def _folds(brand):
    b = brand.unique(); f = {x: i % 5 for i, x in enumerate(np.random.default_rng(SEED).permutation(b))}
    return brand.map(f).values


def _fit(y, base, c, k, grid):
    """Choose the parameter in `grid` on the other folds; correction c(param) added to base."""
    fit = np.zeros(len(y))
    for j in range(5):
        tr, te = k != j, k == j
        g = grid[np.argmin([np.abs(y[tr] - base[tr] - c(a)[tr]).mean() for a in grid])]; fit[te] = base[te] + c(g)[te]
    return fit


def history_variants(d, h):
    """Corrections on posts with an earlier post: mean × a; last post × a; empirical-Bayes sum / (n + k); split by number of earlier posts."""
    out = []
    for p in PLATS:
        s = d[(d.platform == p) & h.hn.notna()]; hh = h.loc[s.index]; k = _folds(s.brand); y, b = s.y.values, s.pred.values
        mean, last, sm, n = (hh.hsum / hh.hn).values, hh.hlast.values, hh.hsum.values, hh.hn.values
        V = {"final model": b,
             "+ a × mean of earlier": _fit(y, b, lambda a: a * mean, k, np.linspace(0, 1, 21)),
             "+ a × last earlier post": _fit(y, b, lambda a: a * last, k, np.linspace(0, 1, 21)),
             "+ sum / (n + k), shrunk by count": _fit(y, b, lambda a: sm / (n + a), k, np.array([0, .25, .5, 1, 2, 3, 5]))}
        for name, v in V.items():
            out.append(pd.DataFrame({"platform": p, "model": name, "brand": s.brand.values, "y": y, "pred": v, "earlier": np.where(n >= 3, "3+", n.astype(int).astype(str))}))
    return pd.concat(out, ignore_index=True)


def posting_time_oof(groups, pmeta, f):
    """Out-of-fold E43 posting-time model (followers + posting-time columns, no reactions), aligned to f.index."""
    pred = pd.Series(np.nan, index=f.index)
    for p in PLATS:
        m = (pmeta.platform == p).values; M = pmeta[m]
        X = pd.concat([groups["followers"][m], groups["posting time"][m]], axis=1); X.columns = [str(c) for c in X.columns]; X = X.reset_index(drop=True)
        v = np.full(len(M), np.nan)
        for tr, te in GroupKFold(5).split(X, M.y, M.brand):
            v[te] = H.catboost().fit(X.iloc[tr], M.y.iloc[tr]).predict(X.iloc[te])
        pred[M.index] = v
    return pred


# ---------- E71 ----------
def author_level(d, kind):
    """Authors with >= 2 posts with views. Target: mean over-performance of all their posts. Features known without history:
    log followers, account kind, language, platform, number of posts in the sample, share of zero-reaction posts.
    CatBoost, folds grouped by the author's main brand; Spearman and R² against the target, and the split-half reliability as ceiling."""
    from catboost import CatBoostRegressor
    from sklearn.metrics import r2_score
    q = d.assign(kind=kind.reindex(d.index).values, zero=((d.likes.fillna(0) == 0) & (d.comments.fillna(0) == 0)).astype(int))
    rows = []
    for p in PLATS:
        z = q[q.platform == p]; g = z.groupby("author")
        A = pd.DataFrame({"n": g.size(), "r": g.r.mean(), "lf": np.log10(g.followers.median().clip(lower=1)), "kind": g.kind.first(), "lang": g.language.agg(lambda s: s.mode().iat[0] if len(s.mode()) else "na").fillna("na"),
                          "zero": g.zero.mean(), "brand": g.brand.agg(lambda s: s.mode().iat[0]), "odd": g.r.apply(lambda s: s.iloc[::2].mean()), "even": g.r.apply(lambda s: s.iloc[1::2].mean())})
        A = A[A.n >= 2].reset_index(drop=True)
        if len(A) < 30: continue
        X = A[["n", "lf", "kind", "lang", "zero"]].astype({"kind": str, "lang": str}); pr = np.zeros(len(A))
        for tr, te in GroupKFold(5).split(X, A.r, A.brand):
            m = CatBoostRegressor(loss_function="RMSE", iterations=300, depth=4, random_seed=SEED, verbose=0, allow_writing_files=False, cat_features=["kind", "lang"], thread_count=4)
            pr[te] = m.fit(X.iloc[tr], A.r.iloc[tr]).predict(X.iloc[te])
        rel = spearmanr(A.odd, A.even)[0]
        rows.append({"platform": p, "authors": len(A), "Spearman (features)": spearmanr(pr, A.r)[0], "R² (features)": r2_score(A.r, pr),
                     "split-half Spearman (history)": rel, "sd of author level": float(A.r.std())})
    return pd.DataFrame(rows)


# ---------- E72 ----------
def _pairs(x, keys):
    """Sum over within-group pairs of x_i x_j, and of (x_i² + x_j²) / 2 (the matching denominator)."""
    g = pd.Series(x).groupby(keys); t, q, n = g.sum(), (pd.Series(x) ** 2).groupby(keys).sum(), g.size()
    return float(((t ** 2 - q) / 2).sum()), float(((n - 1) / 2 * q).sum()), float((n * (n - 1) / 2).sum())


def variance_components(d, col="r", n_boot=300):
    """Pairwise intraclass correlation of `col` (centred per platform): sum of x_i x_j over pairs of posts by the same author (or
    the same brand by different authors) divided by the sum of (x_i² + x_j²) / 2 over the same pairs; bounded by [-1, 1]. Bootstrap by brand
    (the n-th copy of a drawn brand pairs only with n-th copies, so a post never pairs with its own copy)."""
    rng = np.random.default_rng(SEED); rows = []
    for p in PLATS:
        z = d[d.platform == p].reset_index(drop=True); x0 = z[col].values - z[col].mean()
        def comp(idx, rep):
            x = x0[idx]; br = z.brand.values[idx]; au = z.author.values[idx]
            Sa, Da, Na = _pairs(x, [rep, au]); Sb, Db, Nb = _pairs(x, [rep, br]); Sba, Dba, Nba = _pairs(x, [rep, br, au])
            return Sa / Da, (Sb - Sba) / (Db - Dba), Na, Nb - Nba
        a, b, na, nb = comp(np.arange(len(z)), np.zeros(len(z), int))
        brands = z.brand.unique(); ix = {k: np.where(z.brand == k)[0] for k in brands}; B = []
        for _ in range(n_boot):
            pick = rng.choice(brands, len(brands)); occ = pd.Series(pick).groupby(pick).cumcount().values  # copy number of each drawn brand
            idx = np.concatenate([ix[k] for k in pick]); rep = np.concatenate([np.full(len(ix[k]), c) for k, c in zip(pick, occ)])
            B.append(comp(idx, rep)[:2])
        lo, hi = np.nanpercentile(np.array(B), [2.5, 97.5], axis=0)
        rows.append({"platform": p, "quantity": col, "same-author pairs": int(na), "share shared by an author": a, "author low": lo[0], "author high": hi[0],
                     "same-brand other-author pairs": int(nb), "share shared by a brand": b, "brand low": lo[1], "brand high": hi[1]})
    return pd.DataFrame(rows)


# ---------- E73 ----------
def brand_totals(raw, meta, pred, min_posts=5, n_boot=500):
    """Per platform and brand (>= min_posts posts with views): log10 of predicted total / true total. Predicted total = sum of
    10**pred (the median predictions), and calibrated = that times 10**(median brand-level log ratio true/predicted on the other folds). Compared with Brand24's reach on X and with
    the per-post error. Also per brand-month."""
    rows = []; rng = np.random.default_rng(SEED)
    d = pd.DataFrame({"platform": meta.platform, "brand": meta.brand, "y": meta.y, "pred": pred, "b24": raw.b24_estimated_reach.values, "month": raw.created_at.dt.strftime("%Y-%m").values})
    for p in PLATS:
        z = d[d.platform == p].copy(); k = _folds(z.brand); z["tv"] = 10 ** z.y; z["mv"] = 10 ** z.pred; z["fold"] = k; sm = np.zeros(len(z))
        bt = z.groupby("brand").agg(tv=("tv", "sum"), mv=("mv", "sum"), n=("y", "size"), fold=("fold", "first")); bt = bt[bt.n >= min_posts]
        for j in range(5):
            sm[k == j] = 10 ** np.median(np.log10(bt.tv / bt.mv)[bt.fold != j])
        z["pv"] = z.mv * sm
        for unit, keys in [("brand", ["brand"]), ("brand-month", ["brand", "month"])]:
            g = z.groupby(keys).agg(n=("y", "size"), tv=("tv", "sum"), pv=("pv", "sum"), mv=("mv", "sum"), b24=("b24", "sum"))
            g = g[g.n >= min_posts]
            if len(g) < 10: continue
            e = np.log10(g.pv / g.tv); e0 = np.log10(g.mv / g.tv)
            r = {"platform": p, "unit": unit, "units": len(g), "median posts": float(g.n.median()), "per-post MAE": float(np.abs(z.y - z.pred).mean()),
                 "MAE of total (calibrated)": float(np.abs(e).mean()), "bias of total (calibrated)": float(e.mean()), "MAE of total (sum of medians)": float(np.abs(e0).mean()),
                 "bias of total (sum of medians)": float(e0.mean()), "share within ×2": float((np.abs(e) < np.log10(2)).mean()), "Spearman of totals": spearmanr(g.pv, g.tv)[0]}
            if p == "twitter":
                eb = np.log10(g.b24.clip(lower=1) / g.tv); r.update({"Brand24 reach: MAE of total": float(np.abs(eb).mean()), "Brand24 reach: bias of total": float(eb.mean()), "Brand24 reach: Spearman": spearmanr(g.b24, g.tv)[0]})
            B = [np.abs(e.values[i]).mean() for i in (rng.integers(0, len(e), len(e)) for _ in range(n_boot))]; r["MAE total low"], r["MAE total high"] = np.percentile(B, [2.5, 97.5])
            rows.append(r)
    return pd.DataFrame(rows)


# ---------- E74 ----------
def anomalies(d, kind, labels, copy_flag, thr=1.0):
    """Posts with true - predicted >= thr (×10 more views than reactions suggest) or <= -thr. Share of each trait among them vs the rest."""
    q = d.assign(kind=kind.reindex(d.index).values, type=labels.reindex(d.index).values, copy=copy_flag.reindex(d.index).values,
                 zero=((d.likes.fillna(0) == 0) & (d.comments.fillna(0) == 0)), lf=np.log10(d.followers.where(d.followers > 0)))
    n = q.groupby(["platform", "author"]).y.transform("size"); q["repeat_author"] = n >= 2
    q["group"] = np.where(q.r >= thr, "×10 above", np.where(q.r <= -thr, "×10 below", "rest"))
    rows = []
    for p in PLATS:
        z = q[q.platform == p]
        for gname, s in z.groupby("group"):
            rows.append({"platform": p, "group": gname, "posts": len(s), "share of posts": len(s) / len(z), "promo": float((s.type == "promo").mean()), "spam": float((s.type == "spam").mean()),
                         "organic": float((s.type == "organic").mean()), "zero reactions": float(s.zero.mean()), "copied text": float(s["copy"].mean()), "brand or brand-named account": float(s.kind.isin(ac.KINDS[:2]).mean()),
                         "news / media": float((s.kind == ac.KINDS[2]).mean()), "median followers": float(10 ** s.lf.median()) if s.lf.notna().any() else np.nan,
                         "median likes": float(s.likes.median()), "median true views": float(10 ** s.y.median()), "author with >= 2 posts": float(s.repeat_author.mean())})
    return pd.DataFrame(rows)


def anomaly_persistence(d, h, thr=1.0):
    """Of posts with an earlier post by the author: P(×10 above | earlier mean over-performance >= 0.5) vs otherwise."""
    rows = []
    for p in PLATS:
        z = d[(d.platform == p) & h.hn.notna()]; m = (h.loc[z.index].hsum / h.loc[z.index].hn)
        hi = m >= .5
        rows.append({"platform": p, "posts": len(z), "earlier mean >= 0.5": int(hi.sum()), "P(×10 above | high history)": float((z.r[hi] >= thr).mean()) if hi.any() else np.nan,
                     "P(×10 above | other)": float((z.r[~hi] >= thr).mean()), "P(×10 above), all posts": float((d[d.platform == p].r >= thr).mean())})
    return pd.DataFrame(rows)


# ---------- E75 ----------
LIKE_BINS = [0, 1, 10, 100, 1000, 1e9]; LIKE_LAB = ["0", "1-9", "10-99", "100-999", "1000+"]
FOL_BINS = [-1, 1e3, 1e4, 1e5, 1e6, 1e12]; FOL_LAB = ["< 1k", "1k-10k", "10k-100k", "100k-1M", "> 1M"]


def views_per_like(f, meta, raw):
    """Final model fitted on all posts of a platform. Per post: implied views per like = 10**pred / likes (likes >= 1), and the
    marginal effect of doubling likes (and comments) on predicted views. Medians by follower and like tier; real views per like alongside."""
    rows = []
    for p in PLATS:
        m = (meta.platform == p).values; X = f[m][em.ENG].fillna(0).reset_index(drop=True); y = meta.y[m].values
        mod = rt.FINAL().fit(X, y); base = mod.predict(X)
        X2 = X.copy(); X2["log_likes"] = np.log1p(np.expm1(X.log_likes) * 2); X2["log_comments"] = np.log1p(np.expm1(X.log_comments) * 2)
        up = mod.predict(X2)
        likes = raw.likes[m].fillna(0).values; fol = raw.author_followers[m].fillna(-1).values
        z = pd.DataFrame({"likes": likes, "pv": 10 ** base, "tv": 10 ** y, "elast": (up - base) / np.log10(2),
                          "lt": pd.cut(likes, LIKE_BINS, labels=LIKE_LAB, right=False).astype(str), "ft": pd.cut(fol, FOL_BINS, labels=FOL_LAB, right=False).astype(str)})
        z = z[z.ft != "nan"]
        for (ft, lt), s in z.groupby(["ft", "lt"]):
            if len(s) < 15: continue
            pos = s[s.likes >= 1]
            rows.append({"platform": p, "followers": ft, "likes": lt, "posts": len(s), "median predicted views": float(s.pv.median()), "median true views": float(s.tv.median()),
                         "predicted views per like": float((pos.pv / pos.likes).median()) if len(pos) else np.nan, "true views per like": float((pos.tv / pos.likes).median()) if len(pos) else np.nan,
                         "elasticity (doubling reactions)": float(s.elast.median())})
    return pd.DataFrame(rows)


def plot_vpl(V, path=None, lang="en"):
    import matplotlib.pyplot as plt
    pl = lang == "pl"
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for x, p in zip(ax, PLATS):
        q = V[(V.platform == p) & (V.likes != "0")].pivot(index="followers", columns="likes", values="predicted views per like").reindex(index=FOL_LAB, columns=LIKE_LAB[1:])
        im = x.imshow(np.log10(q.values.astype(float)), cmap="viridis", vmin=0, vmax=3, aspect="auto")
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                v = q.values[i, j]
                if np.isfinite(v): x.text(j, i, f"{v:.0f}", ha="center", va="center", color="white" if np.log10(v) < 2 else "black", fontsize=9)
        x.set_xticks(range(q.shape[1])); x.set_xticklabels(q.columns); x.set_yticks(range(q.shape[0])); x.set_yticklabels(q.index if p == PLATS[0] else [])
        x.set_xlabel("lajki" if pl else "likes"); x.set_title(eu.NAMES[p], fontweight="bold")
    ax[0].set_ylabel("obserwujący" if pl else "followers")
    fig.colorbar(im, ax=ax, shrink=.8, label="log10 wyświetleń na lajk (model)" if pl else "log10 views per like (model)")
    fig.suptitle("Ile wyświetleń na lajk przewiduje model końcowy (mediana; puste = < 15 postów)" if pl else "Views per like implied by the final model (median; blank = < 15 posts)", fontweight="bold", y=1.04)
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()
