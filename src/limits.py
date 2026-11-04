"""Limits of the final model (notebooks/models/25_limits.ipynb, E55-E58).
E55 leakage audit: folds grouped by brand ∪ author ∪ identical text (connected components) and plain random folds.
E56 noise ceiling: how different are the views of posts with almost the same inputs? Bounds the MAE of any model on them.
E57 quantile models (10/50/90%) and tail weighting.
E58 abstention: error against the share of posts kept, ranked by the predicted interval width."""
import numpy as np, pandas as pd
from catboost import CatBoostRegressor
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
import eda_utils as eu, extra_models as em, reaction_tiers as rt
from validation import SEED

PLATS = ["tiktok", "youtube", "twitter"]
NUM = ["log_followers", "log_likes", "log_comments", "log_shares"]


def components(meta, raw):
    """Group id joining posts that share a brand, an author or the same normalised text (> 20 characters)."""
    n = len(meta); keys = []
    norm = (raw.title.fillna("") + " " + raw.content.fillna("")).str.lower().str.replace(r"https?://\S+|[^\w ]", "", regex=True).str.split().str.join(" ")
    for col in [meta.brand.values, raw.author_url.values, np.where(norm.str.len() > 20, norm.values, None)]:
        s = pd.Series(col)
        for _, idx in s[s.notna()].groupby(s[s.notna()]).groups.items():
            idx = np.asarray(idx)
            if len(idx) > 1: keys.append(np.c_[idx[:-1], idx[1:]])
    e = np.vstack(keys); A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return connected_components(A, directed=False)[1]


def _cv(X, y, splits):
    pred = np.full(len(y), np.nan)
    for tr, te in splits:
        pred[te] = rt.FINAL().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return pred


def leakage(f, meta, raw):
    comp = components(meta.reset_index(drop=True), raw.reset_index(drop=True)); out = []; info = []
    for p in PLATS:
        m = (meta.platform == p).values; M = meta[m].reset_index(drop=True); X = f[m][em.ENG].fillna(0).reset_index(drop=True); y = M.y.values; c = comp[m]
        info.append({"platform": p, "posts": len(M), "brands": M.brand.nunique(), "components": len(set(c)), "largest component share": float(pd.Series(c).value_counts(normalize=True).iloc[0])})
        for k, sp in {"brand folds (usual)": GroupKFold(5).split(X, y, M.brand), "brand ∪ author ∪ text folds": GroupKFold(5).split(X, y, c),
                      "random folds (brands seen)": KFold(5, shuffle=True, random_state=SEED).split(X)}.items():
            out.append(pd.DataFrame({"platform": p, "model": k, "brand": M.brand.values, "y": y, "pred": _cv(X, y, sp)}))
    return pd.concat(out, ignore_index=True), pd.DataFrame(info)


def noise_ceiling(f, meta, radius=.05):
    """Pairs of different-brand posts on the same platform whose four log inputs are all within `radius` (log10/ln units).
    For two independent draws from the same conditional distribution, E|y1-y2| lies between the best possible MAE
    and twice it, so best MAE >= E|y1-y2| / 2. Reported with the final model's MAE on the same posts."""
    rows = []
    for p in PLATS:
        m = (meta.platform == p).values; M = meta[m].reset_index(drop=True); X = f[m][NUM].fillna(-1).reset_index(drop=True).values; y = M.y.values
        nn = NearestNeighbors(n_neighbors=6, metric="chebyshev").fit(X); d, j = nn.kneighbors(X)
        pair = []
        for i in range(len(X)):
            for dd, jj in zip(d[i, 1:], j[i, 1:]):
                if dd <= radius and M.brand[jj] != M.brand[i] and i < jj: pair.append((i, jj)); break
        pair = np.array(pair); dy = np.abs(y[pair[:, 0]] - y[pair[:, 1]])
        cv = em.cv_predict(rt.FINAL, pd.DataFrame(X, columns=NUM), M.y, M.brand); used = np.unique(pair.ravel())
        rows.append({"platform": p, "pairs": len(pair), "posts in pairs": len(used), "mean |Δ log views|": float(dy.mean()), "lower bound on MAE (½ mean |Δ|)": float(dy.mean() / 2),
                     "final model MAE on these posts": float(np.abs(y[used] - cv[used]).mean()), "final model MAE, all posts": float(np.abs(y - cv).mean())})
    return pd.DataFrame(rows)


QS = [.1, .5, .9]


def quantiles(f, meta, tail_weight=2.0):
    """Out-of-fold 10/50/90% CatBoost quantiles, and a median model with extra weight on the outer fifths of the training target."""
    out = []
    for p in PLATS:
        m = (meta.platform == p).values; M = meta[m].reset_index(drop=True); X = f[m][em.ENG].fillna(0).reset_index(drop=True); y = M.y.values
        Q = np.full((len(y), 3), np.nan); W = np.full(len(y), np.nan); base = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, M.brand):
            mq = CatBoostRegressor(loss_function="MultiQuantile:alpha=0.1,0.5,0.9", iterations=500, random_seed=SEED, verbose=0, allow_writing_files=False, thread_count=4)
            Q[te] = np.sort(mq.fit(X.iloc[tr], y[tr]).predict(X.iloc[te]), axis=1)
            r = pd.Series(y[tr]).rank(pct=True).values; w = np.where((r < .2) | (r > .8), tail_weight, 1.0)
            W[te] = rt.FINAL().fit(X.iloc[tr], y[tr], sample_weight=w).predict(X.iloc[te])
            base[te] = rt.FINAL().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        tier = rt.tier(f[m])
        out.append(pd.DataFrame({"platform": p, "brand": M.brand.values, "y": y, "tier": tier, "q10": Q[:, 0], "q50": Q[:, 1], "q90": Q[:, 2], "tail weighted": W, "final model": base}))
    return pd.concat(out, ignore_index=True)


def tier_width(Qd):
    """E34-style baseline: one width per reaction tier from the 90% quantile of the out-of-fold residuals of the final model."""
    w = np.full(len(Qd), np.nan)
    for p in PLATS:
        m = (Qd.platform == p).values; q = Qd[m]; e = (q.y - q["final model"]).values; t = q.tier.values; ww = np.zeros(m.sum())
        for k in np.unique(t):
            s = t == k; ww[s] = 2 * np.quantile(np.abs(e[s]), .9) if s.sum() >= 30 else 2 * np.quantile(np.abs(e), .9)
        w[m] = ww
    return w


def interval_table(Qd):
    rows = []
    for p in PLATS:
        q = Qd[Qd.platform == p]
        fifth = pd.qcut(q.y.rank(method="first"), 5, labels=["lowest", "2", "3", "4", "highest"])
        for lab, z in [("all", q)] + list(q.groupby(fifth, observed=True)):
            rows.append({"platform": p, "fifth of true views": lab, "posts": len(z), "coverage 10-90%": float(((z.y >= z.q10) & (z.y <= z.q90)).mean()),
                         "median width (×)": float(10 ** np.median(z.q90 - z.q10)), "MAE q50": float(np.abs(z.y - z.q50).mean()),
                         "MAE final": float(np.abs(z.y - z["final model"]).mean()), "MAE tail weighted": float(np.abs(z.y - z["tail weighted"]).mean()),
                         "bias final": float((z["final model"] - z.y).mean()), "bias tail weighted": float((z["tail weighted"] - z.y).mean())})
    return pd.DataFrame(rows)


def abstention(Qd, fracs=np.linspace(.1, 1, 10)):
    """MAE of the final model on the share of posts with the narrowest predicted interval; quantile width vs tier width vs random."""
    rows = []; rng = np.random.default_rng(SEED); Qd = Qd.assign(tierw=tier_width(Qd))
    for p in PLATS:
        q = Qd[Qd.platform == p].reset_index(drop=True); e = np.abs(q.y - q["final model"]).values
        order = {"quantile width": np.argsort((q.q90 - q.q10).values, kind="stable"), "reaction-tier width": np.argsort(q.tierw.values + rng.random(len(q)) * 1e-6),
                 "random": rng.permutation(len(q)), "oracle (true error)": np.argsort(e)}
        for k, o in order.items():
            for fr in fracs:
                n = max(1, int(round(fr * len(q)))); rows.append({"platform": p, "ranking": k, "kept": float(fr), "MAE": float(e[o[:n]].mean())})
    return pd.DataFrame(rows)


def plot_abstention(A, path=None, lang="en"):
    import matplotlib.pyplot as plt
    pl = lang == "pl"
    nm = {"quantile width": "szerokość kwantyli" if pl else "quantile width", "reaction-tier width": "szerokość wg reakcji" if pl else "reaction-tier width",
          "random": "losowo" if pl else "random", "oracle (true error)": "wyrocznia (prawdziwy błąd)" if pl else "oracle (true error)"}
    sty = {"quantile width": dict(color="#1f77b4"), "reaction-tier width": dict(color="#d62728"), "random": dict(color="grey", ls=":"), "oracle (true error)": dict(color="black", ls="--", lw=.8)}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for x, p in zip(ax, PLATS):
        for k in nm:
            r = A[(A.platform == p) & (A.ranking == k)]; x.plot(r.kept * 100, r.MAE, marker="o", ms=3, label=nm[k], **sty[k])
        x.set_title(eu.NAMES[p], fontweight="bold"); x.set_xlabel("% postów, które oceniamy" if pl else "% of posts kept"); x.invert_xaxis()
    ax[0].set_ylabel("MAE log10(wyświetleń)" if pl else "MAE of log10(views)"); ax[-1].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False)
    fig.suptitle("Odrzucanie niepewnych postów: błąd na postach, które zostają" if pl else "Abstaining on uncertain posts: error on the posts that are kept", fontweight="bold")
    plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()
