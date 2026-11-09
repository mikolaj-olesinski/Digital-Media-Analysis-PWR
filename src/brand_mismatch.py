"""Does a brand's mentions sit next to posts from its own industry in the text graph? A low score flags possible false keyword matches (notebooks/models/15_brand_mismatch.ipynb).
Graph as in notebook 09: TF-IDF cosine, 10 nearest neighbours per platform. Neighbours of the same brand are ignored, so a brand cannot vouch for itself."""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import eda_utils as eu, extra_models as em, graph_lab as gl

SEED = 42


def neighbour_agreement(raw, ind, plat, k=10):
    """Per mention: share of other-brand neighbours from the same industry, and the share expected by chance (that industry's share of other-brand posts)."""
    r = raw[raw.platform == plat].reset_index(drop=True); text = (r.title.fillna("") + " " + r.content.fillna("")).values; A, _ = gl.knn_graph(em.text_similarity(text), k); b = r.brand.values; i_ = r.brand.map(ind).fillna("other").values; rows = []
    for j in range(len(r)):
        nb = np.nonzero(A[j])[0]; nb = nb[b[nb] != b[j]]; other = b != b[j]
        rows.append({"platform": eu.NAMES[plat], "brand": b[j], "industry": i_[j], "neighbours": len(nb), "same industry": (i_[nb] == i_[j]).mean() if len(nb) else np.nan, "expected": (i_[other] == i_[j]).mean()})
    return pd.DataFrame(rows), r


def brand_scores(M, min_mentions=10):
    """Per brand and platform: observed agreement, expected, and lift = observed / expected. Brands with fewer mentions, or alone in their industry on the platform, are left out."""
    g = M.dropna(subset=["same industry"]).groupby(["platform", "brand", "industry"]).agg(mentions=("same industry", "size"), observed=("same industry", "mean"), expected=("expected", "mean")).reset_index()
    return g[(g.mentions >= min_mentions) & (g.expected > 0)].assign(lift=lambda d: d.observed / d.expected).sort_values("lift")


def top_terms(r, brand, n=8):
    """Most typical words of one brand's mentions against the rest of the platform (TF-IDF mean)."""
    t = (r.title.fillna("") + " " + r.content.fillna("")); t = t.where(t.str.strip() != "", "emptytext"); v = TfidfVectorizer(min_df=2, sublinear_tf=True, stop_words="english"); X = v.fit_transform(t); w = np.array(v.get_feature_names_out()); m = (r.brand == brand).values
    return ", ".join(w[np.asarray(X[m].mean(0) - X[~m].mean(0)).ravel().argsort()[::-1][:n]])


def flag_auc(S, info):
    """Do names flagged as having another meaning get lower lift? AUC with brand-level bootstrap CI (0.5 = no relation)."""
    d = S.merge(info[["brand", "ambiguous_name"]], on="brand"); y = (d.ambiguous_name == "yes").values; s = -d.lift.values; rng = np.random.default_rng(SEED); bs = []
    for _ in range(1000):
        i = rng.integers(0, len(d), len(d))
        if 0 < y[i].sum() < len(i): bs.append(roc_auc_score(y[i], s[i]))
    return {"AUC": roc_auc_score(y, s), "low": np.percentile(bs, 2.5), "high": np.percentile(bs, 97.5), "brand-platform pairs": len(d), "flagged": int(y.sum())}
