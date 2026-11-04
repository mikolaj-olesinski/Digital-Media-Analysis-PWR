"""Context from other posts, text embeddings and an offset target (notebooks/models/21_context_text.ipynb, E49-E51).
E49: features from other posts of the same author, hashtag and brand, computed on all 12,500 mentions (all five
     platforms) without any view counts, leave-self-out. No label leaks, so test brands may use their own context.
E50: sentence embeddings (local multilingual-e5-small, PCA 16) and TF-IDF SVD 16, added to the final model and to a
     model without reactions.
E51: offset target: the model learns log10(views) - log10(likes + 1) and the offset is added back."""
import re
import numpy as np, pandas as pd
from catboost import CatBoostRegressor
import eda_utils as eu, extra_models as em, reaction_tiers as rt
from validation import SEED

PLATS = ["tiktok", "youtube", "twitter"]
TAG = re.compile(r"#(\w{2,})", re.U)


def _text(d):
    return (d.title.fillna("") + " " + d.content.fillna("")).str.strip()


def _loo_mean(key, val):
    """Mean of `val` over the other rows with the same key (NaN when the row is alone or the key is missing)."""
    s = pd.DataFrame({"k": key, "v": val}); ok = s.k.notna() & s.v.notna()
    g = s[ok].groupby("k").v; tot, cnt = s.k.map(g.sum()), s.k.map(g.count())
    own = np.where(ok, s.v, 0); n = cnt - ok.astype(int)
    return ((tot - own) / n.where(n > 0)).values, cnt.fillna(0).values


def context(root):
    """Unlabelled context features for every mention (index = raw row index)."""
    d = eu.load(root); ll = np.log1p(d.likes); lf = np.log10(d.author_followers.where(d.author_followers > 0))
    c = pd.DataFrame(index=d.index)
    a = d.author_url.where(d.author_url.notna(), None)
    c["author_posts"] = np.log1p(d.groupby("author_url").mention_id.transform("size").fillna(1))
    c["author_brands"] = d.groupby("author_url").brand.transform("nunique").fillna(1)
    c["author_mean_likes"], _ = _loo_mean(a, ll)
    c["likes_vs_author"] = ll - c.author_mean_likes
    # hashtags: frequency in the whole data and mean likes of other posts with the tag
    tags = _text(d).str.lower().apply(lambda t: sorted(set(TAG.findall(t))))
    ex = tags.explode().dropna(); ex = ex.to_frame("tag").assign(ll=ll.reindex(ex.index).values, row=ex.index)
    freq = ex.tag.map(ex.tag.value_counts())
    tsum, tcnt = ex.groupby("tag").ll.sum(), ex.groupby("tag").ll.count()
    ex["other_mean"] = (ex.tag.map(tsum) - ex.ll.fillna(0)) / (ex.tag.map(tcnt) - ex.ll.notna().astype(int)).where(lambda x: x > 0)
    ex["freq"] = freq.values
    c["n_hashtags"] = tags.str.len()
    c["hashtag_max_freq"] = np.log1p(ex.groupby("row").freq.max()).reindex(d.index).fillna(0)
    c["hashtag_mean_likes"] = ex.groupby("row").other_mean.mean().reindex(d.index)
    # brand on the same platform and on the other platforms (FB/IG included), leave-self-out
    bp = d.brand + "|" + d.platform
    c["brand_platform_posts"] = np.log1p(bp.map(bp.value_counts()))
    c["brand_mean_likes"], _ = _loo_mean(bp, ll)
    c["brand_mean_followers"], _ = _loo_mean(bp, lf)
    c["likes_vs_brand"] = ll - c.brand_mean_likes
    other = {}
    for p in d.platform.unique():
        m = d.platform != p; g = pd.DataFrame({"b": d.brand[m], "v": ll[m]}).groupby("b").v.mean()
        other[p] = d.brand[d.platform == p].map(g)
    c["brand_likes_other_platforms"] = pd.concat(other.values()).reindex(d.index)
    # near-duplicates: the same normalised text elsewhere in the data
    norm = _text(d).str.lower().str.replace(r"https?://\S+|[^\w ]", "", regex=True).str.split().str.join(" ")
    c["copies"] = np.log1p(norm.map(norm.value_counts()) - 1).where(norm.str.len() > 20, 0)
    return c


GROUPS = {"author": ["author_posts", "author_brands", "author_mean_likes", "likes_vs_author"],
          "hashtags": ["n_hashtags", "hashtag_max_freq", "hashtag_mean_likes"],
          "brand": ["brand_platform_posts", "brand_mean_likes", "brand_mean_followers", "likes_vs_brand", "brand_likes_other_platforms"],
          "copies": ["copies"]}


def embeddings(root, cache, model_dir=None, batch=64):
    """multilingual-e5-small embeddings of title + content for every mention, cached as a pickle (git-ignored)."""
    import os, pickle
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    from sentence_transformers import SentenceTransformer
    d = eu.load(root); t = ("query: " + _text(d).str.replace(r"https?://\S+", "", regex=True).str.slice(0, 1000)).tolist()
    m = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu"); m.max_seq_length = 256
    E = m.encode(t, batch_size=batch, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    out = pd.DataFrame(E, index=d.index); pickle.dump(out, open(cache, "wb")); return out


def reduce(E, root, dims=16):
    """PCA of the embeddings and TF-IDF SVD of the text, both fitted on all mentions without labels."""
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    d = eu.load(root)
    pe = pd.DataFrame(PCA(dims, random_state=SEED).fit_transform(E.values), index=E.index, columns=[f"emb{i}" for i in range(dims)])
    T = TfidfVectorizer(min_df=3, max_features=30000, ngram_range=(1, 2), sublinear_tf=True).fit_transform(_text(d).str.lower())
    ts = pd.DataFrame(TruncatedSVD(dims, random_state=SEED).fit_transform(T), index=d.index, columns=[f"tfidf{i}" for i in range(dims)])
    return pe, ts


def offset_model():
    """CatBoost on log10(views) - log10(likes + 1); predict adds the offset back."""
    class Offset:
        def fit(self, X, y):
            self.m = rt.FINAL().fit(X, y - self._off(X)); return self
        def predict(self, X):
            return self.m.predict(X) + self._off(X)
        @staticmethod
        def _off(X):
            return np.log10(np.expm1(X.log_likes.clip(lower=0)) + 1).values
    return Offset()


def run(f, meta, C, pe, ts):
    """Out-of-fold predictions for every variant, long table."""
    base = em.ENG
    reactionless = ["log_followers", "na_followers"]
    variants = {"final model": (base, rt.FINAL),
                **{f"+ {k}": (base + v, rt.FINAL) for k, v in GROUPS.items()},
                "+ all context": (base + sum(GROUPS.values(), []), rt.FINAL),
                "+ embeddings": (base + list(pe.columns), rt.FINAL),
                "+ TF-IDF": (base + list(ts.columns), rt.FINAL),
                "offset target": (base, offset_model),
                "offset target + all context": (base + sum(GROUPS.values(), []), offset_model),
                "no reactions: followers": (reactionless, rt.FINAL),
                "no reactions: + embeddings": (reactionless + list(pe.columns), rt.FINAL),
                "no reactions: + TF-IDF": (reactionless + list(ts.columns), rt.FINAL),
                "no reactions: + all context": (reactionless + ["author_posts", "author_brands", "author_mean_likes", "n_hashtags", "hashtag_max_freq", "hashtag_mean_likes", "brand_platform_posts", "brand_mean_likes", "brand_mean_followers", "brand_likes_other_platforms", "copies"], rt.FINAL)}
    F = f.join(C).join(pe).join(ts)
    out = []
    for p in PLATS:
        m = (meta.platform == p).values; M = meta[m].reset_index(drop=True); X = F[m].reset_index(drop=True)
        for k, (cols, mk) in variants.items():
            pred = em.cv_predict(mk, X[cols], M.y, M.brand)
            out.append(pd.DataFrame({"platform": p, "model": k, "brand": M.brand.values, "y": M.y.values, "pred": pred}))
    return pd.concat(out, ignore_index=True)


def fifths(P, models=("final model", "offset target")):
    """Signed error by fifth of true views (E18 style)."""
    rows = []
    for p in PLATS:
        for k in models:
            q = P[(P.platform == p) & (P.model == k)]
            fi = pd.qcut(q.y.rank(method="first"), 5, labels=["lowest", "2", "3", "4", "highest"])
            for lab, z in q.groupby(fi, observed=True):
                rows.append({"platform": p, "model": k, "fifth": lab, "bias": float((z.pred - z.y).mean()), "MAE": float(np.abs(z.pred - z.y).mean())})
    return pd.DataFrame(rows)


def plot_gains(G, path=None, lang="en"):
    import matplotlib.pyplot as plt
    pl = lang == "pl"
    PLN = {"+ author": "+ autor", "+ hashtags": "+ hashtagi", "+ brand": "+ marka", "+ copies": "+ kopie tekstu", "+ all context": "+ cały kontekst",
           "+ embeddings": "+ embeddingi (e5)", "+ TF-IDF": "+ TF-IDF", "offset target": "cel z przesunięciem", "offset target + all context": "przesunięcie + kontekst",
           "no reactions: + embeddings": "bez reakcji: + embeddingi", "no reactions: + TF-IDF": "bez reakcji: + TF-IDF", "no reactions: + all context": "bez reakcji: + kontekst"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for x, p in zip(ax, PLATS):
        r = G[G.platform == p].reset_index(drop=True); i = np.arange(len(r))[::-1]
        x.errorbar(r.gain, i, xerr=[r.gain - r["CI low"], r["CI high"] - r.gain], fmt="o", color=eu.PCOL[p], capsize=3)
        for j, g, hi in zip(i, r.gain, r["CI high"]): x.text(hi + .003, j, f"{g:+.3f}", va="center", fontsize=8)
        x.axvline(0, color="black", lw=.8); x.axvline(.01, color="grey", ls=":", lw=.6)
        x.set_yticks(i); x.set_yticklabels([PLN.get(k, k) if pl else k for k in r.model]); x.set_title(eu.NAMES[p], fontweight="bold")
        x.set_xlabel("spadek MAE względem bazy (>0 = lepiej)" if pl else "fall in MAE vs its base (>0 = better)")
    fig.suptitle("Tekst i kontekst pomagają, gdy nie ma reakcji (YouTube +0,05); obok reakcji prawie nic" if pl else "Text and context help when there are no reactions (YouTube +0.05); next to reactions almost nothing", fontweight="bold")
    plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()
