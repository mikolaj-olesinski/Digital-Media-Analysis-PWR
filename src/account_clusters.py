"""Account types, topic clusters, author recommendation and brand communities (notebooks/models/29_accounts_clusters.ipynb, E66-E69).
E66 account type from the author name (official brand account / brand in the name / news-media / other) and follower tier:
    views per follower, error of the final model, and the label as an extra feature.
E67 topic clusters: k-means on e5 embeddings of all 12,500 posts, described by class-based TF-IDF words; final-model bias per cluster.
E68 author recommendation: does an author's over-performance (views above what the final model expects) persist to their later posts?
E69 brand communities: brands linked by shared authors and by similar text; Louvain; does the bias of similar brands predict a brand's own bias (cold start)?
Author names are used only inside the functions; outputs carry labels and counts, never names or URLs."""
import re
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score
import networkx as nx
import eda_utils as eu, extra_models as em, reaction_tiers as rt, history as H
from validation import SEED

PLATS = ["tiktok", "youtube", "twitter"]
SUFFIX = {"", "official", "us", "usa", "uk", "ph", "au", "australia", "india", "in", "canada", "ca", "global", "jobs", "careers", "news", "hq", "brasil", "mexico", "de", "fr", "pl", "polska",
          "japan", "jp", "sg", "my", "id", "nz", "ie", "za", "tv", "app", "inc", "co", "corp", "group", "support", "help", "team"}
MEDIA = r"news|\btv\b|radio|magazin|gazet|portal|media|times|daily|press|wiadomo|\bfm\b|journal|report|tribune|herald|chronicle|broadcast"
KINDS = ["official brand account", "brand in the name (branch, dealer, fan)", "news / media", "other"]
TIERS = ["followers unknown", "< 1k", "1k-10k", "10k-100k", "100k-1M", "> 1M"]


def _n(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def account_kind(raw):
    """One label per post from the author name and handle compared with the brand name."""
    out = []
    for b, a, u in zip(raw.brand.map(_n), raw.author_name.map(_n), raw.author_url.fillna("").map(lambda s: _n(str(s).rstrip("/").split("/")[-1].lstrip("@")))):
        if len(b) >= 3 and any(x.startswith(b) and x[len(b):] in SUFFIX for x in (a, u) if x):
            out.append(KINDS[0])
        elif len(b) >= 3 and (b in a or b in u):
            out.append(KINDS[1])
        else:
            out.append(None)
    k = pd.Series(out, index=raw.index)
    media = raw.author_name.fillna("").str.contains(MEDIA, case=False, regex=True)
    return k.fillna(pd.Series(np.where(media, KINDS[2], KINDS[3]), index=raw.index))


def tier(followers):
    return pd.cut(followers.fillna(-1), [-2, -0.5, 1e3, 1e4, 1e5, 1e6, 1e12], labels=TIERS, right=False).astype(str)


def oof(f, meta):
    """Out-of-fold final-model predictions for every TikTok / YouTube / X post (5 brand folds per platform)."""
    pred = pd.Series(np.nan, index=f.index)
    for p in PLATS:
        m = (meta.platform == p).values; M = meta[m]
        pred[f.index[m]] = em.cv_predict(rt.FINAL, f[m][em.ENG].fillna(0).reset_index(drop=True), M.y.reset_index(drop=True), M.brand.values)
    return pred


def accounts(raw_all, f, meta, pred):
    """E66 tables: share of each kind on every platform; for posts with views: median views, views per follower, bias and MAE."""
    kind_all = account_kind(raw_all); share = pd.crosstab(raw_all.platform, kind_all, normalize="index")[KINDS]
    raw = raw_all.loc[f.index]; kind = kind_all.loc[f.index]; tr = tier(raw.author_followers)
    d = pd.DataFrame({"platform": meta.platform, "kind": kind, "tier": tr, "y": meta.y, "pred": pred,
                      "vpf": meta.y - np.log10(raw.author_followers.where(raw.author_followers > 0))})
    tabs = []
    for col in ["kind", "tier"]:
        tabs.append(d.groupby(["platform", col]).apply(lambda z: pd.Series({"posts": len(z), "median_views": float(10 ** z.y.median()), "median_views_per_follower": float(10 ** z.vpf.median()),
                                                                           "bias": float((z.pred - z.y).mean()), "MAE": float((z.pred - z.y).abs().mean())})).reset_index())
    return share, tabs[0], tabs[1], d


def kind_gain(f, meta, kind):
    """Final model vs final model + one-hot account kind, same brand folds."""
    out = []
    for p in PLATS:
        m = (meta.platform == p).values; M = meta[m].reset_index(drop=True); X = f[m][em.ENG].fillna(0).reset_index(drop=True)
        K = pd.get_dummies(kind[m].reset_index(drop=True)).reindex(columns=KINDS, fill_value=0).astype(int); K.columns = [f"kind_{i}" for i in range(len(KINDS))]
        for name, Z in [("final model", X), ("final model + account kind", pd.concat([X, K], axis=1))]:
            out.append(pd.DataFrame({"platform": p, "model": name, "brand": M.brand.values, "y": M.y.values, "pred": em.cv_predict(rt.FINAL, Z, M.y, M.brand.values)}))
    return pd.concat(out, ignore_index=True)


def _clean(t):
    return t.str.lower().str.replace(r"https?://\S+|@\w+|\d+", " ", regex=True)


def clusters(E, raw_all, k=16):
    """k-means on L2-normalised embeddings of all posts. Words: class-based TF-IDF (one document per cluster), terms in >= 20 posts."""
    Z = E.loc[raw_all.index].values; Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    c = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
    txt = _clean(raw_all.title.fillna("") + " " + raw_all.content.fillna(""))
    vec = TfidfVectorizer(stop_words="english", min_df=20, max_df=.3, token_pattern=r"(?u)\b[^\W\d_]{3,}\b", sublinear_tf=True)
    vec.fit(txt); docs = pd.Series(txt.values).groupby(c).apply(" ".join)
    W = vec.transform(docs).toarray(); voc = np.array(vec.get_feature_names_out())
    words = {i: ", ".join(voc[np.argsort(-W[j])[:6]]) for j, i in enumerate(docs.index)}
    return pd.Series(c, index=raw_all.index, name="cluster"), words


def cluster_table(cl, words, raw_all, meta, pred, labels):
    """Per cluster: posts, top platforms, brands, promo share, and on posts with views: median views and final-model bias."""
    d = raw_all.assign(cluster=cl, type=labels.reindex(raw_all.mention_id.astype(str)).values)
    t = d.groupby("cluster").agg(posts=("mention_id", "size"), brands=("brand", "nunique"), promo=("type", lambda s: float((s == "promo").mean())), spam=("type", lambda s: float((s == "spam").mean())))
    pm = pd.crosstab(d.cluster, d.platform, normalize="index"); t["main platform"] = pm.idxmax(axis=1).map(eu.NAMES) + pm.max(axis=1).map(lambda v: f" {v:.0%}")
    v = pd.DataFrame({"cluster": cl.loc[meta.index], "y": meta.y, "e": pred - meta.y, "platform": meta.platform})
    t["posts with views"] = v.groupby("cluster").size(); t["median views"] = v.groupby("cluster").y.median().map(lambda s: 10 ** s)
    t["bias (pred - true)"] = v.groupby("cluster").e.mean(); t["words"] = pd.Series(words)
    nmi = {"brand": normalized_mutual_info_score(d.brand, d.cluster), "platform": normalized_mutual_info_score(d.platform, d.cluster), "type": normalized_mutual_info_score(d.type.fillna("na"), d.cluster)}
    return t.reset_index(), nmi, v


def author_persistence(raw, meta, pred, n_boot=1000):
    """E68. Authors with >= 2 posts with views on a platform, posts in time order. Over-performance = true - predicted log views.
    Spearman correlation between the first post's over-performance and the mean of the later ones; and a recommendation test:
    authors in the top third by first-post over-performance vs the bottom third, mean over-performance of their later posts."""
    d = pd.DataFrame({"platform": meta.platform, "author": raw.author_url.fillna(raw.author_name).values, "t": raw.created_at, "r": meta.y - pred, "y": meta.y,
                      "vpf": meta.y - np.log10(raw.author_followers.where(raw.author_followers > 0))})
    rows = []; rng = np.random.default_rng(SEED)
    for p in PLATS:
        q = d[d.platform == p].sort_values("t"); g = q.groupby("author")
        a = pd.DataFrame({"n": g.size(), "first": g.r.first(), "later": g.r.apply(lambda s: s.iloc[1:].mean()), "first_vpf": g.vpf.first(), "later_vpf": g.vpf.apply(lambda s: s.iloc[1:].mean())})
        a = a[a.n >= 2]
        if len(a) < 30: continue
        def stats(b):
            rho = spearmanr(b["first"], b.later)[0]; lo, hi = b["first"].quantile([1 / 3, 2 / 3])
            return rho, b.later[b["first"] >= hi].mean() - b.later[b["first"] <= lo].mean(), spearmanr(b.first_vpf, b.later_vpf, nan_policy="omit")[0]
        s = stats(a); B = np.array([stats(a.iloc[rng.integers(0, len(a), len(a))]) for _ in range(n_boot)])
        lo, hi = np.nanpercentile(B, [2.5, 97.5], axis=0)
        rows.append({"platform": p, "authors": len(a), "later posts": int((a.n - 1).sum()), "rho over-performance": s[0], "rho low": lo[0], "rho high": hi[0],
                     "top third - bottom third (log10)": s[1], "diff low": lo[1], "diff high": hi[1], "top vs bottom (×)": 10 ** s[1],
                     "rho views per follower": s[2], "vpf low": lo[2], "vpf high": hi[2]})
    return pd.DataFrame(rows)


def brand_graph(raw_all, E, k=5):
    """Brands as nodes; edge weight = shared authors (any platform) + cosine similarity of mean text embedding for each brand's k nearest brands."""
    G = nx.Graph(); brands = sorted(raw_all.brand.unique()); G.add_nodes_from(brands)
    au = raw_all.assign(a=raw_all.author_url.fillna(raw_all.author_name)).groupby("a").brand.unique()
    for bs in au:
        bs = sorted(bs)
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                w = G.get_edge_data(bs[i], bs[j], {"shared": 0})["shared"]; G.add_edge(bs[i], bs[j], shared=w + 1)
    Z = E.loc[raw_all.index]; C = Z.groupby(raw_all.brand.values).mean().loc[brands].values; C = C / np.linalg.norm(C, axis=1, keepdims=True); S = C @ C.T; np.fill_diagonal(S, -1)
    for i, b in enumerate(brands):
        for j in np.argsort(-S[i])[:k]:
            e = G.get_edge_data(b, brands[j], {"shared": 0}); G.add_edge(b, brands[j], shared=e["shared"], text=float(S[i, j]))
    for u, v, e in G.edges(data=True): e["weight"] = e.get("shared", 0) + e.get("text", 0)
    return G, pd.DataFrame(S, index=brands, columns=brands)


def communities(G, raw_all, words_of):
    com = nx.community.louvain_communities(G, weight="weight", seed=SEED, resolution=1.0)
    lab = {b: i for i, c in enumerate(sorted(com, key=len, reverse=True)) for b in c}
    cb = raw_all.brand.map(lab)
    txt = _clean(raw_all.title.fillna("") + " " + raw_all.content.fillna("")); vec = TfidfVectorizer(stop_words="english", min_df=20, max_df=.3, token_pattern=r"(?u)\b[^\W\d_]{3,}\b", sublinear_tf=True).fit(txt)
    docs = pd.Series(txt.values).groupby(cb.values).apply(" ".join); W = vec.transform(docs).toarray(); voc = np.array(vec.get_feature_names_out())
    t = pd.DataFrame({"brands": pd.Series(lab).value_counts().sort_index(), "posts": cb.value_counts().sort_index(),
                      "example brands": pd.Series({i: ", ".join(sorted(b for b, j in lab.items() if j == i)[:6]) for i in set(lab.values())}),
                      "words": pd.Series({i: ", ".join(voc[np.argsort(-W[n])[:6]]) for n, i in enumerate(docs.index)})})
    shared = sum(1 for *_, e in G.edges(data=True) if e.get("shared", 0) > 0)
    return lab, t.reset_index(names="community"), {"edges": G.number_of_edges(), "edges with shared authors": shared, "communities": len(com),
                                                    "modularity": nx.community.modularity(G, com, weight="weight")}


def cold_start(S, meta, pred, k=5, n_boot=1000):
    """E69 recommendation test. Per platform, each brand's mean over-performance (true - predicted); correlate it with the mean
    over-performance of its k most similar brands by text (similar brands measured without views)."""
    rows = []; rng = np.random.default_rng(SEED)
    for p in PLATS:
        m = meta.platform == p; r = (meta.y - pred)[m].groupby(meta.brand[m]).agg(["mean", "size"]); r = r[r["size"] >= 5]["mean"]
        b = r.index.tolist(); s = S.loc[b, b].values.copy(); np.fill_diagonal(s, -1)
        nb = np.array([r.values[np.argsort(-s[i])[:k]].mean() for i in range(len(b))])
        rho = spearmanr(r.values, nb)[0]; B = [spearmanr(r.values[i], nb[i])[0] for i in (rng.integers(0, len(b), len(b)) for _ in range(n_boot))]
        rows.append({"platform": p, "brands (>= 5 posts)": len(b), "rho with similar brands": rho, "low": np.percentile(B, 2.5), "high": np.percentile(B, 97.5),
                     "sd of brand over-performance": float(r.std())})
    return pd.DataFrame(rows)


def plot_kinds(K, path=None, lang="en"):
    import matplotlib.pyplot as plt
    pl = lang == "pl"
    nm = {KINDS[0]: "oficjalne konto marki" if pl else "official brand account", KINDS[1]: "marka w nazwie (oddział, dealer, fan)" if pl else "brand in the name (branch, dealer, fan)",
          KINDS[2]: "media / news" if pl else "news / media", KINDS[3]: "pozostali" if pl else "other"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    for j, (col, lab) in enumerate([("median_views_per_follower", "mediana wyświetleń na obserwującego" if pl else "median views per follower"), ("bias", "błąd systematyczny modelu (log10, + = zawyża)" if pl else "model bias (log10, + = overestimates)")]):
        x = ax[j]; w = .25
        for i, p in enumerate(PLATS):
            q = K[(K.platform == p)].set_index("kind").reindex(KINDS)
            x.barh(np.arange(len(KINDS)) + (i - 1) * w, q[col], w, color=eu.PCOL[p], label=eu.NAMES[p])
        x.set_yticks(range(len(KINDS))); x.set_yticklabels([nm[k] for k in KINDS] if j == 0 else []); x.invert_yaxis(); x.set_xlabel(lab)
        if col == "median_views_per_follower": x.set_xscale("log")
        else: x.axvline(0, color="grey", lw=.8)
    ax[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Typ konta: oficjalne konta marek mają mniej wyświetleń na obserwującego, ale więcej, niż sugerują reakcje (mało postów)" if pl else "Account type: official brand accounts get fewer views per follower, yet more than their reactions suggest (few posts)", fontweight="bold")
    plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()


def plot_clusters(T, path=None, lang="en"):
    import matplotlib.pyplot as plt
    pl = lang == "pl"; T = T[T["posts with views"] >= 40].sort_values("bias (pred - true)")
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(T) + 1))
    col = np.where(T["bias (pred - true)"] > 0, "#e4572e", "#2a9d8f")
    ax.barh(range(len(T)), T["bias (pred - true)"], color=col)
    ax.set_yticks(range(len(T))); ax.set_yticklabels([f"{w}  ({n})" for w, n in zip(T.words, T["posts with views"])], fontsize=8); ax.axvline(0, color="grey", lw=.8)
    ax.set_xlabel("średni błąd modelu końcowego (log10; + = zawyża)" if pl else "mean error of the final model (log10; + = overestimates)")
    ax.set_title("Klastry tematów (k-means na embeddingach, ≥ 40 postów z wyświetleniami): błąd modelu końcowego w klastrze" if pl else "Topic clusters (k-means on embeddings, ≥ 40 posts with views): error of the final model per cluster", fontweight="bold", fontsize=10)
    plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()


def _author_frame(raw, meta, pred):
    norm = _clean(raw.title.fillna("") + " " + raw.content.fillna("")).str.split().str.join(" ")
    return pd.DataFrame({"platform": meta.platform, "author": raw.author_url.fillna(raw.author_name).values, "t": raw.created_at, "brand": meta.brand, "text": norm,
                         "url": raw.url.fillna("").str.lower().str.rstrip("/"), "y": meta.y, "pred": pred, "r": meta.y - pred})


def author_robustness(raw, meta, pred, n_boot=1000):
    """Persistence of over-performance after removing later posts that repeat an earlier post of the author (same text or same URL),
    and using only later posts about another brand."""
    d = _author_frame(raw, meta, pred); rows = []; rng = np.random.default_rng(SEED)
    for p in PLATS:
        q = d[d.platform == p].sort_values("t")
        for name in ["all later posts", "later posts with new text and URL", "later posts about another brand"]:
            fr, la = [], []
            for _, z in q.groupby("author"):
                if len(z) < 2: continue
                a = z.iloc[0]; rest = z.iloc[1:]
                if name != "all later posts": rest = rest[(rest.text != a.text) & (rest.url != a.url) & (rest.y != a.y)]
                if name == "later posts about another brand": rest = rest[rest.brand != a.brand]
                if len(rest): fr.append(a.r); la.append(rest.r.mean())
            fr, la = np.array(fr), np.array(la)
            if len(fr) < 20: rows.append({"platform": p, "later posts used": name, "authors": len(fr)}); continue
            B = [spearmanr(fr[i], la[i])[0] for i in (rng.integers(0, len(fr), len(fr)) for _ in range(n_boot))]
            rows.append({"platform": p, "later posts used": name, "authors": len(fr), "rho": spearmanr(fr, la)[0], "low": np.nanpercentile(B, 2.5), "high": np.nanpercentile(B, 97.5)})
    return pd.DataFrame(rows)


def author_history_gain(raw, meta, pred, grid=np.linspace(0, 1, 11)):
    """Use case: predicting a post when the author's earlier posts (and their true views) are known. Correction = mean
    over-performance of the author's earlier posts that are not repeats (other text, URL and view count), times a shrinkage a.
    a is chosen on the other brand folds (grid 0-1). Evaluated on posts that have such an earlier post."""
    d = _author_frame(raw, meta, pred); out = []
    for p in PLATS:
        q = d[d.platform == p].sort_values("t"); prior = pd.Series(np.nan, index=q.index); new = pd.Series(np.nan, index=q.index)
        for _, z in q.groupby("author"):
            for i in range(1, len(z)):
                b = z.iloc[i]; e = z.iloc[:i]; e2 = e[(e.text != b.text) & (e.url != b.url) & (e.y != b.y)]
                if len(e): prior[z.index[i]] = e.r.mean()
                if len(e2): new[z.index[i]] = e2.r.mean()
        for name, pr in [("earlier posts of the author (all)", prior), ("earlier posts of the author (no repeats)", new)]:
            s = q[pr.notna()].assign(c=pr.dropna()); brands = s.brand.unique(); fold = {b: i % 5 for i, b in enumerate(np.random.default_rng(SEED).permutation(brands))}
            k = s.brand.map(fold).values; fit = np.zeros(len(s))
            for j in range(5):
                tr, te = k != j, k == j
                a = grid[np.argmin([np.abs(s.y.values[tr] - s.pred.values[tr] - g * s.c.values[tr]).mean() for g in grid])]; fit[te] = s.pred.values[te] + a * s.c.values[te]
            out.append(pd.DataFrame({"platform": p, "posts": name, "model": "final model", "brand": s.brand.values, "y": s.y.values, "pred": s.pred.values}))
            out.append(pd.DataFrame({"platform": p, "posts": name, "model": "final model + author history", "brand": s.brand.values, "y": s.y.values, "pred": fit}))
    return pd.concat(out, ignore_index=True)
