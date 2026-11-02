"""Graph lab (notebooks/models/05_graph_lab.ipynb): communities, centrality, spectral embeddings, label propagation, hashtag and time graphs, category encodings.
Label-free graph features may use all mentions; every feature built from views is computed from the training fold only (inner out-of-fold residuals)."""
import re
import numpy as np, pandas as pd, networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
import extra_models as em, new_ideas as ni, eda_utils as eu
from validation import mae_ci

SEED = 42; TAG = re.compile(r"#\w+")


# ---------- graphs (label-free) ----------
def knn_graph(S, k=10):
    """Symmetric weighted kNN graph from a similarity matrix; returns adjacency (dense) and a row-normalised copy."""
    A = np.zeros_like(S); top = np.argpartition(-S, k, axis=1)[:, :k]
    for i, js in enumerate(top): A[i, js] = S[i, js]
    A = np.maximum(A, A.T); W = A / np.maximum(A.sum(1, keepdims=True), 1e-9); return A, W


def louvain(A, seed=SEED, resolution=1.0):
    G = nx.from_numpy_array(A); comms = nx.community.louvain_communities(G, weight="weight", seed=seed, resolution=resolution)
    lab = np.zeros(len(A), int)
    for c, nodes in enumerate(comms): lab[list(nodes)] = c
    return lab, G


def structure_features(S, A, G, lab):
    """Local structure of each mention in the text-similarity graph."""
    n = len(A); size = pd.Series(lab).map(pd.Series(lab).value_counts()).values
    pr = nx.pagerank(G, weight="weight"); cl = nx.clustering(G, weight="weight")
    return pd.DataFrame({"comm_size": np.log10(size), "strength": A.sum(1), "pagerank": [pr[i] * n for i in range(n)], "clustering": [cl[i] for i in range(n)],
                         "near_dups": (S > .6).sum(1), "max_sim": S.max(1)})


def spectral(A, dims=8):
    """Leading non-trivial eigenvectors of the normalised adjacency (Laplacian eigenmaps): a node embedding of the text graph."""
    d = np.maximum(A.sum(1), 1e-9); N = A / np.sqrt(d[:, None] * d[None, :]); vals, vecs = eigsh(csr_matrix(N), k=dims + 1, which="LA")
    return pd.DataFrame(vecs[:, np.argsort(-vals)[1:]], columns=[f"spec{i}" for i in range(dims)])


def author_graph_features(M):
    """Bipartite author-brand graph: how many mentions and brands an author spans, and the degree of the brand in the brand projection."""
    a = em.author_ids(M); B = nx.Graph(); B.add_edges_from(zip("a_" + a, "b_" + M.brand.astype(str)))
    authors_brands = {x: len(B[x]) for x in B if x.startswith("a_")}; proj = nx.bipartite.weighted_projected_graph(B, [x for x in B if x.startswith("b_")])
    bdeg = {x: proj.degree(x, weight="weight") for x in proj}
    return pd.DataFrame({"author_brands": ("a_" + a).map(authors_brands).values, "author_mentions": a.map(a.value_counts()).values,
                         "brand_proj_degree": ("b_" + M.brand.astype(str)).map(bdeg).fillna(0).values}).apply(np.log1p)


# ---------- smoothed target encoding of a residual ----------
def smooth_te(keys_tr, res_tr, keys_te, own_tr=True, m=5.0):
    """Smoothed mean residual per key from the training fold. For training rows the row itself is left out."""
    s = pd.Series(res_tr).groupby(np.asarray(keys_tr)).agg(["sum", "count"]); g = res_tr.mean()
    def f(keys, own):
        sm = pd.Series(keys).map(s["sum"]).fillna(0).values; ct = pd.Series(keys).map(s["count"]).fillna(0).values
        if own: sm, ct = sm - res_tr, ct - 1
        return (sm + m * g) / (ct + m), ct
    return f(np.asarray(keys_tr), own_tr), f(np.asarray(keys_te), False)


def inner_residuals(X, y, tr):
    """Out-of-fold residual of a small boosting model inside the training fold, so encoded residuals are not memorised."""
    res = np.zeros(len(tr)); Xt = X.iloc[tr].fillna(0).values; yt = y.iloc[tr].values
    for a, b in KFold(5, shuffle=True, random_state=SEED).split(Xt):
        res[b] = yt[b] - HistGradientBoostingRegressor(max_iter=100, random_state=SEED).fit(Xt[a], yt[a]).predict(Xt[b])
    return res


# ---------- fold-wise target-based features ----------
def target_features(M, X, y, W, comm, bcomm, hashtags, days, cats, S):
    """Returns extra(tr, te) -> (frame_tr, frame_te) with several graph and category encodings of the residual."""
    brands = M.brand.values
    def extra(tr, te):
        res = inner_residuals(X, y, tr); out_tr, out_te = {}, {}
        def put(name, a, b): out_tr[name], out_te[name] = np.asarray(a), np.asarray(b)
        (a, _), (b, _) = te_(comm, tr, te, res); put("comm_te", a, b)
        (a, _), (b, _) = te_(bcomm, tr, te, res, brand_level=brands); put("brand_comm_te", a, b)
        for c, keys in cats.items():
            (a, _), (b, _) = te_(keys, tr, te, res); put(f"{c}_te", a, b)
        # hashtag graph: mean residual of training mentions sharing a hashtag (hashtags seen at least twice)
        tags = pd.Series(hashtags); s = {}
        for i, r in zip(tr, res):
            for t in tags.iloc[i]: s.setdefault(t, []).append(r)
        def hv(rows, own):
            o = []
            for i in rows:
                v = [np.mean(s[t]) if not own else (np.sum(s[t]) - res[np.where(tr == i)[0][0]]) / max(len(s[t]) - 1, 1) for t in tags.iloc[i] if t in s and len(s[t]) > (2 if own else 1)]
                o.append(np.mean(v) if v else 0.0)
            return o
        put("hashtag_te", hv(tr, True), hv(te, False))
        # time graph: mentions within +-3 days on the same platform
        D = np.abs(days[:, None] - days[tr][None, :]) <= 3
        for name, rows, own in [("date_te", tr, True), ("date_te", te, False)]:
            Dm = D[rows].astype(float);
            if own: Dm[np.arange(len(rows)), np.arange(len(rows))] = 0
            v = (Dm @ res) / np.maximum(Dm.sum(1), 1); (out_tr if own else out_te)[name] = np.where(Dm.sum(1) > 0, v, 0.0)
        # propagation: 1 hop over the text graph from training residuals, then a second hop
        R = np.zeros(len(M)); R[tr] = res; msk = np.zeros(len(M)); msk[tr] = 1; num = W * msk[None, :]; h1 = (num @ R) / np.maximum(num.sum(1), 1e-9)
        h1f = np.where(num.sum(1) > 0, h1, res.mean()); h2 = W @ h1f
        put("prop_hop1", h1f[tr], h1f[te]); put("prop_hop2", h2[tr], h2[te])
        # near-duplicate graph: weighted mean residual of training mentions with cosine > 0.8 (copy-paste campaigns), and whether any exist
        Dp = (S[:, tr] > .8).astype(float)
        for rows, own, d in [(tr, True, out_tr), (te, False, out_te)]:
            Dm = Dp[rows].copy()
            if own: Dm[np.arange(len(rows)), np.arange(len(rows))] = 0
            d["dup_te"] = np.where(Dm.sum(1) > 0, (Dm @ res) / np.maximum(Dm.sum(1), 1), 0.0); d["dup_has"] = (Dm.sum(1) > 0).astype(float)
        return pd.DataFrame(out_tr), pd.DataFrame(out_te)
    return extra


def te_(keys, tr, te, res, brand_level=None):
    keys = np.asarray(keys)
    if brand_level is None: return smooth_te(keys[tr], res, keys[te])
    # community of the brand; leave the row's own brand out by using brand-level means
    df = pd.DataFrame({"b": brand_level[tr], "k": keys[tr], "r": res}).groupby("b").agg(k=("k", "first"), r=("r", "mean"), n=("r", "size")).reset_index()
    g = res.mean(); out = []
    for rows in (tr, te):
        v = []
        for b, k in zip(brand_level[rows], keys[rows]):
            d = df[(df.k == k) & (df.b != b)]; v.append((d.r * d.n).sum() / (d.n.sum() + 5) + 5 * g / (d.n.sum() + 5) if len(d) else g)
        out.append((np.asarray(v), None))
    return out[0], out[1]


# ---------- assembling and evaluating ----------
def prepare(f, meta, raw, plat):
    m = (meta.platform == plat).values; M = meta[m].reset_index(drop=True); X = f[m][em.ENG].reset_index(drop=True); r = raw.loc[f.index][m].reset_index(drop=True)
    text = (r.title.fillna("") + " " + r.content.fillna("")); S = em.text_similarity(text.values); A, W = knn_graph(S); comm, G = louvain(A)
    bS = ni.brand_similarity(M, text.values); bA = bS.values.copy(); bl, _ = louvain(np.maximum(bA, bA.T) * (bA > np.sort(bA, 1)[:, [-5]]), resolution=1.0)
    bcomm = pd.Series(M.brand.map(dict(zip(bS.index, bl))).values)
    ts = r.created_at; cats = {"language": r.language.fillna("na").values, "country": r.country.fillna("na").values, "hour": ts.dt.hour.values, "weekday": ts.dt.weekday.values, "month": (ts.dt.year * 12 + ts.dt.month).values}
    free = {"structure": structure_features(S, A, G, comm), "spectral": spectral(A), "author graph": author_graph_features(M)}
    return dict(M=M, X=X, S=S, A=A, W=W, comm=comm, bcomm=bcomm.values, hashtags=text.map(lambda s: [t.lower() for t in TAG.findall(s)]).values, days=((ts - ts.min()).dt.days).values.astype(float), cats=cats, free=free, text=text)


def variants(P):
    X, M = P["X"], P["M"]; y = M.y; ex = target_features(M, X, y, P["W"], P["comm"], P["bcomm"], P["hashtags"], P["days"], P["cats"], P["S"])
    def pick(*cols): return lambda tr, te: tuple(d[list(cols)] for d in ex(tr, te))
    fr = P["free"]; allfree = pd.concat(fr.values(), axis=1)
    V = {"base: followers + engagement": (X, None)}
    for k, v in fr.items(): V[f"+ {k}"] = (pd.concat([X, v], axis=1), None)
    V["+ all label-free graph features"] = (pd.concat([X, allfree], axis=1), None)
    for name, cols in [("+ community mean residual", ["comm_te"]), ("+ brand-community mean residual", ["brand_comm_te"]), ("+ hashtag graph", ["hashtag_te"]), ("+ time graph (+-3 days)", ["date_te"]),
                       ("+ near-duplicate graph (cosine > 0.8)", ["dup_te", "dup_has"]), ("+ label propagation (2 hops)", ["prop_hop1", "prop_hop2"]), ("+ category encodings (language, country, hour, weekday, month)", [f"{c}_te" for c in P["cats"]])]:
        V[name] = (X, pick(*cols))
    V["+ every target-based encoding"] = (X, ex)
    V["+ everything"] = (pd.concat([X, allfree], axis=1), ex)
    return V


def evaluate(P, plat, make=None):
    make = make or (lambda: em.ExtraTreesRegressor(200, min_samples_leaf=5, n_jobs=-1, random_state=SEED)); M = P["M"]; y, g = M.y, M.brand; V = variants(P); rows = []; pb = None
    for name, (Xv, e) in V.items():
        p = em.cv_predict(make, Xv, y, g, e); pb = p if pb is None else pb
        mae, lo, hi = mae_ci((y - p).values, g.values); gn = ni.paired_diff(y.values, pb, p, g.values)
        rows.append({"platform": eu.NAMES[plat], "model": name, "MAE": mae, "CI low": lo, "CI high": hi, "gain vs base": gn[0], "gain low": gn[1], "gain high": gn[2]})
    return pd.DataFrame(rows)


# ---------- part B: what does the graph say about the mentions themselves ----------
def labels_for(meta, root):
    lab = pd.read_csv(root / "data" / "processed" / "mention_labels.csv", dtype={"mention_id": str}).set_index("mention_id")
    ids = eu.load(root).astype({"mention_id": str}).loc[meta.index].mention_id; return lab.reindex(ids).reset_index(drop=True)


def homophily(P, lab, plat):
    """Share of kNN edges joining mentions with the same type label, against the share expected if labels were shuffled."""
    A = P["A"]; t = lab.type.fillna("na").values; i, j = np.nonzero(np.triu(A)); same = (t[i] == t[j]).mean(); p = pd.Series(t).value_counts(normalize=True).values
    G = nx.from_numpy_array(A); nx.set_node_attributes(G, dict(enumerate(t)), "t")
    return {"platform": eu.NAMES[plat], "same-type edges": same, "expected by chance": float((p ** 2).sum()), "assortativity": nx.attribute_assortativity_coefficient(G, "t")}


def community_profile(P, lab, plat, min_size=25, n_terms=5):
    """Louvain communities of the text graph: size, top terms, type mix, median views."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    t = P["text"].where(P["text"].str.strip() != "", "emptytext"); v = TfidfVectorizer(min_df=3, max_features=20000, sublinear_tf=True, stop_words="english"); X = v.fit_transform(t); terms = np.array(v.get_feature_names_out()); rows = []
    for c in pd.Series(P["comm"]).value_counts().index:
        idx = np.where(P["comm"] == c)[0]
        if len(idx) < min_size: continue
        top = terms[np.asarray(X[idx].mean(0)).ravel().argsort()[::-1][:n_terms]]; mix = lab.type.iloc[idx].value_counts(normalize=True)
        rows.append({"platform": eu.NAMES[plat], "community": int(c), "size": len(idx), "top terms": ", ".join(top), "organic": mix.get("organic", 0), "promo": mix.get("promo", 0), "spam": mix.get("spam", 0),
                     "median log10 views": float(P["M"].y.iloc[idx].median()), "brands": int(P["M"].brand.iloc[idx].nunique())})
    return pd.DataFrame(rows)


def duplicates(P, lab, plat, thr=.8):
    """Near-duplicate mentions (cosine > thr): how common, how spammy, and do duplicates get similar views?"""
    S = P["S"]; has = (S > thr).any(1); y = P["M"].y.values; i, j = np.nonzero(np.triu(S > thr)); rng = np.random.default_rng(SEED); r = rng.integers(0, len(y), (len(i), 2))
    return {"platform": eu.NAMES[plat], "share with near-duplicate": has.mean(), "spam share, duplicates": (lab.type[has] == "spam").mean(), "spam share, others": (lab.type[~has] == "spam").mean(),
            "pairs": len(i), "corr of log views, duplicate pairs": np.corrcoef(y[i], y[j])[0, 1] if len(i) > 5 else np.nan, "corr of log views, random pairs": np.corrcoef(y[r[:, 0]], y[r[:, 1]])[0, 1]}


def predict_type(P, lab, plat, k=10):
    """Predict the type label (organic / promo / spam): TF-IDF text only vs text + vote of the most similar TRAINING mentions (graph), brand-grouped CV, macro F1."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold
    from scipy.sparse import hstack, csr_matrix
    keep = lab.type.isin(["organic", "promo", "spam"]).values; idx = np.where(keep)[0]; t = P["text"].values[idx]; y = lab.type.to_numpy(dtype=object)[idx]; g = P["M"].brand.values[idx]; S = P["S"][np.ix_(idx, idx)]
    classes = np.array(["organic", "promo", "spam"]); res = {"text only": np.empty(len(y), object), "text + graph vote": np.empty(len(y), object), "graph vote only": np.empty(len(y), object)}
    for tr, te in GroupKFold(5).split(t, y, g):
        v = TfidfVectorizer(min_df=2, sublinear_tf=True, max_features=30000).fit(t[tr]); Xtr, Xte = v.transform(t[tr]), v.transform(t[te])
        def vote(rows, own):
            Sm = S[np.ix_(rows, tr)].copy()
            if own: Sm[np.arange(len(rows)), np.arange(len(rows))] = 0
            top = np.argpartition(-Sm, k, axis=1)[:, :k]; out = np.zeros((len(rows), 3))
            w = np.take_along_axis(Sm, top, 1); nb = y[tr][top]
            for c, cl in enumerate(classes): out[:, c] = (w * (nb == cl)).sum(1) / np.maximum(w.sum(1), 1e-9)
            return out
        vtr, vte = vote(tr, True), vote(te, False)
        res["text only"][te] = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, y[tr]).predict(Xte)
        res["text + graph vote"][te] = LogisticRegression(max_iter=1000, class_weight="balanced").fit(hstack([Xtr, csr_matrix(vtr * 3)]), y[tr]).predict(hstack([Xte, csr_matrix(vte * 3)]))
        res["graph vote only"][te] = classes[vte.argmax(1)]
    return pd.DataFrame([{"platform": eu.NAMES[plat], "model": k_, "macro F1": f1_score(y, p, average="macro"), "accuracy": (y == p).mean(), "mentions": len(y)} for k_, p in res.items()])


# ---------- plots ----------
def plot_variants(res, path=None, lang="en"):
    import matplotlib.pyplot as plt
    L = res[~res.model.str.startswith("base")]; names = list(dict.fromkeys(L.model)); fig, ax = plt.subplots(figsize=(11, 6)); w = .26
    for j, p in enumerate(em.mo.PLATS):
        r = L[L.platform == eu.NAMES[p]].set_index("model").reindex(names)
        ax.barh(np.arange(len(names)) + (1 - j) * w, r["gain vs base"], w, color=eu.PCOL[p], label=eu.NAMES[p], xerr=[r["gain vs base"] - r["gain low"], r["gain high"] - r["gain vs base"]], ecolor="black", capsize=2)
    ax.axvline(0, color="grey", lw=1); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("MAE gain over followers + engagement, log10 views (positive = better)" if lang == "en" else "zysk MAE względem followersi + reakcji, log10 wyświetleń (dodatni = lepiej)")
    ax.set_ylabel("graph variant added" if lang == "en" else "dodany wariant grafowy"); ax.set_title("Graph lab: what each graph idea adds" if lang == "en" else "Laboratorium grafowe: co dodaje każdy pomysł", fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, title="platform" if lang == "en" else "platforma"); plt.tight_layout()
    if path: fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_map(Ps, labs, path=None, lang="en"):
    """2D t-SNE map of each platform's mentions (30 TF-IDF topics, cosine), coloured by our type label: where do promo and spam sit?"""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    cols = {"organic": "#4c9f70", "promo": "#e0a030", "spam": "#b5403a", "unclear": "#b8b8c4"}; fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for a, p in zip(ax, em.mo.PLATS):
        t = Ps[p]["text"].where(Ps[p]["text"].str.strip() != "", "emptytext"); Z = TruncatedSVD(30, random_state=SEED).fit_transform(TfidfVectorizer(min_df=3, max_features=30000, sublinear_tf=True).fit_transform(t))
        XY = TSNE(2, metric="cosine", init="pca", perplexity=30, random_state=SEED).fit_transform(Z); l = labs[p].type.fillna("unclear").values
        for k in ["unclear", "organic", "promo", "spam"]: m = l == k; a.scatter(XY[m, 0], XY[m, 1], s=5, alpha=.55, color=cols[k], label=k)
        a.set_title(eu.NAMES[p], fontweight="bold"); a.set_xlabel("t-SNE dimension 1" if lang == "en" else "wymiar t-SNE 1"); a.set_ylabel("t-SNE dimension 2" if lang == "en" else "wymiar t-SNE 2")
    ax[-1].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, title="our type label" if lang == "en" else "nasza etykieta typu", markerscale=3); plt.tight_layout()
    if path: fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
