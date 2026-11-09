"""Model zoo, strata and graph features (notebooks/models/03_zoo_strata_graph.ipynb).
Same target and validation as modeling.py: log10(views), 5-fold CV grouped by brand, so a brand never sits in train and test.
Graph features that use the target (author / text-neighbour means) are computed from the training fold only."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import modeling as mo, eda_utils as eu
from validation import mae_ci

SEED = 42
ENG = mo.GROUPS["followers"] + mo.GROUPS["engagement"] + mo.GROUPS["missing flags"]
ALL = [c for g in mo.GROUPS.values() for c in g]


def load(root):
    """Features, meta (platform, brand, y), raw text and author per mention, on TikTok / YouTube / X."""
    f, meta = mo.load_features(root)
    raw = eu.load(root).astype({"mention_id": str}).loc[meta.index]
    meta = meta.assign(author=raw.author_name.fillna("").values, text=raw.content.fillna(raw.title).fillna("").values)
    return f, meta


# ---------- model zoo ----------
def models():
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "kNN (k=25)": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(25, weights="distance")),
        "random forest": lambda: RandomForestRegressor(300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "extra trees": lambda: ExtraTreesRegressor(300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "gradient boosting": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, random_state=SEED),
        "gradient boosting, median loss": lambda: HistGradientBoostingRegressor(loss="absolute_error", max_iter=200, learning_rate=.05, random_state=SEED),
    }


def cv_predict(make, X, y, g, extra=None):
    """Out-of-fold predictions. `extra(tr, te)` may return (Xtr_extra, Xte_extra) built from the training fold only."""
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        Xtr, Xte = X.iloc[tr].fillna(0).reset_index(drop=True), X.iloc[te].fillna(0).reset_index(drop=True)
        if extra is not None:
            a, b = extra(tr, te); Xtr, Xte = pd.concat([Xtr, a], axis=1), pd.concat([Xte, b], axis=1)
        pred[te] = make().fit(Xtr, y.iloc[tr]).predict(Xte)
    return pred


def zoo(f, meta, plat):
    m = (meta.platform == plat).values; X, M = f[m].reset_index(drop=True), meta[m].reset_index(drop=True); y, g = M.y, M.brand
    rows, preds = [], {}
    for name, mk in models().items():
        p = cv_predict(mk, X[ENG], y, g); preds[name] = p
        mae, lo, hi = mae_ci((y - p).values, g.values); rows.append({"platform": eu.NAMES[plat], "model": name, "MAE": mae, "CI low": lo, "CI high": hi})
    p = np.mean([preds[k] for k in ["ridge", "random forest", "gradient boosting"]], axis=0); preds["blend (ridge+forest+boosting)"] = p
    mae, lo, hi = mae_ci((y - p).values, g.values); rows.append({"platform": eu.NAMES[plat], "model": "blend (ridge+forest+boosting)", "MAE": mae, "CI low": lo, "CI high": hi})
    return pd.DataFrame(rows), preds


# ---------- graph features ----------
def text_similarity(texts):
    """Cosine similarity between mentions of one platform from TF-IDF of the text (the content-similarity graph)."""
    t = [s if s.strip() else "emptytext" for s in texts]
    X = TfidfVectorizer(min_df=2, max_features=30000, sublinear_tf=True, stop_words="english").fit_transform(t)
    S = (X @ X.T).toarray(); np.fill_diagonal(S, 0); return S


def author_ids(M):
    a = M.author.where(M.author.str.len() > 0, None)
    return a.fillna(pd.Series([f"_anon{i}" for i in range(len(M))], index=M.index))


def graph_extra(M, y, S, k=10, m_prior=2.0):
    """Returns extra(tr, te) giving target-based graph features from the training fold:
    author_te  - smoothed mean log10 views of the same author's other mentions (author-mention graph)
    knn_te     - similarity-weighted mean log10 views of the k most similar training mentions (content graph)
    knn_sim    - strength of that neighbourhood (mean similarity of the k neighbours)"""
    a = author_ids(M).values; yv = y.values
    def extra(tr, te):
        gm = yv[tr].mean(); out = []
        for idx, own in [(tr, True), (te, False)]:
            ate = np.empty(len(idx)); cnt = np.empty(len(idx)); kn = np.empty(len(idx)); ks = np.empty(len(idx))
            trset = pd.Series(yv[tr], index=a[tr]); sums = trset.groupby(level=0).sum(); ns = trset.groupby(level=0).count()
            Str = S[np.ix_(idx, tr)]
            for r, i in enumerate(idx):
                s, n = sums.get(a[i], 0.0), ns.get(a[i], 0)
                if own: s, n = s - yv[i], n - 1                                  # leave the row itself out
                ate[r] = (s + m_prior * gm) / (n + m_prior); cnt[r] = n
                row = Str[r].copy()
                if own: row[np.where(tr == i)[0]] = 0
                top = np.argpartition(-row, k)[:k]; w = row[top]
                kn[r] = (w * yv[tr][top]).sum() / w.sum() if w.sum() > 0 else gm; ks[r] = w.mean()
            out.append(pd.DataFrame({"author_te": ate, "author_n": cnt, "knn_te": kn, "knn_sim": ks}))
        return out[0].set_index(pd.Index(range(len(tr)))), out[1].set_index(pd.Index(range(len(te))))
    return extra


def smoothed_features(f, S, cols, k=10, hops=2):
    """Label-free graph smoothing (SGC style): average the engagement features over the content-similarity graph, `hops` times.
    Uses no views, so it may include test mentions."""
    A = np.zeros_like(S); top = np.argpartition(-S, k, axis=1)[:, :k]
    for i, js in enumerate(top): A[i, js] = S[i, js]
    A = A + np.eye(len(A)); A = A / A.sum(1, keepdims=True); H = f[cols].fillna(0).values; out = {}
    for h in range(1, hops + 1):
        H = A @ H
        for j, c in enumerate(cols): out[f"{c}_hop{h}"] = H[:, j]
    return pd.DataFrame(out, index=f.index)


def author_degree(M):
    a = author_ids(M); return pd.DataFrame({"author_mentions": a.map(a.value_counts()).values,
                                            "brand_mentions": M.brand.map(M.brand.value_counts()).values}, index=M.index).apply(np.log10)


def graph_ladder(f, meta, plat, mk=None):
    """MAE of the boosting model with graph features added one family at a time."""
    mk = mk or models()["gradient boosting"]
    m = (meta.platform == plat).values; X, M = f[m].reset_index(drop=True), meta[m].reset_index(drop=True); y, g = M.y, M.brand
    S = text_similarity(M.text.values); ex = graph_extra(M, y, S); deg = author_degree(M); sm = smoothed_features(X, S, ["log_likes", "log_comments", "log_followers"])
    variants = {
        "engagement only": (X[ENG], None),
        "+ degree (author / brand mention counts)": (pd.concat([X[ENG], deg], axis=1), None),
        "+ label-free smoothing over text graph": (pd.concat([X[ENG], sm], axis=1), None),
        "+ author target mean": (X[ENG], lambda tr, te: tuple(d[["author_te", "author_n"]] for d in ex(tr, te))),
        "+ text-neighbour target mean": (X[ENG], lambda tr, te: tuple(d[["knn_te", "knn_sim"]] for d in ex(tr, te))),
        "+ all graph features": (pd.concat([X[ENG], deg, sm], axis=1), ex),
    }
    rows, preds = [], {}
    for name, (Xv, e) in variants.items():
        p = cv_predict(mk, Xv, y, g, e); preds[name] = p
        mae, lo, hi = mae_ci((y - p).values, g.values); rows.append({"platform": eu.NAMES[plat], "model": name, "MAE": mae, "CI low": lo, "CI high": hi})
    return pd.DataFrame(rows), preds


# ---------- strata ----------
def strata_table(f, meta, plat, pred, labels):
    """MAE by stratum for one set of out-of-fold predictions. `labels` has the columns type and sentiment aligned to meta rows."""
    m = (meta.platform == plat).values; X, M = f[m].reset_index(drop=True), meta[m].reset_index(drop=True); lab = labels[m].reset_index(drop=True)
    err = np.abs(M.y - pred); rows = []
    def add(var, s):
        for k, idx in sorted(s.groupby(s).groups.items(), key=lambda kv: str(kv[0])): rows.append({"stratifier": var, "stratum": str(k), "mentions": len(idx), "MAE": err.iloc[list(idx)].mean(), "bias": (pred - M.y).iloc[list(idx)].mean()})
    add("our type label", lab.type.fillna("n/a")); add("our sentiment", lab.sentiment.fillna("n/a"))
    add("followers tier", pd.qcut(X.log_followers.rank(method="first"), 4, labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"]).astype(str).where(X.log_followers.notna(), "no followers"))
    add("engagement tier", pd.qcut((X.log_likes + X.log_comments).rank(method="first"), 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"]).astype(object).fillna("missing likes/comments"))
    add("language", pd.Series(np.where(X.english == 1, "english", "other")))
    return pd.DataFrame(rows).assign(platform=eu.NAMES[plat])


def stratum_models(f, meta, plat, by="followers"):
    """Pooled boosting model vs one model per follower tier (tiers cut on the training fold)."""
    m = (meta.platform == plat).values; X, M = f[m].reset_index(drop=True), meta[m].reset_index(drop=True); y, g = M.y, M.brand
    mk = models()["gradient boosting"]; pooled = cv_predict(mk, X[ENG], y, g); per = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        q = np.nanquantile(X.log_followers.iloc[tr], [1 / 3, 2 / 3]); tier = lambda v: np.where(np.isnan(v), 3, np.digitize(v, q))
        ttr, tte = tier(X.log_followers.iloc[tr].values), tier(X.log_followers.iloc[te].values)
        for t in range(4):
            a, b = tr[ttr == t], te[tte == t]
            if len(b) and len(a) > 30: per[b] = mk().fit(X.iloc[a][ENG].fillna(0), y.iloc[a]).predict(X.iloc[b][ENG].fillna(0))
            elif len(b): per[b] = mk().fit(X.iloc[tr][ENG].fillna(0), y.iloc[tr]).predict(X.iloc[b][ENG].fillna(0))
    rows = []
    for name, p in [("one pooled model", pooled), ("one model per follower tier", per)]:
        mae, lo, hi = mae_ci((y - p).values, g.values); rows.append({"platform": eu.NAMES[plat], "model": name, "MAE": mae, "CI low": lo, "CI high": hi})
    return pd.DataFrame(rows)


def plot_bars(res, title, order=None):
    plats = [p for p in mo.PLATS if eu.NAMES[p] in set(res.platform)]
    fig, ax = plt.subplots(1, len(plats), figsize=(5.4 * len(plats), 3.8))
    for a, p in zip(np.atleast_1d(ax), plats):
        r = res[res.platform == eu.NAMES[p]].iloc[::-1]
        a.barh(r.model, r.MAE, xerr=[r.MAE - r["CI low"], r["CI high"] - r.MAE], color=eu.PCOL[p], ecolor="black", capsize=2)
        a.set_xlabel("MAE of log10(views), brand-grouped CV"); a.set_title(f"{eu.NAMES[p]}: {title}", fontweight="bold", fontsize=10)
        if p != plats[0]: a.set_yticklabels([])
    plt.tight_layout(); plt.show()


def plot_strata(st):
    """MAE by stratum, one panel per stratifier, one bar colour per platform; grey line = MAE over all mentions of that platform."""
    vars_ = st.stratifier.unique(); fig, ax = plt.subplots(1, len(vars_), figsize=(3.6 * len(vars_), 4.2), sharey=True)
    for a, v in zip(ax, vars_):
        s = st[st.stratifier == v]; strata = list(dict.fromkeys(s.stratum)); w = .8 / 3
        for j, p in enumerate(mo.PLATS):
            r = s[s.platform == eu.NAMES[p]].set_index("stratum").reindex(strata)
            a.bar(np.arange(len(strata)) + (j - 1) * w, r.MAE, w, color=eu.PCOL[p], label=eu.NAMES[p])
        a.set_xticks(range(len(strata))); a.set_xticklabels(strata, rotation=35, ha="right", fontsize=8); a.set_title(v, fontweight="bold", fontsize=10)
    ax[0].set_ylabel("MAE of log10(views)"); ax[len(ax) // 2].set_xlabel("stratum")
    ax[-1].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, title="platform"); plt.tight_layout(); plt.show()


def summary_figure(zoo_res, graph_res, med, brand24_x, path):
    """One labelled figure: best simple baseline, best model, best graph variant per platform, and Brand24 reach on X."""
    fig, ax = plt.subplots(figsize=(9, 4.4)); labels = ["median views", "best model (engagement)", "best graph variant"]; w = .25
    for j, p in enumerate(mo.PLATS):
        n = eu.NAMES[p]; z = zoo_res[zoo_res.platform == n]; g = graph_res[graph_res.platform == n]
        vals = [med[p], z.MAE.min(), g.MAE.min()]
        ax.bar(np.arange(3) + (j - 1) * w, vals, w, color=eu.PCOL[p], label=n)
        for i, v in enumerate(vals): ax.text(i + (j - 1) * w, v + .01, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(brand24_x, color="grey", ls="--", lw=1); ax.text(2.45, brand24_x + .012, f"Brand24 reach on X: {brand24_x:.2f}", ha="right", fontsize=8, color="dimgrey")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.set_ylabel("MAE of log10(views), brand-grouped CV"); ax.set_xlabel("modelling step")
    ax.set_title("More models and graph features do not beat engagement alone", fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, title="platform"); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.show()


TYPES = ["organic", "promo", "spam"]
TXT = {"en": {"types": ["organic", "promo", "spam"], "p1": "Typical views by our type label", "y1": "median log10(views)", "p2": "Views vs what engagement predicts", "y2": "mean log10(views) minus prediction\n(positive = more views than likes/comments suggest)", "x": "platform", "leg": "our type label", "n": "n"},
       "pl": {"types": ["organiczne", "promocyjne", "spam"], "p1": "Typowe wyświetlenia wg naszej etykiety", "y1": "mediana log10(wyświetleń)", "p2": "Wyświetlenia vs to, co przewidują reakcje", "y2": "średnia log10(wyświetleń) minus predykcja\n(dodatnie = więcej niż wynika z polubień/komentarzy)", "x": "platforma", "leg": "nasza etykieta typu", "n": "n"}}


def _boot(vals, brands, stat, B=400, seed=SEED):
    rng = np.random.default_rng(seed); ub = np.unique(brands); groups = {b: vals[brands == b] for b in ub}
    out = [stat(np.concatenate([groups[b] for b in rng.choice(ub, len(ub))])) for _ in range(B)]
    return stat(vals), np.percentile(out, 2.5), np.percentile(out, 97.5)


def labels_summary(meta, labels, pred_by_plat):
    """Per platform and type label: median log10 views and mean residual (views minus out-of-fold prediction) with brand-cluster bootstrap CI."""
    rows = []
    for p in mo.PLATS:
        m = (meta.platform == p).values; M = meta[m].reset_index(drop=True); lab = labels[m].reset_index(drop=True); res = (M.y - pred_by_plat[p]).values
        for t in TYPES:
            k = (lab.type == t).values; b = M.brand.values[k]
            med = _boot(M.y.values[k], b, np.median); mres = _boot(res[k], b, np.mean)
            rows.append({"platform": eu.NAMES[p], "type": t, "n": int(k.sum()), "median": med[0], "median lo": med[1], "median hi": med[2], "resid": mres[0], "resid lo": mres[1], "resid hi": mres[2]})
    return pd.DataFrame(rows)


def labels_figure(ls, lang, path):
    T = TXT[lang]; cols = ["#4c9f70", "#e0a030", "#b5403a"]; fig, ax = plt.subplots(1, 2, figsize=(12, 4.4)); w = .26
    for a, (c, title, yl) in zip(ax, [("median", T["p1"], T["y1"]), ("resid", T["p2"], T["y2"])]):
        for j, t in enumerate(TYPES):
            r = ls[ls.type == t].set_index("platform").reindex([eu.NAMES[p] for p in mo.PLATS]); x = np.arange(3) + (j - 1) * w
            a.bar(x, r[c], w, color=cols[j], label=T["types"][j], yerr=[r[c] - r[f"{c} lo"], r[f"{c} hi"] - r[c]], ecolor="black", capsize=2)
        a.set_xticks(range(3)); a.set_xticklabels([f"{eu.NAMES[p]}\n(n={int(ls[ls.platform == eu.NAMES[p]].n.sum())})" for p in mo.PLATS]); a.set_ylabel(yl); a.set_xlabel(T["x"]); a.set_title(title, fontweight="bold")
    fig.subplots_adjust(wspace=.4); ax[1].axhline(0, color="grey", lw=1); ax[1].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, title=T["leg"])
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.show()


def summary_figure_pl(root, path):
    """Polish twin of zoo_summary_en.png, drawn from the stored numbers (zoo_numbers.json, results_numbers.json) so notebook 03 need not be re-run."""
    import json
    z = json.load(open(root / "data" / "processed" / "zoo_numbers.json"))["summary"]
    b24 = json.load(open(root / "data" / "processed" / "results_numbers.json"))["ladder"]["twitter"]["Brand24 reach as-is"][0]
    fig, ax = plt.subplots(figsize=(9, 4.4)); labels = ["mediana wyświetleń", "najlepszy model (reakcje)", "najlepszy wariant grafowy"]; w = .25
    for j, p in enumerate(mo.PLATS):
        vals = [z[p]["median"], z[p]["zoo_best"], z[p]["graph_best"]]
        ax.bar(np.arange(3) + (j - 1) * w, vals, w, color=eu.PCOL[p], label=eu.NAMES[p])
        for i, v in enumerate(vals): ax.text(i + (j - 1) * w, v + .01, f"{v:.2f}".replace(".", ","), ha="center", fontsize=8)
    ax.axhline(b24, color="grey", ls="--", lw=1); ax.text(2.45, b24 + .012, f"zasięg Brand24 na X: {b24:.2f}".replace(".", ","), ha="right", fontsize=8, color="dimgrey")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.set_ylabel("MAE log10(wyświetleń), CV po markach"); ax.set_xlabel("krok modelowania")
    ax.set_title("Więcej modeli i cechy grafowe nie biją samych reakcji", fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, title="platforma"); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
