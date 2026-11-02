"""Ablation with our labels, column importance, systematic bias and communities (notebooks/models/09_ablation_labels_bias.ipynb)."""
import numpy as np, pandas as pd, networkx as nx
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import GroupKFold
import eda_utils as eu, extra_features as ef, extra_models as em, graph_lab as gl, modeling as mo, new_ideas as ni
from validation import mae_ci

SEED = 42
ET = lambda: ExtraTreesRegressor(200, min_samples_leaf=5, n_jobs=-1, random_state=SEED)
HGB = lambda: HistGradientBoostingRegressor(random_state=SEED)


def label_frames(root, index):
    """One-hot type and sentiment from our AI instrument, and Brand24's own sentiment (42% missing, so it comes with a flag)."""
    raw = eu.load(root).astype({"mention_id": str}).loc[index]; lab = pd.read_csv(root / "data" / "processed" / "mention_labels.csv", dtype={"mention_id": str}).set_index("mention_id").reindex(raw.mention_id).set_index(raw.index)
    t = pd.get_dummies(lab.type).astype(int).add_prefix("type_"); s = pd.get_dummies(lab.sentiment).astype(int).add_prefix("sent_")
    b = raw.sentiment.astype(str).str.lower(); b24 = pd.DataFrame({"b24_sent": b.map({"negative": -1, "neutral": 0, "positive": 1}).fillna(0), "b24_sent_na": raw.sentiment.isna().astype(int)}, index=raw.index)
    return {"our type label": t, "our sentiment label": s, "Brand24 sentiment": b24}, lab


def groups(f, meta, G, L):
    """All columns grouped for ablation. Ratios are left out because they would leak likes and followers back in after those groups are removed."""
    return {"likes": f[["log_likes", "na_likes"]], "comments": f[["log_comments", "na_comments"]], "shares": f[["log_shares", "na_shares"]], "followers": f[["log_followers", "na_followers"]],
            **L, "text stats": G["text stats"], "TF-IDF topics": G["TF-IDF SVD (30)"], "VADER sentiment": G["VADER sentiment"], "calendar": G["calendar"], "brand size": G["brand size"]}


def three_views(P, meta, plat, make=ET):
    """For each group: error alone, gain when added to reactions + followers, and loss when removed from the full model. All paired on the same folds."""
    m = (meta.platform == plat).values; M = meta[m].reset_index(drop=True); y, g = M.y, M.brand; Q = {k: v[m].reset_index(drop=True) for k, v in P.items()}; cv = lambda X: em.cv_predict(make, X, y, g)
    base_keys = ["likes", "comments", "shares", "followers"]; base = pd.concat([Q[k] for k in base_keys], axis=1); pb, pf = cv(base), cv(pd.concat(Q.values(), axis=1)); rows = []
    for k, v in Q.items():
        alone = cv(v); removed = cv(pd.concat([x for kk, x in Q.items() if kk != k], axis=1)); added = pb if k in base_keys else cv(pd.concat([base, v], axis=1))
        a, l = ni.paired_diff(y.values, pb, added, g.values), ni.paired_diff(y.values, removed, pf, g.values)
        rows.append({"platform": eu.NAMES[plat], "group": k, "MAE alone": np.abs(y - alone).mean(), "add gain": a[0], "add low": a[1], "add high": a[2], "remove loss": l[0], "remove low": l[1], "remove high": l[2]})
    return pd.DataFrame(rows).assign(**{"MAE base": np.abs(y - pb).mean(), "MAE full": np.abs(y - pf).mean()})


def column_importance(P, meta, plat, reps=3, make=ET):
    """Permutation importance of single columns on held-out brands: rise in MAE when one column is shuffled (TF-IDF topics are shuffled as one block)."""
    m = (meta.platform == plat).values; M = meta[m].reset_index(drop=True); y, g = M.y, M.brand; X = pd.concat([v[m].reset_index(drop=True) for v in P.values()], axis=1).fillna(0); blocks = {c: [c] for c in X.columns if not c.startswith("svd")} | {"TF-IDF topics": [c for c in X.columns if c.startswith("svd")]}; rng = np.random.default_rng(SEED); out = {k: [] for k in blocks}
    for tr, te in GroupKFold(5).split(X, y, g):
        mod = make().fit(X.iloc[tr], y.iloc[tr]); Xt, yt = X.iloc[te], y.iloc[te]; base = np.abs(yt - mod.predict(Xt)).mean()
        for k, c in blocks.items():
            for _ in range(reps):
                Xp = Xt.copy(); Xp[c] = Xt[c].values[rng.permutation(len(Xt))]; out[k].append(np.abs(yt - mod.predict(Xp)).mean() - base)
    return pd.Series({k: np.mean(v) for k, v in out.items()}, name=eu.NAMES[plat])


def bias_table(f, meta, raw, lab, plat, make=ET):
    """Signed error (true minus predicted, so + means the model underestimates) of the followers + engagement model per subgroup, with brand-cluster CI."""
    m = (meta.platform == plat).values; M = meta[m].reset_index(drop=True); X = f[m][em.ENG].reset_index(drop=True); y, g = M.y, M.brand; p = em.cv_predict(make, X, y, g); e = (y - p).values; r = raw.loc[f.index][m].reset_index(drop=True); l = lab[m].reset_index(drop=True); rng = np.random.default_rng(SEED)
    react = np.expm1(X.log_likes) + np.expm1(X.log_comments); fol = r.author_followers
    S = {"mention type": l.type, "sentiment": l.sentiment, "language": np.where(r.language == "en", "English", "other"), "reactions": pd.cut(react, [-1, 0, 10, 100, 1000, np.inf], labels=["0", "1-10", "11-100", "101-1000", ">1000"]).astype(str), "followers": pd.cut(fol, [0, 1e3, 1e4, 1e5, np.inf], labels=["<1k", "1k-10k", "10k-100k", ">100k"]).astype(object).where(fol.notna(), "missing").astype(str), "period": np.where(r.created_at < pd.Timestamp("2026-04-01", tz=r.created_at.dt.tz), "before Apr 2026", "from Apr 2026"), "true views": pd.qcut(y.rank(method="first"), 5, labels=["lowest 20%", "20-40%", "40-60%", "60-80%", "top 20%"]).astype(str)}; rows = []
    for name, v in S.items():
        v = pd.Series(np.asarray(v))
        for k in v.unique():
            ok = (v == k).values
            if ok.sum() < 30: continue
            d = pd.DataFrame({"e": e[ok], "g": g[ok].values}).groupby("g").e.agg(["sum", "count"]); idx = np.arange(len(d)); boots = [d.iloc[s]["sum"].sum() / d.iloc[s]["count"].sum() for s in (rng.choice(idx, len(idx)) for _ in range(300))]
            rows.append({"platform": eu.NAMES[plat], "split": name, "group": k, "n": int(ok.sum()), "bias": e[ok].mean(), "low": np.percentile(boots, 2.5), "high": np.percentile(boots, 97.5), "MAE": np.abs(e[ok]).mean()})
    return pd.DataFrame(rows)


def type_lines(f, meta, lab, plat):
    """Own log-log line per mention type: does a reaction mean the same number of views for organic, promo and spam? Brand-cluster CI on slope and level at 100 reactions."""
    m = (meta.platform == plat).values; M = meta[m].reset_index(drop=True); X = f[m].reset_index(drop=True); s = np.log10(1 + np.expm1(X.log_likes.fillna(0)) + np.expm1(X.log_comments.fillna(0))).values; y = M.y.values; t = lab[m].type.values; rng = np.random.default_rng(SEED); rows = []
    for k in ["organic", "promo", "spam"]:
        ok = t == k; b = M.brand.values[ok]; xx, yy = s[ok], y[ok]; ub = np.unique(b); idx = {u: np.where(b == u)[0] for u in ub}; fit = lambda i: LinearRegression().fit(xx[i, None], yy[i]); l = fit(np.arange(len(xx))); bs = []
        for _ in range(300):
            i = np.concatenate([idx[u] for u in rng.choice(ub, len(ub))]); q = fit(i); bs.append((q.coef_[0], q.predict([[2.0]])[0]))
        bs = np.array(bs); rows.append({"platform": eu.NAMES[plat], "type": k, "n": int(ok.sum()), "slope": l.coef_[0], "slope low": np.percentile(bs[:, 0], 2.5), "slope high": np.percentile(bs[:, 0], 97.5), "views at 100 reactions": l.predict([[2.0]])[0], "level low": np.percentile(bs[:, 1], 2.5), "level high": np.percentile(bs[:, 1], 97.5)})
    return pd.DataFrame(rows)


def spam_filter(f, meta, lab, plat, make=ET):
    """Does removing spam from training help on the non-spam test posts? A practical use of the label."""
    m = (meta.platform == plat).values; M = meta[m].reset_index(drop=True); X = f[m][em.ENG].fillna(0).reset_index(drop=True); y, g = M.y, M.brand; t = lab[m].type.values; pa, pb = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        keep = tr[t[tr] != "spam"]; pa[te] = make().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te]); pb[te] = make().fit(X.iloc[keep], y.iloc[keep]).predict(X.iloc[te])
    ok = t != "spam"; d = ni.paired_diff(y.values[ok], pa[ok], pb[ok], g.values[ok]); return {"platform": eu.NAMES[plat], "trained on all": np.abs(y - pa)[ok].mean(), "spam removed": np.abs(y - pb)[ok].mean(), "gain": d[0], "low": d[1], "high": d[2], "spam error": np.abs(y - pa)[~ok].mean(), "non-spam error": np.abs(y - pa)[ok].mean()}


# ---------- communities ----------
def community_report(f, meta, raw, lab, plat):
    """Louvain on the text kNN graph: stability across seeds, overlap with brands and types, the share of view variance communities explain with and without reactions, and the most unusual communities."""
    m = (meta.platform == plat).values; M = meta[m].reset_index(drop=True); r = raw.loc[f.index][m].reset_index(drop=True); l = lab[m].reset_index(drop=True); text = (r.title.fillna("") + " " + r.content.fillna("")); S = em.text_similarity(text.values); A, W = gl.knn_graph(S); c0, G = gl.louvain(A); runs = [gl.louvain(A, seed=s)[0] for s in (1, 2, 3, 4)]
    p = em.cv_predict(ET, f[m][em.ENG].reset_index(drop=True), M.y, M.brand); res = (M.y - p).values
    eta = lambda v: 1 - pd.Series(v).groupby(c0).transform(lambda x: x - x.mean()).pow(2).sum() / ((v - v.mean()) ** 2).sum(); rng = np.random.default_rng(SEED); null = np.mean([1 - pd.Series(M.y.values).groupby(rng.permutation(c0)).transform(lambda x: x - x.mean()).pow(2).sum() / ((M.y.values - M.y.values.mean()) ** 2).sum() for _ in range(20)])
    from sklearn.feature_extraction.text import TfidfVectorizer
    t = text.where(text.str.strip() != "", "emptytext"); v = TfidfVectorizer(min_df=3, max_features=20000, sublinear_tf=True, stop_words="english"); X = v.fit_transform(t); terms = np.array(v.get_feature_names_out()); rows = []
    for c in pd.Series(c0).value_counts().index:
        i = np.where(c0 == c)[0]
        if len(i) < 25: continue
        top = terms[np.asarray(X[i].mean(0)).ravel().argsort()[::-1][:6]]; mix = l.type.iloc[i].value_counts(normalize=True); bs = M.brand.iloc[i].value_counts(normalize=True)
        rows.append({"platform": eu.NAMES[plat], "community": int(c), "size": len(i), "top terms": ", ".join(top), "brands": int(M.brand.iloc[i].nunique()), "top brand share": float(bs.iloc[0]), "organic": mix.get("organic", 0), "promo": mix.get("promo", 0), "spam": mix.get("spam", 0), "unclear": mix.get("unclear", 0), "median log10 views": float(M.y.iloc[i].median()), "median residual": float(np.median(res[i])), "share of platform views": float((10 ** M.y.iloc[i]).sum() / (10 ** M.y).sum())})
    summary = {"platform": eu.NAMES[plat], "communities": int(c0.max() + 1), "with 25+ mentions": len(rows), "stability (ARI across seeds)": float(np.mean([adjusted_rand_score(c0, x) for x in runs])), "NMI with brand": float(normalized_mutual_info_score(M.brand, c0)), "NMI with our type": float(normalized_mutual_info_score(l.type.fillna("na"), c0)),
               "variance of views explained": float(eta(M.y.values)), "same, shuffled communities": float(null), "variance of residual explained": float(eta(res))}
    return summary, pd.DataFrame(rows), dict(A=A, comm=c0, y=M.y.values, type=l.type.fillna("unclear").values)


def plot_community_net(G_, prof, plat, path=None, lang="en", top=8):
    """Force layout of the text graph; each large community gets its own colour and its top words."""
    import matplotlib.pyplot as plt
    A, c = G_["A"], G_["comm"]; G = nx.from_numpy_array(A); pos = nx.spring_layout(G, weight="weight", seed=SEED, iterations=60); XY = np.array([pos[i] for i in range(len(A))]); big = prof.sort_values("size", ascending=False).head(top); cmap = plt.get_cmap("tab10"); col = {int(k): cmap(i % 10) for i, k in enumerate(big.community)}
    fig, a = plt.subplots(figsize=(10, 7)); u, w = np.nonzero(np.triu(A)); s = np.random.default_rng(SEED).choice(len(u), min(3000, len(u)), replace=False)
    for i in s: a.plot(XY[[u[i], w[i]], 0], XY[[u[i], w[i]], 1], color="#d0d0d0", lw=.15, zorder=1)
    a.scatter(XY[:, 0], XY[:, 1], s=7, c=[col.get(int(k), "#c4c4c4") for k in c], zorder=2)
    for _, r in big.iterrows():
        q = XY[c == r.community].mean(0); a.annotate(", ".join(r["top terms"].split(", ")[:3]), q, fontsize=9, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=.2", fc="white", ec="#888888", alpha=.9), zorder=3)
    a.set_xticks([]); a.set_yticks([]); [sp.set_visible(False) for sp in a.spines.values()]; a.set_xlabel("układ siłowy, bez jednostek (blisko = podobny tekst)" if lang == "pl" else "force layout, no units (close = similar text)"); a.set_ylabel("układ siłowy, bez jednostek" if lang == "pl" else "force layout, no units"); a.set_title((f"{eu.NAMES[plat]}: społeczności wykryte z samego tekstu" if lang == "pl" else f"{eu.NAMES[plat]}: communities found from text alone"), fontweight="bold"); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()


PL_NAMES = {"likes": "lajki", "comments": "komentarze", "shares": "udostępnienia", "followers": "followersi", "our type label": "nasz typ wzmianki", "our sentiment label": "nasz sentyment", "Brand24 sentiment": "sentyment Brand24", "text stats": "statystyki tekstu", "TF-IDF topics": "tematy TF-IDF", "VADER sentiment": "sentyment VADER", "calendar": "kalendarz", "brand size": "wielkość marki"}


def plot_views(T, path=None, lang="en"):
    """The three-way ablation: alone, added, removed."""
    import matplotlib.pyplot as plt
    pl = lang == "pl"; groups_ = list(dict.fromkeys(T.group)); fig, ax = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True); w = .26; lab = [PL_NAMES[g] if pl else g for g in groups_]
    for j, p in enumerate(mo.PLATS):
        r = T[T.platform == eu.NAMES[p]].set_index("group").reindex(groups_); yy = np.arange(len(groups_)) + (1 - j) * w
        ax[0].barh(yy, r["MAE alone"], w, color=eu.PCOL[p], label=eu.NAMES[p]); ax[1].barh(yy, r["add gain"], w, color=eu.PCOL[p], xerr=[r["add gain"] - r["add low"], r["add high"] - r["add gain"]], ecolor="black", capsize=1.5); ax[2].barh(yy, r["remove loss"], w, color=eu.PCOL[p], xerr=[r["remove loss"] - r["remove low"], r["remove high"] - r["remove loss"]], ecolor="black", capsize=1.5)
    ax[0].set_yticks(range(len(groups_))); ax[0].set_yticklabels(lab); ax[0].invert_yaxis(); ax[0].set_ylabel("grupa kolumn" if pl else "column group")
    ax[0].set_xlabel("MAE, log10 wyświetleń (mniej = lepiej)" if pl else "MAE, log10 views (lower = better)"); ax[0].set_title("Tylko ta grupa" if pl else "This group alone", fontweight="bold"); ax[1].set_xlabel("spadek MAE (dodatni = pomaga)" if pl else "fall in MAE (positive = helps)"); ax[1].set_title("Dodana do followersi + reakcje" if pl else "Added to followers + reactions", fontweight="bold"); ax[2].set_xlabel("wzrost MAE (symlog; dodatni = była potrzebna)" if pl else "rise in MAE (symlog; positive = was needed)"); ax[2].set_title("Usunięta z pełnego modelu" if pl else "Removed from the full model", fontweight="bold"); ax[2].set_xscale("symlog", linthresh=.01); ax[2].set_xticks([-.01, 0, .01, .03, .1]); ax[2].set_xticklabels(["-0,01" if pl else "-0.01", "0", "0,01" if pl else "0.01", "0,03" if pl else "0.03", "0,1" if pl else "0.1"])
    for a in ax[1:]: a.axvline(0, color="grey", lw=1)
    ax[-1].legend(*ax[0].get_legend_handles_labels(), loc="center left", bbox_to_anchor=(1, .5), frameon=False, title="platforma" if pl else "platform"); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()


def plot_importance(I, path=None, lang="en", top=12):
    import matplotlib.pyplot as plt
    pl = lang == "pl"; order = I.mean(axis=1).sort_values(ascending=False).head(top).index; fig, ax = plt.subplots(figsize=(10, 5.2)); w = .26; nm = {"log_likes": "lajki", "log_comments": "komentarze", "log_shares": "udostępnienia", "log_followers": "followersi", "brand_mentions": "wielkość marki", "hours_to_metrics": "wiek metryk (godziny)", "chars": "długość tekstu (znaki)", "words": "długość tekstu (słowa)", "na_likes": "brak lajków", "na_comments": "brak komentarzy", "na_shares": "brak udostępnień", "TF-IDF topics": "tematy TF-IDF", "type_promo": "typ: promo", "type_organic": "typ: organic", "type_spam": "typ: spam", "type_unclear": "typ: niejasny", "n_mentions": "liczba @wzmianek", "n_urls": "liczba linków", "b24_sent_na": "brak sentymentu Brand24", "n_hashtags": "liczba hashtagów", "n_emoji": "liczba emoji"}
    for j, p in enumerate(mo.PLATS): ax.barh(np.arange(len(order)) + (1 - j) * w, I[eu.NAMES[p]].reindex(order), w, color=eu.PCOL[p], label=eu.NAMES[p])
    ax.set_yticks(range(len(order))); ax.set_yticklabels([nm.get(c, c) if pl else c for c in order]); ax.invert_yaxis(); ax.axvline(0, color="grey", lw=1); ax.set_xscale("symlog", linthresh=.01); ax.set_xticks([0, .01, .03, .1, .3]); ax.set_xticklabels(["0", "0,01" if pl else "0.01", "0,03" if pl else "0.03", "0,1" if pl else "0.1", "0,3" if pl else "0.3"]); ax.set_xlabel("wzrost MAE po przetasowaniu kolumny (symlog, log10 wyświetleń)" if pl else "rise in MAE when the column is shuffled (symlog, log10 views)"); ax.set_ylabel("kolumna" if pl else "column"); ax.set_title("Które kolumny niosą sygnał (marki spoza uczenia)" if pl else "Which columns carry the signal (held-out brands)", fontweight="bold"); ax.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, title="platforma" if pl else "platform"); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()


def plot_bias(B, path=None, lang="en"):
    import matplotlib.pyplot as plt
    pl = lang == "pl"; TR = {"organic": "organic", "promo": "promo", "spam": "spam", "unclear": "niejasne", "negative": "negatywny", "neutral": "neutralny", "positive": "pozytywny", "English": "angielski", "other": "inny", "before Apr 2026": "przed IV 2026", "from Apr 2026": "od IV 2026", "lowest 20%": "najniższe 20%", "top 20%": "najwyższe 20%"}; S = {"mention type": ("typ wzmianki", ["organic", "promo", "spam", "unclear"]), "sentiment": ("sentyment", ["negative", "neutral", "positive"]), "reactions": ("reakcje", ["0", "1-10", "11-100", "101-1000", ">1000"]), "followers": ("followersi", ["<1k", "1k-10k", "10k-100k", ">100k"]), "language": ("język", ["English", "other"]), "period": ("okres", ["before Apr 2026", "from Apr 2026"]), "true views": ("prawdziwe wyświetlenia", ["lowest 20%", "20-40%", "40-60%", "60-80%", "top 20%"])}
    fig, ax = plt.subplots(2, 4, figsize=(17, 7.5)); ax = ax.ravel()
    for a, (k, (npl, order)) in zip(ax, S.items()):
        for j, p in enumerate(mo.PLATS):
            r = B[(B.platform == eu.NAMES[p]) & (B.split == k)].set_index("group").reindex(order); a.errorbar(r.bias, np.arange(len(order)) + (1 - j) * .22, xerr=[r.bias - r.low, r.high - r.bias], fmt="o", ms=4, capsize=2, color=eu.PCOL[p], label=eu.NAMES[p])
        a.set_yticks(range(len(order))); a.set_yticklabels([TR.get(o, o) for o in order] if pl else order); a.invert_yaxis(); a.axvline(0, color="grey", lw=1); a.set_title(npl if pl else k, fontweight="bold"); a.set_xlabel("średni błąd ze znakiem, log10 (+ = model zaniża)" if pl else "mean signed error, log10 (+ = model underestimates)", fontsize=9); a.set_ylabel("grupa" if pl else "group", fontsize=9)
    ax[-1].axis("off"); ax[-1].legend(*ax[0].get_legend_handles_labels(), loc="center", frameon=False, title="platforma" if pl else "platform"); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()


JUNK = {"https", "http", "com", "www", "gt", "039", "quot", "bit", "ly", "amzn", "status", "twitter", "fyp", "viral", "foryou", "viralvideo", "la", "en", "el", "new", "like", "channel", "just"}


def _name(terms, k=3):
    return ", ".join([t for t in terms.split(", ") if t not in JUNK][:k])


def plot_community_bubbles(CP, path=None, lang="en"):
    """Each circle is a community found from text alone. Readable where a force layout of 2,000 nodes is not."""
    import matplotlib.pyplot as plt
    pl = lang == "pl"; fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.2))
    for a, p in zip(ax, mo.PLATS):
        r = CP[CP.platform == eu.NAMES[p]]; sc = a.scatter(r.spam * 100, r["median log10 views"], s=r["size"] * 2.2, c=r.unclear * 100, cmap="YlOrBr", vmin=0, vmax=50, edgecolor="#555555", linewidth=.6, alpha=.85)
        pick = pd.concat([r.nlargest(2, "median log10 views"), r.nsmallest(1, "median log10 views"), r.nlargest(1, "spam"), r.nlargest(1, "unclear")]).drop_duplicates("community"); mid = r.spam.max() * 60
        for i, (_, q) in enumerate(pick.iterrows()):
            right = q.spam * 100 > mid; name = ("transkrypcje filmów" if pl else "video transcripts") if q.unclear > .3 and q["size"] > 100 else _name(q["top terms"])
            a.annotate(name, (q.spam * 100, q["median log10 views"]), xytext=(-7 if right else 7, 7), textcoords="offset points", fontsize=9, ha="right" if right else "left")
        a.margins(x=.12, y=.12); a.set_title(eu.NAMES[p], fontweight="bold"); a.set_xlabel("spam w społeczności (%)" if pl else "spam in the community (%)"); a.set_ylabel("mediana log10(wyświetlenia)" if pl else "median log10(views)"); a.grid(alpha=.2)
    cb = fig.colorbar(sc, ax=ax, fraction=.015, pad=.02); cb.set_label("niejasne wzmianki (%)" if pl else "unclear mentions (%)")
    fig.suptitle("Społeczności wykryte z samego tekstu (rozmiar koła = liczba wzmianek)" if pl else "Communities found from text alone (circle size = mentions)", fontweight="bold")
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()


def promo_flag(f, meta, lab, plat):
    """Does a single is-promo column help? Paired gain per learner, on reactions only and on followers + reactions + flags, plus how promo authors differ."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = (meta.platform == plat).values; M = meta[m].reset_index(drop=True); X = f[m].reset_index(drop=True); y, g = M.y, M.brand; pr = (lab[m].type.values == "promo").astype(int); rows = []
    RID = lambda: make_pipeline(StandardScaler(), Ridge()); sets = {"likes + comments": ["log_likes", "log_comments"], "followers + reactions + flags": em.ENG}
    for sn, cols in sets.items():
        for mn, mk in {"ridge": RID, "boosting (sklearn)": HGB, "extra trees": ET}.items():
            a = em.cv_predict(mk, X[cols], y, g); b = em.cv_predict(mk, X[cols].assign(is_promo=pr), y, g); d = ni.paired_diff(y.values, a, b, g.values); rows.append({"platform": eu.NAMES[plat], "columns": sn, "model": mn, "without": np.abs(y - a).mean(), "with is_promo": np.abs(y - b).mean(), "gain": d[0], "low": d[1], "high": d[2]})
    o = M.author_followers[pr == 0].median(), M.author_followers[pr == 1].median()
    return pd.DataFrame(rows), {"platform": eu.NAMES[plat], "share promo": float(pr.mean()), "median followers, not promo": float(o[0]), "median followers, promo": float(o[1])}


def youtube_error_profile(f, meta, lab, raw):
    """YouTube's error split by the state of the reaction columns and by promo share; and how many zero-reaction posts have many views."""
    m = (meta.platform == "youtube").values; M = meta[m].reset_index(drop=True); X = f[m][em.ENG].reset_index(drop=True); r = raw.loc[f.index][m].reset_index(drop=True); t = lab[m].type.reset_index(drop=True); e = M.y.values - em.cv_predict(ET, X, M.y, M.brand); a = np.abs(e)
    st = np.select([r.likes.isna(), r.likes.eq(0) & r.comments.fillna(0).eq(0)], ["likes hidden", "zero likes and comments"], "has reactions"); d = pd.DataFrame({"state": st, "a": a, "e": e, "promo": t == "promo", "views": 10 ** M.y})
    by = d.groupby("state").agg(n=("a", "size"), MAE=("a", "mean"), bias=("e", "mean"), promo=("promo", "mean"), median_views=("views", "median")); by["share of error"] = d.groupby("state").a.sum() / a.sum()
    err = d.assign(error=pd.cut(e, [-9, -.5, .5, 9], labels=["overestimated over x3", "within x3", "underestimated over x3"])).groupby("error", observed=True).agg(n=("a", "size"), promo=("promo", "mean"), zero=("state", lambda s: (s == "zero likes and comments").mean()))
    z = d[d.state == "zero likes and comments"]; return {"by_state": by.reset_index(), "by_error": err.reset_index(), "zero_posts": {"n": len(z), "share 1k+ views": float((z.views >= 1000).mean()), "n 10k+ views": int((z.views >= 10000).sum()), "max views": float(z.views.max())}}


def drop_zero_reactions(f, meta, raw, plat, make=ET):
    """What happens if posts with no likes and no comments are removed? Separates two effects: the easier test set, and any gain from cleaner training."""
    m = (meta.platform == plat).values; M = meta[m].reset_index(drop=True); X = f[m][em.ENG].fillna(0).reset_index(drop=True); r = raw.loc[f.index][m].reset_index(drop=True); y, g = M.y, M.brand; zero = (r.likes.fillna(0).eq(0) & r.comments.fillna(0).eq(0)).values; a, b = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        keep = tr[~zero[tr]]; a[te] = make().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te]); b[te] = make().fit(X.iloc[keep], y.iloc[keep]).predict(X.iloc[te])
    ok = ~zero; d = ni.paired_diff(y.values[ok], a[ok], b[ok], g.values[ok]); return {"platform": eu.NAMES[plat], "posts": len(y), "zero share": float(zero.mean()), "all posts, trained on all": float(np.abs(y - a).mean()), "zero posts only": float(np.abs(y - a)[zero].mean()), "kept posts, trained on all": float(np.abs(y - a)[ok].mean()), "kept posts, trained on kept": float(np.abs(y - b)[ok].mean()), "training gain": d[0], "low": d[1], "high": d[2], "zero posts if trained on kept": float(np.abs(y - b)[zero].mean())}
