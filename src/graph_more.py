"""More graph ideas (notebooks/models/12_graph_more.ipynb), tested against the same base as graph_lab.evaluate.
Target log10(views) on TikTok, YouTube, X; 5-fold GroupKFold by brand; base = extra trees (200, min_samples_leaf=5) on followers + engagement + flags.
A. Fixed multi-aggregator features over the text kNN graph (mean, max, std, own-minus-neighbours, 2 hops), label-free (Rubio-Madrigal and Burkholz 2026).
B. Cross-platform text neighbours that have real views (training-fold brands only, same brand excluded for every row), also as a bridge for leave-one-platform-out.
C. Hierarchical shrinkage: one shared model plus a shrunken per-community correction (intercept, and intercept + reactions slope) from inner brand-grouped residuals.
Authors, handles and URLs are never used."""
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import extra_models as em, graph_lab as gl, new_ideas as ni, eda_utils as eu, modeling as mo
from validation import mae_ci

SEED = 42; K = 10; PLATS = mo.PLATS; REACT = ["log_likes", "log_comments"]
AGG_COLS = ["log_likes", "log_comments", "log_followers", "log_shares"]


def make_et(): return ExtraTreesRegressor(200, min_samples_leaf=5, n_jobs=4, random_state=SEED)
def make_ridge(): return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


# ---------- data ----------
def build(f, meta, raw):
    """All three platforms in one frame; S is the cosine similarity of TF-IDF text over all of them (label-free, so it may use every mention)."""
    r = raw.loc[f.index]; text = (r.title.fillna("") + " " + r.content.fillna("")).values
    S = em.text_similarity(text).astype(np.float32); empty = np.array([not t.strip() for t in text])
    M = meta.reset_index(drop=True)[["platform", "brand", "y"]]; X = f[em.ENG].reset_index(drop=True).fillna(0)
    return dict(M=M, X=X, S=S, empty=empty, plat=M.platform.to_numpy(dtype=object), brand=M.brand.to_numpy(dtype=object), y=M.y.to_numpy(dtype=float))


def sub(D, p):
    """Platform-local view: global indices, features, targets, brands."""
    gi = np.where(D["plat"] == p)[0]; M = D["M"].iloc[gi].reset_index(drop=True)
    return dict(gi=gi, X=D["X"].iloc[gi].reset_index(drop=True), y=M.y, g=M.brand, S=D["S"][np.ix_(gi, gi)])


# ---------- A: fixed aggregation features ----------
def faf_features(Sp, Xp, k=K):
    """Label-free multi-aggregator features of the k nearest text neighbours of every mention (inside its platform)."""
    A, W = gl.knn_graph(Sp.astype(float), k); nb = np.argpartition(-Sp, k, axis=1)[:, :k]; out = {}
    for c in AGG_COLS:
        h = Xp[c].values; V = h[nb]; m = V.mean(1)
        out[f"{c}_nmean"] = m; out[f"{c}_nmax"] = V.max(1); out[f"{c}_nstd"] = V.std(1); out[f"{c}_rel"] = h - m; out[f"{c}_hop2"] = W @ (W @ h)
    out["nsim"] = np.take_along_axis(Sp, nb, 1).mean(1)
    return pd.DataFrame(out)


# ---------- B: cross-platform neighbours with views ----------
def xp_features(D, rows, pool_plats, allowed, k=K, exclude_own_brand=True):
    """For global rows `rows`: average over pool platforms q of the stats of the k most similar mentions on q whose views are known.
    `allowed` is a boolean mask over all mentions (training-fold brands only). Same-brand neighbours are dropped so every row looks like a new brand."""
    S, y, lk, br = D["S"], D["y"], D["X"].log_likes.values, D["brand"]; feats = []
    for q in pool_plats:
        cand = np.where((D["plat"] == q) & allowed)[0]; Sm = S[np.ix_(rows, cand)].copy()
        Sm[D["empty"][rows][:, None] | D["empty"][cand][None, :]] = 0
        if exclude_own_brand: Sm[br[rows][:, None] == br[cand][None, :]] = 0
        top = np.argpartition(-Sm, k, axis=1)[:, :k]; w = np.take_along_axis(Sm, top, 1).astype(float); sw = np.maximum(w.sum(1), 1e-9); yc = y[cand][top]; rc = (y[cand] - lk[cand])[top]
        ok = w.sum(1) > 0; gm = y[cand].mean(); gr = (y[cand] - lk[cand]).mean()
        feats.append(np.c_[np.where(ok, (w * yc).sum(1) / sw, gm), np.where(ok, (w * rc).sum(1) / sw, gr), w.mean(1)])
    F = np.mean(feats, axis=0); return pd.DataFrame(F, columns=["xp_views", "xp_ratio", "xp_sim"])


def xp_extra(D, P, p):
    """extra(tr, te) for platform-local folds: pool = the other known platforms, training-fold brands only."""
    others = [q for q in PLATS if q != p]
    def extra(tr, te):
        test_b = set(P["g"].iloc[te]); allowed = ~pd.Series(D["brand"]).isin(test_b).values
        return xp_features(D, P["gi"][tr], others, allowed), xp_features(D, P["gi"][te], others, allowed)
    return extra


# ---------- C: hierarchical shrinkage per community ----------
def inner_oof(make, X, y, g, tr):
    """Out-of-fold residuals inside the training fold, grouped by brand (a brand never predicts itself)."""
    res = np.zeros(len(tr)); Xt = X.iloc[tr].reset_index(drop=True); yt = y.iloc[tr].reset_index(drop=True); gt = g.iloc[tr].reset_index(drop=True)
    for a, b in GroupKFold(5).split(Xt, yt, gt): res[b] = yt.iloc[b].values - make().fit(Xt.iloc[a], yt.iloc[a]).predict(Xt.iloc[b])
    return res


def shrunk_fit(comm, z, res, lam, slope):
    """Ridge-shrunk intercept (and slope on z) of the residual per community. lam = pseudo-observations pulling towards 0 (the shared model)."""
    beta = {}
    for c in np.unique(comm):
        i = comm == c; Z = np.c_[np.ones(i.sum()), z[i]] if slope else np.ones((i.sum(), 1))
        beta[c] = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ res[i])
    return beta


def hier_cv(P, comm, lam=10.0):
    """Base predictions, plus corrections: community intercept only, and intercept + slope on log_likes. Returns dict of out-of-fold predictions."""
    X, y, g = P["X"], P["y"], P["g"]; out = {k: np.full(len(y), np.nan) for k in ["base", "intercept", "slope"]}
    for tr, te in GroupKFold(5).split(X, y, g):
        base = make_et().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te]); res = inner_oof(make_et, X, y, g, tr)
        z = X.log_likes.values; mu, sd = z[tr].mean(), max(z[tr].std(), 1e-9); zt, ze = (z[tr] - mu) / sd, (z[te] - mu) / sd
        out["base"][te] = base
        for name, slope in [("intercept", False), ("slope", True)]:
            beta = shrunk_fit(comm[tr], zt, res, lam, slope); corr = np.zeros(len(te))
            for j, (c, zz) in enumerate(zip(comm[te], ze)):
                if c in beta: b = beta[c]; corr[j] = b[0] + (b[1] * zz if slope else 0.0)
            out[name][te] = base + corr
    return out


def slope_heterogeneity(P, comm, pred, min_n=30, lam=10.0, n_perm=300):
    """Do reactions-to-views slopes of the out-of-fold residual differ between communities more than by chance? Permutation of community labels."""
    res = (P["y"] - pred).values; z = P["X"].log_likes.values; z = (z - z.mean()) / z.std(); big = [c for c, n in pd.Series(comm).value_counts().items() if n >= min_n]
    def stat(lab):
        b = shrunk_fit(lab, z, res, lam, True); w = np.array([(lab == c).sum() for c in big]); s = np.array([b[c][1] for c in big])
        return float(np.sqrt(np.average((s - np.average(s, weights=w)) ** 2, weights=w)))
    obs = stat(comm); rng = np.random.default_rng(SEED); null = [stat(rng.permutation(comm)) for _ in range(n_perm)]
    return {"communities": len(big), "slope SD observed": obs, "slope SD null mean": float(np.mean(null)), "p (permutation)": float((np.sum(np.array(null) >= obs) + 1) / (n_perm + 1))}


# ---------- scoring ----------
def score(y, p, g, pb, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask; y, p, g, pb = (np.asarray(a)[m] for a in (y, p, g, pb))
    mae, lo, hi = mae_ci(y - p, g); gn = ni.paired_diff(y, pb, p, g)
    return {"n": int(m.sum()), "MAE": mae, "CI low": lo, "CI high": hi, "gain": gn[0], "gain low": gn[1], "gain high": gn[2], "bias": float((y - p).mean())}


def zero_mask(X): return ((X.log_likes == 0) & (X.log_comments == 0)).values


def run_platform(D, p):
    """All in-platform variants for one platform: returns predictions per variant and the platform view."""
    P = sub(D, p); X, y, g = P["X"], P["y"], P["g"]; preds = {}
    preds["base"] = em.cv_predict(make_et, X, y, g)
    fa = faf_features(P["S"], X); preds["A: fixed aggregation (label-free)"] = em.cv_predict(make_et, pd.concat([X, fa], axis=1), y, g)
    xe = xp_extra(D, P, p); preds["B: cross-platform neighbours with views"] = em.cv_predict(make_et, X, y, g, xe)
    preds["A + B"] = em.cv_predict(make_et, pd.concat([X, fa], axis=1), y, g, xe)
    A_, _ = gl.knn_graph(P["S"].astype(float), K); comm, _ = gl.louvain(A_); P["comm"] = comm
    h = hier_cv(P, comm); preds["C: community intercept (shrunk)"] = h["intercept"]; preds["C: community intercept + reactions slope (shrunk)"] = h["slope"]
    P["hier_base"] = h["base"]; return P, preds


def table(P, preds, p):
    y, g = P["y"].values, P["g"].values; pb = preds["base"]; zm = zero_mask(P["X"]); rows = []
    for name, pr in preds.items():
        for grp, m in [("all posts", None), ("zero reactions", zm)]:
            rows.append({"platform": eu.NAMES[p], "variant": name, "group": grp, **score(y, pr, g, pb, m)})
    return pd.DataFrame(rows)


# ---------- bridge: leave-one-platform-out ----------
def bridge(D, target, use_xp, model="ridge"):
    """Train on the other two known platforms (rows with a test brand removed), test on `target`. use_xp adds cross-platform neighbour features,
    where a source row looks only at the other source platform and a target row looks at both sources (features averaged over pool platforms)."""
    sources = [q for q in PLATS if q != target]; T = np.where(D["plat"] == target)[0]; Xb = D["X"]; cols = REACT if model == "ridge" else em.ENG
    pred = np.full(len(T), np.nan)
    for _, te in GroupKFold(5).split(T, groups=D["brand"][T]):
        test_b = set(D["brand"][T[te]]); allowed = ~pd.Series(D["brand"]).isin(test_b).values
        S_rows = np.where(np.isin(D["plat"], sources) & allowed)[0]; Xtr = Xb.iloc[S_rows][cols].reset_index(drop=True); Xte = Xb.iloc[T[te]][cols].reset_index(drop=True)
        if use_xp:
            parts = []
            for a in sources:
                ra = S_rows[D["plat"][S_rows] == a]; parts.append(xp_features(D, ra, [q for q in sources if q != a], allowed))
            Ftr = pd.concat(parts, ignore_index=True); order = np.concatenate([np.where(D["plat"][S_rows] == a)[0] for a in sources])
            Xtr = Xtr.iloc[order].reset_index(drop=True); Xtr = pd.concat([Xtr, Ftr], axis=1); ytr = D["y"][S_rows][order]
            Xte = pd.concat([Xte, xp_features(D, T[te], sources, allowed)], axis=1)
        else: ytr = D["y"][S_rows]
        mk = make_ridge() if model == "ridge" else make_et(); pred[te] = mk.fit(Xtr, ytr).predict(Xte)
    return T, pred


def bridge_table(D):
    rows = []; zm_all = zero_mask(D["X"])
    for target in PLATS:
        for model in ["ridge", "extra trees"]:
            T, pb = bridge(D, target, False, model); y, g = D["y"][T], D["brand"][T]
            for name, use in [("without bridge", False), ("with cross-platform bridge", True)]:
                pr = pb if not use else bridge(D, target, True, model)[1]
                for grp, m in [("all posts", None), ("zero reactions", zm_all[T])]:
                    rows.append({"platform": eu.NAMES[target], "model": model, "variant": name, "group": grp, **score(y, pr, g, pb, m)})
    return pd.DataFrame(rows)


# ---------- plots ----------
TX = {"en": dict(gain="MAE gain over followers + engagement base, log10 views (positive = better)", ylab="graph idea added", t1="New graph ideas: gain over the base, all posts", t2="New graph ideas: gain over the base, posts with zero reactions",
                 mae="MAE of log10(views), leave-one-platform-out", bias="bias of log10(views), mean(true - predicted)", plat="platform", t3="Cross-platform bridge: error and level shift", x3="test platform (trained on the other two)", ridge="ridge on likes + comments", var="variant"),
      "pl": dict(gain="zysk MAE względem bazy followers + reakcje, log10 wyświetleń (dodatni = lepiej)", ylab="dodany pomysł grafowy", t1="Nowe pomysły grafowe: zysk względem bazy, wszystkie posty", t2="Nowe pomysły grafowe: zysk względem bazy, posty bez reakcji",
                 mae="MAE log10(wyświetleń), leave-one-platform-out", bias="obciążenie log10(wyświetleń), średnia(prawda - predykcja)", plat="platforma", t3="Most międzyplatformowy: błąd i przesunięcie poziomu", x3="platforma testowa (trening na dwóch pozostałych)", ridge="ridge na polubieniach + komentarzach", var="wariant")}


def plot_gains(res, group, path, lang):
    import matplotlib.pyplot as plt
    T = TX[lang]; L = res[(res.group == group) & (res.variant != "base")]; names = list(dict.fromkeys(L.variant)); fig, ax = plt.subplots(figsize=(10.5, 4.8)); w = .26
    for j, p in enumerate(PLATS):
        r = L[L.platform == eu.NAMES[p]].set_index("variant").reindex(names)
        ax.barh(np.arange(len(names)) + (1 - j) * w, r.gain, w, color=eu.PCOL[p], label=f"{eu.NAMES[p]} (n={int(r.n.iloc[0])})", xerr=[r.gain - r["gain low"], r["gain high"] - r.gain], ecolor="black", capsize=2)
    ax.axvline(0, color="grey", lw=1); ax.xaxis.set_major_locator(plt.MaxNLocator(6)); ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.3f")); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9); ax.invert_yaxis(); ax.set_xlabel(T["gain"]); ax.set_ylabel(T["ylab"])
    ax.set_title(T["t1"] if group == "all posts" else T["t2"], fontweight="bold"); ax.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, title=T["plat"]); plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_bridge(bt, path, lang):
    import matplotlib.pyplot as plt
    T = TX[lang]; b = bt[(bt.model == "ridge") & (bt.group == "all posts")]; fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); w = .36
    for a, (c, yl) in zip(ax, [("MAE", T["mae"]), ("bias", T["bias"])]):
        for j, (v, hatch) in enumerate([("without bridge", ""), ("with cross-platform bridge", "//")]):
            r = b[b.variant == v].set_index("platform").reindex([eu.NAMES[p] for p in PLATS]); x = np.arange(3) + (j - .5) * w
            a.bar(x, r[c], w, color=[eu.PCOL[p] for p in PLATS], hatch=hatch, edgecolor="white")
        a.set_xticks(range(3)); a.set_xticklabels([eu.NAMES[p] for p in PLATS]); a.set_xlabel(T["x3"]); a.set_ylabel(yl); a.axhline(0, color="grey", lw=1)
    ax[0].set_title(T["t3"], fontweight="bold"); ax[1].set_title(T["bias"].split(",")[0], fontweight="bold"); from matplotlib.patches import Patch
    ax[1].legend(handles=[Patch(facecolor="#888", label=T["ridge"]), Patch(facecolor="#888", hatch="//", edgecolor="white", label=T["ridge"] + " + bridge")], loc="center left", bbox_to_anchor=(1, .5), frameon=False, title=T["var"]); plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
