"""E234: (1) are the outliers of the view model often off-brand mentions? (2) can the off-brand detector of E224 be improved
honestly (brand-grouped CV on the 300 AI-labelled posts, no tuning on the full label set)?

Task 1 (E224 scores use no views; views only evaluate): outliers = top / bottom 5% of the out-of-fold residual (y - pred) of
results_preds.csv per platform (TikTok / YouTube / X); rows are aligned per platform with em.load (as in size_slices.py).
Task 2: methods a-g + all-features row, repeated brand-grouped 5-fold CV (REPEATS random group splits, OOF predictions
pooled per repeat, AUC averaged over repeats), CI = brand-cluster bootstrap of the repeat-averaged OOF score.
Hyper-parameters that depend on the labels (Louvain resolution, k) are chosen INSIDE each training fold only.
No author, handle, URL or quote is written to any output.

    .venv/bin/python -W ignore src/offbrand_more.py
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, graph_lab as gl, offbrand as ob  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
REPEATS = 5
NBOOT = 500
PROC = ROOT / "data" / "processed"
PLAT = ["tiktok", "youtube", "twitter"]
PNAME = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X"}


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def auc_ci(y, s, rng, n=NBOOT):
    y = np.asarray(y); s = np.asarray(s); a = roc_auc_score(y, s); b = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            b.append(roc_auc_score(y[i], s[i]))
    return r3(a), r3(np.percentile(b, 2.5)), r3(np.percentile(b, 97.5))


# ================= Task 1 =================
def task1(raw, sc, lab, S):
    f, meta = em.load(ROOT); rawm = raw.loc[meta.index]
    d = pd.read_csv(PROC / "results_preds.csv"); rows = []
    for p in PLAT:
        m = (meta.platform == p).values; g = d[d.platform == p]
        assert np.allclose(g.y.values, meta.y[m].values)
        t = pd.DataFrame({"mention_id": rawm.mention_id[m].values, "platform": p, "brand": rawm.brand[m].values,
                          "views": rawm.view_count[m].values, "y": g.y.values, "pred": g.pred.values})
        t["res"] = t.y - t.pred; rows.append(t)
    T = pd.concat(rows, ignore_index=True).merge(sc[["mention_id", "combined", "combined_rank", "decile"]], on="mention_id", validate="one_to_one")
    T["over"] = False; T["under"] = False
    for p in PLAT:
        i = T.platform == p; hi, lo = T.loc[i, "res"].quantile(.95), T.loc[i, "res"].quantile(.05)
        T.loc[i, "over"] = T.loc[i, "res"] >= hi; T.loc[i, "under"] = T.loc[i, "res"] <= lo
    T["out"] = T.over | T.under
    T["top10"] = T.decile == 10
    T["flag5"] = T.combined_rank >= .95
    rng = np.random.default_rng(SEED); out = {"posts": int(len(T))}

    def block(df):
        r = {"n": int(len(df))}
        for nm, col in [("outliers", "out"), ("over (top 5% residual)", "over"), ("under (bottom 5% residual)", "under")]:
            a, b = df[df[col]], df[~df[col]]
            r[nm] = {"n": int(len(a)), "mean combined percentile": r3(a.combined_rank.mean()), "mean combined percentile, rest": r3(b.combined_rank.mean()),
                     "share in top 10% of score": r3(a.top10.mean()), "share in top 10%, rest": r3(b.top10.mean()),
                     "AUC of score for the outlier label": auc_ci(df[col].astype(int).values, df.combined.values, rng, 300)}
        return r
    out["all"] = block(T)
    out["per platform"] = {PNAME[p]: block(T[T.platform == p]) for p in PLAT}
    # labelled posts that are outliers
    L = T.merge(lab[["mention_id", "label_final"]], on="mention_id")
    lo = {"labelled posts on TikTok / YouTube / X": int(len(L))}
    for nm, col in [("outliers", "out"), ("over", "over"), ("under", "under")]:
        a, b = L[L[col]], L[~L[col]]
        lo[nm] = {"labelled": int(len(a)), "not_about_brand": int((a.label_final == "not_about_brand").sum()), "about_brand": int((a.label_final == "about_brand").sum()),
                  "unclear": int((a.label_final == "unclear").sum()),
                  "share not_about_brand": r3((a.label_final == "not_about_brand").mean()) if len(a) else None,
                  "share not_about_brand, other labelled": r3((b.label_final == "not_about_brand").mean())}
    out["labelled outliers"] = lo
    # views lost when flagged posts are dropped, and weight of the over-performers
    v = T.views.astype(float); res = {}
    for nm, fl in [("flag top 5% score", T.flag5), ("flag top 10% score", T.top10)]:
        r = {"flagged posts": int(fl.sum()), "share of views lost": r3(v[fl].sum() / v.sum()), "share of posts": r3(fl.mean()),
             "per platform share of views lost": {PNAME[p]: r3(v[fl & (T.platform == p)].sum() / v[T.platform == p].sum()) for p in PLAT}}
        keep = ~fl
        r["share of top-5% over-performers flagged"] = r3(fl[T.over].mean())
        r["share of under-performers flagged"] = r3(fl[T.under].mean())
        r["over share among kept posts (was 0.05 per platform)"] = r3(T[keep].over.mean())
        r["under share among kept posts"] = r3(T[keep].under.mean())
        r["share of over-performers' views lost"] = r3(v[fl & T.over].sum() / v[T.over].sum())
        # drop in the "top 5% of residual" rate after re-cutting the threshold on the kept posts
        k5 = []
        for p in PLAT:
            i = keep & (T.platform == p); k5.append(T.loc[i, "res"].quantile(.95))
        r["over cut after removal (log10 residual) vs before"] = {PNAME[p]: [r3(T.loc[T.platform == p, "res"].quantile(.95)), r3(k)] for p, k in zip(PLAT, k5)}
        res[nm] = r
    out["drop flagged"] = res
    # examples: labelled not_about_brand with the largest |residual|
    ex = L[L.label_final == "not_about_brand"].assign(a=lambda x: x.res.abs()).sort_values("a", ascending=False).head(12)
    ex = ex.merge(S[["mention_id", "text"]], on="mention_id")
    out["example candidates (ids only, read by hand; text is not stored)"] = [
        {"sample mention": str(r.mention_id), "platform": PNAME[r.platform], "brand": r.brand, "views": int(r.views),
         "estimate": int(round(10 ** r.pred)), "x factor": r3(10 ** r.res)} for r in ex.itertuples()]
    return out, ex


# ================= Task 2 =================
def ambiguity_features(raw, text, amb_yes):
    """Name ambiguity: share of OTHER brands' texts that contain the name as a token sequence (common word), name length,
    number of words, hand-made ambiguous flag."""
    norm_t = [" " + ob.norm(t) + " " for t in text]; feats = {}
    brands = raw.brand.values
    for b in raw.brand.unique():
        full, _, _ = ob.brand_forms(b)
        if not full:
            feats[b] = (0.0, 0, 0, 0); continue
        key = " " + " ".join(full) + " "
        other = [i for i in range(len(raw)) if brands[i] != b]
        samp = np.random.default_rng(SEED).choice(other, 2000, replace=False)
        share = np.mean([key in norm_t[i] for i in samp])
        feats[b] = (np.log10(share + 5e-4), len(" ".join(full)), len(full), int(b in amb_yes))
    return feats


def text_features(raw, text, lex_brand_in):
    rows = []
    for t, b in zip(text, raw.brand):
        tags = re.findall(r"#\w+", t); toks = t.split(); nt = max(len(toks), 1)
        _, joined, _ = ob.brand_forms(b)
        in_tag = any(ob.norm(x).replace(" ", "").startswith(joined) for x in tags) if joined else False
        plain = ob.brand_in_text(re.sub(r"#\w+", " ", t), b)
        asc = np.mean([c.isascii() for c in t]) if t else 1.0
        stop = np.mean([w.lower() in EN_STOP for w in toks]) if toks else 0.0
        rows.append([len(tags), len(tags) / nt, np.log1p(len(t)), np.log1p(len(toks)), float(in_tag and not plain), float(plain), asc, stop])
    return np.array(rows)


EN_STOP = set("the a an and or of to in on for with is are was were be this that it you i we they my your our at by from as not but have has".split())


def author_features(raw, sc):
    a = raw.author_name.fillna("").values; df = pd.DataFrame({"a": a, "brand": raw.brand.values, "c": sc.combined.values})
    df.loc[df.a == "", "a"] = [f"_none{i}" for i in range((df.a == "").sum())]
    nb = df.groupby("a").brand.nunique(); npst = df.groupby("a").size()
    s = df.groupby("a").c.transform("sum"); n = df.a.map(npst)
    loo = np.where(n > 1, (s - df.c) / np.maximum(n - 1, 1), np.nan)   # mean combined score of the author's OTHER posts
    return np.c_[np.log1p(df.a.map(nb).values), np.log1p(n.values), np.where(np.isnan(loo), df.c.mean(), loo), (n.values == 1).astype(float)]


def graph_grid(raw, E, text, ks, ress):
    """E5 and TF-IDF kNN graphs per platform; Louvain resolution x k grid; score = -log lift (as in E224)."""
    cols = {}
    for p in eu.PLATFORMS:
        m = (raw.platform == p).values; idx = np.flatnonzero(m)
        Z = np.asarray(E)[m]; Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
        sims = {"e5": Z @ Z.T, "tfidf": em.text_similarity(text[m].values)}
        for s in sims.values():
            np.fill_diagonal(s, 0.0)
        b = raw.brand.values[m]; base = pd.Series(b).value_counts(normalize=True); pb = pd.Series(b).map(base).values
        for gname, S in sims.items():
            for k in ks:
                A, _ = gl.knn_graph(S, k)
                for rs in ress:
                    comm, _ = gl.louvain(A, resolution=rs)
                    key = f"{gname} k={k} res={rs}"
                    cols.setdefault(key, np.full(len(raw), np.nan))[idx] = -ob._graph_homophily(A, comm, b, pb)
        print("grid platform done", p, flush=True)
    G = pd.DataFrame(cols)
    # rank within platform so platforms are comparable (as in E224)
    for c in G.columns:
        G[c] = G[c].groupby(raw.platform.values).rank(pct=True)
    return G


def label_prop_graph(raw, E, k=10):
    """Block-diagonal (per platform) e5 kNN graph, row-normalised, sparse."""
    n = len(raw); W = sparse.lil_matrix((n, n))
    blocks = []
    for p in eu.PLATFORMS:
        m = (raw.platform == p).values; idx = np.flatnonzero(m)
        Z = np.asarray(E)[m]; Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
        S = Z @ Z.T; np.fill_diagonal(S, 0.0); _, Wp = gl.knn_graph(S, k)
        blocks.append((idx, sparse.csr_matrix(Wp)))
    rows, cols, vals = [], [], []
    for idx, Wp in blocks:
        c = Wp.tocoo(); rows += list(idx[c.row]); cols += list(idx[c.col]); vals += list(c.data)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def propagate(W, y0, alpha=0.9, it=30):
    F = y0.copy()
    for _ in range(it):
        F = (1 - alpha) * y0 + alpha * (W @ F)
    return F


def ece(y, p, bins=10):
    q = pd.qcut(p, bins, duplicates="drop", labels=False); e = 0.0
    for b in np.unique(q):
        i = q == b; e += i.mean() * abs(y[i].mean() - p[i].mean())
    return e


def precision_top(y, s, q):
    n = max(int(round(len(y) * q)), 1); i = np.argsort(-s)[:n]
    return r3(y[i].mean())


def run_cv(Xfun, y, groups, nrep=REPEATS, fit=None):
    """Repeated brand-grouped CV. Xfun(tr_idx, te_idx) -> (Xtr, Xte) or a score vector for te. Returns the mean OOF score."""
    uniq = np.unique(groups); oofs = []
    for r in range(nrep):
        rng = np.random.default_rng(SEED + r); perm = rng.permutation(len(uniq)); gmap = dict(zip(uniq[perm], np.arange(len(uniq)) % 5))
        fold = np.array([gmap[g] for g in groups]); oof = np.full(len(y), np.nan)
        for k in range(5):
            tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            oof[te] = fit(tr, te)
        oofs.append(oof)
    return np.mean(oofs, axis=0), oofs


def task2(raw, sc, lab, S, E):
    text = (raw.title.fillna("") + " " + raw.content.fillna(""))
    amb = pd.read_csv(PROC / "brand_industry.csv"); amb_yes = set(amb.loc[amb.ambiguous_name == "yes", "brand"]) & set(raw.brand)
    pos = {m: i for i, m in enumerate(raw.mention_id)}
    L = lab[lab.label_final.isin(["about_brand", "not_about_brand"])].reset_index(drop=True)
    ix = np.array([pos[m] for m in L.mention_id]); y = (L.label_final == "not_about_brand").astype(int).values
    groups = raw.brand.values[ix]; n = len(y)
    print("labelled for CV", n, int(y.sum()), "brands", len(np.unique(groups)), flush=True)
    sc = sc.set_index("mention_id").loc[raw.mention_id].reset_index()
    P = sc[["lex_pct", "emb_pct", "graph_tfidf_pct", "graph_e5_pct", "clf_pct"]].fillna(.5).values
    comb = sc.combined.values
    X_stack = P[ix]
    feats = ambiguity_features(raw, text.values, amb_yes); X_amb_all = np.array([feats[b] for b in raw.brand]); X_amb = X_amb_all[ix]
    X_txt_all = text_features(raw, text.values, None); X_txt = X_txt_all[ix]
    X_aut_all = author_features(raw, sc); X_aut = X_aut_all[ix]
    print("grid", flush=True)
    G = graph_grid(raw, E, text, ks=[5, 10, 20, 40], ress=[0.5, 1.0, 2.0])
    gcols = list(G.columns); Gm = G.values[ix]
    W = label_prop_graph(raw, E, 10)
    out = {}; rng = np.random.default_rng(SEED)

    def lr_fit(X, C=1.0):
        def fit(tr, te):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            m = LogisticRegression(C=C, max_iter=1000).fit((X[tr] - mu) / sd, y[tr]); return m.predict_proba((X[te] - mu) / sd)[:, 1]
        return fit

    amb_flag = pd.Series(raw.brand.values[ix]).isin(amb_yes).values
    pop = pd.DataFrame({"decile": sc.decile.values, "amb": raw.brand.isin(amb_yes).values}); N = pop.groupby(["decile", "amb"]).size()
    cell = pd.DataFrame({"decile": sc.decile.values[ix], "amb": amb_flag}); nn = cell.groupby(["decile", "amb"]).size()
    w = np.array([N[(d, a)] / nn[(d, a)] for d, a in zip(cell.decile, cell.amb)])

    def report(name, oof, oofs, note=""):
        a, lo, hi = auc_ci(y, oof, rng)
        reps = [roc_auc_score(y, o) for o in oofs]
        out[name] = {"AUC": a, "CI low": lo, "CI high": hi, "AUC per repeat (min-max)": [r3(min(reps)), r3(max(reps))],
                     "precision top 5%": precision_top(y, oof, .05), "precision top 10%": precision_top(y, oof, .10),
                     "AUC design-weighted": r3(ob._w_auc(y, oof, w)), "exceeds 0.75": bool(a > 0.75), "CI low above 0.75": bool(lo > 0.75), "note": note}
        print(f"{name:55s} AUC {a} [{lo}, {hi}]  p5 {out[name]['precision top 5%']} p10 {out[name]['precision top 10%']}", flush=True)

    # reference: the E224 combined score (no training) and its sub-detectors
    report("0 reference: E224 combined (no training)", comb[ix], [comb[ix]], "same 287 posts as E224")
    # (a) stack
    oof, oofs = run_cv(None, y, groups, fit=lr_fit(X_stack)); report("a stack: logistic regression on 5 detector percentiles", oof, oofs)
    oof_a = oof
    # (b) ambiguity
    oof, oofs = run_cv(None, y, groups, fit=lr_fit(X_amb)); report("b ambiguity features alone (name frequency, length, words, hand flag)", oof, oofs)
    X = np.c_[X_stack, X_amb]; oof, oofs = run_cv(None, y, groups, fit=lr_fit(X)); report("b stack + ambiguity", oof, oofs)
    X_ab = X
    Xnf = np.c_[X_stack, X_amb[:, :3]]; oof, oofs = run_cv(None, y, groups, fit=lr_fit(Xnf)); report("b stack + ambiguity without the hand flag", oof, oofs, "name frequency, length, words only")
    # (c) Louvain resolution x k, chosen inside the training folds
    def fit_c(tr, te):
        a = [roc_auc_score(y[tr], Gm[tr, j]) if 0 < y[tr].sum() < len(tr) else .5 for j in range(Gm.shape[1])]
        j = int(np.argmax(a)); return Gm[te, j]
    oof, oofs = run_cv(None, y, groups, fit=fit_c); report("c graph grid, best config picked inside training folds", oof, oofs, "24 configs: 2 graphs x k 5-40 x resolution 0.5-2")
    out["c grid descriptive (optimistic: AUC of each config on all 287 labels, not used for selection)"] = {
        c: r3(roc_auc_score(y, Gm[:, j])) for j, c in enumerate(gcols)}
    # (d) author aggregation
    oof, oofs = run_cv(None, y, groups, fit=lr_fit(X_aut)); report("d author level alone (distinct brands, posts, mean score of other posts)", oof, oofs)
    X = np.c_[X_stack, X_aut]; oof, oofs = run_cv(None, y, groups, fit=lr_fit(X)); report("d stack + author level", oof, oofs)
    # (e) text features
    oof, oofs = run_cv(None, y, groups, fit=lr_fit(X_txt)); report("e text features alone (hashtags, length, name only in hashtag, language proxy)", oof, oofs)
    X = np.c_[X_stack, X_txt]; oof, oofs = run_cv(None, y, groups, fit=lr_fit(X)); report("e stack + text features", oof, oofs)
    # (f) label propagation from TRAINING labels only
    def fit_f(tr, te):
        y0 = np.zeros(len(raw)); y0[ix[tr]] = np.where(y[tr] == 1, 1.0, -1.0)
        F = propagate(W, y0); return F[ix[te]]
    oof, oofs = run_cv(None, y, groups, fit=fit_f); report("f label propagation on the e5 kNN graph (training labels only)", oof, oofs, "k=10, alpha 0.9, 30 iterations")
    Fcache = {}
    def fit_fs(tr, te):
        y0 = np.zeros(len(raw)); y0[ix[tr]] = np.where(y[tr] == 1, 1.0, -1.0); F = propagate(W, y0)
        # propagated score of training posts must exclude their own label to match test conditions: leave-self-out
        own = (1 - .9) * y0
        Xf = np.c_[X_stack, F[ix]]; Xf[tr, -1] = F[ix][tr] - own[ix[tr]]
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-9
        m = LogisticRegression(max_iter=1000).fit((Xf[tr] - mu) / sd, y[tr]); return m.predict_proba((Xf[te] - mu) / sd)[:, 1]
    oof, oofs = run_cv(None, y, groups, fit=fit_fs); report("f stack + propagated label score", oof, oofs, "propagated score of training posts has its own label removed")
    # all-features row
    X = np.c_[X_stack, X_amb, X_aut, X_txt]
    oof, oofs = run_cv(None, y, groups, fit=lr_fit(X, C=0.1)); report("h all features (a + b + d + e), L2 C=0.1", oof, oofs)
    oof_all, oofs_all = oof, oofs
    # (g) calibration of the best-practice stack+ambiguity (b) model: cross-fitted Platt / isotonic on OOF scores
    base = out["b stack + ambiguity"]; oofb, oofsb = run_cv(None, y, groups, fit=lr_fit(X_ab))
    cal = {"model": "b stack + ambiguity", "prevalence": r3(y.mean())}
    cal["Brier raw"] = r3(brier_score_loss(y, oofb)); cal["ECE raw"] = r3(ece(y, oofb))
    cal["Brier constant prevalence"] = r3(brier_score_loss(y, np.full(n, y.mean())))
    # second-level calibration: brand-grouped CV on the OOF scores
    def cal_fit(kind):
        def fit(tr, te):
            if kind == "platt":
                m = LogisticRegression().fit(np.log(oofb[tr] / (1 - oofb[tr] + 1e-9) + 1e-9).reshape(-1, 1), y[tr])
                return m.predict_proba(np.log(oofb[te] / (1 - oofb[te] + 1e-9) + 1e-9).reshape(-1, 1))[:, 1]
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oofb[tr], y[tr]); return ir.predict(oofb[te])
        return fit
    for kind in ["platt", "isotonic"]:
        o, _ = run_cv(None, y, groups, nrep=3, fit=cal_fit(kind))
        cal[f"Brier {kind}"] = r3(brier_score_loss(y, o)); cal[f"ECE {kind}"] = r3(ece(y, o)); cal[f"AUC {kind}"] = r3(roc_auc_score(y, o))
    q = pd.qcut(oofb, 10, duplicates="drop", labels=False)
    cal["reliability (decile of predicted P: mean predicted, observed share, n)"] = [
        [int(b), r3(oofb[q == b].mean()), r3(y[q == b].mean()), int((q == b).sum())] for b in np.unique(q)]
    cal["note"] = "calibration is on the enriched sample (77% off-brand), not the population; AUC is unchanged by a monotone map"
    report("g calibrated (Platt) stack + ambiguity", cal_fit("platt") and run_cv(None, y, groups, nrep=3, fit=cal_fit("platt"))[0], [run_cv(None, y, groups, nrep=1, fit=cal_fit("platt"))[0]], "monotone map: AUC equals the uncalibrated one up to fold effects")
    out["g calibration"] = cal
    print(json.dumps({k: cal[k] for k in cal if k != "reliability (decile of predicted P: mean predicted, observed share, n)"}), flush=True)
    out["_n"] = {"labelled used": n, "not_about_brand": int(y.sum()), "brands": int(len(np.unique(groups))), "repeats": REPEATS}
    return out


def main():
    raw = eu.load(ROOT).astype({"mention_id": str})
    sc = pd.read_csv(PROC / "offbrand_scores.csv", dtype={"mention_id": str})
    lab = pd.read_csv(PROC / "offbrand_labels.csv", dtype={"mention_id": str})
    S = pd.read_csv(PROC / "offbrand_sample.csv", dtype={"mention_id": str})
    E = pd.read_pickle(PROC / "embeddings_e5_cache.pkl"); assert len(E) == len(raw)
    t1, ex = task1(raw, sc, lab, S)
    print(json.dumps({k: t1[k] for k in ["labelled outliers", "drop flagged"]}, indent=1), flush=True)
    # print example texts to the terminal only (for hand reading; never stored)
    for r in ex.head(12).itertuples():
        print("EX", r.platform, r.brand, int(r.views), int(round(10 ** r.pred)), "|", r.text[:260].replace("\n", " "), flush=True)
    t2 = task2(raw, sc, lab, S, E)
    num = {"experiment": "E234", "date": "2026-11-09",
           "design": {"outliers": "top / bottom 5% of the out-of-fold residual per platform (results_preds.csv)",
                      "cv": f"brand-grouped 5-fold, {REPEATS} random group splits, AUC CI = post bootstrap ({NBOOT}) of the repeat-averaged out-of-fold score",
                      "precision top q": "share of not_about_brand among the top q of the OUT-OF-FOLD score within the 287 labelled posts (enriched sample), "
                                         "not the population top 5% of E224",
                      "labels": "AI labels of E224 (unclear excluded)"},
           "task1": t1, "task2": t2}
    (PROC / "offbrand_more_numbers.json").write_text(json.dumps(num, indent=1, ensure_ascii=False))
    print("written", flush=True)


if __name__ == "__main__":
    main()
