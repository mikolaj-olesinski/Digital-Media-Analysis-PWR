"""Two-stage (stacked) models for the extremes (notebooks/models/13_stacked_models.ipynb, plan item G).
Target log10(views) on TikTok, YouTube and X. Outer validation: 5-fold GroupKFold by brand, so a brand is never in train and test.

Leakage rule for every approach: a first-stage output that is fed to a second stage is produced out-of-fold INSIDE the outer training fold
(inner 5-fold GroupKFold by brand on the training rows). The second stage is trained on those out-of-fold values and sees, for the test rows,
the output of a first stage fitted on the whole training fold. Classes defined from views (tiers) use thresholds of the training fold only.
Label-free text features (TF-IDF SVD, kNN text graph and its Louvain communities) use no views and are built on all mentions of a platform, as elsewhere in the repo."""
import json, pickle, warnings
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
import eda_utils as eu, extra_features as ef, extra_models as em, graph_lab as gl, label_ablation as la, modeling as mo, new_ideas as ni
from validation import mae_ci

SEED, NJOBS, SHRINK = 42, 4, 10.0
TUNED = {"tiktok": .27, "youtube": .45, "twitter": .47}          # tuned reference quoted by the team (notebook 07)
APPROACHES = ["base", "base (MAE loss)", "A1 tier probabilities as features", "A2 tier-routed, soft weights", "B1 residual on text", "B2 residual on our labels",
              "B3 residual on text + labels", "C1 hurdle: zero-reaction branch", "C2 hurdle + text in that branch", "D1 promo probability from text",
              "D2 true promo label (oracle)", "E1 community shrunk correction"]


def reg(loss="RMSE"):
    return CatBoostRegressor(iterations=300, loss_function=loss, random_seed=SEED, thread_count=NJOBS, verbose=0, allow_writing_files=False)


def clf():
    return CatBoostClassifier(iterations=300, loss_function="MultiClass", random_seed=SEED, thread_count=NJOBS, verbose=0, allow_writing_files=False)


def proba(model, X, k):
    """Class probabilities with all k columns, even if a class is missing from the fitting data."""
    out = np.zeros((len(X), k)); out[:, model.classes_.astype(int)] = model.predict_proba(X); return out


def prepare(root, f, meta, G, lab, plat):
    m = (meta.platform == plat).values; M = meta[m].reset_index(drop=True); X = f[m][em.ENG].fillna(0).reset_index(drop=True)
    raw = eu.load(root).astype({"mention_id": str}).loc[f.index][m].reset_index(drop=True); text = ef.full_text(raw)
    svd = ef.tfidf_svd(text, 30).reset_index(drop=True)
    labs = pd.concat([pd.get_dummies(lab.type[m].reset_index(drop=True)).astype(int).add_prefix("type_"), pd.get_dummies(lab.sentiment[m].reset_index(drop=True)).astype(int).add_prefix("sent_")], axis=1)
    S = em.text_similarity(M.text.values); A, _ = gl.knn_graph(S); comm, _ = gl.louvain(A)
    zr = (X[["log_likes", "log_comments", "log_shares"]].sum(axis=1) == 0).values
    promo = (lab.type[m].reset_index(drop=True) == "promo").astype(int).values
    return dict(X=X, M=M, y=M.y.values, g=M.brand.values, svd=svd, labs=labs, comm=comm, zr=zr, promo=promo, text=text.values)


def promo_text_model():
    from sklearn.pipeline import make_pipeline
    return make_pipeline(TfidfVectorizer(min_df=2, max_features=30000, sublinear_tf=True, ngram_range=(1, 2)), LogisticRegression(C=1.0, max_iter=1000))


def run_platform(root, f, meta, G, lab, plat):
    P = prepare(root, f, meta, G, lab, plat); X, y, g, n = P["X"], P["y"], P["g"], len(P["y"]); preds = {a: np.full(n, np.nan) for a in APPROACHES}
    pA, tierA, pD = np.full((n, 3), np.nan), np.full(n, -1), np.full(n, np.nan)
    for fold, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
        ytr, Xtr, Xte = y[tr], X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
        inner = [(a, b) for a, b in GroupKFold(5).split(Xtr, ytr, g[tr])]
        base = reg().fit(Xtr, ytr); pb = base.predict(Xte); preds["base"][te] = pb
        preds["base (MAE loss)"][te] = reg("MAE").fit(Xtr, ytr).predict(Xte)
        oof = np.zeros(len(tr))                                                                   # stage-1 out-of-fold predictions inside the training fold
        for a, b in inner: oof[b] = reg().fit(Xtr.iloc[a], ytr[a]).predict(Xtr.iloc[b])
        resid = ytr - oof
        # ---- A: tiers from training-fold quantiles (bottom 20%, middle 60%, top 20%)
        q = np.quantile(ytr, [.2, .8]); ttr, tte = np.digitize(ytr, q), np.digitize(y[te], q); tierA[te] = tte
        poof = np.zeros((len(tr), 3))
        for a, b in inner: poof[b] = proba(clf().fit(Xtr.iloc[a], ttr[a]), Xtr.iloc[b], 3)
        pte = proba(clf().fit(Xtr, ttr), Xte, 3); pA[te] = pte
        cols = lambda P_: pd.DataFrame(P_, columns=["p_low", "p_mid", "p_high"])
        preds["A1 tier probabilities as features"][te] = reg().fit(pd.concat([Xtr, cols(poof)], axis=1), ytr).predict(pd.concat([Xte, cols(pte)], axis=1))
        preds["A2 tier-routed, soft weights"][te] = sum(pte[:, t] * reg().fit(Xtr, ytr, sample_weight=poof[:, t] + 1e-6).predict(Xte) for t in range(3))
        # ---- B: second model predicts the out-of-fold residual from features the first model did not use
        for name, blocks in [("B1 residual on text", [P["svd"]]), ("B2 residual on our labels", [P["labs"]]), ("B3 residual on text + labels", [P["svd"], P["labs"]])]:
            Z = pd.concat(blocks, axis=1).reset_index(drop=True); Ztr = pd.concat([pd.Series(oof, name="stage1").to_frame(), Z.iloc[tr].reset_index(drop=True)], axis=1)
            Zte = pd.concat([pd.Series(pb, name="stage1").to_frame(), Z.iloc[te].reset_index(drop=True)], axis=1)
            preds[name][te] = pb + reg().fit(Ztr, resid).predict(Zte)
        # ---- C: separate branch for posts with no reactions
        zr_tr, zr_te = P["zr"][tr], P["zr"][te]; Xs = P["svd"]
        for name, extra in [("C1 hurdle: zero-reaction branch", None), ("C2 hurdle + text in that branch", Xs)]:
            out = pb.copy()
            for flag in (True, False):
                a, b = np.where(zr_tr == flag)[0], np.where(zr_te == flag)[0]
                if len(b) == 0: continue
                if len(a) < 40: continue                                                           # too few rows: keep the shared model
                Ftr, Fte = Xtr.iloc[a], Xte.iloc[b]
                if extra is not None and flag: Ftr = pd.concat([Ftr.reset_index(drop=True), extra.iloc[tr[a]].reset_index(drop=True)], axis=1); Fte = pd.concat([Fte.reset_index(drop=True), extra.iloc[te[b]].reset_index(drop=True)], axis=1)
                out[b] = reg().fit(Ftr, ytr[a]).predict(Fte)
            preds[name][te] = out
        # ---- D: promo probability from text (no paid labels at run time), plus the oracle with the true label
        ptr = P["promo"][tr]; txt = P["text"]; poof_d = np.zeros(len(tr))
        for a, b in inner:
            if ptr[a].min() != ptr[a].max(): poof_d[b] = promo_text_model().fit(txt[tr[a]], ptr[a]).predict_proba(txt[tr[b]])[:, 1]
        pte_d = promo_text_model().fit(txt[tr], ptr).predict_proba(txt[te])[:, 1]; pD[te] = pte_d
        preds["D1 promo probability from text"][te] = reg().fit(Xtr.assign(p_promo=poof_d), ytr).predict(Xte.assign(p_promo=pte_d))
        preds["D2 true promo label (oracle)"][te] = reg().fit(Xtr.assign(p_promo=ptr), ytr).predict(Xte.assign(p_promo=P["promo"][te]))
        # ---- E: label-free community, shrunk mean of stage-1 out-of-fold residuals
        c = P["comm"]; df = pd.DataFrame({"c": c[tr], "r": resid}).groupby("c").r.agg(["sum", "count"]); corr = df["sum"] / (df["count"] + SHRINK)
        preds["E1 community shrunk correction"][te] = pb + pd.Series(c[te]).map(corr).fillna(0).values
        print(plat, "fold", fold, "done", flush=True)
    n_comm = len(np.unique(P["comm"]))
    return dict(y=y, g=g, preds=preds, pA=pA, tierA=tierA, pD=pD, promo=P["promo"], zr=P["zr"], n_comm=n_comm, comm_sizes=np.bincount(P["comm"]).tolist())


def run_all(root, cache):
    warnings.filterwarnings("ignore")
    if cache.exists(): return pickle.load(open(cache, "rb"))
    f, meta = em.load(root); L, lab = la.label_frames(root, f.index); out = {}
    for plat in mo.PLATS: out[plat] = run_platform(root, f, meta, None, lab, plat)
    pickle.dump(out, open(cache, "wb")); return out


# ---------- summaries ----------
def fifth_index(y):
    return np.digitize(y, np.quantile(y, [.2, .4, .6, .8]))


def summarize(R):
    """Overall MAE with CI, paired gain over base with CI, and the same on the lowest fifth, top fifth and zero-reaction posts (fifths of true views)."""
    rows = []
    for plat, r in R.items():
        y, g, zr = r["y"], r["g"], r["zr"]; fi = fifth_index(y); base = r["preds"]["base"]
        for a in APPROACHES:
            p = r["preds"][a]; mae, lo, hi = mae_ci(y - p, g); gn = ni.paired_diff(y, base, p, g)
            row = {"platform": eu.NAMES[plat], "key": plat, "approach": a, "MAE": mae, "low": lo, "high": hi, "gain": gn[0], "gain low": gn[1], "gain high": gn[2], "pred sd": float(p.std()), "true sd": float(y.std())}
            for name, mask in [("lowest fifth", fi == 0), ("top fifth", fi == 4), ("zero reactions", zr)]:
                if mask.sum() < 30: continue
                gd = ni.paired_diff(y[mask], base[mask], p[mask], g[mask]); row.update({f"{name} MAE": float(np.abs(y[mask] - p[mask]).mean()), f"{name} gain": gd[0], f"{name} gain low": gd[1], f"{name} gain high": gd[2], f"{name} bias": float((p[mask] - y[mask]).mean())})
            for k in range(5): row[f"fifth{k + 1} MAE"] = float(np.abs(y[fi == k] - p[fi == k]).mean())
            row["middle MAE"] = float(np.abs(y[fi == 2] - p[fi == 2]).mean()); row["n zero reactions"] = int(zr.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def stage_quality(R):
    """First-stage quality, pooled out-of-fold: tier classifier (macro F1, macro one-vs-rest AUC) and promo-from-text classifier (AUC, average precision, F1 at 0.5)."""
    rows = []
    for plat, r in R.items():
        t, P = r["tierA"], r["pA"]; pred = P.argmax(1); auc = roc_auc_score(t, P, multi_class="ovr", average="macro")
        rows.append({"platform": eu.NAMES[plat], "stage": "A: tier (low / middle / top)", "macro F1": f1_score(t, pred, average="macro"), "AUC": auc, "chance F1": f1_score(t, np.random.default_rng(SEED).choice(3, len(t), p=np.bincount(t) / len(t)), average="macro"), "n": len(t)})
        pr, pd_ = r["promo"], r["pD"]
        rows.append({"platform": eu.NAMES[plat], "stage": "D: promo from text", "macro F1": f1_score(pr, (pd_ > .5).astype(int), average="macro"), "AUC": roc_auc_score(pr, pd_), "AP": average_precision_score(pr, pd_), "prevalence": float(pr.mean()), "n": len(pr)})
    return pd.DataFrame(rows)


# ---------- figures ----------
TXT = {"en": {"mae": "Two-stage models: overall error", "x": "MAE of log10(views), brand-grouped CV", "gain": "Paired gain over the CatBoost base (positive = better)", "gx": "gain in MAE of log10(views), 95% CI",
              "seg": ["all posts", "lowest fifth", "top fifth", "zero reactions"], "tuned": "tuned reference", "q": "First-stage quality", "qy": "score (pooled out-of-fold)", "approach": "approach"},
       "pl": {"mae": "Modele dwuetapowe: błąd ogólny", "x": "MAE log10(wyświetleń), CV po markach", "gain": "Sparowany zysk względem bazy CatBoost (dodatni = lepiej)", "gx": "zysk MAE log10(wyświetleń), 95% PU",
              "seg": ["wszystkie posty", "najniższa piąta", "najwyższa piąta", "zero reakcji"], "tuned": "wartość po strojeniu", "q": "Jakość pierwszego etapu", "qy": "wynik (zagregowany, poza próbą)", "approach": "podejście"}}


def plot_mae(S, lang="en", path=None):
    import matplotlib.pyplot as plt
    T = TXT[lang]; fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    for a, p in zip(ax, mo.PLATS):
        r = S[S.key == p].set_index("approach").reindex(APPROACHES).iloc[::-1]; yy = np.arange(len(r))
        a.barh(yy, r.MAE, xerr=[r.MAE - r.low, r.high - r.MAE], color=eu.PCOL[p], ecolor="black", capsize=2); a.axvline(TUNED[p], color="grey", ls="--", lw=1, label=T["tuned"])
        a.set_yticks(yy); a.set_yticklabels(r.index); a.set_xlim(r.low.min() - .05, r.high.max() + .03); a.set_xlabel(T["x"]); a.set_title(eu.NAMES[p], fontweight="bold")
    ax[0].set_ylabel(T["approach"]); ax[-1].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False); fig.suptitle(T["mae"], fontweight="bold"); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()


def plot_gain(S, lang="en", path=None):
    import matplotlib.pyplot as plt
    T = TXT[lang]; segs = [("", T["seg"][0]), ("lowest fifth ", T["seg"][1]), ("top fifth ", T["seg"][2]), ("zero reactions ", T["seg"][3])]; apps = APPROACHES[2:]
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for a, p in zip(ax, mo.PLATS):
        r = S[S.key == p].set_index("approach").reindex(apps).iloc[::-1]; yy = np.arange(len(r)); sh = [-.27, -.09, .09, .27]; cl = [eu.PCOL[p], "#555555", "#999999", "#cccccc"]
        for (pre, lab), s, c in zip(segs, sh, cl):
            gcol = "gain" if pre == "" else f"{pre}gain"; lo, hi = ("gain low", "gain high") if pre == "" else (f"{pre}gain low", f"{pre}gain high")
            a.errorbar(r[gcol], yy + s, xerr=[(r[gcol] - r[lo]).clip(lower=0), (r[hi] - r[gcol]).clip(lower=0)], fmt="o", ms=4, color=c, capsize=2, label=lab)
        a.axvline(0, color="black", lw=.8); a.set_yticks(yy); a.set_yticklabels(r.index); a.set_xlabel(T["gx"]); a.set_title(eu.NAMES[p], fontweight="bold"); a.grid(axis="x", alpha=.25)
    ax[0].set_ylabel(T["approach"]); ax[-1].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False); fig.suptitle(T["gain"], fontweight="bold"); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()


def plot_stage(Q, lang="en", path=None):
    import matplotlib.pyplot as plt
    T = TXT[lang]; stages = Q.stage.unique(); fig, ax = plt.subplots(1, len(stages), figsize=(5.6 * len(stages), 4)); w = .35
    for a, st in zip(ax, stages):
        q = Q[Q.stage == st].set_index("platform").reindex([eu.NAMES[p] for p in mo.PLATS]); x = np.arange(3)
        a.bar(x - w / 2, q["macro F1"], w, color=[eu.PCOL[p] for p in mo.PLATS], alpha=.55, label="macro F1"); a.bar(x + w / 2, q.AUC, w, color=[eu.PCOL[p] for p in mo.PLATS], label="AUC")
        for i, (u, v) in enumerate(zip(q["macro F1"], q.AUC)): a.text(i - w / 2, u + .01, f"{u:.2f}", ha="center", fontsize=8); a.text(i + w / 2, v + .01, f"{v:.2f}", ha="center", fontsize=8)
        a.set_xticks(x); a.set_xticklabels(q.index); a.set_ylim(0, 1.05); a.set_ylabel(T["qy"]); a.set_xlabel("platform" if lang == "en" else "platforma"); a.set_title(st, fontweight="bold"); a.legend(frameon=False, loc="lower right")
    fig.suptitle(T["q"], fontweight="bold"); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()
