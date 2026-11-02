"""Paired comparison, engagement-as-target and brand-graph features (notebooks/models/04_extra_features.ipynb)."""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import extra_models as em, modeling as mo, eda_utils as eu
from validation import mae_ci

SEED = 42


def paired_diff(y, p_base, p_new, brands, B=1000):
    """Mean of |err_base| - |err_new| (positive = new is better) with brand-cluster bootstrap 95% CI."""
    d = np.abs(y - p_base) - np.abs(y - p_new); rng = np.random.default_rng(SEED); ub = np.unique(brands); idx = {b: np.where(brands == b)[0] for b in ub}
    boot = [np.concatenate([d[idx[b]] for b in rng.choice(ub, len(ub))]).mean() for _ in range(B)]
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def group_ladder(f, meta, G, plat, make=None):
    """MAE of one model on engagement features + each extra group, and the paired gain over the base."""
    make = make or em.models()["extra trees"]; m = (meta.platform == plat).values; M = meta[m].reset_index(drop=True); y, g = M.y, M.brand
    base = f[m][em.ENG].reset_index(drop=True); R = {k: v[m].reset_index(drop=True) for k, v in G.items()}
    sets = {"base: followers + engagement": base} | {f"+ {k}": pd.concat([base, v], axis=1) for k, v in R.items()} | {"+ all groups": pd.concat([base, *R.values()], axis=1)}
    pb = em.cv_predict(make, base, y, g); rows = []
    for name, X in sets.items():
        p = pb if name.startswith("base") else em.cv_predict(make, X, y, g); mae, lo, hi = mae_ci((y - p).values, g.values); gain = paired_diff(y.values, pb, p, g.values)
        rows.append({"platform": eu.NAMES[plat], "model": name, "MAE": mae, "CI low": lo, "CI high": hi, "gain vs base": gain[0], "gain low": gain[1], "gain high": gain[2]})
    return pd.DataFrame(rows)


# ---------- engagement and followers as targets ----------
def targets(raw):
    return {"likes": np.log1p(raw.likes), "comments": np.log1p(raw.comments), "shares": np.log1p(raw.shares), "followers": np.log10(raw.author_followers.where(raw.author_followers > 0))}


def predict_target(f, meta, raw, G, plat, name):
    """Out-of-fold MAE and R2 for predicting `name` (log scale) from everything except the target itself and views."""
    m = (meta.platform == plat).values; y = targets(raw.loc[f.index])[name][m].reset_index(drop=True); ok = y.notna().values
    if ok.sum() < 300: return None
    own = {"likes": "log_likes", "comments": "log_comments", "shares": "log_shares", "followers": "log_followers"}[name]
    cols = [c for c in f.columns if c not in (own, f"na_{name}") and c not in ("type_code", "sentiment_code", "month", "metrics_age_days")]
    X = pd.concat([f[m][cols].reset_index(drop=True)] + [v[m].reset_index(drop=True) for k, v in G.items() if k != "engagement ratios"], axis=1)[ok].reset_index(drop=True)
    yy, g = y[ok].reset_index(drop=True), meta.brand[m].reset_index(drop=True)[ok].reset_index(drop=True)
    p = em.cv_predict(em.models()["extra trees"], X, yy, g); base = np.full(len(yy), np.nan)
    for tr, te in GroupKFold(5).split(X, yy, g): base[te] = yy.iloc[tr].median()
    mae, lo, hi = mae_ci((yy - p).values, g.values); bmae = np.abs(yy - base).mean()
    return {"platform": eu.NAMES[plat], "target": name, "n": int(ok.sum()), "median baseline MAE": bmae, "model MAE": mae, "CI low": lo, "CI high": hi, "R2": 1 - ((yy - p) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()}


# ---------- brand graph ----------
def brand_similarity(meta, texts):
    """Cosine similarity between brands from TF-IDF of all their mentions' text (label-free)."""
    docs = pd.Series(texts, index=meta.index).groupby(meta.brand).apply(lambda s: " ".join(s)); X = TfidfVectorizer(min_df=2, max_features=40000, sublinear_tf=True, stop_words="english").fit_transform(docs)
    S = (X @ X.T).toarray(); np.fill_diagonal(S, 0); return pd.DataFrame(S, index=docs.index, columns=docs.index)


def brand_neighbour_extra(M, X, y, S, k=5):
    """extra(tr, te): mean engagement-adjusted residual of the k most similar TRAIN brands (brand graph), plus their mean log views.
    Residuals come from a ridge on followers + engagement fitted on the training fold."""
    brands = M.brand.values; yv = y.values
    def extra(tr, te):
        rid = Ridge(1.0).fit(X.iloc[tr].fillna(0), yv[tr]); res = yv[tr] - rid.predict(X.iloc[tr].fillna(0))
        tb = pd.DataFrame({"b": brands[tr], "res": res, "y": yv[tr]}).groupby("b").mean(); sims = S.reindex(index=S.index, columns=tb.index)
        def feat(rows):
            out = []
            for b in brands[rows]:
                row = sims.loc[b].copy()
                if b in row.index: row[b] = 0
                top = row.nlargest(k); w = top.values; out.append([(w * tb.res.loc[top.index].values).sum() / max(w.sum(), 1e-9), (w * tb.y.loc[top.index].values).sum() / max(w.sum(), 1e-9), w.mean()])
            return pd.DataFrame(out, columns=["nbr_resid", "nbr_logviews", "nbr_sim"])
        return feat(tr), feat(te)
    return extra


def brand_graph_gain(f, meta, raw, plat, make=None):
    make = make or em.models()["extra trees"]; m = (meta.platform == plat).values; M = meta[m].reset_index(drop=True); y, g = M.y, M.brand
    base = f[m][em.ENG].reset_index(drop=True); t = (raw.loc[f.index].title.fillna("") + " " + raw.loc[f.index].content.fillna(""))[m].values
    S = brand_similarity(M, t); pb = em.cv_predict(make, base, y, g); pn = em.cv_predict(make, base, y, g, brand_neighbour_extra(M, base, y, S))
    gain = paired_diff(y.values, pb, pn, g.values); return {"platform": eu.NAMES[plat], "base MAE": np.abs(y - pb).mean(), "with brand-graph MAE": np.abs(y - pn).mean(), "gain": gain[0], "gain low": gain[1], "gain high": gain[2]}


def plot_gains(lad, path=None, lang="en"):
    """Paired MAE gain over the engagement-only base for each extra feature group (positive = better), brand-cluster 95% CI."""
    import matplotlib.pyplot as plt
    L = lad[~lad.model.str.startswith("base")]; groups = list(dict.fromkeys(L.model)); fig, ax = plt.subplots(figsize=(10, 4.6)); w = .26
    for j, p in enumerate(mo.PLATS):
        r = L[L.platform == eu.NAMES[p]].set_index("model").reindex(groups)
        ax.barh(np.arange(len(groups)) + (1 - j) * w, r["gain vs base"], w, color=eu.PCOL[p], label=eu.NAMES[p], xerr=[r["gain vs base"] - r["gain low"], r["gain high"] - r["gain vs base"]], ecolor="black", capsize=2)
    ax.axvline(0, color="grey", lw=1); ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups); ax.invert_yaxis()
    ax.set_xlabel("MAE gain over followers + engagement, log10 views (positive = better)" if lang == "en" else "zysk MAE względem followersi + reakcji, log10 wyświetleń (dodatni = lepiej)"); ax.set_ylabel("added feature group" if lang == "en" else "dodana grupa cech")
    ax.set_title("What extra columns add, paired and brand-grouped" if lang == "en" else "Co dodają dodatkowe kolumny (sparowane, CV po markach)", fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, title="platform" if lang == "en" else "platforma"); plt.tight_layout()
    if path: fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()


def ablation(f, meta, G, plat, make=None):
    """Leave-one-group-out from the full model (followers, engagement, missing flags + all extra groups).
    `loss` = MAE without the group minus MAE with the full model, paired (positive = the group helps), brand-cluster 95% CI."""
    make = make or em.models()["extra trees"]; m = (meta.platform == plat).values; M = meta[m].reset_index(drop=True); y, g = M.y, M.brand
    parts = {"followers": f[m][mo.GROUPS["followers"]], "engagement": f[m][mo.GROUPS["engagement"]], "missing flags": f[m][mo.GROUPS["missing flags"]]} | {k: v[m] for k, v in G.items() if k != "engagement ratios"}   # ratios contain likes and followers, so they would leak the removed group back in
    parts = {k: v.reset_index(drop=True) for k, v in parts.items()}; full = pd.concat(parts.values(), axis=1); pf = em.cv_predict(make, full, y, g); rows = []
    for k in parts:
        p = em.cv_predict(make, pd.concat([v for kk, v in parts.items() if kk != k], axis=1), y, g); d = paired_diff(y.values, p, pf, g.values)
        rows.append({"platform": eu.NAMES[plat], "removed group": k, "MAE without": np.abs(y - p).mean(), "loss": d[0], "loss low": d[1], "loss high": d[2]})
    return pd.DataFrame(rows).assign(**{"MAE full": np.abs(y - pf).mean()})


def plot_ablation(ab, path=None, lang="en"):
    import matplotlib.pyplot as plt
    groups = list(dict.fromkeys(ab["removed group"])); fig, ax = plt.subplots(figsize=(10, 5)); w = .26
    for j, p in enumerate(mo.PLATS):
        r = ab[ab.platform == eu.NAMES[p]].set_index("removed group").reindex(groups)
        ax.barh(np.arange(len(groups)) + (1 - j) * w, r["loss"], w, color=eu.PCOL[p], label=eu.NAMES[p], xerr=[r["loss"] - r["loss low"], r["loss high"] - r["loss"]], ecolor="black", capsize=2)
    ax.axvline(0, color="grey", lw=1); ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups); ax.invert_yaxis(); ax.set_xscale("symlog", linthresh=.01); ax.set_xticks([-.01, 0, .01, .03, .1, .3]); ax.set_xticklabels(["-0.01", "0", "0.01", "0.03", "0.1", "0.3"])
    ax.set_xlabel("MAE increase when the group is removed from the full model (symlog, log10 views)" if lang == "en" else "wzrost MAE po usunięciu grupy z pełnego modelu (symlog, log10 wyświetleń)")
    ax.set_ylabel("removed feature group" if lang == "en" else "usunięta grupa cech"); ax.set_title("Ablation: leave one group out" if lang == "en" else "Ablacja: usuwamy po jednej grupie", fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, title="platform" if lang == "en" else "platforma"); plt.tight_layout()
    if path: fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
