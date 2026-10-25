"""First model ladder for views on TikTok, YouTube and X (notebooks/models/01_first_models.ipynb).
Target: log10(view_count). Validation: 5-fold grouped by brand, so a brand never sits in train and test. Errors are MAE on log10 with a brand-cluster bootstrap CI."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import eda_utils as eu
from validation import grouped_cv_predict as _cv, mae_ci

SEED = 42
PLATS = ["tiktok", "youtube", "twitter"]
GROUPS = {   # cumulative feature groups, in the order they are added
    "followers": ["log_followers"],
    "engagement": ["log_likes", "log_comments", "log_shares"],
    "missing flags": ["na_likes", "na_comments", "na_shares", "na_followers"],
    "text": ["english", "text_len"],
    "our labels": ["type_code", "sentiment_code"],
    "time": ["month", "metrics_age_days"],
}
TYPE_CODE = {"organic": 0, "promo": 1, "spam": 2, "unclear": 3}
SENT_CODE = {"negative": -1, "neutral": 0, "positive": 1}


def load_features(root):
    d = eu.load(root).rename(columns={"sentiment": "sentiment_b24"}).astype({"mention_id": str})
    lab = pd.read_csv(root / "data" / "processed" / "mention_labels.csv", dtype={"mention_id": str})
    d = d.merge(lab, on="mention_id")
    d = d[d.platform.isin(PLATS) & (d.view_count > 0)].copy()
    f = pd.DataFrame(index=d.index)
    f["log_followers"] = np.log10(d.author_followers.where(d.author_followers > 0))
    for c in ["likes", "comments", "shares"]:
        f[f"log_{c}"] = np.log1p(d[c]); f[f"na_{c}"] = d[c].isna().astype(int)
    f["na_followers"] = d.author_followers.isna().astype(int)
    text = d.content.fillna(d.title).fillna("")
    f["english"] = (d.language == "en").astype(int); f["text_len"] = np.log10(text.str.len() + 1)
    f["type_code"] = d.type.map(TYPE_CODE); f["sentiment_code"] = d.sentiment.map(SENT_CODE)
    f["month"] = d.created_at.dt.year * 12 + d.created_at.dt.month
    f["metrics_age_days"] = (d.metrics_updated_at - d.created_at).dt.days
    meta = d[["platform", "brand", eu.REACH, "author_followers"]].assign(y=np.log10(d.view_count))
    return f, meta


def hgb():
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, random_state=SEED)


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def ladder(f, meta, plat):
    """MAE (log10 views) for baselines, ridge and the gradient model per cumulative feature group."""
    m = meta.platform == plat; X, M = f[m].reset_index(drop=True), meta[m].reset_index(drop=True); y, g = M.y, M.brand
    rows = []
    def add(name, pred, n_features=None):
        ok = ~np.isnan(pred); mae, lo, hi = mae_ci((y[ok] - pred[ok]).values, g[ok].values)
        rows.append({"platform": eu.NAMES[plat], "model": name, "MAE": mae, "CI low": lo, "CI high": hi, "rows": int(ok.sum())})
    pred_med = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g): pred_med[te] = y.iloc[tr].median()
    add("median views", pred_med)
    has_f = M.author_followers.gt(0).values; pr = np.full(len(y), np.nan)                       # constant x followers, constant fitted on train
    for tr, te in GroupKFold(5).split(X, y, g):
        c = np.median(y.iloc[tr][has_f[tr]] - X.log_followers.iloc[tr][has_f[tr]]); pr[te] = np.where(has_f[te], X.log_followers.iloc[te] + c, np.nan)
    add("constant x followers", pr)
    if plat == "twitter":   # on TikTok/YouTube Brand24 reach IS the view count, so it is not a baseline there
        add("Brand24 reach as-is", np.log10(M[eu.REACH].where(M[eu.REACH] > 0)).values)
    cols = []
    for name, c in GROUPS.items():
        cols += c; add(f"gradient model: + {name}", _cv(hgb, X[cols], y, g))
    add("ridge: followers + engagement", _cv(ridge, X[GROUPS["followers"] + GROUPS["engagement"]].fillna(0), y, g))
    return pd.DataFrame(rows)


def importance(f, meta, plat, n_rep=5):
    """Group permutation importance of the full gradient model on held-out brands: rise in MAE when a group is shuffled."""
    m = meta.platform == plat; X, M = f[m].reset_index(drop=True), meta[m].reset_index(drop=True); y, g = M.y, M.brand; rng = np.random.default_rng(SEED)
    cols = [c for gr in GROUPS.values() for c in gr]; out = {k: [] for k in GROUPS}
    for tr, te in GroupKFold(5).split(X, y, g):
        mod = hgb().fit(X.iloc[tr][cols], y.iloc[tr]); Xt, yt = X.iloc[te][cols], y.iloc[te]; base = np.abs(yt - mod.predict(Xt)).mean()
        for k, c in GROUPS.items():
            for _ in range(n_rep):
                Xp = Xt.copy(); Xp[c] = Xt[c].sample(frac=1, random_state=int(rng.integers(1e9))).values
                out[k].append(np.abs(yt - mod.predict(Xp)).mean() - base)
    return pd.Series({k: np.mean(v) for k, v in out.items()}, name=eu.NAMES[plat])


def plot_ladder(res):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), sharex=False)
    for a, plat in zip(ax, PLATS):
        r = res[res.platform == eu.NAMES[plat]].iloc[::-1]
        a.barh(r.model, r.MAE, xerr=[r.MAE - r["CI low"], r["CI high"] - r.MAE], color=eu.PCOL[plat], ecolor="black", capsize=2)
        a.set_xlabel("MAE of log10(views), brand-grouped CV"); a.set_title(eu.NAMES[plat], fontweight="bold"); a.set_ylabel("model" if plat == PLATS[0] else "")
        if plat != PLATS[0]: a.set_yticklabels([])
    plt.tight_layout(); plt.show()


def plot_importance(imp):
    ax = imp.plot.bar(figsize=(9, 3.6), color=[eu.PCOL[p] for p in PLATS], width=.8)
    ax.set_xlabel("feature group shuffled"); ax.set_ylabel("increase in MAE (log10 views)"); ax.set_title("Group permutation importance on held-out brands", fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, title="platform"); plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.show()
