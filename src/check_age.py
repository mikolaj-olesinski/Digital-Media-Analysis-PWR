"""Age of the view check and newer checks from the history file (notebooks/models/17_check_age.ipynb, E44-E45).
Final in-platform model as in notebook 14: CatBoost with MAE loss on followers + reactions + missing flags.
E44 adds log10(hours from posting to the check). E45 replaces views, reactions and age by the latest check where the
history file has one, and compares how predictable the data are on the same posts and folds."""
import numpy as np, pandas as pd
import eda_utils as eu, extra_models as em, history as H, reaction_tiers as rt

PLATS = ["tiktok", "youtube", "twitter"]
ENG = em.ENG


def load(root):
    """Features, meta and the refreshed copy. Rows: TikTok / YouTube / X posts with views > 0 (as in em.load)."""
    f, meta = em.load(root)
    raw = eu.load(root).loc[f.index]
    age = (raw.metrics_updated_at - raw.created_at).dt.total_seconds() / 3600
    f = f.assign(log_age=np.log10(age.clip(lower=1)))
    h = H.load(root).set_index("mention_id").reindex(raw.mention_id.astype(str).values)
    cur_age = ((h.attributes_last_update - h.created_at).dt.total_seconds() / 3600).values
    has = (h.current_views.notna() & (h.current_views > 0)).values & (cur_age > 0)
    g, m2 = f.copy(), meta.copy()
    for c in ("likes", "comments", "shares"):
        new = np.log1p(h[f"current_{c}"].values)
        g[f"log_{c}"] = np.where(has & ~np.isnan(new), new, f[f"log_{c}"].values)
        g[f"na_{c}"] = np.where(has & ~np.isnan(new), 0, f[f"na_{c}"].values)
    g["log_age"] = np.where(has, np.log10(np.clip(cur_age, 1, None)), f.log_age.values)
    m2["y"] = np.where(has, np.log10(h.current_views.values), meta.y.values)
    meta = meta.assign(refreshed=has, views_grew=has & (m2.y.values > meta.y.values + 1e-9), age_h=age.values)
    return f, meta, g, m2


def predict(f, meta, plat, cols):
    m = (meta.platform == plat).values; M = meta[m].reset_index(drop=True)
    X = f[m][cols].fillna(0).reset_index(drop=True)
    return M, em.cv_predict(rt.FINAL, X, M.y, M.brand)


def run(f, meta, g, m2):
    """Long table of out-of-fold predictions: base, + age, and + age on refreshed data (with its own target)."""
    out = []
    for p in PLATS:
        M, base = predict(f, meta, p, ENG); _, age = predict(f, meta, p, ENG + ["log_age"])
        M2, fresh = predict(g, m2, p, ENG + ["log_age"])
        common = dict(platform=p, brand=M.brand, refreshed=meta[meta.platform == p].refreshed.values, age_h=meta[meta.platform == p].age_h.values)
        out += [pd.DataFrame({**common, "model": "final model", "y": M.y, "pred": base}),
                pd.DataFrame({**common, "model": "final model + check age", "y": M.y, "pred": age}),
                pd.DataFrame({**common, "model": "final model + check age, refreshed data", "y": M2.y, "pred": fresh})]
    return pd.concat(out, ignore_index=True)


def by_age(P, model="final model", bins=(0, 24, 24 * 7, 24 * 30, 24 * 90, 1e7), labels=("<1 d", "1-7 d", "7-30 d", "30-90 d", ">90 d")):
    q = P[P.model == model].assign(age=lambda x: pd.cut(x.age_h, bins, labels=labels))
    t = q.groupby(["platform", "age"], observed=True).apply(lambda z: pd.Series({"posts": len(z), "MAE": float(np.abs(z.y - z.pred).mean()), "bias": float((z.pred - z.y).mean())})).reset_index()
    return t


def subset(P, mask_col="refreshed"):
    return P[P[mask_col]]


def plot_bias(P, path=None, lang="en"):
    """Signed error (prediction - truth) by age of the check, final model with and without the age column."""
    import matplotlib.pyplot as plt
    pl = lang == "pl"; fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    a, b = by_age(P), by_age(P, "final model + check age")
    lab = {"<1 d": "<1 d", "1-7 d": "1-7 d", "7-30 d": "7-30 d", "30-90 d": "30-90 d", ">90 d": ">90 d"}
    for x, p in zip(ax, PLATS):
        r, s = a[a.platform == p], b[b.platform == p]; i = np.arange(len(r)); c = eu.PCOL[p]
        x.bar(i - .2, r.bias, .4, color=c, alpha=.35, label="model końcowy" if pl else "final model")
        x.bar(i + .2, s.bias, .4, color=c, label="+ wiek pomiaru" if pl else "+ age of check")
        x.axhline(0, color="black", lw=.8); x.set_xticks(i); x.set_xticklabels([f"{lab[k]}\nn={int(n)}" for k, n in zip(r.age, r.posts)], fontsize=9)
        x.set_title(eu.NAMES[p], fontweight="bold"); x.set_xlabel("wiek posta przy pomiarze wyświetleń" if pl else "age of the post at the view check")
    ax[0].set_ylabel("średni błąd ze znakiem, log10 (>0 = zawyża)" if pl else "mean signed error, log10 (>0 = overestimates)")
    ax[-1].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False)
    fig.suptitle("TikTok i YouTube: przy pomiarze młodszym niż doba model zawyża; wiek pomiaru to usuwa" if pl else "TikTok and YouTube: checks younger than a day are overestimated; the age column removes it", fontweight="bold"); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()
