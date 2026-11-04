"""Rolling-window validation of the final model (notebooks/models/18_rolling_window.ipynb, E46).
Train on earlier months, test on month M, and compare with brand-grouped CV on the same posts."""
import numpy as np, pandas as pd
import eda_utils as eu, extra_models as em, reaction_tiers as rt

PLATS = ["tiktok", "youtube", "twitter"]
MONTHS = ["2026-05", "2026-06", "2026-07", "2026-08"]
SCHEMES = ["brand CV", "expanding", "expanding, new brands", "sliding (previous month)"]


def load(root):
    f, meta = em.load(root)
    raw = eu.load(root).loc[f.index]
    meta = meta.assign(month=raw.created_at.dt.strftime("%Y-%m").values,
                       vpl=(meta.y - np.log10(np.expm1(f.log_likes.fillna(0)) + 1)).values)
    return f, meta


def _fit_predict(X, y, tr, te):
    if tr.sum() < 30 or te.sum() == 0:
        return np.full(te.sum(), np.nan)
    return rt.FINAL().fit(X[tr], y[tr]).predict(X[te])


def run(f, meta, months=MONTHS):
    """Long table: one row per (scheme, test post)."""
    out = []
    for p in PLATS:
        m = (meta.platform == p).values; M = meta[m].reset_index(drop=True)
        X = f[m][em.ENG].fillna(0).reset_index(drop=True); y = M.y.values
        cv = em.cv_predict(rt.FINAL, X, M.y, M.brand)
        prev = {mo: months[i - 1] if i else "2026-04" for i, mo in enumerate(months)}
        for mo in months:
            te = (M.month == mo).values
            past = (M.month < mo).values
            brands = set(M.brand[te])
            preds = {"brand CV": cv[te],
                     "expanding": _fit_predict(X, y, past, te),
                     "expanding, new brands": _fit_predict(X, y, past & ~M.brand.isin(brands).values, te),
                     "sliding (previous month)": _fit_predict(X, y, (M.month == prev[mo]).values, te)}
            for k, pr in preds.items():
                out.append(pd.DataFrame({"platform": p, "month": mo, "model": k, "brand": M.brand[te].values, "y": y[te], "pred": pr}))
    return pd.concat(out, ignore_index=True)


def by_month(P):
    return (P.groupby(["platform", "month", "model"], sort=False)
             .apply(lambda q: pd.Series({"posts": len(q), "MAE": float(np.abs(q.y - q.pred).mean()), "bias": float((q.pred - q.y).mean())}))
             .reset_index())


def drift(meta, months=["2026-04"] + MONTHS):
    q = meta[meta.month.isin(months)]
    return q.groupby(["platform", "month"]).agg(posts=("y", "size"), median_log_views=("y", "median"), median_views_per_like=("vpl", "median")).reset_index()


def plot(T, path=None, lang="en"):
    import matplotlib.pyplot as plt
    pl = lang == "pl"
    lab = {"brand CV": "CV po markach (zwykły wynik)" if pl else "brand CV (usual number)",
           "expanding": "rosnące okno: uczy się na wcześniejszych miesiącach" if pl else "expanding: trained on earlier months",
           "expanding, new brands": "rosnące okno, nowe marki" if pl else "expanding, new brands",
           "sliding (previous month)": "tylko poprzedni miesiąc" if pl else "previous month only"}
    sty = {"brand CV": dict(color="black", ls="--"), "expanding": dict(color="#1f77b4"), "expanding, new brands": dict(color="#d62728"), "sliding (previous month)": dict(color="#7f7f7f", ls=":")}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for x, p in zip(ax, PLATS):
        for k in SCHEMES:
            r = T[(T.platform == p) & (T.model == k)]
            x.plot(r.month, r.MAE, marker="o", label=lab[k], **sty[k])
        n = T[(T.platform == p) & (T.model == "brand CV")]
        x.set_xticks(range(len(n))); x.set_xticklabels([f"{m}\nn={int(c)}" for m, c in zip(n.month, n.posts)], fontsize=9)
        x.set_title(eu.NAMES[p], fontweight="bold"); x.set_xlabel("miesiąc publikacji (zbiór testowy)" if pl else "month of posting (test set)")
    ax[0].set_ylabel("MAE log10(wyświetleń)" if pl else "MAE of log10(views)")
    ax[-1].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False)
    fig.suptitle("Ucząc się tylko na przeszłości model trzyma poziom CV po markach; gorzej tylko w maju, gdy historii jest mało" if pl else "Trained only on the past, the model stays near brand CV; worse only in May, when there is little history", fontweight="bold"); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()
