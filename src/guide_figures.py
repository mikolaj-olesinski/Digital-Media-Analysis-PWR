"""Plain figures for the guide and slides: the default model against the baselines, and Brand24's rule on X against our model."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import eda_utils as eu, extra_models as em, modeling as mo, new_ideas as ni
from validation import mae_ci

SEED = 42
DEFAULT = lambda: HistGradientBoostingRegressor(random_state=SEED)   # scikit-learn's defaults, nothing chosen by us


def baseline_table(f, meta):
    """Median, constant x followers, and one default model with followers only and with followers + reactions."""
    rows = []
    for p in mo.PLATS:
        m = (meta.platform == p).values; M = meta[m].reset_index(drop=True); X = f[m].reset_index(drop=True); y, g = M.y, M.brand; hf = M.author_followers.gt(0).values; a, b = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, g):
            a[te] = y.iloc[tr].median(); b[te] = np.where(hf[te], X.log_followers.iloc[te] + np.median(y.iloc[tr][hf[tr]] - X.log_followers.iloc[tr][hf[tr]]), np.nan)
        F = mo.GROUPS["followers"]; E = F + mo.GROUPS["engagement"]; c = em.cv_predict(DEFAULT, X[F], y, g); d = em.cv_predict(DEFAULT, X[E], y, g)
        for name, pr in [("median views", a), ("constant x followers", b), ("default model: followers", c), ("default model: followers + reactions", d)]:
            ok = ~np.isnan(pr); v = mae_ci((y[ok] - pr[ok]).values, g[ok].values); rows.append({"platform": eu.NAMES[p], "model": name, "MAE": v[0], "low": v[1], "high": v[2], "rows": int(ok.sum())})
    return pd.DataFrame(rows)


def plot_baseline(T, path=None, lang="en"):
    pl = lang == "pl"; steps = [("median views", "zgadywanie mediany" if pl else "guess the median", "#c9c9c4"), ("constant x followers", "stała × followersi (idea Brand24)" if pl else "constant × followers (Brand24's idea)", "#8d8d88"), ("default model: followers", "model domyślny, tylko followersi" if pl else "default model, followers only", "#7fa3bb"), ("default model: followers + reactions", "model domyślny, followersi + reakcje" if pl else "default model, followers + reactions", "#2f6f93")]; fig, ax = plt.subplots(figsize=(10, 4.8)); w = .2; names = [eu.NAMES[p] for p in mo.PLATS]
    for k, (key, lab, c) in enumerate(steps):
        r = T[T.model == key].set_index("platform").reindex(names); x = np.arange(3) + (k - 1.5) * w; ax.bar(x, r.MAE, w, color=c, label=lab, yerr=[r.MAE - r.low, r.high - r.MAE], ecolor="black", capsize=2)
        for i, v in enumerate(r.MAE): ax.text(x[i], r.high.iloc[i] + .012, (f"{v:.2f}".replace(".", ",") if pl else f"{v:.2f}"), ha="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(names); ax.set_xlabel("platforma" if pl else "platform"); ax.set_ylabel("MAE, log10 wyświetleń" if pl else "MAE, log10 views"); ax.set_title("Już model z domyślnymi ustawieniami jest wyraźnie lepszy od stałej" if pl else "Even a model on default settings clearly beats the constant", fontweight="bold"); ax.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()


def x_rule(f, meta, raw, make=None):
    """On X, on the rows where Brand24 reach and views both exist: reach, the out-of-fold model prediction, the paired gain."""
    make = make or em.models()["extra trees"]; m = (meta.platform == "twitter").values; M = meta[m].reset_index(drop=True); X = f[m][em.ENG].reset_index(drop=True); y, g = M.y, M.brand; p = em.cv_predict(make, X, y, g); r = np.log10(M[eu.REACH].where(M[eu.REACH] > 0)); ok = r.notna().values; ratio = (M[eu.REACH] / M.author_followers.where(M.author_followers > 0))
    a, b = mae_ci((y[ok] - r[ok]).values, g[ok].values), mae_ci((y[ok] - p[ok]).values, g[ok].values); d = ni.paired_diff(y.values[ok], r.values[ok], p[ok], g.values[ok]); zero = float((M[eu.REACH] == 0).mean())
    return {"rows": int(ok.sum()), "zero reach share": zero, "share exactly 0.1 x followers": float(((ratio > .099) & (ratio < .101)).mean()), "Brand24 MAE": a[0], "Brand24 low": a[1], "Brand24 high": a[2], "model MAE": b[0], "model low": b[1], "model high": b[2], "gain": d[0], "gain low": d[1], "gain high": d[2], "Brand24 bias": float((y[ok] - r[ok]).mean()), "model bias": float((y[ok] - p[ok]).mean())}, M.assign(reacted=(np.expm1(X.log_likes.fillna(0)) + np.expm1(X.log_comments.fillna(0)) + np.expm1(X.log_shares.fillna(0)) > 0).values), r, p


def plot_x_rule(s, M, r, p, path=None, lang="en"):
    """Three separate questions: what the rule is, how it compares with true views, how our model compares with true views."""
    pl = lang == "pl"; fig, ax = plt.subplots(1, 3, figsize=(15, 4.8)); c = eu.PCOL["twitter"]; fol = np.log10(M.author_followers.where(M.author_followers > 0)); lim = [0, 7.5]; fm = lambda v: f"{v:.2f}".replace(".", ",") if pl else f"{v:.2f}"
    q = M.reacted.values; ax[0].scatter(fol[~q], r[~q], s=5, alpha=.3, color="#9a9a9a", label="post bez reakcji" if pl else "post without reactions"); ax[0].scatter(fol[q], r[q], s=5, alpha=.3, color=c, label="post z reakcjami" if pl else "post with reactions"); xx = np.array([0, 8]); ax[0].plot(xx, xx - 1, color="black", lw=1, label="0,1 × followersi" if pl else "0.1 × followers"); ax[0].legend(frameon=False, loc="upper left", markerscale=2)
    ax[0].annotate("700 × 1 reakcja" if pl else "700 × 1 reaction", (.3, np.log10(700)), xytext=(.3, 4.4), arrowprops=dict(arrowstyle="->", color="#444444"), fontsize=9); ax[0].set_xlabel("log10(followersi)" if pl else "log10(followers)"); ax[0].set_ylabel("log10(zasięg Brand24)" if pl else "log10(Brand24 reach)"); ax[0].set_title("Reguła: większe z 700 × reakcje i 0,1 × followersi" if pl else "The rule: the larger of 700 × reactions and 0.1 × followers", fontweight="bold", fontsize=11)
    for a, xv, ttl, xl in [(ax[1], r, (f"Reguła Brand24: MAE {fm(s['Brand24 MAE'])} (pomyłka ×{10 ** s['Brand24 MAE']:.0f})" if pl else f"Brand24's rule: MAE {fm(s['Brand24 MAE'])} (off by ×{10 ** s['Brand24 MAE']:.0f})"), "log10(zasięg Brand24)" if pl else "log10(Brand24 reach)"), (ax[2], pd.Series(p).where(r.notna()), (f"Nasz model: MAE {fm(s['model MAE'])} (pomyłka ×{10 ** s['model MAE']:.0f})" if pl else f"Our model: MAE {fm(s['model MAE'])} (off by ×{10 ** s['model MAE']:.0f})"), "przewidywane log10(wyświetlenia)" if pl else "predicted log10(views)")]:
        a.plot(lim, lim, color="black", lw=1); a.fill_between(lim, [l - .3 for l in lim], [l + .3 for l in lim], color="grey", alpha=.15); a.scatter(xv, M.y, s=5, alpha=.25, color=c); a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel(xl); a.set_ylabel("prawdziwe log10(wyświetlenia)" if pl else "true log10(views)"); a.set_title(ttl, fontweight="bold", fontsize=11)
    plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()
