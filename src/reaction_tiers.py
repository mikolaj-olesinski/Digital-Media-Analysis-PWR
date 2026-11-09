"""Error of the final models by number of reactions, and prediction intervals that depend on it (notebooks/models/14_reaction_tiers.ipynb).
Reactions are known when the estimate is made, so tiers of reactions can be used in practice, unlike tiers of true views."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import eda_utils as eu, extra_models as em, modeling as mo
from validation import mae_ci

SEED = 42; EDGES = [-1, 0, 10, 100, 1000, np.inf]; TIERS = ["0", "1-10", "11-100", "101-1000", ">1000"]; MIN_N = 30
FINAL = lambda: CatBoostRegressor(loss_function="MAE", random_seed=SEED, iterations=300, verbose=0, allow_writing_files=False, thread_count=4)
LOPO = lambda: make_pipeline(StandardScaler(), Ridge())


def reactions(f):
    """Likes + comments + shares, with a missing count as zero."""
    return np.expm1(f.log_likes.fillna(0)) + np.expm1(f.log_comments.fillna(0)) + np.expm1(f.log_shares.fillna(0))


def tier(f):
    return pd.cut(reactions(f), EDGES, labels=TIERS).astype(str).values


def in_platform(f, meta, plat):
    """Final in-platform model: CatBoost with MAE loss on followers, reactions and flags, out-of-fold on held-out brands."""
    m = (meta.platform == plat).values; M = meta[m].reset_index(drop=True); X = f[m][em.ENG].fillna(0).reset_index(drop=True); return M, tier(f[m]), em.cv_predict(FINAL, X, M.y, M.brand)


def other_platforms(f, meta, plat):
    """Final out-of-platform model: ridge on likes and comments trained on the other two platforms, without the test fold's brands."""
    m = (meta.platform == plat).values; T = meta[m]; p = pd.Series(np.nan, index=T.index); C = ["log_likes", "log_comments"]; X = f[C].fillna(0); S = meta[(meta.platform != plat)]
    for _, te in GroupKFold(5).split(T, groups=T.brand):
        idx = S.index[~S.brand.isin(set(T.brand.iloc[te]))]; p.iloc[te] = LOPO().fit(X.loc[idx], meta.y.loc[idx]).predict(X.loc[T.index[te]])
    return T.reset_index(drop=True), tier(f[m]), p.values


def table(M, t, p, plat, kind):
    """Typical error and empirical 90% range per tier of reactions. A factor of ×k means true views are within k times the estimate."""
    e = (M.y - p).values; rows = []
    for k in TIERS + ["all"]:
        ok = np.ones(len(e), bool) if k == "all" else t == k
        if ok.sum() < MIN_N: continue
        v = mae_ci(e[ok], M.brand.values[ok]); rows.append({"platform": eu.NAMES[plat], "model": kind, "reactions": k, "posts": int(ok.sum()), "MAE": v[0], "low": v[1], "high": v[2], "typical factor": 10 ** v[0], "90% factor": 10 ** np.quantile(np.abs(e[ok]), .9), "bias": float(e[ok].mean())})
    return pd.DataFrame(rows)


def conformal(f, meta, plat, alpha=.1):
    """Split-conformal inside brand-grouped CV: one interval width for all posts against one per tier of reactions (Mondrian). In every fold a quarter of the training brands calibrates."""
    M, t, _ = in_platform(f, meta, plat); m = (meta.platform == plat).values; X = f[m][em.ENG].fillna(0).reset_index(drop=True); y, g = M.y, M.brand; rng = np.random.default_rng(SEED); w1, w2 = np.full(len(y), np.nan), np.full(len(y), np.nan); e = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        b = pd.unique(g.iloc[tr]); cal_b = rng.choice(b, max(1, len(b) // 4), replace=False); cal = tr[g.iloc[tr].isin(cal_b).values]; fit = np.setdiff1d(tr, cal); mod = FINAL().fit(X.iloc[fit], y.iloc[fit]); r = np.abs(y.iloc[cal] - mod.predict(X.iloc[cal])).values; Q = lambda a: np.quantile(a, min(1, np.ceil((len(a) + 1) * (1 - alpha)) / len(a))); q = Q(r); e[te] = np.abs(y.iloc[te] - mod.predict(X.iloc[te])); w1[te] = q
        for k in TIERS:
            rk = r[t[cal] == k]; w2[te[t[te] == k]] = Q(rk) if len(rk) >= 30 else q
    rows = []
    for k in TIERS + ["all"]:
        ok = np.ones(len(y), bool) if k == "all" else t == k
        if ok.sum() < MIN_N: continue
        rows.append({"platform": eu.NAMES[plat], "reactions": k, "posts": int(ok.sum()), "one width: coverage": float((e[ok] <= w1[ok]).mean()), "one width: factor": float(10 ** w1[ok].mean()), "per tier: coverage": float((e[ok] <= w2[ok]).mean()), "per tier: factor": float(10 ** w2[ok].mean())})
    return pd.DataFrame(rows)


def plot(T, path=None, lang="en"):
    pl = lang == "pl"; fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for a, p in zip(ax, mo.PLATS):
        r = T[(T.platform == eu.NAMES[p]) & (T.model == "same platform") & (T.reactions != "all")].set_index("reactions").reindex(TIERS).dropna(subset=["MAE"]); x = np.arange(len(r)); c = eu.PCOL[p]
        a.bar(x, r["90% factor"], color=c, alpha=.25, label="90% postów w zakresie" if pl else "90% of posts within"); a.bar(x, r["typical factor"], color=c, label="typowa pomyłka (pogrubiona liczba)" if pl else "typical error (bold number)", width=.55)
        for i, (_, q) in enumerate(r.iterrows()):
            a.text(i, q["typical factor"] * 1.04, f"×{q['typical factor']:.1f}".replace(".", ",") if pl else f"×{q['typical factor']:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="white" if q["typical factor"] > 99 else "black"); a.text(i, q["90% factor"] * 1.05, f"×{q['90% factor']:.0f}", ha="center", fontsize=10)
        a.set_yscale("log"); a.set_ylim(1, 60); a.set_yticks([1, 2, 5, 10, 20, 50]); a.set_yticklabels(["×1", "×2", "×5", "×10", "×20", "×50"]); a.minorticks_off(); a.set_xticks(x); a.set_xticklabels([f"{k}\nn={int(q)}" for k, q in zip(r.index, r.posts)], fontsize=9); a.set_xlabel("reakcje (lajki + komentarze + udostępnienia)" if pl else "reactions (likes + comments + shares)"); a.set_title(eu.NAMES[p], fontweight="bold")
    ax[0].set_ylabel("pomyłka (ile razy, skala log)" if pl else "error (how many times, log scale)"); ax[-1].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()


def plot_ranges(T, path=None, lang="en", est=1000):
    """If the model says 1,000 views, where is the truth? Thin bar: 90% of posts. Thick bar: typical error. Symmetric on the log scale."""
    pl = lang == "pl"; fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True); lab = {"0": "brak reakcji" if pl else "no reactions", "1-10": "1–10", "11-100": "11–100", "101-1000": "101–1000", ">1000": "ponad 1000" if pl else "over 1000"}; fm = lambda v: f"{v:,.0f}".replace(",", " " if pl else ",")
    for a, p in zip(ax, mo.PLATS):
        r = T[(T.platform == eu.NAMES[p]) & (T.model == "same platform") & (T.reactions != "all")].set_index("reactions").reindex(TIERS); c = eu.PCOL[p]
        for y, k in enumerate(TIERS):
            q = r.loc[k]
            if np.isnan(q.MAE): a.text(est, y, "za mało postów" if pl else "too few posts", ha="center", va="center", color="#888888", fontsize=10); continue
            k9, kt = q["90% factor"], q["typical factor"]; a.plot([est / k9, est * k9], [y, y], color=c, alpha=.3, lw=9, solid_capstyle="butt"); a.plot([est / kt, est * kt], [y, y], color=c, lw=9, solid_capstyle="butt"); a.text(est * k9 * 1.15, y, (f"×{kt:.1f}".replace(".", ",") if pl else f"×{kt:.1f}"), va="center", fontsize=10.5, fontweight="bold", color=c); a.text(est * k9 * 1.15 * 4.2, y, f"(×{k9:.0f})", va="center", fontsize=10, color="#555555"); a.text(est / k9 / 1.15, y, f"n={int(q.posts)}", va="center", ha="right", fontsize=8.5, color="#777777")
        a.axvline(est, color="black", lw=1); a.set_xscale("log"); a.set_xlim(12, 700000); a.set_xticks([100, 1000, 10000]); a.set_xticklabels(["100", "1000", "10 000" if pl else "10,000"]); a.minorticks_off(); a.set_yticks(range(5)); a.set_yticklabels([lab[k] for k in TIERS]); a.set_ylim(4.6, -.6); a.set_title(eu.NAMES[p], fontweight="bold"); a.set_xlabel("prawdziwe wyświetlenia (skala log)" if pl else "true views (log scale)"); a.grid(axis="x", alpha=.2)
    ax[0].set_ylabel("reakcje pod postem" if pl else "reactions on the post")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color="#555555", lw=9, label="typowa pomyłka (pogrubiona liczba)" if pl else "typical error (bold number)"), Line2D([], [], color="#555555", alpha=.3, lw=9, label="tu mieści się 90% postów (liczba w nawiasie)" if pl else "90% of posts are here (number in brackets)"), Line2D([], [], color="black", lw=1, label="szacunek modelu: 1000" if pl else "model's estimate: 1,000")], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.07)); fig.suptitle("Model uczony na tej samej platformie mówi 1000 wyświetleń. Gdzie jest prawda?" if pl else "A model trained on the same platform says 1,000 views. Where is the truth?", fontweight="bold", fontsize=14); plt.tight_layout()
    if path: fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.show()
