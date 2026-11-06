"""Snapshots of views and reactions over time from the second Brand24 file (notebooks/models/16_history_forecast.ipynb).
The file `reach_samples_students_with_history.csv` has the same 12,500 mentions plus up to three earlier checks (history_1..3)
and the latest check (current_*, attributes_last_update). Those timestamps are local Polish time (2 h after the UTC
metrics_updated_at for the same check in summer), so they are converted from Europe/Warsaw to UTC."""
from pathlib import Path
import numpy as np, pandas as pd

FILE = "reach_samples_students_with_history.csv"
TCOLS = ["created_at", "metrics_updated_at", "attributes_last_update", "history_1_at", "history_2_at", "history_3_at"]
LOCAL = ["attributes_last_update", "history_1_at", "history_2_at", "history_3_at"]
M = ["views", "likes", "comments", "shares"]


def load(root):
    p = Path(root) / "data" / "raw" / FILE
    if not p.exists():
        raise SystemExit(f"Missing data file: {p}\n"
                         "See docs/en/reproduce.md, section 'If a data file is missing'.")
    d = pd.read_csv(p, parse_dates=TCOLS).astype({"mention_id": str})
    for c in LOCAL:
        d[c] = d[c].dt.tz_localize("Europe/Warsaw", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC").dt.tz_localize(None)
    return d


def snapshots(d):
    """Long table: one row per (mention, check) with age in hours and the metrics seen at that check.
    Sources: main (metrics_updated_at, view_count ...), h1..h3 (history), cur (current_*). Duplicate (time, views) pairs are dropped."""
    base = d[["mention_id", "platform", "brand", "created_at", "author_followers"]]
    parts = [base.assign(src="main", t=d.metrics_updated_at, views=d.view_count, likes=d.likes, comments=d.comments, shares=d.shares)]
    for k in (1, 2, 3):
        parts.append(base.assign(src=f"h{k}", t=d[f"history_{k}_at"], **{m: d[f"history_{k}_{m}"] for m in M}))
    parts.append(base.assign(src="cur", t=d.attributes_last_update, **{m: d[f"current_{m}"] for m in M}))
    s = pd.concat(parts, ignore_index=True).dropna(subset=["t", "views"])
    s["age_h"] = (s.t - s.created_at).dt.total_seconds() / 3600
    s = s[s.age_h > 0].sort_values(["mention_id", "t"])
    s["t_round"] = s.t.dt.floor("10min")
    return s.drop_duplicates(["mention_id", "t_round", "views"]).drop(columns="t_round").reset_index(drop=True)


def pairs(s, min_gap_h=1.0):
    """For every mention with >= 2 checks: the earliest check (input) and the latest check (target)."""
    g = s.groupby("mention_id")
    first, last = g.head(1).set_index("mention_id"), g.tail(1).set_index("mention_id")
    p = first.join(last[["age_h"] + M], rsuffix="_end")
    p = p[(p.age_h_end - p.age_h) >= min_gap_h]
    return p.reset_index()


# ---------- forecasting views from the first check ----------
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
from validation import mae_ci

SEED = 42
lg = lambda x: np.log10(np.asarray(x, float) + 1)


def features(p):
    """Inputs known at the first check. Target: log10(views + 1) at the last check."""
    X = pd.DataFrame({"log_views0": lg(p.views), "log_likes0": lg(p.likes), "log_comments0": lg(p.comments), "log_shares0": lg(p.shares),
                      "log_age0": np.log10(p.age_h), "log_age_end": np.log10(p.age_h_end), "log_followers": np.log10(p.author_followers.where(p.author_followers > 0))})
    return X, pd.Series(lg(p.views_end), index=p.index)


def catboost():
    return CatBoostRegressor(loss_function="MAE", iterations=600, learning_rate=.05, depth=4, random_seed=SEED, verbose=0, thread_count=4, allow_writing_files=False)


REACT = ["log_likes0", "log_comments0", "log_shares0", "log_followers", "log_age0", "log_age_end"]


def forecast_models():
    """name -> fn(Xtr, ytr, Xte) returning predictions of log10 views at the last check."""
    def persist(Xtr, ytr, Xte): return Xte.log_views0.values
    def med_growth(Xtr, ytr, Xte): return Xte.log_views0.values + np.median(ytr - Xtr.log_views0)
    def power_law(Xtr, ytr, Xte):   # views ~ age^b: growth = b * (log age_end - log age0)
        dt = lambda X: (X.log_age_end - X.log_age0).values.reshape(-1, 1)
        m = LinearRegression(fit_intercept=False).fit(dt(Xtr), ytr - Xtr.log_views0); return Xte.log_views0.values + m.predict(dt(Xte))
    def cb_growth(Xtr, ytr, Xte):
        m = catboost().fit(Xtr, ytr - Xtr.log_views0); return Xte.log_views0.values + m.predict(Xte)
    def cb_react(Xtr, ytr, Xte):   # no view count at the first check: only reactions, followers and ages
        return catboost().fit(Xtr[REACT], ytr).predict(Xte[REACT])
    return {"first check as-is (persistence)": persist, "first check x median growth": med_growth, "first check x power law in age": power_law,
            "CatBoost: first check + reactions + ages": cb_growth, "CatBoost without first-check views (reactions only)": cb_react}


def forecast_cv(p, n_splits=5):
    """Out-of-fold predictions per platform, folds grouped by brand. Returns long table of predictions."""
    out = []
    for plat, q in p.groupby("platform"):
        q = q.reset_index(drop=True); X, y = features(q); g = q.brand
        pred = {k: np.full(len(y), np.nan) for k in forecast_models()}
        for tr, te in GroupKFold(n_splits).split(X, y, g):
            for k, fn in forecast_models().items():
                pred[k][te] = fn(X.iloc[tr], y.iloc[tr], X.iloc[te])
        for k, v in pred.items():
            out.append(pd.DataFrame({"platform": plat, "model": k, "mention_id": q.mention_id, "brand": g, "y": y, "pred": v, "gap_h": q.age_h_end - q.age_h, "age0_h": q.age_h}))
    return pd.concat(out, ignore_index=True)


def summarise(P):
    rows = []
    for (plat, k), q in P.groupby(["platform", "model"], sort=False):
        mae, lo, hi = mae_ci((q.y - q.pred).values, q.brand.values)
        rows.append({"platform": plat, "model": k, "posts": len(q), "MAE": mae, "CI low": lo, "CI high": hi, "bias": float((q.pred - q.y).mean())})
    return pd.DataFrame(rows)


# ---------- views predicted at posting time (no reactions yet) ----------
import modeling as mo, extra_features as xf, extra_models as em


def posting_time_data(root):
    """TikTok / YouTube / X mentions with views. Feature groups known when the post appears (no likes, comments, shares)."""
    f, meta = em.load(root); g = xf.build(root, f, meta)
    cal = g["calendar"].rename(columns={"hours_to_metrics": "horizon_h"}); cal["horizon_h"] = np.log10(cal.horizon_h.clip(lower=1))
    base = f[["english", "text_len", "month"]]
    groups = {"followers": f[["log_followers", "na_followers"]],
              "posting time": pd.concat([base, f[["type_code", "sentiment_code"]], g["text stats"], g["VADER sentiment"], g["TF-IDF SVD (30)"], cal, g["brand size"]], axis=1),
              "reactions": f[mo.GROUPS["engagement"] + ["na_likes", "na_comments", "na_shares"]]}
    return groups, meta


def author_mean(M, y, tr, te, m_prior=2.0):
    """Smoothed mean log10 views of the same author's mentions in the training fold (0 rows -> training mean)."""
    a = em.author_ids(M).values; mu = y.iloc[tr].mean()
    st = pd.DataFrame({"a": a[tr], "y": y.iloc[tr].values}).groupby("a").y.agg(["sum", "count"])
    def enc(idx, loo):
        s = st.reindex(a[idx]); sm, n = s["sum"].fillna(0).values, s["count"].fillna(0).values
        if loo: sm, n = sm - y.iloc[idx].values, n - 1
        return (sm + m_prior * mu) / (n + m_prior), n
    (te_tr, n_tr), (te_te, n_te) = enc(tr, True), enc(te, False)
    return pd.DataFrame({"author_mean": te_tr, "author_n": n_tr}), pd.DataFrame({"author_mean": te_te, "author_n": n_te})


def posting_time_cv(groups, meta, plat):
    m = (meta.platform == plat).values; M = meta[m].reset_index(drop=True); y, g = M.y, M.brand
    G = {k: v[m].reset_index(drop=True) for k, v in groups.items()}
    sets = {"CatBoost: followers": ["followers"], "CatBoost: posting-time columns": ["followers", "posting time"],
            "CatBoost: posting-time + author history": ["followers", "posting time", "author"],
            "reference: posting-time + reactions": ["followers", "posting time", "reactions"]}
    pred = {"median views": np.full(len(y), np.nan), "constant x followers": np.full(len(y), np.nan), **{k: np.full(len(y), np.nan) for k in sets}}
    hasf = M.author_followers.gt(0).values
    for tr, te in GroupKFold(5).split(M, y, g):
        pred["median views"][te] = y.iloc[tr].median()
        lf = G["followers"].log_followers; c = np.median(y.iloc[tr][hasf[tr]] - lf.iloc[tr][hasf[tr]])
        pred["constant x followers"][te] = np.where(hasf[te], lf.iloc[te] + c, y.iloc[tr].median())
        a_tr, a_te = author_mean(M, y, tr, te)
        for k, parts in sets.items():
            Xtr = pd.concat([a_tr if p == "author" else G[p].iloc[tr].reset_index(drop=True) for p in parts], axis=1)
            Xte = pd.concat([a_te if p == "author" else G[p].iloc[te].reset_index(drop=True) for p in parts], axis=1)
            pred[k][te] = catboost().fit(Xtr, y.iloc[tr]).predict(Xte)
    return pd.concat([pd.DataFrame({"platform": plat, "model": k, "brand": g, "y": y, "pred": v}) for k, v in pred.items()], ignore_index=True)


# ---------- gains, growth exponent, intervals, error by horizon ----------
import matplotlib.pyplot as plt
import eda_utils as eu

PLATS = ["tiktok", "youtube", "twitter"]


def gain_ci(P, plat, base, new, n_boot=400, seed=SEED):
    """Fall in MAE from `base` to `new` on the same posts, 95% CI from resampling brands."""
    a, b = (P[(P.platform == plat) & (P.model == k)].reset_index(drop=True) for k in (base, new))
    d = pd.DataFrame({"g": a.brand, "e": np.abs(a.y - a.pred) - np.abs(b.y - b.pred)}).groupby("g").e.agg(["sum", "count"])
    rng = np.random.default_rng(seed); idx = np.arange(len(d))
    bs = [d.iloc[s]["sum"].sum() / d.iloc[s]["count"].sum() for s in (rng.choice(idx, len(idx)) for _ in range(n_boot))]
    return float(d["sum"].sum() / d["count"].sum()), *np.percentile(bs, [2.5, 97.5])


def growth_exponent(p, n_boot=400, seed=SEED):
    """Exponent b in views ~ age^b between the first and last check, per platform, with a brand bootstrap CI."""
    rows = []; rng = np.random.default_rng(seed)
    for plat, q in p.groupby("platform"):
        dx = np.log10(q.age_h_end / q.age_h).values; dy = lg(q.views_end) - lg(q.views)
        fit = lambda i: float((dx[i] * dy[i]).sum() / (dx[i] ** 2).sum())
        br = q.brand.values; ub = np.unique(br); by = {b: np.where(br == b)[0] for b in ub}
        bs = [fit(np.concatenate([by[b] for b in rng.choice(ub, len(ub))])) for _ in range(n_boot)]
        rows.append({"platform": plat, "posts": len(q), "b": fit(np.arange(len(q))), "CI low": np.percentile(bs, 2.5), "CI high": np.percentile(bs, 97.5),
                     "median growth x": float(10 ** np.median(dy)), "median age first (h)": float(q.age_h.median()), "median age last (h)": float(q.age_h_end.median())})
    return pd.DataFrame(rows)


def oof_intervals(P, model, alpha=.1):
    """Interval from out-of-fold residuals of the other brands (leave-brand-out): |error| quantile of all other brands' posts."""
    rows = []
    for plat, q in P[P.model == model].groupby("platform"):
        r = np.abs(q.y - q.pred).values; br = q.brand.values; ok = []; widths = []
        for b in np.unique(br):
            m = br == b; qq = np.quantile(r[~m], min(1, np.ceil((m.size - m.sum() + 1) * (1 - alpha)) / (m.size - m.sum())))
            ok += list(r[m] <= qq); widths += [qq] * m.sum()
        rows.append({"platform": plat, "coverage": float(np.mean(ok)), "half width log10": float(np.mean(widths)), "factor x": float(10 ** np.mean(widths))})
    return pd.DataFrame(rows)


def by_horizon(P, model, bins=(0, 12, 48, 24 * 7, 24 * 30, 1e6), labels=("<12h", "12-48h", "2-7d", "7-30d", ">30d")):
    q = P[P.model == model].assign(h=lambda x: pd.cut(x.gap_h, bins, labels=labels))
    return q.groupby(["platform", "h"], observed=True).apply(lambda z: pd.Series({"posts": len(z), "MAE": float(np.abs(z.y - z.pred).mean())})).reset_index()


SHORT_PL = {"first check as-is (persistence)": "pierwszy pomiar bez zmian", "first check x median growth": "pierwszy pomiar × mediana wzrostu",
            "first check x power law in age": "pierwszy pomiar × prawo potęgowe wieku", "CatBoost: first check + reactions + ages": "CatBoost: pierwszy pomiar + reakcje + wiek",
            "CatBoost without first-check views (reactions only)": "CatBoost bez wyświetleń (same reakcje)",
            "median views": "mediana wyświetleń", "constant x followers": "stała × obserwujący", "CatBoost: followers": "CatBoost: obserwujący",
            "CatBoost: posting-time columns": "CatBoost: kolumny z chwili publikacji", "CatBoost: posting-time + author history": "CatBoost: publikacja + historia autora",
            "reference: posting-time + reactions": "odniesienie: publikacja + reakcje"}


def plot_mae(T, title, path=None, lang="en"):
    pl = lang == "pl"; plats = [p for p in PLATS if p in set(T.platform)]
    fig, ax = plt.subplots(1, len(plats), figsize=(5 * len(plats), 0.55 * T.model.nunique() + 1.6), sharey=True)
    for a, p in zip(np.atleast_1d(ax), plats):
        r = T[T.platform == p].iloc[::-1]; lab = [SHORT_PL.get(m, m) if pl else m for m in r.model]
        a.barh(lab, r.MAE, xerr=[r.MAE - r["CI low"], r["CI high"] - r.MAE], color=eu.PCOL[p], ecolor="black", capsize=2)
        for i, (v, hi) in enumerate(zip(r.MAE, r["CI high"])): a.text(hi + .012, i, f"×{10 ** v:.1f}".replace(".", ",") if pl else f"×{10 ** v:.1f}", va="center", fontsize=9)
        a.set_title(f"{eu.NAMES[p]} (n={int(r.posts.iloc[0])})", fontweight="bold"); a.set_xlabel("MAE log10(wyświetleń), CV po markach" if pl else "MAE of log10(views), brand-grouped CV")
    fig.suptitle(title, fontweight="bold"); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()


def plot_growth(p, B, path=None, lang="en"):
    pl = lang == "pl"; fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for a, plat in zip(ax, PLATS):
        q = p[p.platform == plat]; dx = np.log10(q.age_h_end / q.age_h); dy = lg(q.views_end) - lg(q.views); b = B.set_index("platform").loc[plat]
        a.scatter(dx, dy, s=10, alpha=.4, color=eu.PCOL[plat], label="post" if not pl else "post")
        xs = np.linspace(0, dx.max(), 50); a.plot(xs, b.b * xs, color="black", lw=1.5, label=(f"wyświetlenia ~ wiek^{b.b:.2f}".replace(".", ",") if pl else f"views ~ age^{b.b:.2f}"))
        a.set_title(f"{eu.NAMES[plat]} (n={int(b.posts)})", fontweight="bold")
        a.set_xlabel("log10(wiek przy ostatnim / pierwszym pomiarze)" if pl else "log10(age at last / first check)")
        a.legend(loc="upper left", frameon=False, fontsize=9)
    ax[0].set_ylabel("log10(wyświetlenia ostatni / pierwszy pomiar)" if pl else "log10(views at last / first check)")
    fig.suptitle("Ile rosną wyświetlenia między pomiarami Brand24" if pl else "How much views grow between Brand24 checks", fontweight="bold"); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()
