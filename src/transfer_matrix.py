"""Full out-of-platform matrix for views (notebooks/models/08_transfer_matrix.ipynb).
Train on any subset of TikTok, YouTube, X, test on a platform left out. Strictly brand-disjoint: a test brand is removed from training even when it appears on another platform.
Also: which feature set survives the move, tuning for transfer, label-based reweighting, calibration with a few target posts, and how much the Facebook/Instagram estimate depends on the source."""
from itertools import combinations
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import eda_utils as eu, extra_features as ef, modeling as mo, tuning as tn
from validation import mae_ci

SEED = 42; KNOWN = ["tiktok", "youtube", "twitter"]; UNKNOWN = ["facebook", "instagram"]
SETS = [c for k in (1, 2) for c in combinations(KNOWN, k)]
FEATURE_SETS = {"reactions": ["log_likes", "log_comments"], "reactions + followers": ["log_likes", "log_comments", "log_followers", "na_followers"], "reactions + followers + shares + flags": ["log_likes", "log_comments", "log_shares", "log_followers", "na_likes", "na_comments", "na_shares", "na_followers"]}
TYPES = ["organic", "promo", "spam", "unclear"]


def models():
    return {"ridge": make_pipeline(StandardScaler(), Ridge()), "random forest": RandomForestRegressor(300, min_samples_leaf=5, n_jobs=-1, random_state=SEED), "extra trees": ExtraTreesRegressor(300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
            "boosting (sklearn)": HistGradientBoostingRegressor(random_state=SEED), "LightGBM": LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1), "CatBoost": CatBoostRegressor(random_seed=SEED, verbose=0, allow_writing_files=False, iterations=300)}


def load(root):
    """Features on all five platforms. A missing like or comment count is 0 for the model and flagged; views exist only on the known three."""
    d = eu.load(root).rename(columns={"sentiment": "sentiment_b24"}).astype({"mention_id": str}); lab = pd.read_csv(root / "data" / "processed" / "mention_labels.csv", dtype={"mention_id": str}); d = d.merge(lab, on="mention_id").reset_index(drop=True)
    f = pd.DataFrame(index=d.index); f["log_followers"] = np.log10(d.author_followers.where(d.author_followers > 0)); f["na_followers"] = d.author_followers.isna().astype(int)
    for c in ["likes", "comments", "shares"]: f[f"log_{c}"] = np.log1p(d[c]); f[f"na_{c}"] = d[c].isna().astype(int)
    f = f.fillna(0); text = ef.text_stats(ef.full_text(d)); y = np.log10(d.view_count.where(d.view_count > 0))
    meta = pd.DataFrame({"platform": d.platform, "brand": d.brand, "y": y, "type": d.type, "b24": np.log10(d[eu.REACH].where(d[eu.REACH] > 0))})
    return f, text, meta


def predict(est, f, meta, source, target, cols, use_brand_filter=True, weights=None, extra=None):
    """Out-of-fold prediction on the target's rows. In-platform when source == (target,): plain brand-grouped CV. Otherwise the model is fitted on the source platforms without the fold's test brands."""
    T = meta[(meta.platform == target) & meta.y.notna()]; S = meta[meta.platform.isin(source) & meta.y.notna()]; pred = pd.Series(np.nan, index=T.index); X = f[cols] if extra is None else pd.concat([f[cols], extra], axis=1)
    for _, te in GroupKFold(5).split(T, groups=T.brand):
        tb = set(T.brand.iloc[te]); idx = S.index[~S.brand.isin(tb)] if use_brand_filter else S.index; kw = {}
        if weights is not None:
            w = weights.loc[idx].values; kw = {f"{est.steps[-1][0]}__sample_weight": w} if hasattr(est, "steps") else {"sample_weight": w}
        pred.iloc[te] = clone(est).fit(X.loc[idx], meta.y.loc[idx], **kw).predict(X.loc[T.index[te]])
    return pred


def score(p, meta):
    t = meta.loc[p.index]; e = (t.y - p).values; mae, lo, hi = mae_ci(e, t.brand.values)
    return {"MAE": mae, "low": lo, "high": hi, "bias": e.mean(), "MAE no bias": np.abs(e - np.median(e)).mean(), "spearman": spearmanr(t.y, p)[0], "n": len(p)}


def matrix(f, meta):
    """Every model × feature set × source set × target, plus the in-platform diagonal. Returns scores and the predictions."""
    rows, preds = [], {}
    for mname, est in models().items():
        for fname, cols in FEATURE_SETS.items():
            for target in KNOWN:
                for source in [(target,)] + [s for s in SETS if target not in s]:
                    p = predict(est, f, meta, source, target, cols); rows.append({"model": mname, "features": fname, "source": "+".join(source), "target": target, "kind": "in-platform" if source == (target,) else f"{len(source)} source(s)", **score(p, meta)}); preds[(mname, fname, source, target)] = p
        print(mname, "done", flush=True)
    return pd.DataFrame(rows), preds


def brand_filter_effect(f, meta, est, cols):
    """How much does it matter that test brands are removed from training on the other platforms?"""
    rows = []
    for t in KNOWN:
        s = tuple(x for x in KNOWN if x != t); a, b = predict(est, f, meta, s, t, cols, True), predict(est, f, meta, s, t, cols, False)
        rows.append({"held-out platform": eu.NAMES[t], "brands removed from training": score(a, meta)["MAE"], "brands kept": score(b, meta)["MAE"]})
    return pd.DataFrame(rows)


def tune_for_transfer(f, meta, cols, n_iter=20):
    """Train on two platforms, test on the third, three ways of choosing settings: defaults; brand-grouped CV inside the sources; leave-one-source-out (the setting that best crosses platforms). No view from the target is used."""
    rows = []; names = ["ridge", "extra trees", "boosting (sklearn)", "LightGBM", "CatBoost"]; fam = tn.families()
    base = {n: fam[n] for n in names}; base["LightGBM"] = (LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1), fam["LightGBM"][1]); base["CatBoost"] = (CatBoostRegressor(random_seed=SEED, verbose=0, allow_writing_files=False, iterations=300), fam["CatBoost"][1]); base["extra trees"] = (ExtraTreesRegressor(random_state=SEED, n_jobs=-1), fam["extra trees"][1])
    for target in KNOWN:
        source = tuple(p for p in KNOWN if p != target); S = meta[meta.platform.isin(source) & meta.y.notna()]; X = f.loc[S.index, cols]; y = S.y
        for name, (est, space) in base.items():
            cand = list(ParameterSampler(space, n_iter=n_iter, random_state=SEED)); res = []
            for params in cand:
                brand_folds = list(GroupKFold(3).split(X, y, S.brand)); src_folds = [(np.where(S.platform != s)[0], np.where(S.platform == s)[0]) for s in source]
                score_of = lambda folds: np.mean([np.abs(y.iloc[b] - clone(est).set_params(**params).fit(X.iloc[a], y.iloc[a]).predict(X.iloc[b])).mean() for a, b in folds])
                res.append((score_of(brand_folds), score_of(src_folds)))
            res = np.array(res); best_b, best_s = cand[res[:, 0].argmin()], cand[res[:, 1].argmin()]
            for how, e in [("default", clone(est)), ("tuned inside sources", clone(est).set_params(**best_b)), ("tuned across sources", clone(est).set_params(**best_s))]:
                rows.append({"model": name, "held-out platform": eu.NAMES[target], "settings": how, **score(predict(e, f, meta, source, target, cols), meta)})
            print(eu.NAMES[target], name, "done", flush=True)
    return pd.DataFrame(rows)


def label_adaptation(f, meta, est, cols):
    """Can our labels help the move? Facebook and Instagram have labels too, so the source can be reweighted to the target's mix of mention types, or the type can be a feature."""
    rows = []; types = pd.get_dummies(meta.type).reindex(columns=TYPES, fill_value=0).astype(int).add_prefix("type_")
    for target in KNOWN:
        source = tuple(p for p in KNOWN if p != target); S = meta[meta.platform.isin(source) & meta.y.notna()]; want = meta[meta.platform == target].type.value_counts(normalize=True); have = S.type.value_counts(normalize=True)
        w = pd.Series(1.0, index=meta.index); w.loc[S.index] = S.type.map(want / have).fillna(0).values
        for how, p in [("no labels", predict(est, f, meta, source, target, cols)), ("source reweighted to target's type mix", predict(est, f, meta, source, target, cols, weights=w)), ("type as a feature", predict(est, f, meta, source, target, cols, extra=types))]:
            rows.append({"held-out platform": eu.NAMES[target], "variant": how, **score(p, meta)})
    return pd.DataFrame(rows)


def calibration_curve(p, meta, sizes=(5, 10, 25, 50, 100, 200, 500), reps=500, per_brand=False):
    """Pretend we get n target posts with true views: shift all predictions by their median error, score on the remaining posts. Returns median and 5-95% range of the resulting MAE."""
    rng = np.random.default_rng(SEED); t = meta.loc[p.index]; e = (t.y - p).values; out = []; brands = t.brand.values; ub = np.unique(brands)
    for n in sizes:
        if n >= len(e): continue
        v = []
        for _ in range(reps):
            if per_brand:   # a real sample would come from a few cooperating brands, not at random
                pool = []; order = rng.permutation(ub); i = 0
                while len(pool) < n and i < len(order): pool += list(np.where(brands == order[i])[0]); i += 1
                idx = rng.choice(pool, min(n, len(pool)), replace=False)
            else: idx = rng.choice(len(e), n, replace=False)
            rest = np.setdiff1d(np.arange(len(e)), idx); v.append(np.abs(e[rest] - np.median(e[idx])).mean())
        out.append({"n": n, "median": float(np.median(v)), "low": float(np.percentile(v, 5)), "high": float(np.percentile(v, 95))})
    return pd.DataFrame(out).assign(uncalibrated=np.abs(e).mean(), oracle=np.abs(e - np.median(e)).mean())


def unknown_sensitivity(f, meta, est, cols):
    """Median estimate on Facebook and Instagram under every source set: the spread shows how much the answer depends on where the model learned."""
    rows = []
    for source in SETS:
        S = meta[meta.platform.isin(source) & meta.y.notna()]; m = clone(est).fit(f.loc[S.index, cols], S.y)
        for t in UNKNOWN:
            r = meta[meta.platform == t]; rows.append({"source": "+".join(eu.NAMES[s] for s in source), "platform": eu.NAMES[t], "median log10 views": float(np.median(m.predict(f.loc[r.index, cols]))), "median Brand24 reach": float(r.b24.median())})
    return pd.DataFrame(rows)


# ---------- plots ----------
SHORT = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X"}


def _txt(s): return "+".join(SHORT[x] for x in s.split("+"))


def plot_grid(preds, meta, mname, fname, path=None, lang="en"):
    """Rows: where the model learned. Columns: where it is tested. Predicted vs true for every pairing."""
    import matplotlib.pyplot as plt
    pl = lang == "pl"; rows = [("tiktok",), ("youtube",), ("twitter",), ("youtube", "twitter"), ("tiktok", "twitter"), ("tiktok", "youtube")]; fig, ax = plt.subplots(len(rows), 3, figsize=(11, 18), sharex=True, sharey=True); lim = [0, 7.5]
    for i, s in enumerate(rows):
        for j, t in enumerate(KNOWN):
            a = ax[i, j]; a.plot(lim, lim, color="black", lw=.8); a.fill_between(lim, [l - .3 for l in lim], [l + .3 for l in lim], color="grey", alpha=.15); a.set_xlim(lim); a.set_ylim(lim)
            if (i, j) in [(0, 0), (1, 1), (2, 2)] or t not in s:
                p = preds[(mname, fname, s, t)]; yy = meta.y.loc[p.index]; a.scatter(p, yy, s=3, alpha=.2, color=eu.PCOL[t]); e = yy - p
                a.text(.03, .97, (f"MAE {np.abs(e).mean():.2f} (×{10 ** np.abs(e).mean():.1f})\nbias {e.mean():+.2f}"), transform=a.transAxes, va="top", fontsize=9)
            else: a.text(.5, .5, "źródło = cel" if pl else "source = target", transform=a.transAxes, ha="center", color="#888888", fontsize=9)
            if i == 0: a.set_title(("test: " if pl else "tested on ") + SHORT[t], fontweight="bold", fontsize=11)
            if j == 0: a.set_ylabel(("uczony: " if pl else "trained: ") + "+".join(SHORT[x] for x in s) + "\n" + ("prawdziwe log10(wyśw.)" if pl else "true log10(views)"), fontsize=9)
            if i == len(rows) - 1: a.set_xlabel("przewidywane log10(wyśw.)" if pl else "predicted log10(views)", fontsize=9)
    fig.suptitle(("Skąd uczony, gdzie testowany: " if pl else "Trained where, tested where: ") + f"{mname}, {fname}", fontweight="bold", y=1.0); plt.tight_layout()
    if path: fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_lopo(preds, meta, mname, fname, path=None, lang="en"):
    """The main picture for phone and slide: trained on two platforms, tested on the third."""
    import matplotlib.pyplot as plt
    pl = lang == "pl"; fig, ax = plt.subplots(1, 3, figsize=(13, 4.6)); lim = [0, 7.5]
    for a, t in zip(ax, KNOWN):
        s = tuple(x for x in KNOWN if x != t); p = preds[(mname, fname, s, t)]; own = preds[(mname, fname, (t,), t)]; yy = meta.y.loc[p.index]; e = yy - p
        a.plot(lim, lim, color="black", lw=1); a.fill_between(lim, [l - .3 for l in lim], [l + .3 for l in lim], color="grey", alpha=.15); a.scatter(p, yy, s=4, alpha=.22, color=eu.PCOL[t]); a.set_xlim(lim); a.set_ylim(lim)
        a.set_title((f"Uczony na {SHORT[s[0]]}+{SHORT[s[1]]}, test na {SHORT[t]}" if pl else f"Trained on {SHORT[s[0]]}+{SHORT[s[1]]}, tested on {SHORT[t]}"), fontweight="bold", fontsize=11)
        a.text(.03, .97, (f"przeniesiony: MAE {np.abs(e).mean():.2f} (×{10 ** np.abs(e).mean():.1f})\nna własnej: MAE {np.abs(yy - own).mean():.2f}\nprzesunięcie: {e.mean():+.2f}".replace(".", ",") if pl else f"transferred: MAE {np.abs(e).mean():.2f} (×{10 ** np.abs(e).mean():.1f})\nown platform: MAE {np.abs(yy - own).mean():.2f}\nlevel shift: {e.mean():+.2f}"), transform=a.transAxes, va="top", fontsize=10)
        a.set_xlabel("przewidywane log10(wyświetlenia)" if pl else "predicted log10(views)"); a.set_ylabel("prawdziwe log10(wyświetlenia)" if pl else "true log10(views)")
    plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()


def plot_heat(res, fname, path=None, lang="en"):
    """MAE for each model and pairing: which pairings cross well, and whether the model family matters."""
    import matplotlib.pyplot as plt
    pl = lang == "pl"; r = res[res.features == fname].copy(); r["pair"] = r.source.map(_txt) + " → " + r.target.map(SHORT.get); order = [f"{_txt(s)} → {SHORT[t]}" for t in KNOWN for s in ["+".join(x) for x in [(t,)] + [z for z in SETS if t not in z]]]
    M = r.pivot(index="pair", columns="model", values="MAE").reindex(order); M = M[M.mean().sort_values().index]; fig, a = plt.subplots(figsize=(9, 6.2)); im = a.imshow(M.values, cmap="YlOrRd", vmin=.2, vmax=1.1, aspect="auto")
    a.set_xticks(range(M.shape[1])); a.set_xticklabels(M.columns, rotation=25, ha="right"); a.set_yticks(range(M.shape[0])); a.set_yticklabels(M.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): a.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    a.set_xlabel("model"); a.set_ylabel("uczony → testowany" if pl else "trained → tested"); a.set_title(("MAE log10 wyświetleń, cechy: " if pl else "MAE of log10 views, features: ") + fname, fontweight="bold"); fig.colorbar(im, ax=a, fraction=.03); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()


def plot_calibration(curves, path=None, lang="en"):
    import matplotlib.pyplot as plt
    pl = lang == "pl"; fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for a, t in zip(ax, KNOWN):
        c = curves[t]["random"]; b = curves[t]["brands"]; a.fill_between(c.n, c.low, c.high, color=eu.PCOL[t], alpha=.2); a.plot(c.n, c["median"], "o-", color=eu.PCOL[t], label="losowe posty" if pl else "random posts"); a.plot(b.n, b["median"], "s--", color="#444444", label="posty z kilku marek" if pl else "posts from a few brands")
        a.axhline(c.uncalibrated.iloc[0], color="#b5403a", ls=":", label="bez kalibracji" if pl else "no calibration"); a.axhline(c.oracle.iloc[0], color="black", ls=":", label="idealna kalibracja" if pl else "perfect calibration"); a.set_xscale("log"); a.set_xlabel("posty z prawdziwymi wyświetleniami (skala log)" if pl else "posts with true views (log scale)"); a.set_ylabel("MAE, log10 wyświetleń" if pl else "MAE, log10 views"); a.set_title(SHORT[t], fontweight="bold")
    ax[-1].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()
