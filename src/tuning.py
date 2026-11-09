"""Hyperparameter search over ten model families (notebooks/models/07_tuning.ipynb).
Nested, brand-grouped CV: the outer 5 folds estimate the error, the inner 3 folds pick the settings, so the reported error never sees the data used for tuning.
Features are followers + engagement + missing flags, the set that carries nearly all the signal."""
import json, warnings
import numpy as np, pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
import eda_utils as eu, extra_models as em, modeling as mo, new_ideas as ni
from validation import mae_ci

SEED, N_ITER = 42, 40


def families():
    """name -> (default-settings estimator, search space). Scaled models are pipelines, so their parameters carry the step prefix."""
    return {
        "ridge": (make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": loguniform(1e-3, 1e3)}),
        "kNN": (make_pipeline(StandardScaler(), KNeighborsRegressor()), {"kneighborsregressor__n_neighbors": randint(3, 200), "kneighborsregressor__weights": ["uniform", "distance"], "kneighborsregressor__p": [1, 2]}),
        "SVR (RBF)": (make_pipeline(StandardScaler(), SVR()), {"svr__C": loguniform(1e-1, 1e2), "svr__gamma": loguniform(1e-3, 1), "svr__epsilon": uniform(.01, .5)}),
        "MLP": (make_pipeline(StandardScaler(), MLPRegressor(max_iter=600, early_stopping=True, random_state=SEED)), {"mlpregressor__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64)], "mlpregressor__alpha": loguniform(1e-5, 1e-1), "mlpregressor__learning_rate_init": loguniform(1e-4, 1e-2)}),
        "random forest": (RandomForestRegressor(random_state=SEED), {"n_estimators": randint(100, 500), "min_samples_leaf": randint(1, 40), "max_features": uniform(.2, .8), "max_depth": [None, 6, 10, 16]}),
        "extra trees": (ExtraTreesRegressor(random_state=SEED), {"n_estimators": randint(100, 500), "min_samples_leaf": randint(1, 40), "max_features": uniform(.2, .8), "max_depth": [None, 6, 10, 16]}),
        "boosting (sklearn)": (HistGradientBoostingRegressor(random_state=SEED), {"loss": ["squared_error", "absolute_error"], "learning_rate": loguniform(.01, .3), "max_iter": randint(50, 600), "max_leaf_nodes": randint(4, 64), "min_samples_leaf": randint(5, 80), "l2_regularization": loguniform(1e-4, 10)}),
        "LightGBM": (LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1), {"objective": ["l2", "l1"], "learning_rate": loguniform(.01, .3), "n_estimators": randint(50, 600), "num_leaves": randint(4, 64), "min_child_samples": randint(5, 80), "subsample": uniform(.5, .5), "subsample_freq": [1], "colsample_bytree": uniform(.5, .5), "reg_lambda": loguniform(1e-4, 10)}),
        "XGBoost": (XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0), {"objective": ["reg:squarederror", "reg:absoluteerror"], "learning_rate": loguniform(.01, .3), "n_estimators": randint(50, 600), "max_depth": randint(2, 9), "min_child_weight": loguniform(.5, 30), "subsample": uniform(.5, .5), "colsample_bytree": uniform(.5, .5), "reg_lambda": loguniform(1e-3, 10)}),
        "CatBoost": (CatBoostRegressor(random_seed=SEED, thread_count=1, verbose=0, allow_writing_files=False, iterations=300), {"loss_function": ["RMSE", "MAE"], "learning_rate": loguniform(.01, .3), "iterations": randint(100, 500), "depth": randint(2, 9), "l2_leaf_reg": loguniform(.1, 30)}),
    }


def clone_default(est):
    from sklearn.base import clone
    return clone(est)


def plain(v):
    """Make a sampled parameter JSON-friendly."""
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)): return round(float(v), 5)
    return v if isinstance(v, (int, str, type(None))) else str(v)


def nested(f, meta, plat, names=None, n_iter=N_ITER, extra=None):
    """For one platform: default-settings OOF, nested-tuned OOF, and the settings picked in each outer fold. `extra` adds label-free columns aligned to f."""
    m = (meta.platform == plat).values; X = (f[em.ENG] if extra is None else pd.concat([f[em.ENG], extra], axis=1))[m].fillna(0).reset_index(drop=True); M = meta[m].reset_index(drop=True); y, g = M.y, M.brand; out, preds = [], {}
    for name, (est, space) in families().items():
        if names and name not in names: continue
        pd_, pt = np.full(len(y), np.nan), np.full(len(y), np.nan); picks, inner = [], []
        for tr, te in GroupKFold(5).split(X, y, g):
            pd_[te] = clone_default(est).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
            s = RandomizedSearchCV(clone_default(est), space, n_iter=n_iter, scoring="neg_mean_absolute_error", cv=list(GroupKFold(3).split(X.iloc[tr], y.iloc[tr], g.iloc[tr])), n_jobs=-1, random_state=SEED, refit=True, error_score="raise")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore"); s.fit(X.iloc[tr], y.iloc[tr])
            pt[te] = s.predict(X.iloc[te]); picks.append({k.split("__")[-1]: plain(v) for k, v in s.best_params_.items()}); inner.append(-s.best_score_)
        a, b = mae_ci((y - pd_).values, g.values), mae_ci((y - pt).values, g.values); gain = ni.paired_diff(y.values, pd_, pt, g.values)
        out.append({"platform": eu.NAMES[plat], "model": name, "default MAE": a[0], "default low": a[1], "default high": a[2], "tuned MAE": b[0], "tuned low": b[1], "tuned high": b[2], "gain": gain[0], "gain low": gain[1], "gain high": gain[2], "inner MAE": float(np.mean(inner)), "picks": picks})
        preds[name] = pt; print(eu.NAMES[plat], name, round(a[0], 3), "->", round(b[0], 3), flush=True)
    t = pd.DataFrame(out); top = t.nsmallest(3, "inner MAE").model.tolist(); pb = np.mean([preds[k] for k in top], axis=0); c = mae_ci((y - pb).values, g.values); best = t.loc[t["tuned MAE"].idxmin()]
    blend = {"platform": eu.NAMES[plat], "members": top, "MAE": c[0], "low": c[1], "high": c[2]}
    return t, blend, {"y": y.values, "pred": preds, "brand": g.values}


def plot_families(tab, path=None, lang="en"):
    """Default vs tuned error per family and platform, with brand-cluster 95% CI."""
    import matplotlib.pyplot as plt
    pl = lang == "pl"; order = tab.groupby("model")["tuned MAE"].mean().sort_values().index.tolist(); fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for a, p in zip(ax, mo.PLATS):
        r = tab[tab.platform == eu.NAMES[p]].set_index("model").reindex(order); yy = np.arange(len(r))
        a.errorbar(r["default MAE"], yy + .16, xerr=[r["default MAE"] - r["default low"], r["default high"] - r["default MAE"]], fmt="o", color="#9a9a9a", ms=5, capsize=2, label="domyślne ustawienia" if pl else "default settings")
        a.errorbar(r["tuned MAE"], yy - .16, xerr=[r["tuned MAE"] - r["tuned low"], r["tuned high"] - r["tuned MAE"]], fmt="o", color=eu.PCOL[p], ms=5, capsize=2, label="po strojeniu" if pl else "tuned")
        a.set_yticks(yy); a.set_yticklabels(order); a.invert_yaxis(); a.grid(axis="x", alpha=.25); a.set_xlabel("MAE, log10 wyświetleń" if pl else "MAE, log10 views"); a.set_title(eu.NAMES[p], fontweight="bold")
    ax[0].set_ylabel("model"); ax[-1].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False)
    fig.suptitle("Dziesięć rodzin modeli przed strojeniem i po nim (zagnieżdżona CV po markach)" if pl else "Ten model families before and after tuning (nested brand-grouped CV)", fontweight="bold"); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()
