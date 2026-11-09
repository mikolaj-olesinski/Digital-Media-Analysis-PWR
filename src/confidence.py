"""E231: a confidence model on top of the final same-platform estimate: which posts can we trust?
Base: final CatBoost out-of-fold predictions (results_preds.csv, E55 component folds), TT / YT / X.
Confidence models (out-of-fold, same component folds, features em.ENG + the estimate itself, all known in use):
  - error model: CatBoost regression of |error| (log10) -> rank posts by predicted error;
  - within-x2 classifier: CatBoost P(|error| < log10 2) -> keep posts with P >= threshold;
  - simple rule: total reactions (likes + comments + shares), more = trusted;
  - oracle: true |error| (upper bound, not usable).
Risk-coverage: keep the most trusted 50-100% of posts, error x10^MAE on kept and on the rest.
Writes data/processed/confidence_numbers.json.
    .venv/bin/python -W ignore src/confidence.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, limits as lm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
L2 = np.log10(2); COV = [.5, .6, .7, .8, .9, 1.]; THR = [.5, .6, .7, .8]
kw = dict(random_seed=42, iterations=300, verbose=0, allow_writing_files=False, thread_count=4)


def fx(e):
    return round(float(10 ** np.abs(e).mean()), 2) if len(e) else None


def oof(X, t, groups, clf):
    p = np.zeros(len(X))
    for tr, te in GroupKFold(5).split(X, t, groups):
        if clf:
            p[te] = CatBoostClassifier(**kw).fit(X.iloc[tr], t[tr]).predict_proba(X.iloc[te])[:, 1]
        else:
            p[te] = CatBoostRegressor(loss_function="RMSE", **kw).fit(X.iloc[tr], t[tr]).predict(X.iloc[te])
    return p


def curve(e, risk):
    o = np.argsort(risk, kind="stable"); r = {}
    for c in COV:
        k = int(round(c * len(e))); r[f"{int(c * 100)}%"] = {"kept": fx(e[o[:k]]), "rest": fx(e[o[k:]]), "kept within x2": round(float((np.abs(e[o[:k]]) < L2).mean()), 3)}
    return r


def main():
    f, meta = em.load(ROOT); raw = eu.load(ROOT).loc[f.index]
    comp = lm.components(meta.reset_index(drop=True), raw.reset_index(drop=True))
    d = pd.read_csv(ROOT / "data/processed/results_preds.csv"); out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values; g = d[d.platform == p]
        assert np.allclose(g.y.values, meta.y[m].values)
        e = (g.y - g.pred).values; a = np.abs(e); ok = (a < L2).astype(int)
        X = f[m][em.ENG].reset_index(drop=True).copy(); X["estimate"] = g.pred.values
        grp = pd.Series(comp[m]).values
        err = oof(X, a, grp, False); pr = oof(X, ok, grp, True)
        n = sum(np.expm1(f[m][c].fillna(0).values) for c in ["log_likes", "log_comments", "log_shares"])
        r = {"posts": int(len(e)), "all": fx(e), "all within x2": round(float(ok.mean()), 3),
             "AUC 'error > x2'": {"error model": round(float(roc_auc_score(1 - ok, err)), 3),
                                  "within-x2 classifier": round(float(roc_auc_score(1 - ok, -pr)), 3),
                                  "reactions rule": round(float(roc_auc_score(1 - ok, -n)), 3)},
             "Spearman predicted vs true |error|": round(float(pd.Series(err).corr(pd.Series(a), "spearman")), 3),
             "risk-coverage": {"error model": curve(e, err), "within-x2 classifier": curve(e, -pr),
                               "reactions rule": curve(e, -n + 1e-9 * err), "oracle (true error)": curve(e, a)},
             "classifier thresholds": {}}
        for t in THR:
            k = pr >= t
            r["classifier thresholds"][f"P >= {t}"] = {"share of posts": round(float(k.mean()), 3), "actual within x2": round(float(ok[k].mean()), 3) if k.any() else None,
                                                     "factor kept": fx(e[k]), "factor rest": fx(e[~k])}
        out[p] = r
    (ROOT / "data/processed/confidence_numbers.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
