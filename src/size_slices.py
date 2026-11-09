"""E228: other views of the same-platform result (final CatBoost predictions, results_preds.csv).
1. Error without the smallest / largest 20% of posts, cut by true views (not known in use) and by the estimate (usable).
2. Error by reaction count (likes + comments + shares, known in use): 0 / 1-5 / 6-100 / >100, and without posts with <= 5.
3. As classification: three ranges (<1k, 1k-10k, >10k) from the regression, accuracy, balanced accuracy, majority baseline.
4. As probability: ROC AUC of the estimate for 'large post' (top 20% of true views) and 'small post' (bottom 20%).
Writes data/processed/size_slices_numbers.json.
    .venv/bin/python src/size_slices.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em  # noqa: E402
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
EDGES = [3, 4]


def f(e):
    return round(float(10 ** np.abs(e).mean()), 2)


def main():
    d = pd.read_csv(ROOT / "data/processed/results_preds.csv"); out = {}; fe_, meta = em.load(ROOT)
    for p, g in d.groupby("platform"):
        e = (g.y - g.pred).values; r = {"posts": int(len(g))}
        m = (meta.platform == p).values; assert np.allclose(g.y.values, meta.y[m].values)
        n = sum(np.expm1(fe_[m][c].fillna(0).values) for c in ["log_likes", "log_comments", "log_shares"])
        G = {"0": n == 0, "1-5": (n >= 1) & (n <= 5), "6-100": (n > 5) & (n <= 100), ">100": n > 100, "without <= 5 (kept)": n > 5, "without 0 (kept)": n > 0}
        r["by reactions"] = {k: {"factor": f(e[v]) if v.any() else None, "share of posts": round(float(v.mean()), 3)} for k, v in G.items()}
        for key, name in (("y", "cut by true views (not known in use)"), ("pred", "cut by the estimate (usable)")):
            lo, hi = g[key].quantile(.2), g[key].quantile(.8); v = g[key].values
            r[name] = {"all": f(e), "without smallest 20%": f(e[v > lo]), "without largest 20%": f(e[v < hi]), "middle 60%": f(e[(v > lo) & (v < hi)])}
        t, q = np.digitize(g.y, EDGES), np.digitize(g.pred, EDGES)
        r["three ranges"] = {"accuracy": round(float((t == q).mean()), 3), "balanced accuracy": round(float(balanced_accuracy_score(t, q)), 3),
                             "majority baseline": round(float(pd.Series(t).value_counts(normalize=True).iloc[0]), 3),
                             "true share per range": [round(float((t == k).mean()), 3) for k in range(3)]}
        r["AUC"] = {"large post (top 20%)": round(float(roc_auc_score(g.y >= g.y.quantile(.8), g.pred)), 3),
                    "small post (bottom 20%)": round(float(roc_auc_score(g.y <= g.y.quantile(.2), -g.pred)), 3)}
        out[p] = r
    (ROOT / "data/processed/size_slices_numbers.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
