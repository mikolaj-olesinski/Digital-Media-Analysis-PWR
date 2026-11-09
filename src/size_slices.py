"""E228: other views of the same-platform result (final CatBoost predictions, results_preds.csv).
1. Error without the smallest / largest 20% of posts, cut by true views (not known in use) and by the estimate (usable).
2. Error without posts with 0 reactions / with < 5 reactions is in the slide note (E228 card).
3. As classification: three ranges (<1k, 1k-10k, >10k) from the regression, accuracy, balanced accuracy, majority baseline.
4. As probability: ROC AUC of the estimate for 'large post' (top 20% of true views) and 'small post' (bottom 20%).
Writes data/processed/size_slices_numbers.json.
    .venv/bin/python src/size_slices.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
EDGES = [3, 4]


def f(e):
    return round(float(10 ** np.abs(e).mean()), 2)


def main():
    d = pd.read_csv(ROOT / "data/processed/results_preds.csv"); out = {}
    for p, g in d.groupby("platform"):
        e = (g.y - g.pred).values; r = {"posts": int(len(g))}
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
