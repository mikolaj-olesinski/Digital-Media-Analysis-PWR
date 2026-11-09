"""Alert budget (E106): with a limited number of alerts per period, how many of the truly viral posts (top 1%) does each
score catch? The score is the views model, raw likes, or Brand24's reach. Complements E104 (precision at one k).

    .venv/bin/python src/alert_budget.py

Writes data/processed/alert_budget_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAT = ["tiktok", "youtube", "twitter"]
SEED = 42
BUDGETS = [10, 25, 50, 100, 200]


def main():
    d = eu.load(ROOT)
    out = {}
    for p in PLAT:
        q = d[(d.platform == p) & (pd.to_numeric(d.view_count, errors="coerce") > 0)].copy()
        q["log_likes"] = np.log1p(q.likes.fillna(0))
        q["log_comments"] = np.log1p(q.comments.fillna(0))
        label = (q.view_count >= q.view_count.quantile(.99)).astype(int).values
        pred = np.zeros(len(q))
        for tr, te in GroupKFold(5).split(q, label, q.brand):
            m = CatBoostClassifier(iterations=300, depth=4, verbose=0, random_seed=SEED, allow_writing_files=False, thread_count=4)
            m.fit(q.iloc[tr][["log_likes", "log_comments"]], label[tr])
            pred[te] = m.predict_proba(q.iloc[te][["log_likes", "log_comments"]])[:, 1]
        scores = {"model": pred, "likes": q.likes.fillna(0).values, "brand24": q.b24_estimated_reach.fillna(0).values}
        out[p] = {"positives": int(label.sum()), "budget": {}}
        for k in BUDGETS:
            row = {}
            for name, s in scores.items():
                top = np.argsort(-s)[:k]
                row[name] = round(float(label[top].sum() / label.sum()), 3)  # recall
            out[p]["budget"][str(k)] = row
        # alerts needed to catch half of the top 1%
        half = {}
        for name, s in scores.items():
            order = np.argsort(-s); cum = np.cumsum(label[order])
            half[name] = int(np.searchsorted(cum, 0.5 * label.sum()) + 1)
        out[p]["alerts_for_50pct_recall"] = half
        print(p, json.dumps(out[p], indent=1))
    (ROOT / "data" / "processed" / "alert_budget_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
