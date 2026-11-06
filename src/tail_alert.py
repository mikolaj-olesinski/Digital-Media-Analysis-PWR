"""Viral alert at the top 1% (E104): reach is dominated by a handful of posts (E96). Can a model flag, from reactions
and comments, the posts that land in the top 1% of views? Compared with ranking by Brand24's reach and by raw likes.

    .venv/bin/python src/tail_alert.py

Writes data/processed/tail_alert_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAT = ["tiktok", "youtube", "twitter"]
SEED = 42


def cv_score(q, score, label):
    """Precision in the k highest scores, k = number of positives; recall and lift."""
    k = int(label.sum())
    top = np.argsort(-np.asarray(score))[:k]
    prec = float(label.iloc[top].mean())
    return {"precision_at_k": round(prec, 3), "recall": round(prec, 3), "lift": round(prec / label.mean(), 1)}


def main():
    d = eu.load(ROOT)
    out = {}
    for p in PLAT:
        q = d[(d.platform == p) & (pd.to_numeric(d.view_count, errors="coerce") > 0)].copy()
        q["log_likes"] = np.log1p(q.likes.fillna(0))
        q["log_comments"] = np.log1p(q.comments.fillna(0))
        out[p] = {}
        for frac in (0.01, 0.05):
            label = (q.view_count >= q.view_count.quantile(1 - frac)).astype(int)
            pred = np.zeros(len(q))
            for tr, te in GroupKFold(5).split(q, label, q.brand):
                m = CatBoostClassifier(iterations=300, depth=4, verbose=0, random_seed=SEED, allow_writing_files=False, thread_count=4)
                m.fit(q.iloc[tr][["log_likes", "log_comments"]], label.iloc[tr])
                pred[te] = m.predict_proba(q.iloc[te][["log_likes", "log_comments"]])[:, 1]
            auc = roc_auc_score(label, pred)
            row = {"positives": int(label.sum()), "base_rate": round(float(label.mean()), 3), "AUC": round(float(auc), 3),
                   **cv_score(q, pred, label)}
            # baselines: raw likes; Brand24 reach (only meaningful on X, elsewhere it is the view count itself)
            row["likes"] = cv_score(q, q.likes.fillna(0).values, label)
            row["brand24"] = cv_score(q, q.b24_estimated_reach.fillna(0).values, label)
            out[p][str(frac)] = row
        print(p, json.dumps(out[p], indent=1))
    (ROOT / "data" / "processed" / "tail_alert_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
