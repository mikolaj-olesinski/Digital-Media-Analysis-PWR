"""Confidence flags (E108): can the metadata known before the check (missing fields, age, zero reactions, platform,
reach) tell which posts will have a large error? If yes, a product can flag or re-check the risky mentions.

    .venv/bin/python src/confidence_flags.py

Writes data/processed/confidence_flags_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import validation as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
C2 = ["log_likes", "log_comments"]


def main():
    d = eu.load(ROOT)
    q = d[d.platform.isin(["tiktok", "youtube", "twitter"]) & (pd.to_numeric(d.view_count, errors="coerce") > 0)].copy()
    q["y"] = np.log10(q.view_count)
    q["age_days"] = (pd.to_datetime(q.metrics_updated_at, errors="coerce") - pd.to_datetime(q.created_at, errors="coerce")).dt.total_seconds() / 86400
    q["log_likes"] = np.log1p(q.likes.fillna(0)); q["log_comments"] = np.log1p(q.comments.fillna(0))
    q["log_followers"] = np.log10(q.author_followers.where(q.author_followers > 0)).fillna(0)
    q["zero_reactions"] = ((q.likes.fillna(0) + q.comments.fillna(0) + q.shares.fillna(0)) == 0).astype(int)
    q["has_likes"] = q.likes.notna().astype(int); q["has_comments"] = q.comments.notna().astype(int); q["has_shares"] = q.shares.notna().astype(int)
    q["log_reach"] = np.log10(q.b24_estimated_reach.clip(lower=1))
    q["plat"] = q.platform.map({"tiktok": 0, "youtube": 1, "twitter": 2})
    # views model, out-of-fold by brand
    pred = va.grouped_cv_predict(lambda: make_pipeline(StandardScaler(), Ridge()), q[C2], q.y, q.brand)
    q["err"] = np.abs(q.y - pred)
    q["big"] = (q.err > 0.5).astype(int)  # worse than x3
    F = ["log_likes", "log_comments", "log_followers", "zero_reactions", "has_likes", "has_comments", "has_shares",
         "age_days", "log_reach", "plat"]
    risk = np.zeros(len(q))
    for tr, te in GroupKFold(5).split(q[F], q.big, q.brand):
        m = CatBoostClassifier(iterations=300, depth=4, verbose=0, random_seed=SEED, allow_writing_files=False, thread_count=4)
        m.fit(q.iloc[tr][F], q.big.iloc[tr]); risk[te] = m.predict_proba(q.iloc[te][F])[:, 1]
    out = {"posts": int(len(q)), "big_error_share": round(float(q.big.mean()), 3),
           "AUC_big_error": round(float(roc_auc_score(q.big, risk)), 3), "flag": {}}
    for share in (0.2, 0.3, 0.5):
        k = int(share * len(q)); flag = np.argsort(-risk)[:k]
        removed = float(q.err.values[flag].sum() / q.err.sum())
        out["flag"][str(share)] = {"flagged": k, "share_of_total_error_flagged": round(removed, 3),
                                   "MAE_all": round(float(q.err.mean()), 3),
                                   "MAE_unflagged": round(float(q.err.values[np.setdiff1d(np.arange(len(q)), flag)].mean()), 3)}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "confidence_flags_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
