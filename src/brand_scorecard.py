"""Brand scorecard (E102): if Brand24 ranked brands and posts by reach, how often would it be right? Compare the
Brand24 reach, a views model (ridge on likes + comments, out-of-fold by brand) and the measured views on the platforms
that expose views.

    .venv/bin/python src/brand_scorecard.py

Writes data/processed/brand_scorecard_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import validation as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAT = ["tiktok", "youtube", "twitter"]
C2 = ["log_likes", "log_comments"]


def ridge():
    return make_pipeline(StandardScaler(), Ridge())


def main():
    d = eu.load(ROOT)
    out = {}
    for p in PLAT:
        q = d[(d.platform == p) & (pd.to_numeric(d.view_count, errors="coerce") > 0)].copy()
        q["y"] = np.log10(q.view_count)
        q["log_likes"] = np.log1p(q.likes.fillna(0))
        q["log_comments"] = np.log1p(q.comments.fillna(0))
        q["pred"] = va.grouped_cv_predict(ridge, q[C2], q.y, q.brand)
        q["b24"] = np.log10(q.b24_estimated_reach.clip(lower=1))
        # brand totals, brands with >= 5 posts
        g = q.groupby("brand").agg(n=("y", "size"), views=("view_count", "sum"),
                                   b24=("b24_estimated_reach", "sum"),
                                   pred=("pred", lambda s: float(np.sum(10 ** s))))
        g = g[g.n >= 5]
        actual = g.views.values
        t_b24 = kendalltau(actual, g.b24.values)[0]
        t_model = kendalltau(actual, g.pred.values)[0]
        top5 = lambda v: set(g.index[np.argsort(-np.asarray(v))[:5]])
        # alert quality: top 10% of posts by score against the true top 10%
        k = max(1, int(round(.1 * len(q))))
        viral = set(q.nlargest(k, "view_count").index)
        def prec(score):
            top = set(q.nlargest(k, score).index)
            return round(len(top & viral) / len(top), 3)
        out[p] = {"posts": int(len(q)), "brands": int(len(g)),
                  "kendall_b24": round(float(t_b24), 3), "kendall_model": round(float(t_model), 3),
                  "top5_overlap_b24": len(top5(actual) & top5(g.b24.values)),
                  "top5_overlap_model": len(top5(actual) & top5(g.pred.values)),
                  "alert_precision_b24": prec("b24_estimated_reach"),
                  "alert_precision_model": prec("pred"),
                  "alert_precision_reactions": prec("likes"),
                  "auc_model": round(float(roc_auc_score(q.index.isin(viral), q.pred)), 3),
                  "auc_b24": round(float(roc_auc_score(q.index.isin(viral), q.b24)), 3)}
        print(p, out[p])
    (ROOT / "data" / "processed" / "brand_scorecard_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
