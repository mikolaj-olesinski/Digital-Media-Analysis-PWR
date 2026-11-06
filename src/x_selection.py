"""Selection in X view counts (E105): 25% of X posts have no view count. Are they different, can we predict the
missingness, and how far does training on the observed posts shift an average-reach estimate for the whole X sample?

    .venv/bin/python src/x_selection.py

Writes data/processed/x_selection_numbers.json.
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

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
C2 = ["log_likes", "log_comments"]


def main():
    d = eu.load(ROOT)
    x = d[d.platform == "twitter"].copy()
    x["age_days"] = (pd.to_datetime(x.metrics_updated_at, errors="coerce") - pd.to_datetime(x.created_at, errors="coerce")).dt.total_seconds() / 86400
    x["reactions"] = x.likes.fillna(0) + x.comments.fillna(0) + x.shares.fillna(0)
    x["log_likes"] = np.log1p(x.likes.fillna(0))
    x["log_comments"] = np.log1p(x.comments.fillna(0))
    x["log_reactions"] = np.log1p(x.reactions)
    x["log_followers"] = np.log10(x.author_followers.where(x.author_followers > 0))
    x["has_views"] = pd.to_numeric(x.view_count, errors="coerce").notna()
    x["y"] = np.log10(pd.to_numeric(x.view_count, errors="coerce").where(lambda v: v > 0))

    obs, miss = x[x.has_views], x[~x.has_views]
    out = {"posts": int(len(x)), "observed": int(len(obs)), "missing": int(len(miss)),
           "missing_share": round(float(1 - len(obs) / len(x)), 3)}

    def med(q, c):
        return round(float(pd.to_numeric(q[c], errors="coerce").median()), 2)
    out["medians"] = {c: {"observed": med(obs, c), "missing": med(miss, c)}
                      for c in ["reactions", "author_followers", "age_days"]}
    out["missing_by_year"] = {str(k): round(float(v), 3) for k, v in x.groupby(pd.to_datetime(x.created_at).dt.year).has_views.apply(lambda s: 1 - s.mean()).items()}

    # selection model: can we tell a missing post from an observed one?
    Z = x[["log_reactions", "log_followers", "age_days"]].fillna(0)
    lab = x.has_views.astype(int).values
    pred = np.zeros(len(x))
    for tr, te in GroupKFold(5).split(Z, lab, x.brand):
        m = CatBoostClassifier(iterations=300, depth=4, verbose=0, random_seed=SEED, allow_writing_files=False, thread_count=4)
        m.fit(Z.iloc[tr], lab[tr]); pred[te] = m.predict_proba(Z.iloc[te])[:, 1]
    out["selection_auc"] = round(float(roc_auc_score(lab, pred)), 3)

    # views model on the observed posts, applied to every X post
    tr = obs[obs.y.notna()]
    mod = make_pipeline(StandardScaler(), Ridge()).fit(tr[C2], tr.y)
    x["pred"] = mod.predict(x[C2].fillna(0))
    out["predicted_views"] = {
        "observed_median": round(float((10 ** x.loc[x.has_views, "pred"]).median()), 1),
        "missing_median": round(float((10 ** x.loc[~x.has_views, "pred"]).median()), 1),
        "zero_median": round(float((10 ** x.loc[x.has_views & (pd.to_numeric(x.view_count, errors="coerce") == 0), "pred"]).median()), 1),
    }
    # inverse-probability weighting: does the observed subset over/under-state the average predicted reach?
    w = 1 / np.clip(pred[x.has_views.values], .05, .95)
    obs_pred = 10 ** x.loc[x.has_views, "pred"]
    out["mean_reach"] = {"observed_unweighted": round(float(obs_pred.mean()), 1),
                         "observed_ipw": round(float(np.average(obs_pred, weights=w)), 1),
                         "full_sample_model": round(float((10 ** x.pred).mean()), 1)}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "x_selection_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
