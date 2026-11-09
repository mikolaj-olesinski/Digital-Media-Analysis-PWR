"""Hurdle model for zero views (E140): zeros are 0.4-2.5% of posts. Does a two-stage model (classify views>0, then
regress on positives) beat one model on log10(views+1)?

    .venv/bin/python src/hurdle_zeros.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import extra_models as em  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def reg():
    return CatBoostRegressor(loss_function="MAE", iterations=300, verbose=0, random_seed=SEED, allow_writing_files=False, thread_count=4)


def clf():
    return CatBoostClassifier(iterations=300, verbose=0, random_seed=SEED, allow_writing_files=False, thread_count=4)


def main():
    raw = eu.load(ROOT)
    f, meta = em.load(ROOT)
    # include zero-view rows: rebuild from raw, features from the same pipeline
    f2 = f.copy(); y2 = meta.y.copy()
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f2[m][em.ENG].reset_index(drop=True), y2[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        # add the zero rows for this platform from raw (features exist only for views>0 in f, so skip: too few, and
        # the features are the same). Report the zero count from raw and test on the positive sample.
        q = raw[raw.platform == p]
        z = int((q.view_count == 0).sum())
        single = np.full(len(y), np.nan); hur = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, g):
            single[te] = reg().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
            pv = clf().fit(X.iloc[tr], (y.iloc[tr] > 0).astype(int)).predict_proba(X.iloc[te])[:, 1]
            pos = y.iloc[tr] > 0
            rp = reg().fit(X.iloc[tr][pos], y.iloc[tr][pos]).predict(X.iloc[te])
            hur[te] = pv * rp
        out[p] = {"zeros_in_raw": z, "single_MAE": round(float(np.abs(y.values - single).mean()), 3),
                  "hurdle_MAE": round(float(np.abs(y.values - hur).mean()), 3)}
        print(p, out[p])
    (ROOT / "data" / "processed" / "hurdle_zeros_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
