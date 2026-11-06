"""Distributional model (E129): CatBoost RMSEWithUncertainty predicts a mean and a variance per post. Compare its MAE
and its 90% band (mean +- 1.645 sd) with the MAE point model plus CQR (E118).

    .venv/bin/python src/dist_uncertainty.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
Z = 1.6449


def main():
    f, meta = em.load(ROOT)
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        oof_mu = np.full(len(y), np.nan); oof_sd = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, g):
            mod = CatBoostRegressor(loss_function="RMSEWithUncertainty", iterations=300, verbose=0,
                                    random_seed=SEED, allow_writing_files=False, thread_count=4)
            mod.fit(X.iloc[tr], y.iloc[tr])
            pr = mod.predict(X.iloc[te]); oof_mu[te] = pr[:, 0]; oof_sd[te] = np.sqrt(np.maximum(pr[:, 1], 1e-6))
        lo, hi = oof_mu - Z * oof_sd, oof_mu + Z * oof_sd
        out[p] = {"posts": int(len(y)), "MAE_mean": round(float(np.abs(y.values - oof_mu).mean()), 3),
                  "coverage90": round(float(((y.values >= lo) & (y.values <= hi)).mean()), 3),
                  "median_width": round(float(np.median(hi - lo)), 2),
                  "bias": round(float((y.values - oof_mu).mean()), 3)}
        print(p, out[p])
    (ROOT / "data" / "processed" / "dist_uncertainty_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
