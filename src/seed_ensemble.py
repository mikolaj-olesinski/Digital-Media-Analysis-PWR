"""Seed averaging (E134): CatBoost is a single-seed model. Average 5 seeds per fold; does the variance drop enough to
lower MAE?

    .venv/bin/python src/seed_ensemble.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44, 45, 46]


def main():
    f, meta = em.load(ROOT)
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        one = np.full(len(y), np.nan); ens = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, g):
            preds = []
            for s in SEEDS:
                mod = CatBoostRegressor(loss_function="MAE", iterations=300, verbose=0, random_seed=s,
                                        allow_writing_files=False, thread_count=4)
                preds.append(mod.fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te]))
            one[te] = preds[0]; ens[te] = np.mean(preds, axis=0)
        out[p] = {"posts": int(len(y)), "single_MAE": round(float(np.abs(y.values - one).mean()), 3),
                  "ensemble_MAE": round(float(np.abs(y.values - ens).mean()), 3),
                  "mean_abs_change": round(float(np.abs(ens - one).mean()), 3)}
        print(p, out[p])
    (ROOT / "data" / "processed" / "seed_ensemble_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
