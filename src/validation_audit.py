"""Validation audit (E114): how stable is the final model's MAE, is it calibrated, and do two cheap corrections
(isotonic recalibration and a per-reaction-tier median shift) improve it?

    .venv/bin/python src/validation_audit.py

Writes data/processed/validation_audit_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em  # noqa: E402
import reaction_tiers as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAT = ["tiktok", "youtube", "twitter"]
SEED = 42


def main():
    f, meta = em.load(ROOT)
    out = {}
    for p in PLAT:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG], meta.y[m], meta.brand[m]
        # 1. repeated grouped 80/20 splits: variance of the headline
        maes = []
        for seed in range(5):
            for tr, te in GroupShuffleSplit(n_splits=5, test_size=.2, random_state=seed).split(X, y, g):
                mod = rt.FINAL().fit(X.iloc[tr], y.iloc[tr])
                maes.append(float(np.abs(y.iloc[te].values - mod.predict(X.iloc[te])).mean()))
        # 2. out-of-fold predictions
        oof = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, g):
            oof[te] = rt.FINAL().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        base = float(np.abs(y.values - oof).mean())
        # calibration by decile of the prediction
        dec = pd.qcut(oof, 10, labels=False)
        cal = pd.DataFrame({"pred": oof, "y": y.values, "d": dec}).groupby("d").apply(lambda q: (q.y - q.pred).mean(), include_groups=False)
        # 3. isotonic recalibration, fitted out-of-fold
        iso = np.full(len(y), np.nan)
        tier = np.full(len(y), np.nan)
        tiers = pd.cut(rt.reactions(f)[m], rt.EDGES, labels=rt.TIERS)
        for tr, te in GroupKFold(5).split(X, y, g):
            ir = IsotonicRegression(out_of_bounds="clip").fit(oof[tr], y.values[tr])
            iso[te] = ir.predict(oof[te])
            med = pd.Series(y.values[tr] - oof[tr]).groupby(tiers.iloc[tr].values).median()
            tier[te] = oof[te] + pd.Series(tiers.iloc[te].values).map(med).astype(float).fillna(0).values
        out[p] = {"posts": int(len(y)),
                  "repeated_cv": {"mean": round(float(np.mean(maes)), 3), "sd": round(float(np.std(maes)), 3),
                                  "p10": round(float(np.percentile(maes, 10)), 3), "p90": round(float(np.percentile(maes, 90)), 3)},
                  "oof_MAE": round(base, 3),
                  "isotonic_MAE": round(float(np.abs(y.values - iso).mean()), 3),
                  "tier_shift_MAE": round(float(np.abs(y.values - tier).mean()), 3),
                  "bias_by_decile": {str(k): round(float(v), 2) for k, v in cal.items()},
                  "residual_mean": round(float((y.values - oof).mean()), 3),
                  "residual_sd": round(float((y.values - oof).std()), 3)}
        print(p, out[p])
    (ROOT / "data" / "processed" / "validation_audit_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
