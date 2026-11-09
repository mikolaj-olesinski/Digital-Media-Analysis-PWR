"""Calibrated brand totals with intervals (E109): E107 showed the X brand totals are low by x1.5 and that 49% of brands
are still off by >2x after the model. Here a one-number calibration is fitted out-of-fold and an 80% interval is built
from brand-level residuals, with coverage on held-out brands.

    .venv/bin/python src/brand_totals_cal.py

Writes data/processed/brand_totals_cal_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import validation as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C2 = ["log_likes", "log_comments"]


def main():
    d = eu.load(ROOT)
    q = d[(d.platform == "twitter") & (pd.to_numeric(d.view_count, errors="coerce") > 0)].copy()
    q["y"] = np.log10(q.view_count)
    q["log_likes"] = np.log1p(q.likes.fillna(0)); q["log_comments"] = np.log1p(q.comments.fillna(0))
    q["pred"] = va.grouped_cv_predict(lambda: make_pipeline(StandardScaler(), Ridge()), q[C2], q.y, q.brand)
    g = q.groupby("brand").agg(n=("y", "size"), true=("view_count", "sum"),
                               model=("pred", lambda s: float(np.sum(10 ** s))), b24=("b24_estimated_reach", "sum"))
    g = g[g.n >= 5].copy()
    g["r_model"] = np.log10(g.model / g.true)
    g["r_b24"] = np.log10(g.b24 / g.true)
    out = {"brands": int(len(g)),
           "uncorrected": {"share_off_2x": round(float((g.r_model.abs() > np.log10(2)).mean()), 2),
                           "median_err": round(float(g.r_model.abs().median()), 2),
                           "median_ratio": round(float(g.r_model.median()), 2)},
           "b24": {"share_off_2x": round(float((g.r_b24.abs() > np.log10(2)).mean()), 2),
                   "median_err": round(float(g.r_b24.abs().median()), 2)}}
    # out-of-fold calibration + interval
    r = np.zeros(len(g)); lo = np.zeros(len(g)); hi = np.zeros(len(g)); covered = []
    for tr, te in GroupKFold(5).split(g, g.true, g.index):
        c = float(np.median(np.log10(g.true.values[tr] / g.model.values[tr])))
        res = np.log10(g.true.values[tr] / g.model.values[tr]) - c
        hw = float(np.quantile(np.abs(res), .8))
        r[te] = np.log10(g.true.values[te] / g.model.values[te]) - c
        lo[te] = np.log10(g.model.values[te]) + c - hw
        hi[te] = np.log10(g.model.values[te]) + c + hw
        covered.append(float((np.abs(r[te]) <= hw).mean()))
    out["calibrated"] = {"share_off_2x": round(float((np.abs(r) > np.log10(2)).mean()), 2),
                         "median_err": round(float(np.median(np.abs(r))), 2),
                         "bias_after_calibration": round(float(r.mean()), 2),
                         "interval80_coverage": round(float(np.mean(covered)), 3)}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "brand_totals_cal_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
