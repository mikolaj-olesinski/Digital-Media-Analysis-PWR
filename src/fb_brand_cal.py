"""FB brand totals after calibration (E158): E157 found the totals are x2 low. Fit a one-number factor leave-one-brand-out
and build an 80% interval from brand residuals; does the share off by >2x fall and does the interval cover?

    .venv/bin/python src/fb_brand_cal.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import extra_models as em  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C2 = ["log_likes", "log_comments"]


def main():
    f, meta = em.load(ROOT)
    src = meta.platform.isin(["twitter", "youtube"]).values
    fb = pd.read_csv(ROOT / "data" / "processed" / "fb_all_fetched.csv", dtype={"mention_id": str})
    fb["views"] = pd.to_numeric(fb.views, errors="coerce")
    fb = fb[(fb.status == "ok") & (fb.views > 0)].reset_index(drop=True)
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    fb["brand"] = raw.loc[fb.mention_id, "brand"].values
    fb["log_likes"] = np.log1p(pd.to_numeric(fb.reactions, errors="coerce").fillna(0))
    fb["log_comments"] = np.log1p(pd.to_numeric(fb.comments, errors="coerce").fillna(0))
    mod = make_pipeline(StandardScaler(), Ridge()).fit(
        pd.DataFrame({"log_likes": f[src].log_likes.fillna(0).values, "log_comments": f[src].log_comments.fillna(0).values}), meta.y[src])
    fb["pred"] = mod.predict(fb[C2])
    br = fb.groupby("brand").agg(n=("views", "size"), true=("views", "sum"), est=("pred", lambda s: float(np.sum(10 ** s))))
    br = br[br.n >= 5].reset_index()
    r = np.log10(br.true / br.est)
    cal = np.zeros(len(br)); cov = []; off = []
    for i in range(len(br)):
        others = np.delete(np.arange(len(br)), i)
        c = float(np.median(np.log10(br.true.values[others] / br.est.values[others])))
        res = np.log10(br.true.values[others] / br.est.values[others]) - c
        hw = float(np.quantile(np.abs(res), .8))
        cal[i] = c
        cov.append(float(abs(r.values[i] - c) <= hw)); off.append(float(abs(r.values[i] - c) > np.log10(2)))
    out = {"brands": int(len(br)),
           "uncorrected": {"median_ratio": round(float(r.median()), 2), "share_off_2x": round(float((r.abs() > np.log10(2)).mean()), 2),
                           "kendall": round(float(kendalltau(br.true, br.est)[0]), 3)},
           "calibrated": {"median_ratio": round(float(np.median(r - cal)), 2), "share_off_2x": round(float(np.mean(off)), 2),
                          "kendall": round(float(kendalltau(br.true, br.est * 10 ** cal)[0]), 3)},
           "interval80_coverage": round(float(np.mean(cov)), 3)}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "fb_brand_cal_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
