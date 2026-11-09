"""Out-of-platform CQR matrix (E131): train quantile models on two platforms, calibrate the 90% band on a quarter of the
target brands and evaluate on the held-out target brands. How wide is the band when the target platform has no training
data?

    .venv/bin/python src/transfer_cqr.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
C2 = ["log_likes", "log_comments"]


def qm(a):
    return LGBMRegressor(objective="quantile", alpha=a, n_estimators=300, learning_rate=.05, num_leaves=31,
                         subsample=.9, colsample_bytree=.9, random_state=SEED, verbose=-1)


def main():
    d = eu.load(ROOT)
    d = d[d.platform.isin(["tiktok", "youtube", "twitter"]) & (pd.to_numeric(d.view_count, errors="coerce") > 0)].copy()
    d["y"] = np.log10(d.view_count)
    d["log_likes"] = np.log1p(d.likes.fillna(0)); d["log_comments"] = np.log1p(d.comments.fillna(0))
    d["plat"] = d.platform.map({"tiktok": 0, "youtube": 1, "twitter": 2})
    out = {}
    for h in ["tiktok", "youtube", "twitter"]:
        src = d[d.platform != h]; tgt = d[d.platform == h]
        Xs = src[C2 + ["plat"]].reset_index(drop=True); ys = src.y.reset_index(drop=True)
        Xt = tgt[C2 + ["plat"]].reset_index(drop=True); yt = tgt.y.reset_index(drop=True); gt = tgt.brand.reset_index(drop=True)
        cov, wid, mae = [], [], []
        for tr, te in GroupKFold(5).split(Xt, yt, gt):
            rng = np.random.default_rng(SEED)
            brands = np.array(sorted(set(gt.iloc[tr])))
            cal_b = set(rng.choice(brands, max(1, int(.25 * len(brands))), replace=False))
            cal = tr[gt.iloc[tr].isin(cal_b).values]
            lo = qm(.05).fit(Xs, ys); hi = qm(.95).fit(Xs, ys); med = qm(.5).fit(Xs, ys)
            E = np.maximum(lo.predict(Xt.iloc[cal]) - yt[cal].values, yt[cal].values - hi.predict(Xt.iloc[cal]))
            q = float(np.quantile(E, .9))
            L, H = lo.predict(Xt.iloc[te]) - q, hi.predict(Xt.iloc[te]) + q
            cov.append(((yt[te].values >= L) & (yt[te].values <= H)).mean()); wid.append(np.median(H - L))
            mae.append(np.abs(yt[te].values - med.predict(Xt.iloc[te])).mean())
        out[h] = {"source": [x for x in ["tiktok", "youtube", "twitter"] if x != h],
                  "MAE": round(float(np.mean(mae)), 3), "coverage90": round(float(np.mean(cov)), 3),
                  "median_width": round(float(np.mean(wid)), 2)}
        print(h, out[h])
    (ROOT / "data" / "processed" / "transfer_cqr_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
