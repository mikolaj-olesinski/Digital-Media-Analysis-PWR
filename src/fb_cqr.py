"""CQR for the Facebook transfer (E123): train quantile models on X + YouTube, calibrate the 90% band on a subset of
Facebook brands, evaluate coverage on the held-out brands. Compares with split conformal around the point model.

    .venv/bin/python src/fb_cqr.py
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
import extra_models as em  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C2 = ["log_likes", "log_comments"]
SEED = 42


def qm(a):
    return LGBMRegressor(objective="quantile", alpha=a, n_estimators=300, learning_rate=.05, num_leaves=31,
                         subsample=.9, colsample_bytree=.9, random_state=SEED, verbose=-1)


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
    Xs, ys = f[src][C2].reset_index(drop=True), meta.y[src].reset_index(drop=True)
    Xf, yf, gf = fb[C2], np.log10(fb.views.values), fb.brand
    cov = {"cqr": [], "split": []}; wid = {"cqr": [], "split": []}
    for tr, te in GroupKFold(5).split(Xf, yf, gf):
        rng = np.random.default_rng(SEED)
        brands = np.array(sorted(set(gf.iloc[tr])))
        cal_brands = set(rng.choice(brands, max(1, int(.25 * len(brands))), replace=False))
        cal = tr[gf.iloc[tr].isin(cal_brands).values]; fit = tr[~gf.iloc[tr].isin(cal_brands).values]
        lo_m, hi_m = qm(.05).fit(Xs, ys), qm(.95).fit(Xs, ys)
        E = np.maximum(lo_m.predict(Xf.iloc[cal]) - yf[cal], yf[cal] - hi_m.predict(Xf.iloc[cal]))
        q = float(np.quantile(E, .9))
        lo, hi = lo_m.predict(Xf.iloc[te]) - q, hi_m.predict(Xf.iloc[te]) + q
        cov["cqr"].append(((yf[te] >= lo) & (yf[te] <= hi)).mean()); wid["cqr"].append(np.median(hi - lo))
        med = qm(.5).fit(Xs, ys)
        r = np.abs(yf[cal] - med.predict(Xf.iloc[cal])); qs = float(np.quantile(r, .9))
        mp = med.predict(Xf.iloc[te])
        cov["split"].append(((yf[te] >= mp - qs) & (yf[te] <= mp + qs)).mean()); wid["split"].append(float(2 * qs))
    out = {k: {"coverage": round(float(np.mean(cov[k])), 3), "median_width": round(float(np.mean(wid[k])), 2)} for k in cov}
    out["posts"] = int(len(yf))
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "fb_cqr_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
