"""FB intervals on the 130 hand-checked posts (E166): the 402-post CQR covers 0.925 (E123). Does the same procedure hold
on the smaller hand sample, or is 130 posts too few for a calibrated band?

    .venv/bin/python src/fb_cqr_hand.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import extra_models as em  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C2 = ["log_likes", "log_comments"]
SEED = 42
REPS = 20


def qm(a):
    return LGBMRegressor(objective="quantile", alpha=a, n_estimators=300, learning_rate=.05, num_leaves=31,
                         subsample=.9, colsample_bytree=.9, random_state=SEED, verbose=-1)


def main():
    f, meta = em.load(ROOT)
    src = meta.platform.isin(["twitter", "youtube"]).values
    fb = pd.read_csv(ROOT / "data" / "processed" / "fb_fetched.csv", dtype={"mention_id": str})
    fb["views"] = pd.to_numeric(fb.views, errors="coerce")
    fb = fb[(fb.status == "ok") & (fb.views > 0)].reset_index(drop=True)
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    fb["brand"] = raw.loc[fb.mention_id, "brand"].values
    fb["log_likes"] = np.log1p(pd.to_numeric(fb.reactions, errors="coerce").fillna(0))
    fb["log_comments"] = np.log1p(pd.to_numeric(fb.comments, errors="coerce").fillna(0))
    Xs, ys = f[src][C2].reset_index(drop=True), meta.y[src].reset_index(drop=True)
    Xf, yf, gf = fb[C2], np.log10(fb.views.values), fb.brand
    lo, hi = qm(.05).fit(Xs, ys), qm(.95).fit(Xs, ys)
    L, H = lo.predict(Xf), hi.predict(Xf)
    rng = np.random.default_rng(SEED); cov, wid = [], []
    for _ in range(REPS):
        brands = np.array(sorted(set(gf)))
        cal_b = set(rng.choice(brands, max(1, int(.25 * len(brands))), replace=False))
        cal = gf.isin(cal_b).values; te = ~cal
        if cal.sum() < 5 or te.sum() < 5: continue
        E = np.maximum(L[cal] - yf[cal], yf[cal] - H[cal]); q = float(np.quantile(E, .9))
        cov.append(((yf[te] >= L[te] - q) & (yf[te] <= H[te] + q)).mean()); wid.append(np.median((H[te] - L[te]) + 2 * q))
    out = {"posts": int(len(fb)), "coverage": round(float(np.mean(cov)), 3), "width": round(float(np.mean(wid)), 2)}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "fb_cqr_hand_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
