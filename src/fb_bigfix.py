"""Why widening does not fix the large FB pages (E147): a symmetric widening barely moves coverage (E146). Test a level
shift for the >1M tier instead: shift + widened band, calibrated on held-out brands.

    .venv/bin/python src/fb_bigfix.py
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
    fb = pd.read_csv(ROOT / "data" / "processed" / "fb_all_fetched.csv", dtype={"mention_id": str})
    fb["views"] = pd.to_numeric(fb.views, errors="coerce")
    fb = fb[(fb.status == "ok") & (fb.views > 0)].reset_index(drop=True)
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    fb["brand"] = raw.loc[fb.mention_id, "brand"].values
    fb["followers"] = pd.to_numeric(raw.loc[fb.mention_id, "author_followers"].values, errors="coerce")
    fb["log_likes"] = np.log1p(pd.to_numeric(fb.reactions, errors="coerce").fillna(0))
    fb["log_comments"] = np.log1p(pd.to_numeric(fb.comments, errors="coerce").fillna(0))
    Xs, ys = f[src][C2].reset_index(drop=True), meta.y[src].reset_index(drop=True)
    Xf, yf, gf = fb[C2], np.log10(fb.views.values), fb.brand
    big = (fb.followers >= 1e6).values
    lo, hi, med = qm(.05).fit(Xs, ys), qm(.95).fit(Xs, ys), qm(.5).fit(Xs, ys)
    L, H, M = lo.predict(Xf), hi.predict(Xf), med.predict(Xf)
    rng = np.random.default_rng(SEED)
    out = {"shift": [], "shift_widen": [], "raw": []}; shifts = []
    for _ in range(REPS):
        brands = np.array(sorted(set(gf)))
        cal_b = set(rng.choice(brands, max(1, int(.25 * len(brands))), replace=False))
        cal = gf.isin(cal_b).values; te = ~cal
        E = np.maximum(L[cal] - yf[cal], yf[cal] - H[cal]); qg = float(np.quantile(E, .9))
        kc = cal & big
        sh = float(np.median(yf[kc] - M[kc])) if kc.sum() >= 5 else 0.0
        shifts.append(sh)
        k = te & big
        raw = (yf[k] >= L[k] - qg) & (yf[k] <= H[k] + qg)
        shf = (yf[k] >= L[k] - qg + sh) & (yf[k] <= H[k] + qg + sh)
        shw = (yf[k] >= L[k] - 1.5 * qg + sh) & (yf[k] <= H[k] + 1.5 * qg + sh)
        out["raw"].append(raw.mean()); out["shift"].append(shf.mean()); out["shift_widen"].append(shw.mean())
    res = {k: round(float(np.mean(v)), 3) for k, v in out.items()} | {"median_shift": round(float(np.median(shifts)), 2)}
    print(json.dumps(res, indent=1))
    (ROOT / "data" / "processed" / "fb_bigfix_numbers.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
