"""Calibration budget (E115, plan B): how many posts with true views does a client need before the estimate is good
enough? For a target platform, train the ridge on the other platforms, calibrate with n target posts (median shift) and
measure the MAE on the rest. Gives the concrete "collect N posts" recommendation.

    .venv/bin/python src/calibration_curve.py

Writes data/processed/calibration_curve_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import extra_models as em  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
C2 = ["log_likes", "log_comments"]
NS = [0, 5, 10, 25, 50, 100, 200]


def curve(pred, y, ns=NS, draws=200, rng=None):
    rng = rng or np.random.default_rng(SEED)
    out = {}
    for n in ns:
        if n >= len(y):
            continue
        vals = []
        for _ in range(draws):
            i = rng.permutation(len(y)); cal, rest = i[:n], i[n:]
            shift = np.median(y[cal] - pred[cal]) if n else 0.0
            vals.append(np.abs(y[rest] - (pred[rest] + shift)).mean())
        out[str(n)] = {"MAE": round(float(np.mean(vals)), 3), "p10": round(float(np.percentile(vals, 10)), 3),
                       "p90": round(float(np.percentile(vals, 90)), 3)}
    return out


def main():
    d = eu.load(ROOT)
    d = d[d.platform.isin(["tiktok", "youtube", "twitter"]) & (pd.to_numeric(d.view_count, errors="coerce") > 0)].copy()
    d["y"] = np.log10(d.view_count)
    d["log_likes"] = np.log1p(d.likes.fillna(0)); d["log_comments"] = np.log1p(d.comments.fillna(0))
    out = {}
    rng = np.random.default_rng(SEED)
    for h in ["tiktok", "youtube", "twitter"]:
        tr = d[d.platform != h]
        mod = make_pipeline(StandardScaler(), Ridge()).fit(tr[C2], tr.y)
        te = d[d.platform == h]
        pred = mod.predict(te[C2])
        out[h] = curve(pred, te.y.values, NS, rng=rng)
    # Facebook: the same model trained on X + YouTube, calibrated on n FB posts (402 available)
    fb = pd.read_csv(ROOT / "data" / "processed" / "fb_all_fetched.csv", dtype={"mention_id": str})
    fb["views"] = pd.to_numeric(fb.views, errors="coerce")
    fb = fb[(fb.status == "ok") & (fb.views > 0)].reset_index(drop=True)
    fb["log_likes"] = np.log1p(pd.to_numeric(fb.reactions, errors="coerce").fillna(0))
    fb["log_comments"] = np.log1p(pd.to_numeric(fb.comments, errors="coerce").fillna(0))
    tr = d[d.platform.isin(["twitter", "youtube"])]
    mod = make_pipeline(StandardScaler(), Ridge()).fit(tr[C2], tr.y)
    out["facebook"] = curve(mod.predict(fb[C2]), np.log10(fb.views.values), [0, 5, 10, 25, 50, 100], rng=rng)
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "calibration_curve_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
