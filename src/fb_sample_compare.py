"""Hand-check sample vs the full fetched sample (E153): are the 130 hand-checked FB posts representative of the 402
fetched ones? Rule and model errors on both, and the overlap.

    .venv/bin/python src/fb_sample_compare.py
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
C2 = ["log_likes", "log_comments"]


def load(name):
    d = pd.read_csv(ROOT / "data" / "processed" / name, dtype={"mention_id": str})
    d["views"] = pd.to_numeric(d.views, errors="coerce")
    return d[(d.status == "ok") & (d.views > 0)].copy()


def main():
    a = load("fb_fetched.csv"); b = load("fb_all_fetched.csv")
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    f, meta = em.load(ROOT); src = meta.platform.isin(["twitter", "youtube"]).values
    mod = make_pipeline(StandardScaler(), Ridge()).fit(
        pd.DataFrame({"log_likes": f[src].log_likes.fillna(0).values, "log_comments": f[src].log_comments.fillna(0).values}), meta.y[src])
    out = {"hand_n": int(len(a)), "full_n": int(len(b)), "overlap": int(len(set(a.mention_id) & set(b.mention_id)))}
    for name, d in [("hand", a), ("full", b)]:
        d["b24"] = pd.to_numeric(raw.loc[d.mention_id, "b24_estimated_reach"].values, errors="coerce")
        d["log_likes"] = np.log1p(pd.to_numeric(d.reactions, errors="coerce").fillna(0))
        d["log_comments"] = np.log1p(pd.to_numeric(d.comments, errors="coerce").fillna(0))
        y = np.log10(d.views.values); pred = mod.predict(d[C2])
        ok = d.b24 > 0
        out[name] = {"model_MAE": round(float(np.abs(y - pred).mean()), 3),
                     "rule_MAE": round(float(np.abs(y[ok.values] - np.log10(d.b24.values[ok.values])).mean()), 3),
                     "median_views": round(float(d.views.median()), 0)}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "fb_sample_compare_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
