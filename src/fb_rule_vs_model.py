"""FB rule vs model by follower tier (E165): Brand24's reach against the reaction-based transfer, per tier. Saves the
numbers and a figure (EN/PL).

    .venv/bin/python src/fb_rule_vs_model.py
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
TIERS = ["<1k", "1k-10k", "10k-100k", "100k-1M", ">1M"]


def main():
    f, meta = em.load(ROOT)
    src = meta.platform.isin(["twitter", "youtube"]).values
    fb = pd.read_csv(ROOT / "data" / "processed" / "fb_all_fetched.csv", dtype={"mention_id": str})
    fb["views"] = pd.to_numeric(fb.views, errors="coerce")
    fb = fb[(fb.status == "ok") & (fb.views > 0)].reset_index(drop=True)
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    fb["b24"] = pd.to_numeric(raw.loc[fb.mention_id, "b24_estimated_reach"].values, errors="coerce")
    fb["followers"] = pd.to_numeric(raw.loc[fb.mention_id, "author_followers"].values, errors="coerce")
    fb["log_likes"] = np.log1p(pd.to_numeric(fb.reactions, errors="coerce").fillna(0))
    fb["log_comments"] = np.log1p(pd.to_numeric(fb.comments, errors="coerce").fillna(0))
    fb["tier"] = pd.cut(fb.followers, [0, 1e3, 1e4, 1e5, 1e6, 1e9], labels=TIERS, right=False)
    mod = make_pipeline(StandardScaler(), Ridge()).fit(
        pd.DataFrame({"log_likes": f[src].log_likes.fillna(0).values, "log_comments": f[src].log_comments.fillna(0).values}), meta.y[src])
    y = np.log10(fb.views.values); pred = mod.predict(fb[C2])
    out = {}
    for t in TIERS:
        k = (fb.tier == t).values
        if k.sum() < 10: continue
        out[t] = {"n": int(k.sum()), "rule_MAE": round(float(np.abs(y[k] - np.log10(fb.b24.values[k])).mean()), 3),
                  "model_MAE": round(float(np.abs(y[k] - pred[k]).mean()), 3)}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "fb_rule_vs_model_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
