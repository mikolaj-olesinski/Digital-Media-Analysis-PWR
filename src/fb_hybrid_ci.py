"""Is the hybrid FB gain real? (E155) Paired brand bootstrap of the MAE difference between the global and hybrid models.

    .venv/bin/python src/fb_hybrid_ci.py
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
C3 = C2 + ["log_followers"]


def main():
    f, meta = em.load(ROOT)
    src = meta.platform.isin(["twitter", "youtube"]).values
    big_src = src & (meta.author_followers.fillna(0) >= 1e6).values
    fb = pd.read_csv(ROOT / "data" / "processed" / "fb_all_fetched.csv", dtype={"mention_id": str})
    fb["views"] = pd.to_numeric(fb.views, errors="coerce")
    fb = fb[(fb.status == "ok") & (fb.views > 0)].reset_index(drop=True)
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    fb["brand"] = raw.loc[fb.mention_id, "brand"].values
    fb["followers"] = pd.to_numeric(raw.loc[fb.mention_id, "author_followers"].values, errors="coerce")
    fb["log_likes"] = np.log1p(pd.to_numeric(fb.reactions, errors="coerce").fillna(0))
    fb["log_comments"] = np.log1p(pd.to_numeric(fb.comments, errors="coerce").fillna(0))
    fb["log_followers"] = np.log10(fb.followers.fillna(1).clip(lower=1))
    big = (fb.followers >= 1e6).values; y = np.log10(fb.views.values)
    g = make_pipeline(StandardScaler(), Ridge()).fit(pd.DataFrame({"log_likes": f[src].log_likes.fillna(0).values, "log_comments": f[src].log_comments.fillna(0).values}), meta.y[src])
    b = make_pipeline(StandardScaler(), Ridge()).fit(pd.DataFrame({"log_likes": f[big_src].log_likes.fillna(0).values, "log_comments": f[big_src].log_comments.fillna(0).values, "log_followers": np.log10(meta.author_followers[big_src].fillna(1).clip(lower=1).values)}), meta.y[big_src])
    e_g = np.abs(y - g.predict(fb[C2])); e_h = np.abs(y - np.where(big, b.predict(fb[C3]), g.predict(fb[C2])))
    rng = np.random.default_rng(42); diffs = []
    for _ in range(2000):
        pick = rng.choice(fb.brand.unique(), len(fb.brand.unique()), replace=True)
        idx = np.concatenate([np.where(fb.brand.values == b0)[0] for b0 in pick])
        diffs.append(e_h[idx].mean() - e_g[idx].mean())
    out = {"MAE_global": round(float(e_g.mean()), 3), "MAE_hybrid": round(float(e_h.mean()), 3),
           "diff": round(float(e_h.mean() - e_g.mean()), 3), "CI_low": round(float(np.percentile(diffs, 2.5)), 3),
           "CI_high": round(float(np.percentile(diffs, 97.5)), 3)}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "fb_hybrid_ci_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
