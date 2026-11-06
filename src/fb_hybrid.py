"""Hybrid FB model (E154): global transfer for pages below 1M followers, large-page model above 1M. Does the hybrid
beat the global model on all 402 posts?

    .venv/bin/python src/fb_hybrid.py
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
    fb["followers"] = pd.to_numeric(raw.loc[fb.mention_id, "author_followers"].values, errors="coerce")
    fb["log_likes"] = np.log1p(pd.to_numeric(fb.reactions, errors="coerce").fillna(0))
    fb["log_comments"] = np.log1p(pd.to_numeric(fb.comments, errors="coerce").fillna(0))
    fb["log_followers"] = np.log10(fb.followers.fillna(1).clip(lower=1))
    big = (fb.followers >= 1e6).values
    yf = np.log10(fb.views.values)
    # global model: all sources, likes + comments
    g = make_pipeline(StandardScaler(), Ridge()).fit(
        pd.DataFrame({"log_likes": f[src].log_likes.fillna(0).values, "log_comments": f[src].log_comments.fillna(0).values}), meta.y[src])
    pg = g.predict(fb[C2])
    # large-page model: big sources, likes + comments + followers
    b = make_pipeline(StandardScaler(), Ridge()).fit(
        pd.DataFrame({"log_likes": f[big_src].log_likes.fillna(0).values, "log_comments": f[big_src].log_comments.fillna(0).values,
                      "log_followers": np.log10(meta.author_followers[big_src].fillna(1).clip(lower=1).values)}), meta.y[big_src])
    pb = b.predict(fb[C3])
    hy = np.where(big, pb, pg)
    out = {"global": {"MAE_all": round(float(np.abs(yf - pg).mean()), 3), "MAE_big": round(float(np.abs(yf[big] - pg[big]).mean()), 3),
                      "MAE_small": round(float(np.abs(yf[~big] - pg[~big]).mean()), 3)},
           "hybrid": {"MAE_all": round(float(np.abs(yf - hy).mean()), 3), "MAE_big": round(float(np.abs(yf[big] - pb[big]).mean()), 3),
                      "MAE_small": round(float(np.abs(yf[~big] - pg[~big]).mean()), 3)}}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "fb_hybrid_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
