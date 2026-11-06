"""A dedicated model for the largest FB pages (E150): train only on source posts from pages above 1M followers and
apply to the FB >1M pages. Does it beat the global transfer (MAE 1.049)?

    .venv/bin/python src/fb_big_model.py
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
    big = (fb.followers >= 1e6).values
    yf = np.log10(fb.views.values)
    out = {"source_big_posts": int(big_src.sum()), "target_big_posts": int(big.sum())}
    for name, mask in [("all sources", src), ("big sources", big_src)]:
        if mask.sum() < 20: continue
        Xs = pd.DataFrame({"log_likes": f[mask].log_likes.fillna(0).values, "log_comments": f[mask].log_comments.fillna(0).values})
        ys = meta.y[mask]
        mod = make_pipeline(StandardScaler(), Ridge()).fit(Xs, ys)
        pred = mod.predict(fb.loc[big, C2])
        out[name] = {"MAE_big": round(float(np.abs(yf[big] - pred).mean()), 3), "bias": round(float((yf[big] - pred).mean()), 3)}
        print(name, out[name])
    (ROOT / "data" / "processed" / "fb_big_model_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
