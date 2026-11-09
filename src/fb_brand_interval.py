"""FB brand-total interval scaled by the number of posts (E159): a constant 80% width covers only 0.737 (E158). Scale the
width by sqrt(n_median / n_posts) and test the leave-one-brand-out coverage.

    .venv/bin/python src/fb_brand_interval.py
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
    fb = pd.read_csv(ROOT / "data" / "processed" / "fb_all_fetched.csv", dtype={"mention_id": str})
    fb["views"] = pd.to_numeric(fb.views, errors="coerce")
    fb = fb[(fb.status == "ok") & (fb.views > 0)].reset_index(drop=True)
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    fb["brand"] = raw.loc[fb.mention_id, "brand"].values
    fb["log_likes"] = np.log1p(pd.to_numeric(fb.reactions, errors="coerce").fillna(0))
    fb["log_comments"] = np.log1p(pd.to_numeric(fb.comments, errors="coerce").fillna(0))
    mod = make_pipeline(StandardScaler(), Ridge()).fit(
        pd.DataFrame({"log_likes": f[src].log_likes.fillna(0).values, "log_comments": f[src].log_comments.fillna(0).values}), meta.y[src])
    fb["pred"] = mod.predict(fb[C2])
    br = fb.groupby("brand").agg(n=("views", "size"), true=("views", "sum"), est=("pred", lambda s: float(np.sum(10 ** s))))
    br = br[br.n >= 5].reset_index()
    out = {}
    for min_n in (5, 10):
        b = br[br.n >= min_n].reset_index(drop=True)
        r = np.log10(b.true.values / b.est.values)
        nmed = float(np.median(b.n.values))
        cov_const, cov_scaled = [], []
        for i in range(len(b)):
            others = np.delete(np.arange(len(b)), i)
            c = float(np.median(r[others]))
            res = np.abs(r[others] - c)
            w = float(np.quantile(res, .8))
            w_i = w * np.sqrt(nmed / b.n.values[i])
            cov_const.append(float(abs(r[i] - c) <= w)); cov_scaled.append(float(abs(r[i] - c) <= w_i))
        out[f"min_{min_n}"] = {"brands": int(len(b)), "coverage_const": round(float(np.mean(cov_const)), 3),
                               "coverage_scaled": round(float(np.mean(cov_scaled)), 3), "n_median": nmed}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "fb_brand_interval_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
