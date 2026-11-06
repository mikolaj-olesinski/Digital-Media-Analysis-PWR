"""FB brand totals and industry errors (E156/E157): with the global and hybrid models, do brand totals and their order
hold? And is the residual concentrated in an industry?

    .venv/bin/python src/fb_brand_totals.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
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
    big = (fb.followers >= 1e6).values
    g = make_pipeline(StandardScaler(), Ridge()).fit(pd.DataFrame({"log_likes": f[src].log_likes.fillna(0).values, "log_comments": f[src].log_comments.fillna(0).values}), meta.y[src])
    b = make_pipeline(StandardScaler(), Ridge()).fit(pd.DataFrame({"log_likes": f[big_src].log_likes.fillna(0).values, "log_comments": f[big_src].log_comments.fillna(0).values, "log_followers": np.log10(meta.author_followers[big_src].fillna(1).clip(lower=1).values)}), meta.y[big_src])
    fb["pg"] = g.predict(fb[C2]); fb["ph"] = np.where(big, b.predict(fb[C3]), fb["pg"])
    br = fb.groupby("brand").agg(n=("views", "size"), true=("views", "sum"), eg=("pg", lambda s: float(np.sum(10 ** s))),
                                 eh=("ph", lambda s: float(np.sum(10 ** s))))
    br = br[br.n >= 5]
    out = {"brands": int(len(br))}
    for name, col in [("global", "eg"), ("hybrid", "eh")]:
        r = np.log10(br[col] / br.true)
        out[name] = {"median_ratio": round(float(r.median()), 2), "share_off_2x": round(float((r.abs() > np.log10(2)).mean()), 2),
                     "kendall": round(float(kendalltau(br.true, br[col])[0]), 3)}
    ind = pd.read_csv(ROOT / "data" / "processed" / "brand_industry.csv").set_index("brand")["industry"]
    fb["ind"] = fb.brand.map(ind)
    fb["err"] = np.abs(np.log10(fb.views.values) - fb.ph)
    ii = fb.groupby("ind").agg(n=("err", "size"), err=("err", "mean"))
    ii = ii[ii.n >= 20].sort_values("err")
    out["industry"] = {"best": {k: round(float(v), 3) for k, v in ii.err.head(3).items()},
                       "worst": {k: round(float(v), 3) for k, v in ii.err.tail(3).items()}}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "fb_brand_totals_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
