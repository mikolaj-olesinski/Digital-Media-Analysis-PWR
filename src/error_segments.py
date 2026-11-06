"""Where the model fails by brand and industry (E152): out-of-fold residuals of the final model, aggregated per brand
(min 10 posts) and per industry; relation with brand size.

    .venv/bin/python src/error_segments.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em  # noqa: E402
import reaction_tiers as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    f, meta = em.load(ROOT)
    ind = pd.read_csv(ROOT / "data" / "processed" / "brand_industry.csv").set_index("brand")["industry"]
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        pred = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, g):
            pred[te] = rt.FINAL().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        d = pd.DataFrame({"brand": g, "err": np.abs(y.values - pred), "fol": meta.author_followers[m].values})
        b = d.groupby("brand").agg(n=("err", "size"), err=("err", "mean"), fol=("fol", "median"))
        b = b[b.n >= 10]
        rho = spearmanr(b.fol.fillna(0), b.err)
        b = b.assign(industry=b.index.map(ind))
        i = b.groupby("industry").agg(brands=("err", "size"), err=("err", "mean"), posts=("n", "sum"))
        i = i[i.posts >= 20].sort_values("err")
        out[p] = {"brands": int(len(b)), "spearman_size_error": round(float(rho[0]), 3),
                  "corr_p": round(float(rho[1]), 4),
                  "best_industries": {k: round(float(v), 3) for k, v in i.err.head(3).items()},
                  "worst_industries": {k: round(float(v), 3) for k, v in i.err.tail(3).items()},
                  "industry_spread": round(float(i.err.max() - i.err.min()), 3)}
        print(p, out[p])
    (ROOT / "data" / "processed" / "error_segments_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
