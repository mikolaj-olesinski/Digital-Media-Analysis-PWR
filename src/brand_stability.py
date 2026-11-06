"""How stable is a brand total? (E112) Reach is dominated by a few posts (E96), so a brand total computed from n posts is
unstable. For brands with >= 20 measured posts, bootstrap n posts and measure the p90/p10 spread of the total and how
often it is within 2x of the full-brand total.

    .venv/bin/python src/brand_stability.py

Writes data/processed/brand_stability_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def main():
    d = eu.load(ROOT)
    q = d[d.platform.isin(["tiktok", "youtube", "twitter"]) & (pd.to_numeric(d.view_count, errors="coerce") > 0)].copy()
    q["views"] = pd.to_numeric(q.view_count)
    rng = np.random.default_rng(SEED)
    rows = []
    for (b, p), g in q.groupby(["brand", "platform"]):
        if len(g) < 20:
            continue
        v = g.views.values; total = v.sum()
        for n in (5, 10, 20):
            if n > len(v):
                continue
            tots = np.array([v[rng.choice(len(v), n, replace=True)].sum() for _ in range(400)])
            rows.append({"brand": b, "platform": p, "n_posts": len(v), "sample": n,
                         "p90_p10": float(np.percentile(tots, 90) / max(np.percentile(tots, 10), 1)),
                         "within_2x": float(np.mean((tots > total / 2) & (tots < total * 2)))})
    R = pd.DataFrame(rows)
    out = {"brands": int(R.brand.nunique()), "groups": int(len(R) / 3)}
    for n, g in R.groupby("sample"):
        out[str(n)] = {"median_p90_p10": round(float(g.p90_p10.median()), 2),
                       "median_within_2x": round(float(g.within_2x.median()), 2)}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "brand_stability_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
