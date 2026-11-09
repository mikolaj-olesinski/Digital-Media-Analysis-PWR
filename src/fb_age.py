"""Facebook transfer with the age of the post (E143): FB posts differ in age (2025 vs 2026) and views grow with age.
Add log(age) to the X+YouTube ridge; does the FB MAE improve?

    .venv/bin/python src/fb_age.py
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


def main():
    f, meta = em.load(ROOT)
    raw = eu.load(ROOT)
    age = ((raw.metrics_updated_at - raw.created_at).dt.total_seconds() / 86400).loc[f.index].clip(0.1, 3650)
    la = np.log10(age).values
    src = meta.platform.isin(["twitter", "youtube"]).values
    fb = pd.read_csv(ROOT / "data" / "processed" / "fb_all_fetched.csv", dtype={"mention_id": str})
    fb["views"] = pd.to_numeric(fb.views, errors="coerce")
    fb = fb[(fb.status == "ok") & (fb.views > 0)].reset_index(drop=True)
    fb["log_likes"] = np.log1p(pd.to_numeric(fb.reactions, errors="coerce").fillna(0))
    fb["log_comments"] = np.log1p(pd.to_numeric(fb.comments, errors="coerce").fillna(0))
    fb["age"] = (pd.to_datetime(fb.fetched_at, errors="coerce", utc=True) - pd.to_datetime(fb.posted_at, errors="coerce", utc=True)).dt.total_seconds() / 86400
    fb["log_age"] = np.log10(fb.age.clip(0.1, 3650)).fillna(fb["log_age"].median() if "log_age" in fb else 0)
    yf = np.log10(fb.views.values)
    out = {}
    for name, cols in [("likes+comments", ["log_likes", "log_comments"]), ("+age", ["log_likes", "log_comments", "log_age"])]:
        Xs = pd.DataFrame({"log_likes": f[src].log_likes.fillna(0).values, "log_comments": f[src].log_comments.fillna(0).values, "log_age": np.nan_to_num(la[src])})
        mod = make_pipeline(StandardScaler(), Ridge()).fit(Xs[cols], meta.y[src])
        pred = mod.predict(fb[cols])
        out[name] = {"MAE": round(float(np.abs(yf - pred).mean()), 3), "bias": round(float((yf - pred).mean()), 3)}
        print(name, out[name])
    (ROOT / "data" / "processed" / "fb_age_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
