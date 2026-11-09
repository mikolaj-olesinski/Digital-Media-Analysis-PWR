"""Which signal does the FB transfer use? (E162) Ridge on the source with followers only, reactions only, or both;
MAE on the 402 FB posts overall and by follower tier.

    .venv/bin/python src/fb_signal.py
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
    src = meta.platform.isin(["twitter", "youtube"]).values
    fb = pd.read_csv(ROOT / "data" / "processed" / "fb_all_fetched.csv", dtype={"mention_id": str})
    fb["views"] = pd.to_numeric(fb.views, errors="coerce")
    fb = fb[(fb.status == "ok") & (fb.views > 0)].reset_index(drop=True)
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    fb["followers"] = pd.to_numeric(raw.loc[fb.mention_id, "author_followers"].values, errors="coerce")
    fb["log_likes"] = np.log1p(pd.to_numeric(fb.reactions, errors="coerce").fillna(0))
    fb["log_comments"] = np.log1p(pd.to_numeric(fb.comments, errors="coerce").fillna(0))
    fb["log_followers"] = np.log10(fb.followers.fillna(1).clip(lower=1))
    big = (fb.followers >= 1e6).values; y = np.log10(fb.views.values)
    out = {}
    for name, cols in [("followers only", ["log_followers"]), ("reactions only", ["log_likes", "log_comments"]),
                       ("both", ["log_followers", "log_likes", "log_comments"])]:
        Xs = pd.DataFrame({"log_followers": np.log10(meta.author_followers[src].fillna(1).clip(lower=1).values),
                           "log_likes": f[src].log_likes.fillna(0).values, "log_comments": f[src].log_comments.fillna(0).values})
        mod = make_pipeline(StandardScaler(), Ridge()).fit(Xs[cols], meta.y[src])
        pred = mod.predict(fb[cols])
        out[name] = {"MAE_all": round(float(np.abs(y - pred).mean()), 3), "MAE_big": round(float(np.abs(y[big] - pred[big]).mean()), 3),
                     "MAE_small": round(float(np.abs(y[~big] - pred[~big]).mean()), 3)}
        print(name, out[name])
    (ROOT / "data" / "processed" / "fb_signal_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
