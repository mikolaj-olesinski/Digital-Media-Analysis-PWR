"""Text for the Facebook transfer (E100): do multilingual-e5 embeddings of the post text add anything to likes and
comments when the model is trained on X + YouTube and applied to Facebook?

    .venv/bin/python src/fb_text.py

Writes data/processed/fb_text_numbers.json. Embeddings are the cached e5 vectors (PCA 16, fitted on the source posts).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import validation as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def main():
    raw = eu.load(ROOT).reset_index(drop=True)
    raw["mid"] = raw.mention_id.astype(str)
    pos = {m: i for i, m in enumerate(raw.mid)}
    E = pickle.load(open(ROOT / "data" / "processed" / "embeddings_e5_cache.pkl", "rb")).values
    v = pd.to_numeric(raw.view_count, errors="coerce")
    src = raw.platform.isin(["twitter", "youtube"]) & (v > 0)
    pca = PCA(16, random_state=SEED).fit(E[src.values])
    out = {"posts_source": int(src.sum()), "pca_explained": round(float(pca.explained_variance_ratio_.sum()), 3)}
    for name, path, only_sample in [("sample_130", ROOT / "data" / "processed" / "fb_fetched.csv", True),
                                    ("all_402", ROOT / "data" / "processed" / "fb_all_fetched.csv", False)]:
        fb = pd.read_csv(path, dtype={"mention_id": str})
        fb = fb[(fb.status == "ok") & (pd.to_numeric(fb.views, errors="coerce") > 0)].copy()
        if only_sample:
            s = pd.read_csv(eu.require(ROOT / "data" / "manual" / "fbig_sample.csv"), dtype={"mention_id": str})
            keep = set(s.loc[(s.platform == "facebook") & (pd.to_numeric(s.views, errors="coerce") > 0), "mention_id"])
            fb = fb[fb.mention_id.isin(keep)]
        idx = np.array([pos[m] for m in fb.mention_id])
        fbE = pca.transform(E[idx])
        X = pd.DataFrame({"log_likes": np.log1p(pd.to_numeric(fb.reactions, errors="coerce").fillna(0)),
                          "log_comments": np.log1p(pd.to_numeric(fb.comments, errors="coerce").fillna(0))})
        for i in range(16):
            X[f"e{i}"] = fbE[:, i]
        y = np.log10(pd.to_numeric(fb.views, errors="coerce")).values
        S = pd.DataFrame({"log_likes": np.log1p(raw.likes.fillna(0))[src.values].values,
                          "log_comments": np.log1p(raw.comments.fillna(0))[src.values].values})
        SE = pca.transform(E[src.values])
        for i in range(16):
            S[f"e{i}"] = SE[:, i]
        yS = np.log10(v[src.values].values)
        rows = {}
        for label, cols in [("base", ["log_likes", "log_comments"]),
                            ("+ text", ["log_likes", "log_comments"] + [f"e{i}" for i in range(16)]),
                            ("text only", [f"e{i}" for i in range(16)])]:
            mod = make_pipeline(StandardScaler(), Ridge()).fit(S[cols].fillna(0), yS)
            p = mod.predict(X[cols].fillna(0))
            e = y - p
            mae, lo, hi = va.mae_ci(e, raw.brand.values[idx])
            rows[label] = {"MAE": round(float(mae), 3), "CI": [round(float(lo), 3), round(float(hi), 3)],
                           "bias": round(float(e.mean()), 3), "factor": round(float(10 ** mae), 2)}
        out[name] = rows
        print(name, json.dumps(rows, indent=1))
    (ROOT / "data" / "processed" / "fb_text_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
