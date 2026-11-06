"""First measured transfer error on Facebook (E93): models trained on X / YouTube / all three scored on the 130 FB
Reels and videos whose public view count we fetched on 2026-11-05 (B14, B24).

Features are the only ones every platform has: log1p(likes), log1p(comments). For the target they are the counts seen at
the fetch (likes_now = total reactions, comments_now), so features and views come from the same moment. Source rows keep
Brand24's counts and views from collection time.

    .venv/bin/python src/fb_transfer_check.py

Writes data/processed/fb_transfer_numbers.json. Numbers are a transfer assumption measured on a sample stratified by
reactions, not a random sample of Facebook.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import extra_models as em  # noqa: E402
import validation as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
C2 = ["log_likes", "log_comments"]


def ridge():
    return make_pipeline(StandardScaler(), Ridge())


def frame(root):
    f, meta = em.load(root)
    s = pd.read_csv(eu.require(Path(root) / "data" / "manual" / "fbig_sample.csv"), dtype={"mention_id": str})
    s = s[(s.platform == "facebook") & (pd.to_numeric(s.views, errors="coerce") > 0)].copy()
    raw = eu.load(root).astype({"mention_id": str}).set_index("mention_id")
    r = raw.loc[s.mention_id]
    X = pd.DataFrame({"log_likes": np.log1p(pd.to_numeric(s.likes_now, errors="coerce").fillna(0)),
                      "log_comments": np.log1p(pd.to_numeric(s.comments_now, errors="coerce").fillna(0))})
    y = np.log10(pd.to_numeric(s.views))
    b24 = np.log10(r[eu.REACH].clip(lower=1).values)
    return f, meta, X, y, s, r, b24


def score(e, groups):
    mae, lo, hi = va.mae_ci(e, groups)
    return {"MAE": round(float(mae), 4), "CI low": round(float(lo), 4), "CI high": round(float(hi), 4),
            "bias": round(float(np.mean(e)), 4), "factor": round(float(10 ** mae), 2)}


def recalibrate(pred, y, k, n_draw=200, rng=None):
    """MAE on the posts NOT used for calibration, after: (a) median shift, (b) slope+intercept, fitted on k posts."""
    rng = rng or np.random.default_rng(SEED)
    e = y - pred
    rows = {"shift": [], "slope": []}
    for _ in range(n_draw):
        i = rng.permutation(len(y))
        cal, rest = i[:k], i[k:]
        rows["shift"].append(np.abs(e[rest] - np.median(e[cal])).mean())
        b, a = np.polyfit(pred[cal], y[cal], 1)
        rows["slope"].append(np.abs(y[rest] - (a + b * pred[rest])).mean())
    return {m: {"mean": round(float(np.mean(v)), 3), "p10": round(float(np.percentile(v, 10)), 3),
                "p90": round(float(np.percentile(v, 90)), 3)} for m, v in rows.items()}


def main():
    f, meta, X, y, s, raw, b24 = frame(ROOT)
    groups = raw.brand.values
    print(f"FB posts with views: {len(y)}, brands: {len(set(groups))}")
    out = {"posts": int(len(y)), "brands": int(len(set(groups))), "models": {}}

    sources = {"X + YouTube": ["twitter", "youtube"], "YouTube": ["youtube"], "X": ["twitter"],
               "all three": ["tiktok", "youtube", "twitter"]}
    preds = {}
    for name, src in sources.items():
        m = meta.platform.isin(src).values
        mod = ridge().fit(f[m][C2].fillna(0), meta.y[m])
        p = mod.predict(X[C2].fillna(0))
        preds[name] = p
        e = y.values - p
        out["models"][name] = {**score(e, groups), "Spearman": round(float(spearmanr(y, p)[0]), 3)}
        print(name, out["models"][name])

    e24 = y.values - b24
    out["models"]["Brand24 reach"] = {**score(e24, groups), "Spearman": round(float(spearmanr(y, b24)[0]), 3)}

    # adaptation: a few FB posts fix the level (shift) or level and slope
    out["recalibration"] = {}
    rng = np.random.default_rng(SEED)
    for name in ["X + YouTube", "X", "YouTube", "all three"]:
        out["recalibration"][name] = {str(k): recalibrate(preds[name], y.values, k, rng=rng) for k in (5, 10, 25, 50)}
    # in-platform ceiling: 5-fold brand-grouped CV of the same two features on FB itself
    from sklearn.model_selection import GroupKFold
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        oof[te] = ridge().fit(X.iloc[tr][C2].fillna(0), y.iloc[tr]).predict(X.iloc[te][C2].fillna(0))
    out["ceiling_in_platform"] = {**score(y.values - oof, groups), "Spearman": round(float(spearmanr(y, oof)[0]), 3)}
    print("ceiling", out["ceiling_in_platform"])

    Path(ROOT / "data" / "processed" / "fb_transfer_numbers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["recalibration"]["X + YouTube"], indent=1))


def main_followers():
    """E94: does adding the author's followers improve the measured transfer to Facebook? Median imputation for the
    missing followers, plus a missing flag, fitted on X + YouTube and scored on the same 130 FB posts."""
    f, meta, X, y, s, raw, b24 = frame(ROOT)
    groups = raw.brand.values
    src = meta.platform.isin(["twitter", "youtube"]).values
    fol = pd.to_numeric(raw.author_followers.values, errors="coerce")
    lf = np.log10(np.where(fol > 0, fol, np.nan))
    med = float(np.nanmedian(f.log_followers[src]))
    Xf = X.assign(log_followers=np.where(np.isfinite(lf), lf, med),
                  na_followers=(~np.isfinite(lf)).astype(float)).fillna(0)
    tr = f[src].copy()
    tr["log_followers"] = tr.log_followers.fillna(med)
    tr = tr.fillna(0)
    out = {"posts": int(len(y)), "brands": int(len(set(groups))), "followers_known": int(np.isfinite(lf).sum()),
           "median_log_followers_source": round(med, 3), "models": {}}
    preds = {}
    for name, cols in [("base (likes + comments)", C2),
                       ("+ followers", C2 + ["log_followers", "na_followers"]),
                       ("followers only", ["log_followers", "na_followers"])]:
        p = ridge().fit(tr[cols], meta.y[src]).predict(Xf[cols])
        preds[name] = p
        out["models"][name] = score(y.values - p, groups)
        print(name, out["models"][name])
    rng = np.random.default_rng(SEED)
    d = []
    e0, e1 = np.abs(y.values - preds["base (likes + comments)"]), np.abs(y.values - preds["+ followers"])
    for _ in range(2000):
        i = rng.choice(len(y), len(y))
        d.append(e0[i].mean() - e1[i].mean())
    out["gain_plus_followers"] = {"mean": round(float(np.mean(d)), 3),
                                  "CI": [round(float(x), 3) for x in np.percentile(d, [2.5, 97.5])]}
    k = np.isfinite(lf)
    out["subset_with_followers"] = {"n": int(k.sum()), "base": round(float(e0[k].mean()), 3),
                                    "plus": round(float(e1[k].mean()), 3)}
    print(out["gain_plus_followers"], out["subset_with_followers"])
    (ROOT / "data" / "processed" / "fb_followers_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main_followers() if "--followers" in sys.argv else main()
