"""Facebook beyond the 130-post sample (E95) and intervals for the FB estimate (E98).

E95: the same X + YouTube ridge scored on all 402 public Facebook Reels/videos (not only the 130 hand-check sample),
error by kind and by age at fetch, and whether the 22 unavailable posts look different from the measured ones.
E98: 80/90% intervals for the FB estimate - widths taken from the source platforms against widths calibrated on k FB
posts (split conformal), coverage on the remaining posts.

    .venv/bin/python src/fb_extend.py

Writes data/processed/fb_extend_numbers.json.
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


def fb_frame(root):
    d = pd.read_csv(root / "data" / "processed" / "fb_all_fetched.csv", dtype={"mention_id": str})
    raw = eu.load(root).astype({"mention_id": str}).set_index("mention_id")
    r = raw.loc[d.mention_id]
    d = d.assign(brand=r.brand.values, kind=r.url.str.extract(r"/(reels?|videos?)/", expand=False).values,
                 created_at=pd.to_datetime(r.created_at.values, errors="coerce"),
                 fetched_at=pd.to_datetime(d.fetched_at, errors="coerce", utc=True).dt.tz_convert(None))
    d["age_days"] = (d.fetched_at - d.created_at).dt.days
    d["log_likes"] = np.log1p(pd.to_numeric(d.reactions, errors="coerce").fillna(0))
    d["log_comments"] = np.log1p(pd.to_numeric(d.comments, errors="coerce").fillna(0))
    d["y"] = np.log10(pd.to_numeric(d.views, errors="coerce"))
    return d


def train_source(root):
    f, meta = em.load(root)
    m = meta.platform.isin(["twitter", "youtube"]).values
    return ridge().fit(f[m][C2].fillna(0), meta.y[m])


def score(pred, y, groups):
    e = y - pred
    mae, lo, hi = va.mae_ci(e, groups)
    return {"MAE": round(float(mae), 3), "CI": [round(float(lo), 3), round(float(hi), 3)],
            "bias": round(float(e.mean()), 3), "factor": round(float(10 ** mae), 2),
            "Spearman": round(float(spearmanr(y, pred)[0]), 3), "posts": int(len(y))}


def e95(root):
    d = fb_frame(root)
    ok = d[d.status == "ok"].dropna(subset=["y"]).reset_index(drop=True)
    gone = d[d.status != "ok"]
    mod = train_source(root)
    pred = mod.predict(ok[C2].fillna(0))
    out = {"all_402": score(pred, ok.y.values, ok.brand.values)}
    sample = pd.read_csv(root / "data" / "manual" / "fbig_sample.csv", dtype={"mention_id": str})
    sub = sample[(sample.platform == "facebook") & (pd.to_numeric(sample.views, errors="coerce") > 0)]
    k = ok.mention_id.isin(sub.mention_id).values
    out["sample_130"] = score(pred[k], ok.y.values[k], ok.brand.values[k])
    out["by_kind"] = {kk: score(pred[g], ok.y.values[g], ok.brand.values[g]) for kk, g in ok.groupby("kind").groups.items()}
    # age at fetch, in quartiles of the FB posts
    ok["age_q"] = pd.qcut(ok.age_days, 4, labels=["young", "mid-young", "mid-old", "old"], duplicates="drop")
    out["by_age"] = {str(kk): score(pred[g], ok.y.values[g], ok.brand.values[g]) for kk, g in ok.groupby("age_q", observed=True).groups.items()}
    # the 22 unavailable posts: are the measured ones special? compare reactions and followers
    raw = eu.load(root).astype({"mention_id": str}).set_index("mention_id")
    def react(ids):
        q = raw.loc[ids]
        return pd.to_numeric(q.likes, errors="coerce").fillna(0) + pd.to_numeric(q.comments, errors="coerce").fillna(0)
    out["unavailable_vs_measured"] = {
        "unavailable": int(len(gone)), "measured": int(len(ok)),
        "median_reactions_unavailable": float(react(gone.mention_id).median()),
        "median_reactions_measured": float(react(ok.mention_id).median()),
        "median_followers_unavailable": float(pd.to_numeric(raw.loc[gone.mention_id].author_followers, errors="coerce").median()),
        "median_followers_measured": float(pd.to_numeric(raw.loc[ok.mention_id].author_followers, errors="coerce").median()),
    }
    return out, ok, pred


def e98(root):
    """Intervals on the FB posts: source-residual width vs split-conformal width calibrated on k FB posts."""
    d = fb_frame(root)
    ok = d[d.status == "ok"].dropna(subset=["y"]).reset_index(drop=True)
    mod = train_source(root)
    pred = mod.predict(ok[C2].fillna(0))
    e = ok.y.values - pred
    rng = np.random.default_rng(SEED)
    out = {"posts": int(len(e)), "source_width": {}, "conformal": {}}
    # source-platform residuals: leave-one-platform-out ridge residuals on X and YouTube
    f, meta = em.load(root)
    src = meta.platform.isin(["twitter", "youtube"]).values
    from sklearn.model_selection import GroupKFold
    oof = np.zeros(src.sum())
    yy = meta.y.values[src]
    for tr, te in GroupKFold(5).split(f[src][C2].fillna(0), yy, meta.brand.values[src]):
        oof[te] = ridge().fit(f[src][C2].fillna(0).iloc[tr], yy[tr]).predict(f[src][C2].fillna(0).iloc[te])
    for a in (0.8, 0.9):
        q = float(np.quantile(np.abs(yy - oof), a))
        out["source_width"][str(a)] = {"half_width": round(q, 3), "coverage": round(float((np.abs(e) <= q).mean()), 3)}
    for a in (0.8, 0.9):
        rows = {}
        for k in (10, 25, 50):
            cov = []
            for _ in range(400):
                i = rng.permutation(len(e)); cal, rest = i[:k], i[k:]
                q = float(np.quantile(np.abs(e[cal]), a))
                cov.append((np.abs(e[rest]) <= q).mean())
            rows[str(k)] = {"coverage": round(float(np.mean(cov)), 3), "p10": round(float(np.percentile(cov, 10)), 3),
                            "p90": round(float(np.percentile(cov, 90)), 3)}
        out["conformal"][str(a)] = rows
    return out


def main():
    o95, ok, pred = e95(ROOT)
    o99 = e98(ROOT)
    out = {"E95": o95, "E98": o99}
    print(json.dumps(o95, indent=1)); print(json.dumps(o99, indent=1))
    (ROOT / "data" / "processed" / "fb_extend_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
