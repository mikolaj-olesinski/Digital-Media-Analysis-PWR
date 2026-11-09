"""Age-normalised views (E101): views keep growing with the age of the check, and the check age differs by platform
and over time. Normalise the target to 30 days with a growth exponent b, train on the normalised target, and convert
back to the target's own age. Does that lower the error or the transfer bias?

    .venv/bin/python src/age_target.py

Writes data/processed/age_target_numbers.json.
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
import validation as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C2 = ["log_likes", "log_comments"]
PLAT = ["tiktok", "youtube", "twitter"]


def ridge():
    return make_pipeline(StandardScaler(), Ridge())


def prep(d):
    d = d.copy()
    d["age"] = (pd.to_datetime(d.metrics_updated_at, errors="coerce") - pd.to_datetime(d.created_at, errors="coerce")).dt.total_seconds() / 86400
    d["log_likes"] = np.log1p(d.likes.fillna(0))
    d["log_comments"] = np.log1p(d.comments.fillna(0))
    d["y"] = np.log10(pd.to_numeric(d.view_count, errors="coerce").where(lambda x: x > 0))
    return d


def growth_b(d):
    """b in log10(views/likes) = b * log10(age) + c, per platform, on posts with likes > 0."""
    out = {}
    for p in PLAT:
        q = d[(d.platform == p) & (d.age > 0) & (d.likes > 0) & d.y.notna()]
        if len(q) < 50:
            out[p] = np.nan; continue
        out[p] = float(np.polyfit(np.log10(q.age), np.log10(q.view_count / q.likes), 1)[0])
    return out


def main():
    d = prep(eu.load(ROOT))
    b = growth_b(d)
    out = {"b": {k: round(v, 3) for k, v in b.items()}, "lopo": {}, "facebook": {}}
    for h in PLAT:
        tr = d[d.platform.isin([p for p in PLAT if p != h]) & d.y.notna() & (d.age > 0)]
        te = d[(d.platform == h) & d.y.notna() & (d.age > 0)]
        bh = float(np.mean([b[p] for p in PLAT if p != h]))
        # standard
        m = ridge().fit(tr[C2].fillna(0), tr.y)
        e0 = te.y.values - m.predict(te[C2].fillna(0))
        # normalised to 30 days
        tr30 = tr.y - tr.platform.map(b) * np.log10(tr.age / 30)
        m30 = ridge().fit(tr[C2].fillna(0), tr30)
        pred = m30.predict(te[C2].fillna(0)) + bh * np.log10(te.age / 30)
        e1 = te.y.values - pred
        out["lopo"][h] = {"b_target_used": round(bh, 3), "standard": round(float(np.abs(e0).mean()), 3),
                          "age_normalised": round(float(np.abs(e1).mean()), 3),
                          "bias_standard": round(float(e0.mean()), 3), "bias_normalised": round(float(e1.mean()), 3),
                          "posts": int(len(te))}
    # Facebook: use the mean b of X and YouTube
    fb = pd.read_csv(ROOT / "data" / "processed" / "fb_all_fetched.csv", dtype={"mention_id": str})
    fb = fb[(fb.status == "ok") & (pd.to_numeric(fb.views, errors="coerce") > 0)].copy()
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    r = raw.loc[fb.mention_id]
    fb["created_at"] = pd.to_datetime(r.created_at.values, errors="coerce")
    fb["fetched_at"] = pd.to_datetime(fb.fetched_at, errors="coerce", utc=True).dt.tz_convert(None)
    fb["age"] = (fb.fetched_at - fb.created_at).dt.total_seconds() / 86400
    fb["log_likes"] = np.log1p(pd.to_numeric(fb.reactions, errors="coerce").fillna(0))
    fb["log_comments"] = np.log1p(pd.to_numeric(fb.comments, errors="coerce").fillna(0))
    fb["y"] = np.log10(pd.to_numeric(fb.views))
    fb = fb[fb.age > 0]
    tr = d[d.platform.isin(["twitter", "youtube"]) & d.y.notna() & (d.age > 0)]
    bh = float(np.mean([b["twitter"], b["youtube"]]))
    tr30 = tr.y - tr.platform.map(b) * np.log10(tr.age / 30)
    m0 = ridge().fit(tr[C2].fillna(0), tr.y)
    m30 = ridge().fit(tr[C2].fillna(0), tr30)
    e0 = fb.y.values - m0.predict(fb[["log_likes", "log_comments"]].fillna(0))
    e1 = fb.y.values - (m30.predict(fb[["log_likes", "log_comments"]].fillna(0)) + bh * np.log10(fb.age / 30))
    out["facebook"] = {"b_used": round(bh, 3), "standard": round(float(np.abs(e0).mean()), 3),
                       "age_normalised": round(float(np.abs(e1).mean()), 3),
                       "bias_standard": round(float(e0.mean()), 3), "bias_normalised": round(float(e1.mean()), 3),
                       "posts": int(len(fb))}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "age_target_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
