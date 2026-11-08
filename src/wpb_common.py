"""Shared helpers for work package B (E214-E217): re-read loader, export loaders, scoring and paired brand bootstrap.
The re-read loader is a copy of the one in the E210-E221 plan, so WP-B does not import WP-A files.
Re-read: data/processed/{youtube,x,tiktok}_full_fetched.csv, status == "ok" with views; X deduplicated by mention_id (keep
last); y = log10(views.clip(1)); log_likes = log1p(likes), log_comments = log1p(comments); NaN reactions are missing
(primary: drop the row; sensitivity: fill 0). Brand from the raw export via mention_id; platform "twitter" for x."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import validation as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
C2 = ["log_likes", "log_comments"]
L2 = np.log10(2)
PLAT3 = ["tiktok", "youtube", "twitter"]


def reread(fill0=False, plats=("youtube", "twitter", "tiktok")):
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    out = []
    for p in plats:
        name = "x" if p == "twitter" else p
        d = pd.read_csv(ROOT / f"data/processed/{name}_full_fetched.csv", dtype={"mention_id": str})
        d = d.drop_duplicates("mention_id", keep="last")
        for c in ["views", "likes", "comments", "shares"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d[(d.status == "ok") & d.views.notna()].copy()
        if fill0:
            d[["likes", "comments"]] = d[["likes", "comments"]].fillna(0)
        else:
            d = d.dropna(subset=["likes", "comments"])
        d = d[d.mention_id.isin(raw.index)]
        d["platform"] = p
        d["brand"] = raw.loc[d.mention_id, "brand"].fillna("unknown").values
        d["y"] = np.log10(d.views.clip(lower=1))
        d["log_likes"] = np.log1p(d.likes); d["log_comments"] = np.log1p(d.comments)
        out.append(d[["mention_id", "platform", "brand", "y", "log_likes", "log_comments"]])
    return pd.concat(out, ignore_index=True)


def score(y, p, g):
    """MAE (log10), brand-bootstrap CI, factor, median bias (+ = under), within x2, top-fifth and top-tenth bias by true y and
    by predicted y (mean of y - p in the group; the predicted-y split avoids the regression-to-the-mean artefact, E114)."""
    y, p, g = np.asarray(y, float), np.asarray(p, float), np.asarray(g)
    e = y - p; mae, lo, hi = va.mae_ci(e, g)
    out = {"n": int(len(y)), "MAE": round(mae, 4), "CI": [round(float(lo), 4), round(float(hi), 4)], "factor": round(10 ** mae, 3),
           "median bias (+ = under)": round(float(np.median(e)), 4), "within x2": round(float((np.abs(e) < L2).mean()), 3)}
    for q, lab in [(0.8, "top fifth"), (0.9, "top tenth")]:
        out[f"{lab} bias by true y"] = round(float(e[y >= np.quantile(y, q)].mean()), 4)
        out[f"{lab} bias by predicted y"] = round(float(e[p >= np.quantile(p, q)].mean()), 4)
    return out


def paired_gain(y, p_base, p_new, g, n_boot=300, seed=SEED):
    """Gain = MAE(base) - MAE(new) (positive = new is better) with a 95% brand-cluster bootstrap CI."""
    y = np.asarray(y, float); g = pd.Series(np.asarray(g))
    d = pd.DataFrame({"b": np.abs(y - np.asarray(p_base)), "n": np.abs(y - np.asarray(p_new)), "g": g.values}).groupby("g").agg(
        b=("b", "sum"), n=("n", "sum"), c=("b", "size"))
    rng = np.random.default_rng(seed); k = len(d); B, N, C = d.b.values, d.n.values, d.c.values
    w = rng.multinomial(k, np.full(k, 1 / k), size=n_boot)
    gains = (w @ B - w @ N) / (w @ C)
    gain = (B.sum() - N.sum()) / C.sum()
    return {"gain": round(float(gain), 4), "CI": [round(float(np.percentile(gains, 2.5)), 4), round(float(np.percentile(gains, 97.5)), 4)]}


def raw_eng():
    """ENG features (followers, reactions, missing flags) + english + text_len for all 12,500 export rows, with view_count.
    Same formulas as modeling.load_features, but without the views filter (used for unlabelled rows)."""
    d = eu.load(ROOT).astype({"mention_id": str})
    f = pd.DataFrame(index=d.index)
    f["log_followers"] = np.log10(d.author_followers.where(d.author_followers > 0))
    for c in ["likes", "comments", "shares"]:
        f[f"log_{c}"] = np.log1p(d[c]); f[f"na_{c}"] = d[c].isna().astype(int)
    f["na_followers"] = d.author_followers.isna().astype(int)
    text = d.content.fillna(d.title).fillna("")
    f["english"] = (d.language == "en").astype(int); f["text_len"] = np.log10(text.str.len() + 1)
    meta = pd.DataFrame({"mention_id": d.mention_id, "platform": d.platform, "brand": d.brand.fillna("unknown"),
                         "view_count": pd.to_numeric(d.view_count, errors="coerce")}, index=d.index)
    return f, meta
