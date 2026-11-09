"""E229: do Instagram follower counts (read from the owners' public profiles, 2026-11-09) improve the IG estimate?
Reels: the 642 scored IG reels (ig_transfer.frame), owner from data/manual/ig_work/ig_reel_owners.csv, followers from
data/manual/ig_owner_profiles.csv (both private, git-ignored; only counts and errors are written).
1. Transfer: ridge trained on export X + YouTube with log likes + log comments (ours) vs + log followers.
2. Own platform (IG only): ridge with and without log followers, 5-fold CV grouped by owner and by brand.
Followers were read months after the views (followers now vs views then): an upper bound for "followers at read time".
Writes data/processed/ig_followers_numbers.json.
    .venv/bin/python src/ig_followers.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em, fb_extend as fe, ig_transfer as ig, reread_transfer as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C3 = fe.C2 + ["log_followers"]


def sc(y, p, g):
    e = y - p; return {"MAE": round(float(np.abs(e).mean()), 3), "factor": round(float(10 ** np.abs(e).mean()), 2), "within x2": round(float((np.abs(e) < np.log10(2)).mean()), 3)}


def cv(d, cols, groups):
    p = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(d, d.y, groups):
        p[te] = fe.ridge().fit(d.iloc[tr][cols], d.y.iloc[tr]).predict(d.iloc[te][cols])
    return p


def main():
    r = ig.frame(); r = r[(r.status == "ok") & r.views.notna() & r.likes.notna()].copy()
    o = pd.read_csv(ROOT / "data/manual/ig_work/ig_reel_owners.csv")[["id", "owner"]]
    pr = pd.read_csv(ROOT / "data/manual/ig_owner_profiles.csv"); pr = pr[pr.status == "ok"][["owner", "followers"]]
    d = r.merge(o, on="id", how="left").merge(pr, on="owner", how="left")
    d["y"] = np.log10(d.views.clip(lower=1)); d["log_likes"] = np.log1p(d.likes); d["log_comments"] = np.log1p(d.comments.fillna(0))
    n_all = len(d); d = d[d.followers.notna()].reset_index(drop=True); d["log_followers"] = np.log1p(d.followers)
    f, meta = em.load(ROOT); m = (meta.platform.isin(["twitter", "youtube"]) & f.log_followers.notna()).values
    X = f[m].copy(); X[fe.C2] = X[fe.C2].fillna(0); y = meta.y[m]
    a, b = fe.ridge().fit(X[fe.C2], y), fe.ridge().fit(X[C3], y)
    g = d.brand.fillna("unknown").values; Y = d.y.values
    pa, pb = a.predict(d[fe.C2]), b.predict(d[C3])
    out = {"reels scored": n_all, "reels with owner followers": int(len(d)), "owners": int(d.owner.nunique()),
           "followers median / p10 / p90": [int(d.followers.median()), int(d.followers.quantile(.1)), int(d.followers.quantile(.9))],
           "Spearman log followers vs log views": round(float(d[["log_followers", "y"]].corr("spearman").iloc[0, 1]), 3),
           "transfer X + YT": {"likes + comments (ours)": sc(Y, pa, g), "+ followers": sc(Y, pb, g), "+ followers - ours": rt.paired_diff(Y - pb, Y - pa, g),
                               "source rows with followers": int(m.sum())}}
    for name, grp in (("grouped by owner", d.owner.values), ("grouped by brand", g)):
        p2, p3, p1 = cv(d, fe.C2, grp), cv(d, C3, grp), cv(d, ["log_followers"], grp)
        out[f"own platform, {name}"] = {"likes + comments": sc(Y, p2, g), "+ followers": sc(Y, p3, g), "followers only": sc(Y, p1, g),
                                        "+ followers - likes + comments": rt.paired_diff(Y - p3, Y - p2, g)}
    (ROOT / "data/processed/ig_followers_numbers.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
