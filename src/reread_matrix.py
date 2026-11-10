"""E238: one clean grid for the slides: ridge on log likes + comments, in platform (brand-grouped CV on the platform's own posts)
and out of platform (train on the other platforms), each with export labels and with same-moment re-read labels.
In platform: export = all export posts of the platform, re-read = re-read posts (TikTok only 426). Out of platform: numbers
from reread_transfer_numbers.json (E201 / E210 grid). Writes data/processed/reread_matrix_numbers.json.
    .venv/bin/python -W ignore src/reread_matrix.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em, fb_extend as fe, reread_transfer as rt, validation as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]; C2 = fe.C2


def cv(d):
    X = d[C2].fillna(0); y = d.y.values; g = d.brand.values; p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g): p[te] = fe.ridge().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return {"factor": round(float(10 ** va.mae_ci(y - p, g)[0]), 2), "posts": int(len(y))}


def main():
    f, meta = em.load(ROOT); rr = rt.reread(); out = {"in platform": {"export": {}, "re-read": {}}, "out of platform": {"export": {}, "re-read": {}}}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        e = pd.DataFrame({"y": meta.y[m].values, "log_likes": f[m].log_likes.fillna(0).values, "log_comments": f[m].log_comments.fillna(0).values, "brand": meta.brand[m].fillna("unknown").values})
        out["in platform"]["export"][p] = cv(e); out["in platform"]["re-read"][p] = cv(rr[rr.platform == p].reset_index(drop=True))
    r = json.loads((ROOT / "data/processed/reread_transfer_numbers.json").read_text()); g = r["grid (31 training sets)"]; v = r["variants"]
    ex, re = g["export TikTok/YouTube/X"]["combinations"], g["re-read TikTok/YouTube/X"]["combinations"]
    for key, c, base, prim in (("export", ex, v["export X + YT (base)"], None), ("re-read", re, v["re-read X + YT, drop missing (primary)"], None)):
        out["out of platform"][key] = {"tiktok": round(c["youtube + twitter"]["tiktok"]["factor"], 2), "youtube": round(c["tiktok + twitter"]["youtube"]["factor"], 2),
                                       "twitter": round(c["tiktok + youtube"]["twitter"]["factor"], 2), "facebook": base["facebook"]["factor"], "instagram": base["instagram"]["factor"]}
    (ROOT / "data/processed/reread_matrix_numbers.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out))


if __name__ == "__main__":
    main()
