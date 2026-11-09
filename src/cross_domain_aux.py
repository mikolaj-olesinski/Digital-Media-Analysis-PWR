"""E203: two ideas from the cross-domain popularity literature, tested on the Facebook (402) and Instagram (338) transfer.
(a) Post age at read (Shulman et al. 2016 / SMTPD 2025: temporal signals carry and transfer): log age added to the
    X + YouTube ridge on log likes + log comments.
(b) Brand prominence on other platforms (Tatar et al. 2014 "cross domain"; Roy et al. 2013): the brand's median residual
    of the same ridge on TikTok / YouTube / X, added to the FB / IG prediction with weight 0.5 or 1.
Writes data/processed/cross_domain_aux_numbers.json. Statistics only.
    python src/cross_domain_aux.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, fb_extend as fe, ig_transfer as it  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def targets():
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    r = it.frame(); ig = r[(r.status == "ok") & r.views.notna() & r.likes.notna()].copy()
    ig["y"] = np.log10(ig.views.clip(lower=1)); ig["log_likes"] = np.log1p(ig.likes); ig["log_comments"] = np.log1p(ig.comments.fillna(0))
    age = (pd.to_datetime(ig.checked_at) - pd.to_datetime(raw.loc[ig.mention_id].created_at.values, errors="coerce")).dt.days
    ig["log_age"] = np.log1p(age.clip(lower=0)); ig["age"] = age
    fb = fe.fb_frame(ROOT); fb = fb[fb.status == "ok"].dropna(subset=["y"]).copy(); fb["log_age"] = np.log1p(fb.age_days.clip(lower=0)); fb["age"] = fb.age_days
    return {"facebook": fb.reset_index(drop=True), "instagram": ig.reset_index(drop=True)}


def sc(d, p):
    s = fe.score(p, d.y.values, d.brand.values); s["median bias (+ = under)"] = round(float(np.median(d.y.values - p)), 3); return s


def main():
    f, meta = em.load(ROOT); m = meta.platform.isin(["twitter", "youtube"]).values
    X = f[m].copy(); X["log_age"] = np.log1p(X.metrics_age_days.clip(lower=0)); y = meta.y[m]
    T = targets(); out = {"post age at read, median days": {k: float(d.age.median()) for k, d in T.items()} | {"X + YouTube training": float(f[m].metrics_age_days.median())}}
    base = fe.ridge().fit(X[fe.C2].fillna(0), y); aged = fe.ridge().fit(X[fe.C2 + ["log_age"]].fillna(0), y)
    out["(a) age"] = {k: {"likes + comments": sc(d, base.predict(d[fe.C2])), "+ log age": sc(d, aged.predict(d[fe.C2 + ["log_age"]]))} for k, d in T.items()}
    allp = meta.platform.isin(["tiktok", "youtube", "twitter"]).values
    res = pd.DataFrame({"brand": meta.brand[allp].values, "plat": meta.platform[allp].values, "r": meta.y[allp].values - base.predict(f[allp][fe.C2].fillna(0))})
    out["(b) brand residual from other platforms"] = {}
    for src in (["tiktok"], ["youtube"], ["twitter"], ["tiktok", "youtube", "twitter"]):
        br = res[res.plat.isin(src)].groupby("brand").r.median(); key = " + ".join(src); out["(b) brand residual from other platforms"][key] = {}
        for k, d in T.items():
            p = base.predict(d[fe.C2]); b = d.brand.map(br).fillna(0).values
            out["(b) brand residual from other platforms"][key][k] = {"brand coverage": round(float(d.brand.isin(br.index).mean()), 3), **{f"weight {w}": sc(d, p + w * b) for w in (0.5, 1.0)}}
    (ROOT / "data/processed/cross_domain_aux_numbers.json").write_text(json.dumps({"E203": out}, indent=1))
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
