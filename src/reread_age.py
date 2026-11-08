"""E211: is the E210 re-read gain just post age? Growth exponent b per platform from export / re-read pairs of the same
post (Δlog10 views = b · Δlog10 age, through the origin), export X + YouTube labels moved to the re-read age, ridge on
log likes + log comments, scored on FB / IG. Decomposition on the paired posts. b is never estimated from FB / IG.
Writes data/processed/reread_age_numbers.json.
    .venv/bin/python src/reread_age.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, fb_extend as fe, ig_transfer as ig, results_b as rb  # noqa: E402
from reread_transfer import reread, paired_diff, full_score, coefs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
E210_IG_GAIN = 0.053


def export_frame():
    f, meta = em.load(ROOT); raw = eu.load(ROOT).loc[meta.index]
    age = (pd.to_datetime(raw.metrics_updated_at) - pd.to_datetime(raw.created_at)).dt.total_seconds() / 86400
    return pd.DataFrame({"mention_id": raw.mention_id.astype(str).values, "platform": meta.platform.values,
                         "brand": meta.brand.fillna("unknown").values, "y": meta.y.values,
                         "log_likes": f.log_likes.fillna(0).values, "log_comments": f.log_comments.fillna(0).values,
                         "created_at": pd.to_datetime(raw.created_at).values, "age_ex": age.clip(lower=1).values})


def slope(dx, dy, intercept=False):
    if intercept:
        b, a = np.polyfit(dx, dy, 1); return float(b), float(a)
    return float((dx * dy).sum() / (dx * dx).sum()), 0.0


def main():
    T = rb.targets(); ex = export_frame(); rr = reread()
    rr["age_rr"] = ((rr.fetched_at - rr.created_at).dt.total_seconds() / 86400).clip(lower=1)
    pr = ex.merge(rr[["mention_id", "y", "log_likes", "log_comments", "age_rr"]], on="mention_id", suffixes=("", "_rr"))
    pr["dla"] = np.log10(pr.age_rr) - np.log10(pr.age_ex); pr["dy"] = pr.y_rr - pr.y
    pr["dl"] = pr.log_likes_rr - pr.log_likes; pr["dc"] = pr.log_comments_rr - pr.log_comments
    out = {"pairs": pr.platform.value_counts().to_dict(),
           "median read age, days": {"export X + YT": float(ex[ex.platform.isin(["twitter", "youtube"])].age_ex.median()),
                                     "re-read X + YT": float(rr[rr.platform.isin(["twitter", "youtube"])].age_rr.median())}}
    # FB / IG read age (descriptive only)
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    fb = fe.fb_frame(ROOT); fb = fb[fb.status == "ok"].dropna(subset=["y"]); out["median read age, days"]["facebook"] = float(fb.age_days.median())
    r = ig.frame(); r = r[(r.status == "ok") & r.views.notna() & r.likes.notna()]
    out["median read age, days"]["instagram"] = float(((pd.to_datetime(r.checked_at) - pd.to_datetime(raw.loc[r.mention_id].created_at.values)).dt.days).median())
    B = {}
    for p, d in pr.groupby("platform"):
        b0, _ = slope(d.dla.values, d.dy.values); b1, a1 = slope(d.dla.values, d.dy.values, True)
        bl, _ = slope(d.dla.values, d.dl.values); bc, _ = slope(d.dla.values, d.dc.values)
        B[p] = {"b (through origin)": round(b0, 3), "b with intercept": round(b1, 3), "intercept": round(a1, 3),
                "b likes": round(bl, 3), "b comments": round(bc, 3), "median Δlog10 age": round(float(d.dla.median()), 3),
                "median Δlog10 views": round(float(d.dy.median()), 3), "pairs": int(len(d))}
    out["growth exponents (source pairs only)"] = B
    # move every export post to its age at the re-read date
    fetch = rr.groupby("platform").fetched_at.median()
    src = ex[ex.platform.isin(["twitter", "youtube"])].copy()
    src["age_new"] = ((src.platform.map(fetch) - src.created_at).dt.total_seconds() / 86400).clip(lower=1)
    src["dla"] = np.log10(src.age_new) - np.log10(src.age_ex)
    bo = src.platform.map({p: B[p]["b (through origin)"] for p in B}); bi = src.platform.map({p: B[p]["b with intercept"] for p in B})
    ai = src.platform.map({p: B[p]["intercept"] for p in B})
    bl = src.platform.map({p: B[p]["b likes"] for p in B}); bc = src.platform.map({p: B[p]["b comments"] for p in B})
    V = {"export X + YT (base)": src,
         "export moved to re-read age (primary)": src.assign(y=src.y + bo * src.dla),
         "moved, b with intercept": src.assign(y=src.y + ai + bi * src.dla),
         "moved views + likes + comments": src.assign(y=src.y + bo * src.dla, log_likes=src.log_likes + bl * src.dla, log_comments=src.log_comments + bc * src.dla)}
    pxy = pr[pr.platform.isin(["twitter", "youtube"])]
    pb = pxy.platform.map({p: B[p]["b (through origin)"] for p in B})
    V |= {"pairs: export y + export reactions": pxy,
          "pairs: moved y + export reactions": pxy.assign(y=pxy.y + pb * pxy.dla),
          "pairs: re-read y + export reactions": pxy.assign(y=pxy.y_rr),
          "pairs: re-read y + re-read reactions": pxy.assign(y=pxy.y_rr, log_likes=pxy.log_likes_rr, log_comments=pxy.log_comments_rr)}
    res, preds = {}, {}
    for n, s in V.items():
        m = fe.ridge().fit(s[fe.C2], s.y); res[n] = {"rows": int(len(s)), "coefficients (raw scale)": coefs(m)}
        for p, d in T.items():
            preds[(n, p)] = m.predict(d[fe.C2]); res[n][p] = full_score(preds[(n, p)], d.y.values, d.brand.values)
            if n != "export X + YT (base)":
                res[n][p]["paired diff vs base"] = paired_diff(d.y.values - preds[(n, p)], d.y.values - preds[("export X + YT (base)", p)], d.brand.values)
    out["variants"] = res
    g = -res["export moved to re-read age (primary)"]["instagram"]["paired diff vs base"]["diff"]
    out["primary IG gain / E210 IG gain"] = round(g / E210_IG_GAIN, 2)
    # views per like by age bin
    bins = [0, 7, 30, 90, 180, 365, 10000]; lab = ["<7d", "7-30d", "30-90d", "90-180d", "180-365d", ">365d"]
    vpl = {}
    for name, d, a, yy, ll in [("export", ex, "age_ex", "y", "log_likes"), ("re-read", rr.assign(age_ex=rr.age_rr), "age_ex", "y", "log_likes")]:
        for p in ["tiktok", "youtube", "twitter"]:
            s = d[d.platform == p]; c = pd.cut(s[a], bins, labels=lab)
            vpl[f"{name} {p}"] = {str(k): {"n": int(len(v)), "median log10(views/(likes+1))": round(float((v[yy] - np.log10(np.expm1(v[ll]) + 1)).median()), 3)}
                                  for k, v in s.groupby(c, observed=True)}
    out["views per like by age bin"] = vpl
    (ROOT / "data/processed/reread_age_numbers.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ["pairs", "median read age, days", "growth exponents (source pairs only)", "primary IG gain / E210 IG gain"]}, indent=1))
    for n, v in res.items():
        print(n, v["rows"], v["coefficients (raw scale)"])
        for p in T: print("  ", p, {k: v[p][k] for k in ("MAE", "CI", "factor", "median bias (+ = under)", "top 20% by predicted y, mean bias", "paired diff vs base") if k in v[p]})
    print(json.dumps(vpl, indent=0))


if __name__ == "__main__":
    main()
