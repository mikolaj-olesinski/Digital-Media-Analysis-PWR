"""E202: Instagram transfer on the hand-collected reels (views, likes and comments read at the same moment).
Data: data/manual/ig_browser_batch.csv (500 random reels from the Brand24 IG slice) + ig_browser_results_*.csv,
collected by a browser agent on a team member's logged-in account (read-only; values from the web API).
Scored like E93: ridge on log likes + log comments trained on X + YouTube, never seeing Instagram.
  - error with brand-bootstrap CI, factor, bias, over / under counts, share within x2 and x3;
  - posts with hidden likes: a comments-only ridge from X + YouTube;
  - 25-post calibration (median shift, mean of 200 draws);
  - own-platform reference: the same ridge trained on Instagram with brand-grouped 5-fold CV;
  - brand concentration: per brand, without the three largest brands.
Writes data/processed/ig_transfer_numbers.json. Statistics only.
    python src/ig_transfer.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, fb_extend as fe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
L2, L3 = np.log10(2), np.log10(3)


def frame():
    b = pd.read_csv(ROOT / "data/manual/ig_browser_batch.csv", dtype={"mention_id": str})
    r = pd.concat([pd.read_csv(p) for p in sorted((ROOT / "data/manual").glob("ig_browser_results_*.csv"))]).merge(b, on="id")
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    r["brand"] = raw.loc[r.mention_id].brand.values
    return r


def detail(y, p, g):
    e = y - p; s = fe.score(p, y, g)
    s.update({"median bias (+ = under)": round(float(np.median(e)), 3), "within x2": round(float((np.abs(e) < L2).mean()), 3),
              "within x3": round(float((np.abs(e) < L3).mean()), 3), "under-predicted": int((e > 0).sum()), "over-predicted": int((e < 0).sum()),
              "under by more than x3": int((e > L3).sum()), "over by more than x3": int((e < -L3).sum()), "median |error| factor": round(float(10 ** np.median(np.abs(e))), 2)})
    return s


def main():
    r = frame(); out = {"rows": int(len(r)), "status": r.status.value_counts().to_dict(), "likes hidden": int((r.status.eq("ok") & r.likes.isna()).sum())}
    ok = r[(r.status == "ok") & r.views.notna()].copy()
    ok["y"] = np.log10(ok.views.clip(lower=1)); ok["log_likes"] = np.log1p(ok.likes); ok["log_comments"] = np.log1p(ok.comments.fillna(0))
    vis = ok[ok.likes.notna()].reset_index(drop=True); hid = ok[ok.likes.isna()].reset_index(drop=True)
    out["posts with views"] = int(len(ok)); out["brands"] = int(vis.brand.nunique())
    f, meta = em.load(ROOT); m = meta.platform.isin(["twitter", "youtube"]).values
    src = fe.ridge().fit(f[m][fe.C2].fillna(0), meta.y[m]); src_c = fe.ridge().fit(f[m][["log_comments"]].fillna(0), meta.y[m])
    p = src.predict(vis[fe.C2]); e = vis.y.values - p
    out["X + YouTube -> IG, likes + comments"] = detail(vis.y.values, p, vis.brand.values)
    if len(hid): out["X + YouTube -> IG, comments only (likes hidden)"] = detail(hid.y.values, src_c.predict(hid[["log_comments"]]), hid.brand.values)
    out["X + YouTube -> IG, comments only, on posts with visible likes"] = detail(vis.y.values, src_c.predict(vis[["log_comments"]]), vis.brand.values)
    rng = np.random.default_rng(42); cal = []
    for _ in range(200):
        i = rng.permutation(len(e)); z = e[i[25:]] - np.median(e[i[:25]]); cal.append((np.abs(z).mean(), (np.abs(z) < L2).mean()))
    c = np.mean(cal, 0); out["25-post median shift (mean of 200 draws)"] = {"MAE": round(float(c[0]), 3), "factor": round(float(10 ** c[0]), 2), "within x2": round(float(c[1]), 3)}
    pr = np.zeros(len(vis))
    for tr, te in GroupKFold(min(5, vis.brand.nunique())).split(vis, vis.y, vis.brand):
        pr[te] = fe.ridge().fit(vis.iloc[tr][fe.C2], vis.y.iloc[tr]).predict(vis.iloc[te][fe.C2])
    out["own platform: IG ridge, brand CV"] = detail(vis.y.values, pr, vis.brand.values)
    vc = vis.brand.value_counts(); top = vc.index[:3]
    out["top 3 brands share"] = round(float(vc.iloc[:3].sum() / len(vis)), 3)
    k = ~vis.brand.isin(top).values; out["transfer without the 3 largest brands"] = detail(vis.y.values[k], p[k], vis.brand.values[k])
    out["per brand (posts >= 10)"] = sorted([{"posts": int(n), "MAE": round(float(np.abs(e[vis.brand.values == b]).mean()), 3), "bias": round(float(np.median(e[vis.brand.values == b])), 3)} for b, n in vc.items() if n >= 10], key=lambda d: -d["posts"])
    (ROOT / "data/processed/ig_transfer_numbers.json").write_text(json.dumps({"E202": out}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
