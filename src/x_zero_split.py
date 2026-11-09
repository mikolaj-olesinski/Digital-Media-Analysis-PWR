"""E219: are X's zero-like posts what breaks X as a transfer source and target?
Ridge on log likes + log comments (fe.ridge, C2), rows pooled as in E201. Three variants per (training set, target):
base (all rows), drop (zero-like source rows removed) and two-part (target zero-like rows get the median y of the
source zero-like rows, other target rows the ridge fitted on source non-zero rows). Zero-like = likes == 0 (primary),
sensitivity likes == 0 and comments == 0. Sources / targets: train_combos.data() (export TikTok / YouTube / X, FB, IG)
and the same-moment re-read of X. Every target is also split into zero-like vs non-zero rows.
Writes data/processed/x_zero_split_numbers.json.
    .venv/bin/python src/x_zero_split.py"""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, fb_extend as fe, train_combos as tc, validation as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C2 = fe.C2
N_PAIR = 300
MIN_ZERO = 10


def reread(plat, drop_nan=True):
    """Same-moment re-read (E204): status ok, views present, dedup by mention_id (last row), NaN reactions = missing."""
    f = {"youtube": "youtube", "twitter": "x", "tiktok": "tiktok"}[plat]
    d = pd.read_csv(ROOT / f"data/processed/{f}_full_fetched.csv", dtype={"mention_id": str}).drop_duplicates("mention_id", keep="last")
    d = d[(d.status == "ok") & d.views.notna()].copy()
    if drop_nan: d = d.dropna(subset=["likes", "comments"])
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id"); r = raw.loc[d.mention_id]
    d["platform"] = plat; d["brand"] = r.brand.fillna("unknown").values
    d["followers"] = pd.to_numeric(r.author_followers, errors="coerce").values
    d["created_at"] = pd.to_datetime(r.created_at.values, errors="coerce")
    d["y"] = np.log10(d.views.clip(lower=1)); d["log_likes"] = np.log1p(d.likes.fillna(0)); d["log_comments"] = np.log1p(d.comments.fillna(0))
    return d.reset_index(drop=True)


def zero(d, kind):
    return (d.log_likes.values == 0) if kind == "likes" else ((d.log_likes.values == 0) & (d.log_comments.values == 0))


def predict(train, test, kind):
    zt, ze = zero(train, kind), zero(test, kind)
    base = fe.ridge().fit(train[C2], train.y).predict(test[C2])
    nz = fe.ridge().fit(train[~zt][C2], train.y[~zt]).predict(test[C2])
    two = nz.copy()
    two[ze] = np.median(train.y[zt]) if zt.sum() >= MIN_ZERO else base[ze]
    return {"base": base, "drop": nz, "two-part": two}, ze


def stats(y, p, g):
    e = y - p; mae, lo, hi = va.mae_ci(e, g)
    return {"n": int(len(y)), "MAE": round(mae, 3), "factor": round(10 ** mae, 2), "CI": [round(lo, 3), round(hi, 3)], "median bias (+ = under)": round(float(np.median(e)), 3)}


def paired(y, pb, pn, g, seed=42):
    """MAE(new) - MAE(base) on the same posts, 95% CI from resampling brands (N_PAIR draws). Negative = new is better."""
    d = pd.DataFrame({"g": g, "e": np.abs(y - pn) - np.abs(y - pb)}).groupby("g").e.agg(["sum", "count"])
    rng = np.random.default_rng(seed); idx = np.arange(len(d))
    bs = [d.iloc[s]["sum"].sum() / d.iloc[s]["count"].sum() for s in (rng.choice(idx, len(idx)) for _ in range(N_PAIR))]
    return {"diff": round(float(d["sum"].sum() / d["count"].sum()), 3), "CI": [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])]}


def evaluate(train, test, kind):
    P, ze = predict(train, test, kind); y, g = test.y.values, test.brand.values
    out = {"zero-like share source": round(float(zero(train, kind).mean()), 3), "zero-like share target": round(float(ze.mean()), 3)}
    for v, p in P.items():
        s = stats(y, p, g)
        for part, m in (("zero-like rows", ze), ("non-zero rows", ~ze)):
            if m.sum() >= 5: s[part] = stats(y[m], p[m], g[m])
        if v != "base": s["paired diff vs base"] = paired(y, P["base"], p, g)
        out[v] = s
    return out


def main():
    d = tc.data(); rx = reread("twitter"); rx = rx.assign(platform="x_reread")[["platform", "brand", "y"] + C2]
    allf = pd.concat([d[["platform", "brand", "y"] + C2], rx], ignore_index=True)
    sel = lambda ps: allf[allf.platform.isin(ps)].reset_index(drop=True)
    res = {"n": allf.platform.value_counts().to_dict()}
    for kind in ("likes", "likes and comments"):
        r = {}
        # X as a source
        for src in (["twitter"], ["x_reread"], ["twitter", "youtube"], ["x_reread", "youtube"]):
            for tgt in ("facebook", "instagram", "tiktok", "youtube"):
                if tgt in src: continue
                r[f"{' + '.join(src)} -> {tgt}"] = evaluate(sel(src), sel([tgt]), kind)
        # X as a target: every combination of the other four platforms
        others = ["tiktok", "youtube", "facebook", "instagram"]
        for k in range(1, 5):
            for c in itertools.combinations(others, k):
                for tgt in ("twitter", "x_reread"):
                    r[f"{' + '.join(c)} -> {tgt}"] = evaluate(sel(list(c)), sel([tgt]), kind)
        res[f"zero = {kind}"] = r
    p = res["zero = likes"]
    prim = {
        "source X export -> facebook": {v: p["twitter -> facebook"][v]["paired diff vs base"] for v in ("drop", "two-part")},
        "source X export -> instagram": {v: p["twitter -> instagram"][v]["paired diff vs base"] for v in ("drop", "two-part")},
        "target TT + YT + FB + IG -> X export": {v: p["tiktok + youtube + facebook + instagram -> twitter"][v]["paired diff vs base"] for v in ("drop", "two-part")},
    }
    res["primary (zero = likes, paired diff = new - base)"] = prim
    for k in ("twitter", "x_reread"):
        rows = [v for kk, v in p.items() if kk.endswith(f"-> {k}")]
        res[f"mean over 15 combinations -> {k}"] = {v: round(float(np.mean([r[v]["MAE"] for r in rows])), 3) for v in ("base", "drop", "two-part")}
    (ROOT / "data/processed/x_zero_split_numbers.json").write_text(json.dumps({"E219": res}, indent=1))
    print(json.dumps(prim, indent=1))
    for kk in ("twitter -> facebook", "twitter -> instagram", "x_reread -> facebook", "x_reread -> instagram", "twitter + youtube -> facebook",
               "twitter + youtube -> instagram", "x_reread + youtube -> facebook", "x_reread + youtube -> instagram",
               "tiktok + youtube + facebook + instagram -> twitter", "tiktok + youtube + facebook + instagram -> x_reread", "tiktok + youtube -> twitter", "tiktok + youtube -> x_reread"):
        print(kk, {v: (p[kk][v]["MAE"], p[kk][v].get("zero-like rows", {}).get("MAE"), p[kk][v].get("non-zero rows", {}).get("MAE")) for v in ("base", "drop", "two-part")}, p[kk]["zero-like share source"], p[kk]["zero-like share target"])
    for k in ("twitter", "x_reread"): print(k, res[f"mean over 15 combinations -> {k}"])


if __name__ == "__main__":
    main()
