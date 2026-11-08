"""E220: can borrowed large-account residuals calibrate the FB 90% band for pages above 1M followers?
E145 protocol: LightGBM quantile 0.05 / 0.95 (and 0.5) on export X + YouTube C2, CQR conformity scores on FB, 20 random
splits with 25% of FB brands for calibration, tiers by export followers, per-tier quantile with fallback to global under
20 calibration posts. New: the >1M quantile also uses scores of re-read TikTok / YouTube / X posts from >=1M-follower
accounts, their y shifted to the FB level (delta from calibration brands only). Also the E150 large-page point model
retrained on re-read sources.
Writes data/processed/fb_big_borrow_numbers.json.
    .venv/bin/python src/fb_big_borrow.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, fb_extend as fe, validation as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C2 = fe.C2
SEED, REPS, N_PAIR = 42, 20, 300
TIERS = ["<10k", "10k-1M", ">1M"]


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


def qm(a):
    return LGBMRegressor(objective="quantile", alpha=a, n_estimators=300, learning_rate=.05, num_leaves=31,
                         subsample=.9, colsample_bytree=.9, random_state=SEED, verbose=-1)


def fb():
    d = pd.read_csv(ROOT / "data/processed/fb_all_fetched.csv", dtype={"mention_id": str})
    d["views"] = pd.to_numeric(d.views, errors="coerce"); d = d[(d.status == "ok") & (d.views > 0)].reset_index(drop=True)
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    d["brand"] = raw.loc[d.mention_id, "brand"].values
    d["followers"] = pd.to_numeric(raw.loc[d.mention_id, "author_followers"].values, errors="coerce")
    d["log_likes"] = np.log1p(pd.to_numeric(d.reactions, errors="coerce").fillna(0))
    d["log_comments"] = np.log1p(pd.to_numeric(d.comments, errors="coerce").fillna(0))
    d["tier"] = pd.cut(d.followers, [0, 1e4, 1e6, 1e12], labels=TIERS, right=False).astype(str)
    d["y"] = np.log10(d.views.values)
    return d


def paired(y, pb, pn, g, seed=42):
    d = pd.DataFrame({"g": g, "e": np.abs(y - pn) - np.abs(y - pb)}).groupby("g").e.agg(["sum", "count"])
    rng = np.random.default_rng(seed); idx = np.arange(len(d))
    bs = [d.iloc[s]["sum"].sum() / d.iloc[s]["count"].sum() for s in (rng.choice(idx, len(idx)) for _ in range(N_PAIR))]
    return {"diff": round(float(d["sum"].sum() / d["count"].sum()), 3), "CI": [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])]}


FINITE = False   # E145 used the plain 0.9 quantile; the finite-sample (n+1) correction is reported as a sensitivity


def q90(E):
    return float(np.quantile(E, min(1, np.ceil((len(E) + 1) * .9) / len(E)) if FINITE else .9))


def bands(L, H, M, yf, gf, tf, bL, bH, by, br):
    V = ["global (E145)", "per-tier FB only (E145)", "borrowed, delta from FB >1M cal (primary)", "borrowed, delta from all FB cal",
         "borrowed, unshifted", "borrowed only, delta from FB >1M cal"]
    cov = {v: {t: [] for t in TIERS} for v in V}; wid = {v: {t: [] for t in TIERS} for v in V}; deltas = []; ncal_big = []
    rng = np.random.default_rng(SEED); brands = np.array(sorted(set(gf)))
    for _ in range(REPS):
        cal_b = set(rng.choice(brands, max(1, int(.25 * len(brands))), replace=False))
        cal = gf.isin(cal_b).values; te = ~cal
        E = np.maximum(L[cal] - yf[cal], yf[cal] - H[cal]); qg = q90(E)
        qt = {t: q90(E[tf[cal] == t]) if (tf[cal] == t).sum() >= 20 else qg for t in TIERS}
        kb = cal & (tf == ">1M"); ncal_big.append(int(kb.sum()))
        fb_res_big = np.median(yf[kb] - M[kb]) if kb.sum() >= 5 else np.median(yf[cal] - M[cal])
        d_big = fb_res_big - np.median(br); d_all = np.median(yf[cal] - M[cal]) - np.median(br); deltas.append(d_big)
        bE = lambda dl: np.maximum(bL - (by + dl), (by + dl) - bH)
        Ebig = E[tf[cal] == ">1M"]
        qbig = {V[2]: q90(np.r_[Ebig, bE(d_big)]), V[3]: q90(np.r_[Ebig, bE(d_all)]), V[4]: q90(np.r_[Ebig, bE(0.0)]), V[5]: q90(bE(d_big))}
        for t in TIERS:
            k = te & (tf == t)
            if k.sum() == 0: continue
            for v in V:
                q = qg if v == V[0] else (qt[t] if (v == V[1] or t != ">1M") else qbig[v])
                cov[v][t].append(((yf[k] >= L[k] - q) & (yf[k] <= H[k] + q)).mean()); wid[v][t].append(((H[k] + q) - (L[k] - q)).mean())
    band = {v: {t: {"coverage": round(float(np.mean(cov[v][t])), 3), "width log10": round(float(np.mean(wid[v][t])), 3),
                    "width vs global": round(float(np.mean(wid[v][t]) / np.mean(wid[V[0]][t])), 2)} for t in TIERS} for v in V}
    return band, deltas, ncal_big


def main():
    global FINITE
    f, meta = em.load(ROOT); src = meta.platform.isin(["twitter", "youtube"]).values
    Xs, ys = f[src][C2].fillna(0).reset_index(drop=True), meta.y[src].reset_index(drop=True)
    lo, hi, med = qm(.05).fit(Xs, ys), qm(.95).fit(Xs, ys), qm(.5).fit(Xs, ys)
    d = fb(); L, H, M = lo.predict(d[C2]), hi.predict(d[C2]), med.predict(d[C2]); yf, gf, tf = d.y.values, d.brand, d.tier.values
    bor = pd.concat([reread(p) for p in ("tiktok", "youtube", "twitter")], ignore_index=True)
    bor = bor[bor.followers >= 1e6].reset_index(drop=True)
    bL, bH, bM = lo.predict(bor[C2]), hi.predict(bor[C2]), med.predict(bor[C2]); by = bor.y.values
    br = by - bM
    band, deltas, ncal_big = bands(L, H, M, yf, gf, tf, bL, bH, by, br)
    FINITE = True; band_f, _, _ = bands(L, H, M, yf, gf, tf, bL, bH, by, br); FINITE = False
    brands = sorted(set(gf))
    out = {"FB posts per tier": {t: int((tf == t).sum()) for t in TIERS}, "FB brands": int(len(brands)),
           "borrowed posts (>=1M followers)": bor.platform.value_counts().to_dict(),
           "borrowed median residual y - M": round(float(np.median(br)), 3),
           "FB >1M median residual y - M (all posts)": round(float(np.median(yf[tf == ">1M"] - M[tf == ">1M"])), 3),
           "delta (primary) median over splits": round(float(np.median(deltas)), 3),
           "FB >1M calibration posts per split, median": float(np.median(ncal_big)), "reps": REPS, "band 90% (plain quantile, E145 protocol)": band,
           "band 90% (finite-sample n+1 quantile, sensitivity)": band_f}
    # E150 point model on re-read sources
    big = tf == ">1M"; yb, gb = yf[big], gf.values[big]
    def fit(Xt, yt): return fe.ridge().fit(Xt[C2], yt).predict(d.loc[big, C2])
    ex = lambda m: (pd.DataFrame({"log_likes": f[m].log_likes.fillna(0).values, "log_comments": f[m].log_comments.fillna(0).values}), meta.y.values[m])
    big_src = src & (meta.author_followers.fillna(0) >= 1e6).values
    rr = pd.concat([reread(p) for p in ("tiktok", "youtube", "twitter")], ignore_index=True)
    P = {"export X + YouTube, all pages (global)": fit(*ex(src)), "export X + YouTube, >=1M pages (E150)": fit(*ex(big_src)),
         "re-read X + YouTube, >=1M pages": fit(bor[bor.platform.isin(["twitter", "youtube"])], bor[bor.platform.isin(["twitter", "youtube"])].y),
         "re-read TikTok + YouTube + X, >=1M pages": fit(bor, bor.y),
         "re-read X + YouTube, all pages": fit(rr[rr.platform.isin(["twitter", "youtube"])], rr[rr.platform.isin(["twitter", "youtube"])].y)}
    pm = {}
    for k, p in P.items():
        e = yb - p; mae, l, h = va.mae_ci(e, gb)
        pm[k] = {"MAE": round(mae, 3), "factor": round(10 ** mae, 2), "CI": [round(l, 3), round(h, 3)], "mean bias (+ = under)": round(float(e.mean()), 3)}
        if k != "export X + YouTube, all pages (global)": pm[k]["paired diff vs global"] = paired(yb, P["export X + YouTube, all pages (global)"], p, gb)
    out["E150 point model on FB >1M"] = {"posts": int(big.sum()), "brands": int(len(set(gb))), "source >=1M posts export X + YT": int(big_src.sum()), **pm}
    (ROOT / "data/processed/fb_big_borrow_numbers.json").write_text(json.dumps({"E220": out}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
