"""One estimator for whatever fields arrive, with calibrated confidence (E176), and Facebook on Brand24's own fields (E177).
E176  Four input patterns, each with its own point model (CatBoost MAE), its own CQR bands (50% and 90%) and a
      calibrated probability that the truth is within x2 of the point:
        full          followers + reactions + missing flags (+ log age on X)
        no reactions  followers + text length, language, our labels, month, age
        no followers  reactions + flags (+ age on X)
        nothing       text length, language, our labels, month, age
      The probability within x2 comes from held-out calibration brands: share of |error| < log10(2) within quintiles of
      the raw quantile width. Checked on held-out brands: coverage, width, reliability of the probability (Brier, bins).
      Also: what happens without the router (the full model fed a post whose reactions are missing).
E177  Facebook, 402 fetched posts: the X + YouTube ridge fed Brand24's own likes / comments (what Brand24 has in
      production) instead of the fresh counts fetched with the views.
Writes data/processed/fallback_estimator_numbers.json. Statistics only.
    python src/fallback_estimator.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, fb_extend as fe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLATS = ["tiktok", "youtube", "twitter"]
SEED = 42; L2 = np.log10(2)
REACT = ["log_likes", "log_comments", "log_shares", "na_likes", "na_comments", "na_shares"]
CTX = ["english", "text_len", "type_code", "sentiment_code", "month", "metrics_age_days"]


def patterns(p):
    age = ["metrics_age_days"] if p == "twitter" else []
    return {"full": em.ENG + age, "no reactions": ["log_followers", "na_followers"] + CTX, "no followers": REACT + age, "nothing": CTX}


def cb():
    return CatBoostRegressor(loss_function="MAE", iterations=300, random_seed=SEED, verbose=0, allow_writing_files=False, thread_count=4)


def qm(a):
    return LGBMRegressor(objective="quantile", alpha=a, n_estimators=300, learning_rate=.05, num_leaves=31, subsample=.9, subsample_freq=1, colsample_bytree=.9, random_state=SEED, verbose=-1)


def fit_pattern(X, y, fit, cal):
    """Point, quantile models and the calibration pieces for one pattern."""
    pt = cb().fit(X.iloc[fit], y[fit]); Q = {a: qm(a).fit(X.iloc[fit], y[fit]) for a in (.05, .25, .75, .95)}
    c = {a: m.predict(X.iloc[cal]) for a, m in Q.items()}; yc = y[cal]
    q90 = float(np.quantile(np.maximum(c[.05] - yc, yc - c[.95]), .9)); q50 = float(np.quantile(np.maximum(c[.25] - yc, yc - c[.75]), .5))
    w = c[.95] - c[.05]; edges = np.quantile(w, [.2, .4, .6, .8]); hit = np.abs(yc - pt.predict(X.iloc[cal])) < L2
    p2 = np.array([hit[np.digitize(w, edges) == k].mean() for k in range(5)])
    return pt, Q, q90, q50, edges, p2


def apply(model, X):
    pt, Q, q90, q50, edges, p2 = model; o = {a: m.predict(X) for a, m in Q.items()}
    return {"point": pt.predict(X), "lo90": o[.05] - q90, "hi90": o[.95] + q90, "lo50": o[.25] - q50, "hi50": o[.75] + q50, "p2": p2[np.digitize(o[.95] - o[.05], edges)]}


def score(r, y):
    e = np.abs(y - r["point"]); hit = (e < L2).astype(float); bins = pd.qcut(r["p2"], 5, duplicates="drop")
    rel = pd.DataFrame({"pred": r["p2"], "obs": hit, "b": bins}).groupby("b", observed=True).agg(pred=("pred", "mean"), obs=("obs", "mean"), n=("obs", "size"))
    return {"MAE": float(e.mean()), "coverage 90": float(((y >= r["lo90"]) & (y <= r["hi90"])).mean()), "width 90 (log10)": float(np.median(r["hi90"] - r["lo90"])),
            "coverage 50": float(((y >= r["lo50"]) & (y <= r["hi50"])).mean()), "width 50 (log10)": float(np.median(r["hi50"] - r["lo50"])),
            "within x2 observed": float(hit.mean()), "within x2 predicted": float(r["p2"].mean()), "Brier": float(np.mean((r["p2"] - hit) ** 2)),
            "Brier constant": float(np.mean((hit.mean() - hit) ** 2)), "reliability": rel.round(3).to_dict("records")}


def e176(f, meta):
    out = {}; rng = np.random.default_rng(SEED)
    for p in PLATS:
        m = (meta.platform == p).values; F = f[m].reset_index(drop=True); y = meta.y[m].values; g = meta.brand[m].values
        res = {k: {kk: np.full(len(y), np.nan) for kk in ["point", "lo90", "hi90", "lo50", "hi50", "p2"]} for k in list(patterns(p)) + ["full model, reactions missing"]}
        for tr, te in GroupKFold(5).split(F, y, g):
            a, b = next(GroupShuffleSplit(1, test_size=.25, random_state=SEED).split(tr, groups=g[tr])); fit, cal = tr[a], tr[b]
            for name, cols in patterns(p).items():
                mod = fit_pattern(F[cols], y, fit, cal); r = apply(mod, F[cols].iloc[te])
                for kk, v in r.items(): res[name][kk][te] = v
                if name == "full":   # no router: the same model gets a post whose reactions are missing
                    Xm = F[cols].iloc[te].copy(); Xm[["log_likes", "log_comments", "log_shares"]] = np.nan; Xm[["na_likes", "na_comments", "na_shares"]] = 1
                    for kk, v in apply(mod, Xm).items(): res["full model, reactions missing"][kk][te] = v
        out[p] = {k: score(v, y) for k, v in res.items()}
        # a realistic mix routed by available fields: 60% full, 20% no reactions, 10% no followers, 10% nothing
        pick = rng.choice(list(patterns(p)), len(y), p=[.6, .2, .1, .1]); mix = {kk: np.choose(pd.Categorical(pick, categories=list(patterns(p))).codes, [res[k][kk] for k in patterns(p)]) for kk in res["full"]}
        out[p]["routed mix (60/20/10/10)"] = score(mix, y)
        print(p, {k: (round(v["MAE"], 3), round(v["coverage 90"], 3), round(v["within x2 observed"], 3), round(v["Brier"], 4), round(v["Brier constant"], 4)) for k, v in out[p].items()}, flush=True)
    return out


def e177(f, meta):
    d = fe.fb_frame(ROOT); d = d[d.status == "ok"].dropna(subset=["y"]).reset_index(drop=True)
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id").loc[d.mention_id]
    b24 = pd.DataFrame({"log_likes": np.log1p(pd.to_numeric(raw.likes, errors="coerce").fillna(0).values), "log_comments": np.log1p(pd.to_numeric(raw.comments, errors="coerce").fillna(0).values)})
    m = meta.platform.isin(["twitter", "youtube"]).values; mod = fe.ridge().fit(f[m][fe.C2].fillna(0), meta.y[m])
    reach = np.log10(pd.to_numeric(raw[eu.REACH], errors="coerce").clip(lower=1).values)
    age_b24 = (pd.to_datetime(raw.metrics_updated_at.values) - pd.to_datetime(raw.created_at.values)).days
    out = {"fresh likes + comments (fetched with views)": fe.score(mod.predict(d[fe.C2].fillna(0)), d.y.values, d.brand.values),
           "Brand24 likes + comments": fe.score(mod.predict(b24), d.y.values, d.brand.values),
           "Brand24 likes + comments, shift fitted on 25 posts (mean of 200 draws)": None,
           "Brand24 reach": fe.score(reach, d.y.values, d.brand.values),
           "Brand24 counts vs fresh": {"likes ratio median": float(np.median(np.expm1(d.log_likes) / np.maximum(np.expm1(b24.log_likes), 1))),
                                       "share of posts with zero Brand24 comments": float((b24.log_comments == 0).mean()),
                                       "Brand24 check age median days": float(np.median(age_b24))}}
    pb = mod.predict(b24); e = d.y.values - pb; rng = np.random.default_rng(SEED)
    out["Brand24 likes + comments, shift fitted on 25 posts (mean of 200 draws)"] = float(np.mean([np.abs(e[i[25:]] - np.median(e[i[:25]])).mean() for i in (rng.permutation(len(e)) for _ in range(200))]))
    print("FB", json.dumps(out, default=float)[:600], flush=True)
    return out


def main():
    f, meta = em.load(ROOT)
    out = {"E176": e176(f, meta), "E177": e177(f, meta)}
    (ROOT / "data" / "processed" / "fallback_estimator_numbers.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
