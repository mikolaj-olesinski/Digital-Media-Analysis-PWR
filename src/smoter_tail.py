"""E214: SMOTER / SMOGN oversampling of large posts (top decile of training y), against under-prediction of large posts.
(a) Transfer: ridge on log likes + log comments, trained on re-read X + YouTube (primary source) or export X + YouTube
(sensitivity), scored on FB 402 / IG reels (results_b.targets). (b) In-platform: rt.FINAL on ENG, brand-grouped 5-fold CV
on TikTok / YouTube / X, synthetic rows added to the training fold only. Base = same model without augmentation.
Writes data/processed/smoter_tail_numbers.json.
    python -W ignore src/smoter_tail.py"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em, fb_extend as fe, reaction_tiers as rt, results_b as rb, wpb_common as w  # noqa: E402

VARIANTS = {"SMOTER 1x": dict(ratio=1), "SMOTER 2x (primary)": dict(ratio=2), "SMOTER 5x": dict(ratio=5),
            "SMOGN 2x": dict(ratio=2, noise=0.05), "SMOTER 2x + undersample middle": dict(ratio=2, under=True)}
PRIMARY = "SMOTER 2x (primary)"


def augment(X, y, cont, ratio, rng, noise=0.0, under=False, k=5, q=0.9):
    """X: DataFrame (already NaN-filled), y: array. Returns augmented (X, y) with synthetic rare rows appended."""
    X = X.reset_index(drop=True); y = np.asarray(y, float)
    rare = np.where(y > np.quantile(y, q))[0]
    sd = X[cont].std().replace(0, 1).values; sy = y.std()
    Z = np.column_stack([X[cont].values[rare] / sd, y[rare] / sy])
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(rare))).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    n = int(round(ratio * len(rare)))
    a = rng.integers(0, len(rare), n); b = nn[a, rng.integers(0, nn.shape[1], n)]
    lam = rng.uniform(0, 1, n)[:, None]
    ia, ib = rare[a], rare[b]
    S = X.iloc[ia].reset_index(drop=True).copy()
    S[cont] = X[cont].values[ia] + lam * (X[cont].values[ib] - X[cont].values[ia])
    ys = y[ia] + lam[:, 0] * (y[ib] - y[ia])
    if noise:
        S[cont] = S[cont].values + rng.normal(0, 1, (n, len(cont))) * noise * sd
        ys = ys + rng.normal(0, noise * sy, n)
    keep = np.ones(len(y), bool)
    if under:
        mid = np.where((y >= np.quantile(y, .25)) & (y <= np.quantile(y, .75)))[0]
        keep[rng.choice(mid, len(mid) // 2, replace=False)] = False
    return pd.concat([X[keep], S], ignore_index=True), np.concatenate([y[keep], ys])


def transfer(src_name, src):
    targets = rb.targets(); out = {"source rows": int(len(src))}
    base = fe.ridge().fit(src[w.C2], src.y)
    preds = {"base": {p: base.predict(d[w.C2]) for p, d in targets.items()}}
    for v, kw in VARIANTS.items():
        rng = np.random.default_rng(w.SEED)
        Xa, ya = augment(src[w.C2], src.y.values, w.C2, rng=rng, **kw)
        m = fe.ridge().fit(Xa, ya); preds[v] = {p: m.predict(d[w.C2]) for p, d in targets.items()}
    for p, d in targets.items():
        out[p] = {}
        for v in preds:
            s = w.score(d.y, preds[v][p], d.brand)
            if v != "base":
                s["gain vs base"] = w.paired_gain(d.y, preds["base"][p], preds[v][p], d.brand)
            out[p][v] = s
    return out


def in_platform(f, meta):
    out = {}
    cont = ["log_followers", "log_likes", "log_comments", "log_shares"]
    for plat in w.PLAT3:
        m = (meta.platform == plat).values; M = meta[m].reset_index(drop=True); X = f[m][em.ENG].fillna(0).reset_index(drop=True)
        y, g = M.y.values, M.brand.values
        preds = {v: np.zeros(len(y)) for v in ["base"] + list(VARIANTS)}
        for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
            preds["base"][te] = rt.FINAL().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            for v, kw in VARIANTS.items():
                rng = np.random.default_rng(w.SEED + k)
                Xa, ya = augment(X.iloc[tr], y[tr], cont, rng=rng, **kw)
                preds[v][te] = rt.FINAL().fit(Xa, ya).predict(X.iloc[te])
        out[plat] = {}
        for v in preds:
            s = w.score(y, preds[v], g)
            if v != "base":
                s["gain vs base"] = w.paired_gain(y, preds["base"], preds[v], g)
            out[plat][v] = s
        print(plat, {v: out[plat][v]["MAE"] for v in out[plat]}, flush=True)
    return out


def verdict(res):
    """Primary rule: top-tenth bias by true y reduced by >= 0.1 and MAE worse by <= 0.01, on >= 3 of 5 targets."""
    rows = {}
    for p in ["facebook", "instagram"]:
        rows[p] = res["transfer, re-read X + YouTube source (primary)"][p]
    for p in w.PLAT3:
        rows[p] = res["in-platform rt.FINAL on ENG"][p]
    ok = {}
    for p, r in rows.items():
        b, n = r["base"], r[PRIMARY]
        dbias = abs(b["top tenth bias by true y"]) - abs(n["top tenth bias by true y"])
        ok[p] = {"bias reduction (|base| - |new|), top tenth by true y": round(dbias, 4), "MAE change": round(n["MAE"] - b["MAE"], 4),
                 "passes": bool(dbias >= 0.1 and n["MAE"] - b["MAE"] <= 0.01)}
    return {"per target": ok, "targets passing": int(sum(v["passes"] for v in ok.values())), "hypothesis supported (>= 3 of 5)": bool(sum(v["passes"] for v in ok.values()) >= 3)}


def main():
    res = {"note": "MAE of log10 views; gain = MAE(base) - MAE(variant), positive = better, brand bootstrap 300 draws; bias = mean(y - pred), + = under-predicted."}
    rr = w.reread(); src = rr[rr.platform.isin(["twitter", "youtube"])].reset_index(drop=True)
    res["transfer, re-read X + YouTube source (primary)"] = transfer("reread", src)
    f, meta = em.load(w.ROOT); m = meta.platform.isin(["twitter", "youtube"]).values
    ex = pd.DataFrame({"log_likes": f[m].log_likes.fillna(0).values, "log_comments": f[m].log_comments.fillna(0).values, "y": meta.y[m].values})
    res["transfer, export X + YouTube source (sensitivity)"] = transfer("export", ex)
    print("transfer done", flush=True)
    res["in-platform rt.FINAL on ENG"] = in_platform(f, meta)
    res["verdict (primary variant)"] = verdict(res)
    (w.ROOT / "data/processed/smoter_tail_numbers.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["verdict (primary variant)"], indent=1))


if __name__ == "__main__":
    main()
