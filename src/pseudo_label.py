"""E217: self-training with pseudo-labels.
(a) Transductive transfer: ridge teacher on re-read X + YouTube (likes + comments) labels FB / IG posts (views hidden);
CatBoost student on source + pseudo-labelled target rows; noisy-student variant. Sensitivity: export teacher.
(b) In-platform X: export X posts with missing views are labelled by the fold teacher (rt.FINAL on ENG, brand CV) and added
to training; pseudo-labels checked against the re-read views.
Writes data/processed/pseudo_label_numbers.json.
    python -W ignore src/pseudo_label.py"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em, fb_extend as fe, reaction_tiers as rt, results_b as rb, wpb_common as w  # noqa: E402

SIG = 0.1


def transductive(src):
    targets = rb.targets(); teacher = fe.ridge().fit(src[w.C2], src.y)
    plain = rt.FINAL().fit(src[w.C2], src.y)
    out = {"source rows": int(len(src))}
    for p, d in targets.items():
        y, g = d.y.values, d.brand.values
        pt = teacher.predict(d[w.C2]); preds = {"teacher ridge (base)": pt, "CatBoost without pseudo-labels": plain.predict(d[w.C2])}
        X = np.vstack([src[w.C2].values, d[w.C2].values]); Y = np.concatenate([src.y.values, pt])
        preds["(a) student, hard pseudo-labels (primary)"] = rt.FINAL().fit(X, Y).predict(d[w.C2])
        rng = np.random.default_rng(w.SEED)
        preds["(a) noisy student, sigma 0.1"] = rt.FINAL().fit(X + rng.normal(0, SIG, X.shape), Y).predict(d[w.C2])
        preds["(a) student on pseudo-labelled target rows only"] = rt.FINAL().fit(d[w.C2].values, pt).predict(d[w.C2])
        o = {}
        for k, v in preds.items():
            s = w.score(y, v, g)
            if k != "teacher ridge (base)":
                s["gain vs teacher"] = w.paired_gain(y, pt, v, g)
            o[k] = s
        out[p] = o
    return out


def x_inplatform(f, meta):
    F, M = w.raw_eng()
    lab = (meta.platform == "twitter").values
    X = f[lab][em.ENG].fillna(0).reset_index(drop=True); ML = meta[lab].reset_index(drop=True); y, g = ML.y.values, ML.brand.values
    ul = ((M.platform == "twitter") & M.view_count.isna()).values
    XU = F[ul][em.ENG].fillna(0).reset_index(drop=True); MU = M[ul].reset_index(drop=True)
    cont = ["log_followers", "log_likes", "log_comments", "log_shares"]
    preds = {k: np.zeros(len(y)) for k in ["base (labelled only)", "(b) hard pseudo-labels (primary)", "(b) noisy pseudo rows, sigma 0.1", "(b) pseudo rows at weight 0.5"]}
    used = []
    for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
        t = rt.FINAL().fit(X.iloc[tr], y[tr]); preds["base (labelled only)"][te] = t.predict(X.iloc[te])
        keep = ~MU.brand.isin(set(g[te])).values; U = XU[keep]; yu = t.predict(U); used.append(int(keep.sum()))
        XA = pd.concat([X.iloc[tr], U], ignore_index=True); YA = np.concatenate([y[tr], yu])
        preds["(b) hard pseudo-labels (primary)"][te] = rt.FINAL().fit(XA, YA).predict(X.iloc[te])
        Un = U.copy(); rng = np.random.default_rng(w.SEED + k); Un[cont] = Un[cont].values + rng.normal(0, SIG, (len(Un), len(cont)))
        preds["(b) noisy pseudo rows, sigma 0.1"][te] = rt.FINAL().fit(pd.concat([X.iloc[tr], Un], ignore_index=True), YA).predict(X.iloc[te])
        wt = np.concatenate([np.ones(len(tr)), np.full(len(U), 0.5)])
        preds["(b) pseudo rows at weight 0.5"][te] = rt.FINAL().fit(XA, YA, sample_weight=wt).predict(X.iloc[te])
    out = {"labelled X posts": int(len(y)), "unlabelled X posts (missing export views)": int(ul.sum()), "unlabelled rows used per fold": used}
    base = preds["base (labelled only)"]
    for k, v in preds.items():
        s = w.score(y, v, g)
        if k != "base (labelled only)":
            s["gain vs base"] = w.paired_gain(y, base, v, g)
        out[k] = s
    # pseudo-label check against the re-read (X has no missing reactions in the re-read)
    rr = w.reread(plats=("twitter",)).set_index("mention_id")
    tfull = rt.FINAL().fit(X, y); pu = pd.Series(tfull.predict(XU), index=MU.mention_id.values)
    iu = [m for m in MU.mention_id if m in rr.index]
    eu_ = rr.loc[iu, "y"].values - pu.loc[iu].values
    mid_l = pd.Series(base, index=pd.Index(M.loc[meta.index[lab], "mention_id"].values))
    il = [m for m in mid_l.index if m in rr.index]
    el = rr.loc[il, "y"].values - mid_l.loc[il].values
    out["pseudo-label check against re-read views"] = {
        "unlabelled posts with a re-read": len(iu), "MAE vs re-read": round(float(np.abs(eu_).mean()), 4), "median bias vs re-read (+ = under)": round(float(np.median(eu_)), 4),
        "reference: labelled posts with a re-read (out-of-fold teacher)": len(il), "reference MAE vs re-read": round(float(np.abs(el).mean()), 4),
        "reference median bias vs re-read": round(float(np.median(el)), 4)}
    return out


def main():
    res = {"note": "MAE of log10 views; gain = MAE(base) - MAE(variant), + = better, brand bootstrap 300 draws."}
    rr = w.reread(); src = rr[rr.platform.isin(["twitter", "youtube"])].reset_index(drop=True)
    res["(a) transductive, re-read X + YouTube teacher (primary)"] = transductive(src)
    f, meta = em.load(w.ROOT); m = meta.platform.isin(["twitter", "youtube"]).values
    ex = pd.DataFrame({"log_likes": f[m].log_likes.fillna(0).values, "log_comments": f[m].log_comments.fillna(0).values, "y": meta.y[m].values})
    res["(a) transductive, export X + YouTube teacher (sensitivity)"] = transductive(ex)
    print("transductive done", flush=True)
    res["(b) in-platform X"] = x_inplatform(f, meta)
    gains = []
    for sec in res:
        if not isinstance(res[sec], dict):
            continue
        for p, o in res[sec].items():
            if isinstance(o, dict) and "MAE" not in o:
                for k, s in o.items():
                    gg = s.get("gain vs teacher") if isinstance(s, dict) else None
                    if gg: gains.append((f"{sec} | {p} | {k}", gg))
            elif isinstance(o, dict) and "gain vs base" in o:
                gains.append((f"{sec} | {p}", o["gain vs base"]))
    clear = [n for n, gg in gains if gg["gain"] >= 0.01 and gg["CI"][0] > 0 and "without pseudo" not in n]
    res["verdict"] = {"variants with gain >= 0.01 and CI above zero": clear, "refuted": not clear}
    (w.ROOT / "data/processed/pseudo_label_numbers.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
