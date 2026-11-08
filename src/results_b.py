"""Preprint section 5 (Results B): transfer to Facebook and Instagram, never seen in training.
Ridge on log likes + log comments trained on X + YouTube (pooled, as E93 / E201 / E202), scored on our same-moment
reads: Facebook 402 videos (fb_extend.fb_frame), Instagram reels with visible likes (ig_transfer.frame).
Reports MAE with brand-bootstrap CI, factor, share within x2, under / over share, a 25-post median shift (mean of 200
draws) and the same ridge trained on the target platform with brand-grouped 5-fold CV.
Writes data/processed/results_b_numbers.json and figures/results_b_{en,pl}.png.
    python src/results_b.py"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em, fb_extend as fe, ig_transfer as ig, results_a as ra  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
L2 = np.log10(2)


def targets():
    fb = fe.fb_frame(ROOT); fb = fb[fb.status == "ok"].dropna(subset=["y"]).reset_index(drop=True)
    r = ig.frame(); r = r[(r.status == "ok") & r.views.notna() & r.likes.notna()].copy()
    r["y"] = np.log10(r.views.clip(lower=1)); r["log_likes"] = np.log1p(r.likes); r["log_comments"] = np.log1p(r.comments.fillna(0))
    return {"facebook": fb[["y", "brand"] + fe.C2], "instagram": r[["y", "brand"] + fe.C2].reset_index(drop=True)}


def main():
    f, meta = em.load(ROOT); m = meta.platform.isin(["twitter", "youtube"]).values
    src = fe.ridge().fit(f[m][fe.C2].fillna(0), meta.y[m]); src_c = fe.ridge().fit(f[m][["log_comments"]].fillna(0), meta.y[m])
    out, res = {}, {}
    for p, d in targets().items():
        y, g = d.y.values, d.brand.values; pr = src.predict(d[fe.C2]); e = y - pr
        s = fe.score(pr, y, g)
        s.update({"within x2": round(float((np.abs(e) < L2).mean()), 3), "under-predicted": round(float((e > 0).mean()), 3),
                  "median bias (+ = under)": round(float(np.median(e)), 3), "brands": int(len(set(g)))})
        rng = np.random.default_rng(42); c = []
        for _ in range(200):
            i = rng.permutation(len(e)); c.append(np.abs(e[i[25:]] - np.median(e[i[:25]])).mean())
        s["25-post median shift"] = {"MAE": round(float(np.mean(c)), 3), "factor": round(float(10 ** np.mean(c)), 2)}
        own = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(d, y, g):
            own[te] = fe.ridge().fit(d.iloc[tr][fe.C2], y[tr]).predict(d.iloc[te][fe.C2])
        mo = float(np.abs(y - own).mean()); s["own platform ridge, brand CV"] = {"MAE": round(mo, 3), "factor": round(10 ** mo, 2)}
        s["comments only (X + YouTube ridge)"] = fe.score(src_c.predict(d[["log_comments"]]), y, g)
        out[p] = s; res[p] = ({"n": len(y), "factor": s["factor"], "within x2": s["within x2"]}, y, pr)
    import pandas as pd
    pd.concat([pd.DataFrame({"platform": p, "y": r[1], "pred": r[2]}) for p, r in res.items()]).to_csv(ROOT / "data/processed/results_b_preds.csv", index=False)
    (ROOT / "data/processed/results_b_numbers.json").write_text(json.dumps({"results B": out}, indent=1))
    T = {"en": ("Trained only on X and YouTube: typical error ×2.43 on Facebook, ×2.65 on Instagram",
                "Each dot is one post; the model never saw this platform. Band: prediction within ×2.",
                "true views (our read)", "predicted views", "typical error", "within ×2: {w} of posts", "posts"),
         "pl": ("Uczony tylko na X i YouTube: błąd ×2,43 na Facebooku, ×2,65 na Instagramie",
                "Każda kropka to jeden post; model nie widział tej platformy. Pas: oszacowanie w granicach ×2.",
                "prawdziwe wyświetlenia (nasz odczyt)", "oszacowane wyświetlenia", "typowy błąd", "w granicach ×2: {w} postów", "postów")}
    for lang in ["en", "pl"]: ra.figure(res, lang, plats=["facebook", "instagram"], name="results_b", T=T[lang])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
