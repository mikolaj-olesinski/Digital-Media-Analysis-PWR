"""E225: does the X + YouTube ridge (log1p likes + log1p comments, as results_b) estimate Stack Exchange question views?
Stack Exchange is used only for evaluation. Mapping fixed in the card before scoring:
  primary  : likes = up_vote_count (fallback max(score, 0)), comments = answers + comments
  variant A: likes = max(score, 0),                          comments = answers + comments
  variant B: likes = up_vote_count (fallback max(score, 0)), comments = answers only
Reference: same ridge trained on Stack Exchange, brand-grouped 5-fold CV. Grid row: each single platform and the pooled
others (train_combos.data()) as training set. Views accumulate over months and votes are not likes: the level shift
(median bias) is reported, not tuned on.
Writes data/processed/stackexchange_numbers.json and figures/stackexchange_{en,pl}.png.
    python src/stackexchange_transfer.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em, fb_extend as fe, results_a as ra, train_combos as tc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
L2 = np.log10(2)


def frame(mapping="primary"):
    d = pd.read_csv(ROOT / "data/processed/stackexchange_posts.csv")
    d = d[d.views.notna()].copy()
    pos = d.score.clip(lower=0)
    likes = pos if mapping == "variant_score" else d.up_votes.fillna(pos)
    comm = d.answers if mapping == "variant_answers" else d.answers + d.comments
    d["log_likes"], d["log_comments"] = np.log1p(likes), np.log1p(comm.fillna(0))
    d["y"] = np.log10(d.views.clip(lower=1))
    return d.reset_index(drop=True)


def stats(model, d):
    y, g = d.y.values, d.brand.values; pr = model.predict(d[fe.C2]); e = y - pr
    s = fe.score(pr, y, g)
    s.update({"within x2": round(float((np.abs(e) < L2).mean()), 3), "under-predicted": round(float((e > 0).mean()), 3),
              "median bias (+ = under)": round(float(np.median(e)), 3), "brands": int(len(set(g)))})
    return s, y, pr


def main():
    f, meta = em.load(ROOT); m = meta.platform.isin(["twitter", "youtube"]).values
    src = fe.ridge().fit(f[m][fe.C2].fillna(0), meta.y[m])
    d = frame(); out = {"questions": int(len(d)), "brands": int(d.brand.nunique()), "sites": d.site.value_counts().to_dict(),
                        "median views": float(d.views.median()), "median log10 views": round(float(d.y.median()), 3)}
    s, y, pr = stats(src, d); out["primary: X + YouTube ridge"] = s
    for k in ("variant_score", "variant_answers"): out[k] = stats(src, frame(k))[0]
    out["by site (primary)"] = {k: stats(src, g)[0] for k, g in d.groupby("site") if len(g) >= 30}
    own = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(d, d.y, d.brand):
        own[te] = fe.ridge().fit(d.iloc[tr][fe.C2], d.y.iloc[tr]).predict(d.iloc[te][fe.C2])
    so = fe.score(own, d.y.values, d.brand.values); eo = d.y.values - own
    so.update({"within x2": round(float((np.abs(eo) < L2).mean()), 3), "under-predicted": round(float((eo > 0).mean()), 3), "median bias (+ = under)": round(float(np.median(eo)), 3)})
    out["own platform ridge, brand CV"] = so
    out["ratio factor transfer / own"] = round(s["factor"] / so["factor"], 3)
    out["hypothesis (ratio <= 1.5) holds"] = bool(s["factor"] / so["factor"] <= 1.5)
    # E201 grid row: Stack Exchange as target
    D = tc.data(); grid = {}
    for name, plats in [(p, [p]) for p in tc.P] + [("all five pooled", tc.P), ("X + YouTube", ["twitter", "youtube"])]:
        mod = tc.fit(D, plats); g, _, _ = stats(mod, d); grid[name] = {k: g[k] for k in ("MAE", "CI", "factor", "median bias (+ = under)", "within x2", "Spearman")}
    out["E201 grid row (target = Stack Exchange)"] = grid
    pd.DataFrame({"y": y, "pred": pr, "own": own, "site": d.site, "brand": d.brand}).to_csv(ROOT / "data/processed/stackexchange_preds.csv", index=False)
    (ROOT / "data/processed/stackexchange_numbers.json").write_text(json.dumps({"E225": out}, indent=1))
    ra.LAB.update({"se_t": "Stack Exchange", "se_o": "Stack Exchange, own ridge"}); ra.COL.update({"se_t": "#6a4fc2", "se_o": "#9b88d9"})
    res = {"se_t": ({"n": len(y), "factor": s["factor"], "within x2": s["within x2"]}, y, pr),
           "se_o": ({"n": len(y), "factor": so["factor"], "within x2": so["within x2"]}, d.y.values, own)}
    T = {"en": (f"Trained on X and YouTube: typical error ×{s['factor']:.2f} on Stack Exchange question views",
                "Each dot is one question; the model never saw this platform. Right: the same ridge trained on Stack Exchange (brand folds). Band: within ×2.",
                "true views (public count)", "predicted views", "typical error", "within ×2: {w} of questions", "questions"),
         "pl": (f"Uczony na X i YouTube: typowy błąd ×{str(s['factor']).replace('.', ',')[:4]} na wyświetleniach pytań Stack Exchange",
                "Każda kropka to jedno pytanie; model nie widział tej platformy. Po prawej ten sam ridge uczony na Stack Exchange (foldy po markach). Pas: w granicach ×2.",
                "prawdziwe wyświetlenia (licznik publiczny)", "oszacowane wyświetlenia", "typowy błąd", "w granicach ×2: {w} pytań", "pytań")}
    names = {"en": ("Stack Exchange, X + YouTube ridge", "Stack Exchange, own ridge"), "pl": ("Stack Exchange, ridge z X + YouTube", "Stack Exchange, własny ridge")}
    for lang in ["en", "pl"]:
        ra.LAB.update({"se_t": names[lang][0], "se_o": names[lang][1]}); ra.figure(res, lang, plats=["se_t", "se_o"], name="stackexchange", T=T[lang])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
