"""E201: every combination of training platforms, tested on every platform left out.
Ground truth on all five platforms: Brand24 export views (TikTok, YouTube, X), Facebook re-measured (402 fetched, with
fresh reactions, taken from fb_extend.fb_frame for the brand column) and Instagram re-measured (hand batches, fresh likes and comments). Only the Brand24 frame is used
(own-search rows are a different population).
Model: ridge on log likes and log comments (the two fields every platform has), training rows pooled (as E93);
sensitivity: each training platform weighted equally. For each of the 31 non-empty training sets, every platform not in it is scored: MAE in log10, factor 10^MAE,
median bias (+ = under-predicts), share within x2. Reference: the same ridge in-platform with brand-grouped 5-fold CV.
Writes data/processed/train_combos_numbers.json and figures/train_combos_{en,pl}.png.
    python src/train_combos.py"""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, fb_extend as fe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ["tiktok", "youtube", "twitter", "facebook", "instagram"]
LAB = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X", "facebook": "Facebook", "instagram": "Instagram"}
L2 = np.log10(2)


def data():
    d = pd.read_csv(ROOT / "data/processed/all_views_dataset.csv", low_memory=False)
    d = d[d.frame.str.startswith("brand24")][["platform", "brand", "views", "likes", "comments"]]
    b = pd.read_csv(ROOT / "data/manual/ig_browser_batch.csv", dtype={"mention_id": str})
    ig = pd.concat([pd.read_csv(r) for r in sorted((ROOT / "data/manual").glob("ig_browser_results_*.csv"))]).merge(b, on="id")
    ig = ig[(ig.status == "ok") & ig.likes.notna() & ig.views.notna()]
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    ig = pd.DataFrame({"platform": "instagram", "brand": raw.loc[ig.mention_id].brand.values, "views": ig.views.values, "likes": ig.likes.values, "comments": ig.comments.values})
    fb = fe.fb_frame(ROOT); fb = fb[fb.status == "ok"].dropna(subset=["y"])   # brand and fresh reactions come from the fetch file
    fb = pd.DataFrame({"platform": "facebook", "brand": fb.brand.values, "views": 10 ** fb.y.values, "likes": np.expm1(fb.log_likes.values), "comments": np.expm1(fb.log_comments.values)})
    d = pd.concat([d[d.platform != "facebook"], fb, ig], ignore_index=True)
    d = d[pd.to_numeric(d.views, errors="coerce").notna()].copy()
    d["log_likes"] = np.log1p(pd.to_numeric(d.likes, errors="coerce").fillna(0)); d["log_comments"] = np.log1p(pd.to_numeric(d.comments, errors="coerce").fillna(0))
    d["y"] = np.log10(pd.to_numeric(d.views).clip(lower=1)); d["brand"] = d.brand.fillna("unknown")
    return d.reset_index(drop=True)


def stats(y, p):
    e = y - p
    return {"n": int(len(y)), "MAE": float(np.abs(e).mean()), "factor": float(10 ** np.abs(e).mean()), "bias": float(np.median(e)), "within x2": float((np.abs(e) < L2).mean())}


def fit(d, plats, equal=False):
    t = d[d.platform.isin(plats)]; w = (1 / t.platform.map(t.platform.value_counts())).values if equal else np.ones(len(t))
    return fe.ridge().fit(t[fe.C2], t.y, ridge__sample_weight=w)


def run(d):
    own = {}
    for p in P:
        s = d[d.platform == p].reset_index(drop=True); pr = np.zeros(len(s))
        for tr, te in GroupKFold(5).split(s, s.y, s.brand):
            pr[te] = fe.ridge().fit(s.iloc[tr][fe.C2], s.y.iloc[tr]).predict(s.iloc[te][fe.C2])
        own[p] = stats(s.y.values, pr)
    combos, eq = {}, {}
    for k in range(1, 5):
        for c in itertools.combinations(P, k):
            for store, flag in ((combos, False), (eq, True)):
                m = fit(d, c, flag); store[" + ".join(c)] = {q: stats(d[d.platform == q].y.values, m.predict(d[d.platform == q][fe.C2])) for q in P if q not in c}
    return {"combinations, platforms weighted equally": eq, "n per platform": d.platform.value_counts().to_dict(), "brands per platform": d.groupby("platform").brand.nunique().to_dict(), "own platform (brand CV)": own, "combinations": combos}


def figure(out, lang):
    T = {"en": ("Trained on", "Tested on", "Typical error (×) when the test platform was not in training", "own platform (brand CV)", "grey cell = platform was in training"),
         "pl": ("Trening na", "Test na", "Typowy błąd (×), gdy platformy testowej nie było w treningu", "własna platforma (CV po markach)", "szare pole = platforma była w treningu")}[lang]
    rows = sorted(out["combinations"], key=lambda r: (r.count("+"), np.mean([v["MAE"] for v in out["combinations"][r].values()])))
    M = np.full((len(rows) + 1, len(P)), np.nan)
    M[0] = [out["own platform (brand CV)"][q]["factor"] for q in P]
    for i, r in enumerate(rows, 1):
        for j, q in enumerate(P):
            if q in out["combinations"][r]: M[i, j] = out["combinations"][r][q]["factor"]
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"]); cmap.set_bad("#f0efec")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(8.6, 11), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.04}, sharey=True)
    for i, r in enumerate(rows, 1):
        inc = r.split(" + ")
        xs = [j for j, q in enumerate(P) if q in inc]
        a0.plot(xs, [i] * len(xs), color="#3a3a38", lw=2, zorder=1)
        for j, q in enumerate(P):
            a0.scatter(j, i, s=60, color="#3a3a38" if q in inc else "#dddcd8", zorder=2)
    a0.text(2, 0, T[3], ha="center", va="center", fontsize=8.5, color="#55544f")
    a0.set_xticks(range(5)); a0.set_xticklabels([LAB[p] for p in P], rotation=45, ha="right"); a0.set_title(T[0], fontsize=10)
    a0.set_xlim(-.6, 4.6); a0.tick_params(left=False); [a0.spines[s].set_visible(False) for s in a0.spines]
    im = a1.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=1.5, vmax=6, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                a1.text(j, i, f"×{M[i, j]:.1f}", ha="center", va="center", fontsize=7.5, color="white" if M[i, j] > 3.3 else "#1d1d1b")
    a1.axhline(0.5, color="white", lw=3)
    a1.set_xticks(range(5)); a1.set_xticklabels([LAB[p] for p in P], rotation=45, ha="right"); a1.set_title(T[1], fontsize=10)
    a1.set_yticks([]); [a1.spines[s].set_visible(False) for s in a1.spines]
    cb = fig.colorbar(im, ax=a1, fraction=.04, pad=.02); cb.set_label("×10^MAE"); cb.outline.set_visible(False)
    fig.suptitle(T[2], fontsize=11, y=.915)
    fig.text(.5, .03, T[4] + "  ·  ridge(log likes, log comments)", ha="center", fontsize=8, color="#8a8984")
    a0.set_ylim(len(rows) + .5, -.5)
    fig.savefig(ROOT / "figures" / f"train_combos_{lang}.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def main():
    out = run(data())
    (ROOT / "data/processed/train_combos_numbers.json").write_text(json.dumps({"E201": out}, indent=1))
    for lang in ["en", "pl"]: figure(out, lang)
    print(json.dumps(out["own platform (brand CV)"], indent=0)[:600])
    for r, v in out["combinations"].items(): print(r, {q: round(s["factor"], 2) for q, s in v.items()})


if __name__ == "__main__":
    main()
