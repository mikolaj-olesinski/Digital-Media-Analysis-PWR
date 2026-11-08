"""Preprint section 4 (Results A): the final in-platform model on TikTok, YouTube and X, re-run to get per-post
out-of-fold predictions for the figure. Same setup as E124 base (CatBoost MAE on em.ENG, 5-fold CV grouped by brand ∪ author ∪ identical text (E55 components,
so no author or copied text is in both training and test; E222);
CQR 90% band with LightGBM 5/95 quantiles calibrated on a held-out quarter of training brands).
Reports MAE with brand-bootstrap CI, factor 10^MAE, median bias, share within x2 / x3, under / over counts,
90% coverage and median band width (as a factor).
Writes data/processed/results_a_numbers.json and figures/results_a_{en,pl}.png.
    python src/results_a.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, limits as lm, reaction_tiers as rt, validation as va  # noqa: E402
from final_v2 import qm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ["tiktok", "youtube", "twitter"]
LAB = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X", "facebook": "Facebook", "instagram": "Instagram"}
COL = {"tiktok": "#2a78d6", "youtube": "#eb6834", "twitter": "#1baf7a", "facebook": "#c98500", "instagram": "#d55181"}
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
L2, L3 = np.log10(2), np.log10(3)


def run(f, meta, p, comp):
    m = (meta.platform == p).values
    X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
    c = pd.Series(comp[m])
    pr, lo, hi = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, c):
        inner = next(iter(GroupShuffleSplit(1, test_size=.25, random_state=42).split(X.iloc[tr], y.iloc[tr], c.iloc[tr])))
        fit, cal = np.array(tr)[inner[0]], np.array(tr)[inner[1]]
        pr[te] = rt.FINAL().fit(X.iloc[fit], y.iloc[fit]).predict(X.iloc[te])
        ql, qh = qm(.05).fit(X.iloc[fit], y.iloc[fit]), qm(.95).fit(X.iloc[fit], y.iloc[fit])
        E = np.maximum(ql.predict(X.iloc[cal]) - y.iloc[cal].values, y.iloc[cal].values - qh.predict(X.iloc[cal])); q = float(np.quantile(E, .9))
        lo[te], hi[te] = ql.predict(X.iloc[te]) - q, qh.predict(X.iloc[te]) + q
    y = y.values; e = y - pr; mae, a, b = va.mae_ci(e, g.values)
    s = {"n": int(len(y)), "brands": int(g.nunique()), "MAE": round(mae, 3), "CI": [round(a, 3), round(b, 3)], "factor": round(10 ** mae, 2),
         "median bias (+ = under)": round(float(np.median(e)), 3), "within x2": round(float((np.abs(e) < L2).mean()), 3),
         "within x3": round(float((np.abs(e) < L3).mean()), 3), "under-predicted": round(float((e > 0).mean()), 3),
         "over-predicted": round(float((e < 0).mean()), 3), "90% coverage": round(float(((y >= lo) & (y <= hi)).mean()), 3),
         "90% median width (log10)": round(float(np.median(hi - lo)), 2), "90% median width (factor, top/bottom)": round(float(10 ** np.median(hi - lo)), 1)}
    return s, y, pr


def figure(res, lang, plats=P, name="results_a", T=None):
    T = T or {"en": ("Typical error: ×1.9 on TikTok, ×2.9 on YouTube, ×3.0 on X",
                "Each dot is one post, predicted for a brand the model did not see. Band: prediction within ×2.",
                "true views", "predicted views", "typical error", "within ×2: {w} of posts", "posts"),
         "pl": ("Typowy błąd: ×1,9 na TikToku, ×2,9 na YouTube i ×3,0 na X",
                "Każda kropka to jeden post, oszacowany dla marki, której model nie widział. Pas: oszacowanie w granicach ×2.",
                "prawdziwe wyświetlenia", "oszacowane wyświetlenia", "typowy błąd", "w granicach ×2: {w} postów", "postów")}[lang]
    from matplotlib import font_manager as fm
    for f in Path("/usr/share/fonts/truetype/liberation").glob("LiberationSans-*.ttf"): fm.fontManager.addfont(str(f))
    plt.rcParams.update({"font.family": "Liberation Sans" if any(f.name == "Liberation Sans" for f in fm.fontManager.ttflist) else "DejaVu Sans"})
    dec = lambda v, k=1: f"{v:.{k}f}".replace(".", "," if lang == "pl" else ".")
    box = dict(boxstyle="round,pad=0.25", fc=SURF, ec="none", alpha=.85)
    fig, ax = plt.subplots(1, len(plats), figsize=(3.2 * len(plats) if len(plats) > 2 else 8, 3.7), sharex=True, sharey=True, gridspec_kw={"wspace": 0.07}); fig.patch.set_facecolor(SURF)
    xx = np.array([-.5, 8])
    for a, p in zip(ax, plats):
        s, y, pr = res[p]; a.set_facecolor(SURF)
        a.fill_between(xx, xx - L2, xx + L2, color="#dddbd3", lw=0, zorder=0)
        a.plot(xx, xx, color=INK, lw=1, zorder=1)
        a.scatter(y, pr, s=6 if len(y) > 1000 else 10, color=COL[p], alpha=.3 if len(y) > 1000 else .6, linewidth=0, rasterized=True, zorder=2)
        a.text(0, 1.13, LAB[p], transform=a.transAxes, fontsize=12, color=COL[p], fontweight="bold")
        a.text(0, 1.04, f"{s['n']:,} {T[6]}".replace(",", " "), transform=a.transAxes, fontsize=8.5, color=MUTED)
        a.text(.96, .19, "×" + dec(s["factor"], 2), transform=a.transAxes, ha="right", fontsize=17, fontweight="bold", color=COL[p], bbox=box)
        a.text(.96, .12, T[4], transform=a.transAxes, ha="right", fontsize=8, color=INK, bbox=box)
        a.text(.96, .04, T[5].format(w=f"{s['within x2']:.0%}"), transform=a.transAxes, ha="right", fontsize=8, color=MUTED, bbox=box)
        a.grid(color="#ebeae6", lw=.7); a.set_axisbelow(True); [a.spines[q].set_visible(False) for q in a.spines]
        a.tick_params(length=0, labelsize=8, colors=MUTED)
    lab = ["1", "10", "100", "1k", "10k", "100k", "1M", "10M"]
    ax[0].set_xticks(range(8)); ax[0].set_xticklabels(lab); ax[0].set_yticks(range(8)); ax[0].set_yticklabels(lab)
    ax[0].set_xlim(-.3, 7.5); ax[0].set_ylim(-.3, 7.5)
    fig.supxlabel(T[2], fontsize=9.5, color=MUTED, y=.0); ax[0].set_ylabel(T[3], fontsize=9.5, color=MUTED)
    fig.text(.125 if len(plats) > 2 else .13, 1.13, T[0], fontsize=13.5, fontweight="bold", color=INK); fig.text(.125, 1.065, T[1], fontsize=9.5, color=MUTED)
    fig.savefig(ROOT / "figures" / f"{name}_{lang}.png", dpi=170, bbox_inches="tight", facecolor=SURF); plt.close(fig)


def main():
    f, meta = em.load(ROOT); raw = eu.load(ROOT).loc[f.index]
    comp = lm.components(meta.reset_index(drop=True), raw.reset_index(drop=True))
    res = {p: run(f, meta, p, comp) for p in P}
    (ROOT / "data/processed/results_a_numbers.json").write_text(json.dumps({"results A": {p: r[0] for p, r in res.items()}}, indent=1))
    pd.concat([pd.DataFrame({"platform": p, "y": r[1], "pred": r[2]}) for p, r in res.items()]).to_csv(ROOT / "data/processed/results_preds.csv", index=False)
    for lang in ["en", "pl"]: figure(res, lang)
    print(json.dumps({p: r[0] for p, r in res.items()}, indent=1))


if __name__ == "__main__":
    main()
