"""Data overview for the preprint (section 2): Table 1 and Figure 1. Descriptive only, no model.
Ground truth per platform as in E201 (train_combos.data): Brand24 export views for TikTok, YouTube and X; our own
same-moment reads for Facebook (402 fetched) and Instagram (hand-collected reels with visible likes).
Writes data/processed/data_overview_numbers.json and figures/data_overview_{en,pl}.png.
    python src/data_overview.py"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, train_combos as tc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P, LAB = tc.P, tc.LAB
SRC = {"en": {"tiktok": "Brand24 export", "youtube": "Brand24 export", "twitter": "Brand24 export", "facebook": "own read", "instagram": "own read (reels)"},
       "pl": {"tiktok": "eksport Brand24", "youtube": "eksport Brand24", "twitter": "eksport Brand24", "facebook": "własny odczyt", "instagram": "własny odczyt (rolki)"}}
BLUE, LIGHT, INK, MUTED, SURF = "#2a78d6", "#b7d3f6", "#0b0b0b", "#52514e", "#fcfcfb"


def table(d, raw):
    rows = {}
    for p in P:
        s = d[d.platform == p]; v = 10 ** s.y
        rows[p] = {"mentions in export": int((raw.platform == p).sum()), "brands in export": int(raw[raw.platform == p].brand.nunique()),
                   "posts with views (used)": int(len(s)), "brands with views": int(s.brand.nunique()),
                   "median views": float(np.median(v)), "median likes": float(np.median(np.expm1(s.log_likes))),
                   "median comments": float(np.median(np.expm1(s.log_comments))),
                   "views p10-p90 (log10)": [round(float(np.percentile(s.y, 10)), 2), round(float(np.percentile(s.y, 90)), 2)],
                   "median views per like": float(np.median(v / np.expm1(s.log_likes).clip(lower=1)))}
    return rows


def figure(d, rows, lang):
    T = {"en": ("Views per post", "Views per like", "median", "posts"),
         "pl": ("Wyświetlenia na post", "Wyświetlenia na lajk", "mediana", "postów")}[lang]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, gridspec_kw={"wspace": 0.12})
    fig.patch.set_facecolor(SURF)
    for k, a in enumerate(ax):
        a.set_facecolor(SURF)
        for i, p in enumerate(P):
            s = d[d.platform == p]
            v = s.y.values if k == 0 else s.y.values - np.log10(np.maximum(np.expm1(s.log_likes.values), 1))
            q10, q25, q50, q75, q90 = np.percentile(v, [10, 25, 50, 75, 90])
            a.plot([q10, q90], [i, i], color=LIGHT, lw=9, solid_capstyle="round", zorder=1)
            a.plot([q25, q75], [i, i], color=BLUE, lw=9, solid_capstyle="round", zorder=2)
            a.scatter(q50, i, s=70, color=INK, edgecolor=SURF, linewidth=2, zorder=3)
            a.text(q50, i - 0.36, f"{10 ** q50:,.0f}".replace(",", " "), ha="center", va="bottom", fontsize=8, color=INK)
        ticks = range(0, 6) if k == 0 else range(0, 4)
        a.set_xticks(list(ticks)); a.set_xticklabels([f"{10 ** t:,}".replace(",", " ") for t in ticks], fontsize=8, color=MUTED)
        a.set_xlim(-0.2, 5.3 if k == 0 else 3.2)
        a.grid(axis="x", color="#e6e5e0", lw=0.8); a.set_axisbelow(True)
        a.set_title(T[k], fontsize=10, color=INK, loc="left")
        [a.spines[x].set_visible(False) for x in a.spines]; a.tick_params(length=0)
    ax[0].set_yticks(range(len(P)))
    ax[0].set_yticklabels([f"{LAB[p]}  ({rows[p]['posts with views (used)']})" for p in P], fontsize=9, color=INK)
    ax[0].set_ylim(len(P) - 0.5, -0.7)
    fig.savefig(ROOT / "figures" / f"data_overview_{lang}.png", dpi=160, bbox_inches="tight", facecolor=SURF); plt.close(fig)
    if lang == "pl":
        out = ROOT / "docs/pl/figures"; out.mkdir(exist_ok=True)
        (out / "data_overview_pl.png").write_bytes((ROOT / "figures/data_overview_pl.png").read_bytes())


def main():
    d = tc.data(); raw = eu.load(ROOT)
    rows = table(d, raw)
    (ROOT / "data/processed/data_overview_numbers.json").write_text(json.dumps({"data overview": rows}, indent=1))
    for lang in ["en", "pl"]: figure(d, rows, lang)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
