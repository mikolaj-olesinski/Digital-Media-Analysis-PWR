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
BLUE, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"


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
    T = {"en": ("Views per post on each platform", "views (log scale)", "median", "posts"),
         "pl": ("Wyświetlenia na post na każdej platformie", "wyświetlenia (skala log)", "mediana", "postów")}[lang]
    fig, ax = plt.subplots(len(P), 1, figsize=(7, 6.2), sharex=True, gridspec_kw={"hspace": 0.15})
    bins = np.arange(0, 8.01, 0.25)
    for a, p in zip(ax, P):
        y = d[d.platform == p].y
        a.hist(y, bins=bins, color=BLUE, edgecolor="#fcfcfb", linewidth=1, density=True)
        m = np.median(y); a.axvline(m, color=INK, lw=1.2)
        a.text(m + 0.08, a.get_ylim()[1] * 0.82, f"{T[2]} {rows[p]['median views']:,.0f}".replace(",", " "), fontsize=8, color=INK)
        a.text(-0.02, 0.5, f"{LAB[p]}\n{len(y)} {T[3]}\n{SRC[lang][p]}", transform=a.transAxes, ha="right", va="center", fontsize=8.5, color=MUTED)
        a.set_facecolor("#fcfcfb"); a.set_yticks([]); [a.spines[s].set_visible(False) for s in ("left", "right", "top")]
        a.spines["bottom"].set_color("#bdbcb6"); a.tick_params(colors=MUTED, labelsize=8)
    ax[-1].set_xlabel(T[1], color=MUTED, fontsize=9); ax[-1].set_xticks(range(9))
    ax[-1].set_xticklabels([f"{10 ** k:,}".replace(",", " ") if k < 7 else ("10 mln" if lang == "pl" else "10M") if k == 7 else "" for k in range(9)])
    fig.suptitle(T[0], fontsize=11, x=0.42, y=0.95, color=INK)
    fig.savefig(ROOT / "figures" / f"data_overview_{lang}.png", dpi=160, bbox_inches="tight", facecolor="#fcfcfb"); plt.close(fig)


def main():
    d = tc.data(); raw = eu.load(ROOT)
    rows = table(d, raw)
    (ROOT / "data/processed/data_overview_numbers.json").write_text(json.dumps({"data overview": rows}, indent=1))
    for lang in ["en", "pl"]: figure(d, rows, lang)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
