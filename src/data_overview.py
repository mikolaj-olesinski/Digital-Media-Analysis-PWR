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


COL = {"tiktok": "#2a78d6", "youtube": "#eb6834", "twitter": "#1baf7a", "facebook": "#c98500", "instagram": "#d55181"}


def figure(d, rows, lang):
    T = {"en": ("Views grow with likes on every platform, from a different level on each",
                "likes", "views", "median", "views per like", "posts", "dashed: views = median views per like × likes"),
         "pl": ("Na każdej platformie wyświetlenia rosną razem z lajkami, ale od innego poziomu",
                "lajki", "wyświetlenia", "mediana", "wyśw. na lajk", "postów", "linia przerywana: wyświetlenia = mediana wyświetleń na lajk × lajki")}[lang]
    fig, ax = plt.subplots(1, len(P), figsize=(11, 3.3), sharex=True, sharey=True, gridspec_kw={"wspace": 0.08})
    fig.patch.set_facecolor(SURF)
    rng = np.random.default_rng(0)
    for a, p in zip(ax, P):
        s = d[d.platform == p]
        x = np.log10(np.expm1(s.log_likes.values) + 1) + rng.uniform(-.06, .06, len(s)); y = s.y.values
        a.set_facecolor(SURF)
        a.scatter(x, y, s=7, color=COL[p], alpha=.35 if len(s) > 1000 else .6, linewidth=0, rasterized=True)
        k = np.log10(rows[p]["median views per like"]); xx = np.array([0, 5])
        a.plot(xx, xx + k, color=INK, lw=1.2, ls="--")
        a.set_title(LAB[p], fontsize=11, color=COL[p], fontweight="bold", loc="left")
        a.text(0.03, 0.97, f"{len(s)} {T[5]}\n{T[3]}: {rows[p]['median views']:,.0f}\n{rows[p]['median views per like']:.0f} {T[4]}".replace(",", " "),
               transform=a.transAxes, va="top", fontsize=8, color=INK, linespacing=1.4)
        a.grid(color="#e6e5e0", lw=.7); a.set_axisbelow(True)
        [a.spines[q].set_visible(False) for q in ("top", "right")]; [a.spines[q].set_color("#bdbcb6") for q in ("left", "bottom")]
        a.tick_params(length=0, labelsize=8, colors=MUTED); a.set_xlabel(T[1], fontsize=9, color=MUTED)
    lab = lambda t: ["1", "10", "100", "1k", "10k", "100k", "1M", "10M"][t]
    ax[0].set_xticks(range(0, 6)); ax[0].set_xticklabels(["0"] + [lab(t) for t in range(1, 6)])
    ax[0].set_yticks(range(0, 8)); ax[0].set_yticklabels([lab(t) for t in range(0, 8)])
    ax[0].set_xlim(-.2, 5.2); ax[0].set_ylim(-.2, 7.6); ax[0].set_ylabel(T[2], fontsize=9, color=MUTED)
    fig.suptitle(T[0], fontsize=11.5, x=.125, ha="left", y=1.04, color=INK)
    fig.text(.125, -.06, T[6], fontsize=8, color=MUTED)
    fig.savefig(ROOT / "figures" / f"data_overview_{lang}.png", dpi=170, bbox_inches="tight", facecolor=SURF); plt.close(fig)
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
