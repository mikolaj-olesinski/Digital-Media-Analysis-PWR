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
SRC = {"en": {"tiktok": "Brand24 export", "youtube": "Brand24 export", "twitter": "Brand24 export", "facebook": "own read (videos)", "instagram": "own read (reels)"},
       "pl": {"tiktok": "eksport Brand24", "youtube": "eksport Brand24", "twitter": "eksport Brand24", "facebook": "własny odczyt (filmy)", "instagram": "własny odczyt (rolki)"}}
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
        l = np.expm1(s.log_likes.values); m = l >= 1; b, c = np.polyfit(np.log10(l[m]), s.y.values[m], 1)
        rows[p].update({"share with 0 likes": float((~m).mean()), "slope log views on log likes (likes >= 1)": float(b), "intercept": float(c),
                        "r (likes >= 1)": float(np.corrcoef(np.log10(l[m]), s.y.values[m])[0, 1])})
    return rows


COL = {"tiktok": "#2a78d6", "youtube": "#eb6834", "twitter": "#1baf7a", "facebook": "#c98500", "instagram": "#d55181"}


def figure(d, rows, lang):
    T = {"en": ("10× more likes means about 6–12× more views on every platform",
                "Each dot is one post. Line: fit on posts with at least one like.",
                "likes", "views", "more views", "posts", "for 10× more likes", "no likes: {z} of posts"),
         "pl": ("10× więcej lajków to ok. 6–12× więcej wyświetleń na każdej platformie",
                "Każda kropka to jeden post. Linia: dopasowanie na postach z co najmniej jednym lajkiem.",
                "lajki", "wyświetlenia", "więcej wyświetleń", "postów", "przy 10× więcej lajków", "bez lajków: {z} postów")}[lang]
    num = lambda v: f"{v:,.0f}".replace(",", "\u2009")
    dec = lambda v, k=2: f"{v:.{k}f}".replace(".", "," if lang == "pl" else ".")
    from matplotlib import font_manager as fm
    for f in Path("/usr/share/fonts/truetype/liberation").glob("LiberationSans-*.ttf"): fm.fontManager.addfont(str(f))
    plt.rcParams.update({"font.family": "Liberation Sans" if any(f.name == "Liberation Sans" for f in fm.fontManager.ttflist) else "DejaVu Sans"})
    box = dict(boxstyle="round,pad=0.25", fc=SURF, ec="none", alpha=.85)
    fig, ax = plt.subplots(1, len(P), figsize=(11, 3.6), sharex=True, sharey=True, gridspec_kw={"wspace": 0.07})
    fig.patch.set_facecolor(SURF)
    rng = np.random.default_rng(0)
    for a, p in zip(ax, P):
        s = d[d.platform == p]; r = rows[p]
        x = np.log10(np.expm1(s.log_likes.values) + 1) + rng.uniform(-.06, .06, len(s)); y = s.y.values
        a.set_facecolor(SURF)
        a.scatter(x, y, s=7, color=COL[p], alpha=.3 if len(s) > 1000 else .55, linewidth=0, rasterized=True)
        sl, ic = r["slope log views on log likes (likes >= 1)"], r["intercept"]
        xx = np.array([np.log10(2), np.log10(np.expm1(s.log_likes.values).max())])
        a.plot(xx, ic + sl * xx, color=SURF, lw=4, solid_capstyle="round"); a.plot(xx, ic + sl * xx, color=INK, lw=1.8, solid_capstyle="round")
        a.text(0, 1.13, LAB[p], transform=a.transAxes, fontsize=12, color=COL[p], fontweight="bold")
        a.text(0, 1.04, f"{num(len(s))} {T[5]}", transform=a.transAxes, fontsize=8.5, color=MUTED)
        a.text(.96, .27, "×" + dec(10 ** sl, 1), transform=a.transAxes, ha="right", fontsize=17, fontweight="bold", color=COL[p], bbox=box)
        a.text(.96, .19, T[4], transform=a.transAxes, ha="right", fontsize=8, color=INK, bbox=box)
        a.text(.96, .12, T[6], transform=a.transAxes, ha="right", fontsize=8, color=INK, bbox=box)
        a.text(.96, .04, T[7].format(z=f"{r['share with 0 likes']:.0%}"), transform=a.transAxes, ha="right", fontsize=8, color=MUTED, bbox=box)
        a.grid(color="#ebeae6", lw=.7); a.set_axisbelow(True)
        [a.spines[q].set_visible(False) for q in a.spines]
        a.tick_params(length=0, labelsize=8, colors=MUTED)
    lab = ["0", "10", "100", "1k", "10k", "100k", "1M", "10M"]
    ax[0].set_xticks(range(0, 6)); ax[0].set_xticklabels(lab[:6])
    ax[0].set_yticks(range(0, 8)); ax[0].set_yticklabels(["1"] + lab[1:])
    ax[0].set_xlim(-.3, 5.3); ax[0].set_ylim(-.3, 7.6)
    fig.supxlabel(T[2], fontsize=9.5, color=MUTED, y=.0); ax[0].set_ylabel(T[3], fontsize=9.5, color=MUTED)
    fig.text(.125, 1.13, T[0], fontsize=13.5, fontweight="bold", color=INK)
    fig.text(.125, 1.065, T[1], fontsize=9.5, color=MUTED)
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
