"""Preprint figure: the point estimate cut into three view ranges (<1k, 1k-10k, >10k), per platform.
Reads data/processed/results_preds.csv (TikTok / YouTube / X, author-safe CV, src/results_a.py) and
data/processed/results_b_preds.csv (Facebook / Instagram transfer, src/results_b.py). Same model as the headline result;
no refit. Reports accuracy, share within one range and the majority-range baseline.
Writes data/processed/preprint_bins_numbers.json and figures/preprint_bins_{en,pl}.png.
    python src/preprint_bins.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
P = ["tiktok", "youtube", "twitter", "facebook", "instagram"]
LAB = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X", "facebook": "Facebook", "instagram": "Instagram"}
COL = {"tiktok": "#2a78d6", "youtube": "#eb6834", "twitter": "#1baf7a", "facebook": "#c98500", "instagram": "#d55181"}
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
EDGES = [3, 4]  # log10: 1k, 10k


def main():
    d = pd.concat([pd.read_csv(ROOT / "data/processed/results_preds.csv"), pd.read_csv(ROOT / "data/processed/results_b_preds.csv")])
    out, M = {}, {}
    for p in P:
        g = d[d.platform == p]; t, q = np.digitize(g.y, EDGES), np.digitize(g.pred, EDGES)
        C = np.array([[((t == i) & (q == j)).sum() for j in range(3)] for i in range(3)])
        maj = np.bincount(t, minlength=3).max() / len(t)
        out[p] = {"n": int(len(t)), "accuracy": round(float((t == q).mean()), 3), "within one range": round(float((abs(t - q) <= 1).mean()), 3),
                  "two ranges off": round(float((abs(t - q) == 2).mean()), 3), "majority baseline": round(float(maj), 3),
                  "true share per range": [round(float(x), 3) for x in np.bincount(t, minlength=3) / len(t)], "confusion (rows true)": C.tolist()}
        M[p] = C
    (ROOT / "data/processed/preprint_bins_numbers.json").write_text(json.dumps({"preprint bins": out}, indent=1))
    for f in Path("/usr/share/fonts/truetype/liberation").glob("LiberationSans-*.ttf"): fm.fontManager.addfont(str(f))
    plt.rcParams.update({"font.family": "Liberation Sans" if any(f.name == "Liberation Sans" for f in fm.fontManager.ttflist) else "DejaVu Sans"})
    for lang in ["en", "pl"]:
        T = {"en": ("Cut into three ranges, the estimate lands in the right one for {lo}-{hi}% of posts",
                    "Rows: true range; columns: estimated range; share of each true row. Grey: always guessing the most common range.",
                    ["<1k", "1k-10k", ">10k"], "true", "estimated", "right range", "most common range"),
             "pl": ("Po podziale na trzy przedziały oszacowanie trafia w dobry przedział w {lo}–{hi}% postów",
                    "Wiersze: prawdziwy przedział, kolumny: oszacowany; udział w wierszu. Szare: zawsze najczęstszy przedział.",
                    ["<1 tys.", "1–10 tys.", ">10 tys."], "prawdziwe", "oszacowane", "trafiony przedział", "najczęstszy przedział")}[lang]
        acc = [out[p]["accuracy"] for p in P]
        fig, ax = plt.subplots(1, 5, figsize=(15, 3.6), gridspec_kw={"wspace": .12}); fig.patch.set_facecolor(SURF)
        for a, p in zip(ax, P):
            C = M[p]; R = C / C.sum(1, keepdims=True).clip(1)
            cmap = LinearSegmentedColormap.from_list("c", ["#f4f3ef", COL[p]])
            a.imshow(R, cmap=cmap, vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    a.text(j, i, f"{R[i, j]:.0%}" + f"\n({C[i, j]})", ha="center", va="center", fontsize=8.5, color="white" if R[i, j] > .55 else INK,
                           fontweight="bold" if i == j else "normal")
            a.set_xticks(range(3)); a.set_xticklabels(T[2], fontsize=8, color=MUTED); a.set_yticks(range(3))
            a.set_yticklabels(T[2] if p == "tiktok" else [], fontsize=8, color=MUTED); a.tick_params(length=0)
            [a.spines[s].set_visible(False) for s in a.spines]
            a.set_xlabel(T[4], fontsize=8, color=MUTED)
            if p == "tiktok": a.set_ylabel(T[3], fontsize=8, color=MUTED)
            s = out[p]; dec = (lambda v: f"{v:.0%}")
            a.text(0, 1.24, LAB[p], transform=a.transAxes, fontsize=12, color=COL[p], fontweight="bold")
            a.text(0, 1.12, f"{T[5]}: {dec(s['accuracy'])}", transform=a.transAxes, fontsize=9, color=INK)
            a.text(0, 1.03, f"{T[6]}: {dec(s['majority baseline'])}", transform=a.transAxes, fontsize=8.5, color="#8a8984")
        title = T[0].format(lo=round(min(acc) * 100), hi=round(max(acc) * 100))
        fig.text(.125, 1.13, title, fontsize=13.5, fontweight="bold", color=INK); fig.text(.125, 1.06, T[1], fontsize=9.5, color=MUTED)
        fig.savefig(ROOT / "figures" / f"preprint_bins_{lang}.png", dpi=170, bbox_inches="tight", facecolor=SURF); plt.close(fig)
    print(json.dumps({p: {k: out[p][k] for k in ["n", "accuracy", "within one range", "majority baseline", "true share per range"]} for p in P}, indent=0))


if __name__ == "__main__":
    main()
