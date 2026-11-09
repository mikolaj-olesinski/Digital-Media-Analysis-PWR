"""Simplified E201 figure for the preprint: train on one platform (or on all four others), test on each platform.
Reads data/processed/train_combos_numbers.json (run src/train_combos.py first). Diagonal = the same ridge trained on the
platform itself with brand-grouped CV. Writes figures/train_combos_simple_{en,pl}.png.
    python src/train_combos_simple.py"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
P = ["tiktok", "youtube", "twitter", "facebook", "instagram"]
LAB = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X", "facebook": "Facebook", "instagram": "Instagram"}
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"


def main():
    d = json.loads((ROOT / "data/processed/train_combos_numbers.json").read_text())["E201"]
    c, own = d["combinations"], d["own platform (brand CV)"]
    M = np.full((6, 5), np.nan)
    for i, a in enumerate(P):
        for j, b in enumerate(P):
            M[i, j] = own[b]["factor"] if a == b else c[a][b]["factor"]
    for j, b in enumerate(P):
        M[5, j] = c[" + ".join(q for q in P if q != b)][b]["factor"]
    for f in Path("/usr/share/fonts/truetype/liberation").glob("LiberationSans-*.ttf"): fm.fontManager.addfont(str(f))
    plt.rcParams.update({"font.family": "Liberation Sans" if any(f.name == "Liberation Sans" for f in fm.fontManager.ttflist) else "DejaVu Sans"})
    cmap = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
    for lang in ["en", "pl"]:
        T = {"en": ("Which platform to train on? Typical error (×) on each test platform",
                    "Ridge on likes + comments. Outlined: trained on the platform itself (unseen brands). Darker = worse than that.",
                    "trained on", "tested on", "all four others"),
             "pl": ("Na czym uczyć? Typowy błąd (×) na każdej platformie testowej",
                    "Ridge na lajkach i komentarzach. W ramce: trening na tej samej platformie (nowe marki). Ciemniej = gorzej niż w ramce.",
                    "trening na", "test na", "pozostałe cztery")}[lang]
        dec = lambda v: f"×{v:.1f}".replace(".", "," if lang == "pl" else ".")
        fig, ax = plt.subplots(figsize=(6.6, 4.6)); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
        R = np.log10(M) / np.log10(np.diag(M[:5]))[None, :]   # error relative to training on the platform itself
        ax.imshow(R, cmap=cmap, vmin=1, vmax=1.8, aspect="auto")
        for i in range(6):
            for j in range(5):
                ax.text(j, i, dec(M[i, j]), ha="center", va="center", fontsize=10.5, fontweight="bold" if i == j or i == 5 else "normal",
                        color="white" if R[i, j] > 1.4 else INK)
        for k in range(5):
            ax.add_patch(plt.Rectangle((k - .5, k - .5), 1, 1, fill=False, ec=INK, lw=1.6))
        ax.axhline(4.5, color=SURF, lw=5)
        ax.set_xticks(range(5)); ax.set_xticklabels([LAB[p] for p in P], fontsize=9.5, color=INK); ax.xaxis.tick_top()
        ax.set_yticks(range(6)); ax.set_yticklabels([LAB[p] for p in P] + [T[4]], fontsize=9.5, color=INK)
        ax.set_xlabel(T[3], fontsize=9, color=MUTED, labelpad=8); ax.xaxis.set_label_position("top"); ax.set_ylabel(T[2], fontsize=9, color=MUTED)
        ax.tick_params(length=0); [ax.spines[q].set_visible(False) for q in ax.spines]
        fig.text(.02, 1.06, T[0], fontsize=13, fontweight="bold", color=INK); fig.text(.02, 1.005, T[1], fontsize=8.8, color=MUTED)
        fig.savefig(ROOT / "figures" / f"train_combos_simple_{lang}.png", dpi=170, bbox_inches="tight", facecolor=SURF); plt.close(fig)
    print(np.round(M, 2))


if __name__ == "__main__":
    main()
