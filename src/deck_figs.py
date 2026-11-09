"""Two simple figures for the Polish slide deck (no new experiment, numbers read from data/processed).
- deck_author_pl.png: final model vs + author history (shrunk mean of earlier posts), posts with an earlier post (E70-E75).
- deck_reread_pl.png: transfer error with export labels vs views and reactions re-read at the same moment (E210-E213).
    .venv/bin/python src/deck_figs.py"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker  # noqa: E401
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
BEFORE, AFTER, INK = "#c4c2bb", "#3d3c39", "#55544f"


def bars(groups, a, b, labels, title, out, note):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x = np.arange(len(groups)); w = .36
    for off, v, c, lab in ((-w / 2, a, BEFORE, labels[0]), (w / 2, b, AFTER, labels[1])):
        ax.bar(x + off, v, w, color=c, label=lab)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + .05, f"×{vi:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=10.5, color="#1d1d1b")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=10.5)
    ax.set_ylim(1, max(max(a), max(b)) * 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _: f"{t:.1f}".replace(".", ","))); ax.set_ylabel("typowy błąd (×)", fontsize=10, color=INK)
    ax.set_title(title, fontsize=11.5, loc="left", pad=26); ax.legend(frameon=False, fontsize=9.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK); ax.grid(axis="y", color="#e6e5e1", lw=.8); ax.set_axisbelow(True)
    fig.text(.01, -.1, note, fontsize=8.5, color="#8a8984")
    fig.savefig(ROOT / "figures" / out, dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    h = {(r["platform"], r["model"]): r for r in json.loads((P / "author_more_numbers.json").read_text())["history_models"]}
    pl = ["tiktok", "youtube", "twitter"]; nm = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X"}
    a = [10 ** h[(p, "final model")]["MAE"] for p in pl]; b = [10 ** h[(p, "+ sum / (n + k), shrunk by count")]["MAE"] for p in pl]
    bars([f"{nm[p]}\n({h[(p, 'final model')]['posts']} postów)" for p in pl], a, b, ["bez historii autora", "z historią autora (wcześniejsze wyświetlenia)"],
         "Znając wcześniejsze posty autora, błąd wyraźnie spada", "deck_author_pl.png", "Posty, których autor ma wcześniejszy post w danych; poprawka = średnia wcześniejszych odchyleń autora, ściągnięta do zera przy małej liczbie postów.")
    r = json.loads((P / "reread_transfer_numbers.json").read_text())
    v, g = r["variants"], r["grid (31 training sets)"]
    ex, rr = g["export TikTok/YouTube/X"]["combinations"], g["re-read TikTok/YouTube/X"]["combinations"]
    base, prim = v["export X + YT (base)"], v["re-read X + YT, drop missing (primary)"]
    a = [ex["tiktok + twitter"]["youtube"]["factor"], ex["tiktok + youtube"]["twitter"]["factor"], base["facebook"]["factor"], base["instagram"]["factor"]]
    b = [rr["tiktok + twitter"]["youtube"]["factor"], rr["tiktok + youtube"]["twitter"]["factor"], prim["facebook"]["factor"], prim["instagram"]["factor"]]
    bars(["YouTube\n(trening TT + X)", "X\n(trening TT + YT)", "Facebook\n(trening X + YT)", "Instagram\n(trening X + YT)"], a, b,
         ["etykiety z eksportu", "wyświetlenia i reakcje odczytane razem"], "Poza platformą: odczyt w tej samej chwili zmniejsza błąd",
         "deck_reread_pl.png", "Ten sam model (ridge na lajkach i komentarzach); zmienia się tylko moment odczytu (dla YouTube i X także etykiet testowych).")


if __name__ == "__main__":
    main()
