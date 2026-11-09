"""Two simple figures for the Polish slide deck (no new experiment, numbers read from data/processed).
- deck_author_pl.png: final model vs + author history (shrunk mean of earlier posts), posts with an earlier post (E70-E75).
- deck_tails_pl.png: signed error of the final CatBoost (results_preds.csv) by quintile of true views, per platform.
- deck_calib_pl.png: coverage of the 90% interval, raw quantiles vs CQR (cqr_numbers.json, E118).
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
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(groups)); w = .36
    for off, v, c, lab in ((-w / 2, a, BEFORE, labels[0]), (w / 2, b, AFTER, labels[1])):
        ax.bar(x + off, v, w, color=c, label=lab)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + .05, f"×{vi:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=13.5, color="#1d1d1b")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=13.5)
    ax.set_ylim(1, max(max(a), max(b)) * 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _: f"{t:.1f}".replace(".", ","))); ax.set_ylabel("typowy błąd (×)", fontsize=13, color=INK)
    ax.legend(frameon=False, fontsize=12.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK); ax.grid(axis="y", color="#e6e5e1", lw=.8); ax.set_axisbelow(True)
    fig.savefig(ROOT / "figures" / out, dpi=170, bbox_inches="tight"); plt.close(fig)


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
    bars(["YouTube\n(TT + X)", "X\n(TT + YT)", "Facebook\n(X + YT)", "Instagram\n(X + YT)"], a, b,
         ["etykiety z eksportu", "wyświetlenia i reakcje odczytane razem"], "Poza platformą: odczyt w tej samej chwili zmniejsza błąd",
         "deck_reread_pl.png", "Ten sam model (ridge na lajkach i komentarzach); zmienia się tylko moment odczytu (dla YouTube i X także etykiet testowych).")


def tails():
    import pandas as pd
    d = pd.read_csv(P / "results_preds.csv")
    C = {"tiktok": ("TikTok", "#3b6ea8"), "youtube": ("YouTube", "#c4573a"), "twitter": ("X", "#3d3c39")}
    fig, ax = plt.subplots(figsize=(8, 4.6)); x = np.arange(5)
    ax.axhspan(-np.log10(2), np.log10(2), color="#efeee9", zorder=0); ax.axhline(0, color="#9c9a93", lw=1)
    for p, (nm, c) in C.items():
        g = d[d.platform == p]; q = pd.qcut(g.y, 5, labels=False); e = (g.y - g.pred).values
        m = [e[q.values == i].mean() for i in range(5)]
        ax.plot(x, m, "-o", color=c, lw=2.2, ms=7, label=nm, zorder=3)
        ax.text(4.12, m[-1], nm, color=c, fontsize=13.5, va="center")
    t = [-.6, -.3, 0, .3, .6]
    ax.set_yticks(t); ax.set_yticklabels(["×4 za dużo", "×2 za dużo", "trafnie", "×2 za mało", "×4 za mało"], fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(["20% najmniejszych", "", "środek", "", "20% największych"], fontsize=13)
    ax.set_xlabel("posty według prawdziwych wyświetleń", fontsize=13, color=INK); ax.set_xlim(-.3, 4.7); ax.set_ylim(-.75, .75)
    ax.text(-.25, .69, "model zaniża duże posty", fontsize=12.5, color=INK, va="top")
    ax.text(.35, -.72, "model zawyża małe posty", fontsize=12.5, color=INK, va="bottom")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.tick_params(colors=INK)
    fig.savefig(ROOT / "figures" / "deck_tails_pl.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def calib():
    d = json.loads((P / "cqr_numbers.json").read_text()); pl = ["tiktok", "youtube", "twitter"]
    nm = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X"}
    fig, ax = plt.subplots(figsize=(8, 4.4)); x = np.arange(3); w = .36
    for off, k, c, lab in ((-w / 2, "raw", BEFORE, "surowy przedział modelu"), (w / 2, "cqr", AFTER, "po kalibracji (CQR)")):
        v = [100 * d[p][k]["coverage"] for p in pl]; ax.bar(x + off, v, w, color=c, label=lab)
        for xi, vi in zip(x + off, v): ax.text(xi, (vi + 1) if vi < 85 else 92.2, f"{vi:.0f}%", ha="center", fontsize=13.5)
    ax.axhline(90, color="#c4573a", lw=1.5, ls="--"); ax.set_xlim(-.6, 2.95); ax.text(2.93, 90.8, "cel 90%", color="#c4573a", fontsize=12.5, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([f"{nm[p]}\n1/{10 ** (d[p]['cqr']['median_width'] / 2):.1f} … ×{10 ** (d[p]['cqr']['median_width'] / 2):.1f}".replace(".", ",") for p in pl], fontsize=13)
    ax.set_ylim(50, 97); ax.set_ylabel("ile prawdziwych wyników\nwpada w przedział", fontsize=12.5, color=INK)
    ax.legend(frameon=False, fontsize=12.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.tick_params(colors=INK); ax.grid(axis="y", color="#e6e5e1", lw=.8); ax.set_axisbelow(True)
    fig.savefig(ROOT / "figures" / "deck_calib_pl.png", dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    tails(); calib()
    main()
