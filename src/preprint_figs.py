"""Preprint figures from stored predictions and numbers (no model fitting). Run results_a.py and results_b.py first.
  fig_main:       predicted vs true, TikTok / YouTube / X in-platform and Facebook / Instagram out-of-platform
  fig_size:       median error by true-views range (who is under- or over-estimated)
  fig_conditions: Facebook / Instagram error under different inputs (same-moment reactions, comments only, export reactions)
  fig_missing:    error by which fields arrive (E176)
  fig_side:       side uses: alert precision (E185), brand ranking (E186), EMV (E188)
Writes figures/preprint_<name>_{en,pl}.png.
    python src/preprint_figs.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_a as ra  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P5 = ["tiktok", "youtube", "twitter", "facebook", "instagram"]
LAB, COL = ra.LAB, ra.COL
INK, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#ebeae6"
L2 = np.log10(2)
for f in Path("/usr/share/fonts/truetype/liberation").glob("LiberationSans-*.ttf"): fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Liberation Sans" if any(f.name == "Liberation Sans" for f in fm.fontManager.ttflist) else "DejaVu Sans"})


def dec(v, lang, k=1): return f"{v:.{k}f}".replace(".", "," if lang == "pl" else ".")


def style(a):
    a.set_facecolor(SURF); a.grid(color=GRID, lw=.7); a.set_axisbelow(True)
    [a.spines[q].set_visible(False) for q in a.spines]; a.tick_params(length=0, labelsize=8.5, colors=MUTED)


def head(fig, title, sub, x=.08, y=1.1):
    fig.text(x, y, title, fontsize=13, fontweight="bold", color=INK); fig.text(x, y - .06, sub, fontsize=9, color=MUTED)


def save(fig, name, lang):
    fig.savefig(ROOT / "figures" / f"preprint_{name}_{lang}.png", dpi=170, bbox_inches="tight", facecolor=SURF); plt.close(fig)


def preds():
    d = pd.concat([pd.read_csv(ROOT / "data/processed/results_preds.csv"), pd.read_csv(ROOT / "data/processed/results_b_preds.csv")])
    return {p: d[d.platform == p] for p in P5}


def fig_main(D, lang):
    T = {"pl": ("Uczony na tej samej platformie (TikTok, YouTube, X) i na innych (Facebook, Instagram)",
                "Każda kropka to post z marki, której model nie widział. Pas: w granicach ×2. Liczba: typowy błąd.",
                "prawdziwe wyświetlenia", "oszacowane", "ta sama platforma", "inne platformy (X + YouTube)"),
         "en": ("Trained on the same platform (TikTok, YouTube, X) and on others (Facebook, Instagram)",
                "Each dot is a post from a brand the model did not see. Band: within ×2. Number: typical error.",
                "true views", "predicted", "same platform", "other platforms (X + YouTube)")}[lang]
    fig, ax = plt.subplots(1, 5, figsize=(12, 3.3), sharex=True, sharey=True, gridspec_kw={"wspace": .06}); fig.patch.set_facecolor(SURF)
    xx = np.array([-.5, 8]); box = dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=.85)
    for a, p in zip(ax, P5):
        s = D[p]; e = s.y - s.pred; style(a)
        a.fill_between(xx, xx - L2, xx + L2, color="#dddbd3", lw=0); a.plot(xx, xx, color=INK, lw=.9)
        a.scatter(s.y, s.pred, s=5 if len(s) > 1000 else 9, color=COL[p], alpha=.3 if len(s) > 1000 else .6, lw=0, rasterized=True)
        a.text(0, 1.04, LAB[p], transform=a.transAxes, fontsize=11.5, color=COL[p], fontweight="bold")
        a.text(.96, .06, "×" + dec(10 ** np.abs(e).mean(), lang, 2), transform=a.transAxes, ha="right", fontsize=15, fontweight="bold", color=COL[p], bbox=box)
    lab = ["1", "10", "100", "1k", "10k", "100k", "1M", "10M"]
    ax[0].set_xticks(range(0, 8, 2)); ax[0].set_xticklabels(lab[::2]); ax[0].set_yticks(range(0, 8, 2)); ax[0].set_yticklabels(lab[::2])
    ax[0].set_xlim(-.3, 7.5); ax[0].set_ylim(-.3, 7.5); ax[0].set_ylabel(T[3], fontsize=9, color=MUTED)
    fig.supxlabel(T[2], fontsize=9, color=MUTED, y=-.04)
    for a0, a1, t in ((ax[0], ax[2], T[4]), (ax[3], ax[4], T[5])):
        x0, x1 = a0.get_position().x0, a1.get_position().x1
        fig.add_artist(plt.Line2D([x0, x1], [.99, .99], color=MUTED, lw=1)); fig.text((x0 + x1) / 2, 1.0, t, ha="center", fontsize=9, color=MUTED)
    head(fig, T[0], T[1], x=.125, y=1.17); save(fig, "main", lang)


def fig_size(D, lang):
    T = {"pl": ("Model zawyża małe posty i zaniża duże, na każdej platformie",
                "Mediana błędu (prawda / oszacowanie) w przedziałach prawdziwych wyświetleń; powyżej 1 = model zaniża.",
                "prawdziwe wyświetlenia", "prawda / oszacowanie (mediana)"),
         "en": ("The model overestimates small posts and underestimates large ones, on every platform",
                "Median error (true / predicted) by true-views range; above 1 = the model underestimates.",
                "true views", "true / predicted (median)")}[lang]
    bins = [-1, 2, 3, 4, 5, 9]; names = ["<100", "100–1k", "1k–10k", "10k–100k", "100k+"]
    fig, a = plt.subplots(figsize=(7.4, 3.6)); fig.patch.set_facecolor(SURF); style(a)
    a.axhline(0, color=INK, lw=1); a.axhspan(-L2, L2, color="#e9e8e3", lw=0, zorder=0)
    for k, p in enumerate(P5):
        s = D[p]; b = pd.cut(s.y, bins, labels=False); m = (s.y - s.pred).groupby(b).median(); n = b.value_counts()
        m = m[n.reindex(m.index) >= 10]
        a.plot(m.index + (k - 2) * .05, m.values, "-o", color=COL[p], lw=2, ms=6, mec=SURF, mew=1.5, label=LAB[p])
    ticks = [-1, -L2, 0, L2, 1, np.log10(30)]
    a.set_yticks(ticks); a.set_yticklabels(["1/10", "1/2", "1", "2", "10", "30"]); a.set_xticks(range(5)); a.set_xticklabels(names)
    a.set_xlabel(T[2], fontsize=9, color=MUTED); a.set_ylabel(T[3], fontsize=9, color=MUTED)
    a.legend(frameon=False, fontsize=8.5, ncol=5, loc="upper left", bbox_to_anchor=(0, 1.02))
    head(fig, T[0], T[1], x=.02); save(fig, "size", lang)


def fig_conditions(lang):
    b = json.loads((ROOT / "data/processed/results_b_numbers.json").read_text())["results B"]
    st = json.loads((ROOT / "data/processed/fallback_estimator_numbers.json").read_text())["E177"]["Brand24 likes + comments"]["factor"]
    T = {"pl": ("Przeniesienie działa tylko z lajkami i komentarzami odczytanymi razem z wyświetleniami",
                "Typowy błąd na Facebooku i Instagramie przy różnych danych wejściowych.",
                ["model uczony na tej platformie", "z X + YouTube: lajki + komentarze", "z X + YouTube: tylko komentarze", "z X + YouTube: reakcje z eksportu\n(odczytane w innym momencie)"],
                "typowy błąd (×)", "brak danych"),
         "en": ("Transfer works only with likes and comments read together with the views",
                "Typical error on Facebook and Instagram with different inputs.",
                ["model trained on this platform", "from X + YouTube: likes + comments", "from X + YouTube: comments only", "from X + YouTube: export reactions\n(read at another time)"],
                "typical error (×)", "no data")}[lang]
    rows = {"facebook": [b["facebook"]["own platform ridge, brand CV"]["factor"], b["facebook"]["factor"], b["facebook"]["comments only (X + YouTube ridge)"]["factor"], st],
            "instagram": [b["instagram"]["own platform ridge, brand CV"]["factor"], b["instagram"]["factor"], b["instagram"]["comments only (X + YouTube ridge)"]["factor"], np.nan]}
    fig, a = plt.subplots(figsize=(7.4, 3.2)); fig.patch.set_facecolor(SURF); style(a); a.grid(axis="y", visible=False)
    for k, p in enumerate(["facebook", "instagram"]):
        for i, v in enumerate(rows[p]):
            yy = i + (k - .5) * .36
            if np.isnan(v):
                a.text(1.05, yy, T[4], va="center", fontsize=8, color=MUTED); continue
            a.barh(yy, v - 1, left=1, height=.32, color=COL[p], label=LAB[p] if i == 0 else None)
            a.text(v + .1, yy, "×" + dec(v, lang), va="center", fontsize=8.5, color=INK)
    a.set_yticks(range(4)); a.set_yticklabels(T[2], fontsize=8.5, color=INK); a.invert_yaxis(); a.set_xlim(1, 8.2)
    a.set_xlabel(T[3], fontsize=9, color=MUTED); a.legend(frameon=False, fontsize=8.5, loc="upper right")
    head(fig, T[0], T[1], x=.0); save(fig, "conditions", lang)


def fig_missing(lang):
    d = json.loads((ROOT / "data/processed/fallback_estimator_numbers.json").read_text())["E176"]
    K = ["full", "no followers", "no reactions", "nothing"]
    T = {"pl": ("Gdy brakuje reakcji, błąd rośnie najbardziej; brak obserwujących prawie nie szkodzi",
                "Typowy błąd na TikToku, YouTube i X, gdy dociera tylko część pól; przedział 90% trafia w każdym przypadku w 88–90% postów.",
                ["pełne dane", "bez obserwujących", "bez reakcji", "bez reakcji\ni obserwujących"], "typowy błąd (×)"),
         "en": ("Missing reactions hurt most; missing followers barely matter",
                "Typical error on TikTok, YouTube and X when only some fields arrive; the 90% band covers 88–90% of posts in every case.",
                ["all fields", "no followers", "no reactions", "no reactions\nor followers"], "typical error (×)")}[lang]
    fig, a = plt.subplots(figsize=(7.4, 3.2)); fig.patch.set_facecolor(SURF); style(a); a.grid(axis="x", visible=False)
    for k, p in enumerate(["tiktok", "youtube", "twitter"]):
        v = [10 ** d[p][q]["MAE"] for q in K]; x = np.arange(4) + (k - 1) * .26
        a.bar(x, np.array(v) - 1, bottom=1, width=.24, color=COL[p], label=LAB[p])
        for xi, vi in zip(x, v): a.text(xi, vi + .05, dec(vi, lang), ha="center", fontsize=7.5, color=INK)
    a.set_xticks(range(4)); a.set_xticklabels(T[2], fontsize=8.5, color=INK); a.set_ylim(1, 9); a.set_ylabel(T[3], fontsize=9, color=MUTED)
    a.legend(frameon=False, fontsize=8.5, loc="upper left")
    head(fig, T[0], T[1], x=.0); save(fig, "missing", lang)


def fig_side(lang):
    c = json.loads((ROOT / "data/processed/critical_review_numbers.json").read_text())["E185"]
    al = {p: (c[p]["CatBoost: first check + reactions + ages"]["top 5%, alerts = target size"]["precision"],
              c[p]["CatBoost without first-check views (reactions only)"]["top 5%, alerts = target size"]["precision"]) for p in ["tiktok", "youtube", "twitter"]}
    kendall = {"tiktok": .73, "youtube": .70, "twitter": .60}      # E186 card (no numbers file)
    emv = {"tiktok": .61, "youtube": .57, "twitter": .47}          # E188 card (no numbers file)
    T = {"pl": ("Zastosowania: alert o wybiciu, ranking marek i wartość medialna",
                "Wszystko na markach, których model nie widział.",
                ("Alert: trafność wśród 5% postów\nwskazanych jako wybijające się", "z pierwszym odczytem wyświetleń", "same reakcje"),
                ("Ranking marek w branży i miesiącu\n(Kendall: 1 = idealna kolejność)",),
                ("Wartość medialna marki (EMV):\nudział marek w granicach ±50%",)),
         "en": ("Uses: early viral alert, brand ranking and media value",
                "All on brands the model did not see.",
                ("Alert: precision among the 5% of posts\nflagged as breaking out", "with a first view read", "reactions only"),
                ("Brand ranking within industry and month\n(Kendall: 1 = perfect order)",),
                ("Brand media value (EMV):\nshare of brands within ±50%",))}[lang]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw={"wspace": .25}); fig.patch.set_facecolor(SURF)
    P3 = ["tiktok", "youtube", "twitter"]; x = np.arange(3)
    a = ax[0]; style(a); a.grid(axis="x", visible=False)
    a.bar(x - .18, [al[p][0] for p in P3], .34, color=[COL[p] for p in P3], label=T[2][1])
    a.bar(x + .18, [al[p][1] for p in P3], .34, color=[COL[p] for p in P3], alpha=.35, label=T[2][2])
    for i, p in enumerate(P3):
        a.text(i - .18, al[p][0] + .02, dec(al[p][0], lang, 2), ha="center", fontsize=8, color=INK); a.text(i + .18, al[p][1] + .02, dec(al[p][1], lang, 2), ha="center", fontsize=8, color=MUTED)
    a.set_title(T[2][0], fontsize=9.5, color=INK, loc="left"); a.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
    for a, vals, t in ((ax[1], kendall, T[3][0]), (ax[2], emv, T[4][0])):
        style(a); a.grid(axis="x", visible=False); a.bar(x, [vals[p] for p in P3], .5, color=[COL[p] for p in P3])
        for i, p in enumerate(P3): a.text(i, vals[p] + .02, dec(vals[p], lang, 2) if a is ax[1] else f"{vals[p]:.0%}", ha="center", fontsize=8.5, color=INK)
        a.set_title(t, fontsize=9.5, color=INK, loc="left")
    for a in ax: a.set_xticks(x); a.set_xticklabels([LAB[p] for p in P3]); a.set_ylim(0, 1.18); a.set_yticks([0, .5, 1]); a.set_yticklabels(["0", dec(.5, lang), "1"])
    head(fig, T[0], T[1], x=.125); save(fig, "side", lang)


def main():
    D = preds()
    for lang in ["pl", "en"]:
        fig_main(D, lang); fig_size(D, lang); fig_conditions(lang); fig_missing(lang); fig_side(lang)


if __name__ == "__main__":
    main()
