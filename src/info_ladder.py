"""E226: how much each input buys, step by step (preprint figure replacing the old "what the model needs").
A, in platform (TikTok / YouTube / X): CatBoost rt.FINAL, folds grouped by brand ∪ author ∪ text (E55 / E222);
   steps: training-fold median -> followers -> + likes -> + comments -> + shares (= final model, em.ENG).
B, out of platform (Facebook / Instagram): ridge trained on X + YouTube (fe.ridge), never seeing the target;
   steps: source median -> comments only -> likes only -> likes + comments (export source) -> likes + comments
   (same-moment re-read source, E210). Facebook also: likes + comments as stored in the export (E177, read at another time).
Writes data/processed/info_ladder_numbers.json; `--plot` renders figures/preprint_ladder_{en,pl}.png from the JSON.
    python src/info_ladder.py          (Kueue)
    python src/info_ladder.py --plot   (local, for the font)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/info_ladder_numbers.json"
STEPS_A = {"median": [], "followers": ["log_followers", "na_followers"], "+ likes": ["log_followers", "na_followers", "log_likes", "na_likes"],
           "+ comments": ["log_followers", "na_followers", "log_likes", "na_likes", "log_comments", "na_comments"], "+ shares (final)": None}
COL = {"tiktok": "#2a78d6", "youtube": "#eb6834", "twitter": "#1baf7a", "facebook": "#c98500", "instagram": "#d55181"}
LAB = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X", "facebook": "Facebook", "instagram": "Instagram"}
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"


def run():
    import eda_utils as eu, extra_models as em, fb_extend as fe, limits as lm, reaction_tiers as rt, results_b as rb, validation as va
    from reread_transfer import reread
    f, meta = em.load(ROOT); raw = eu.load(ROOT).loc[f.index]
    comp = lm.components(meta.reset_index(drop=True), raw.reset_index(drop=True))
    A = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values; X = f[m].reset_index(drop=True); y = meta.y[m].values; g = meta.brand[m].values; c = comp[m]
        A[p] = {}
        for k, cols in STEPS_A.items():
            cols = em.ENG if cols is None else cols; pr = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(X, y, c):
                pr[te] = np.median(y[tr]) if not cols else rt.FINAL().fit(X.iloc[tr][cols], y[tr]).predict(X.iloc[te][cols])
            s = fe.score(pr, y, g); A[p][k] = {"MAE": s["MAE"], "factor": s["factor"], "CI": s.get("CI")}
            print(p, k, s["factor"], flush=True)
    m = meta.platform.isin(["twitter", "youtube"]).values; S = f[m][fe.C2].fillna(0); ys = meta.y[m].values
    rr = reread(); rr = rr[rr.platform.isin(["twitter", "youtube"])]
    models = {"median": None, "comments only": fe.ridge().fit(S[["log_comments"]], ys), "likes only": fe.ridge().fit(S[["log_likes"]], ys),
              "likes + comments": fe.ridge().fit(S, ys), "likes + comments, fresh source": fe.ridge().fit(rr[fe.C2], rr.y)}
    cols = {"comments only": ["log_comments"], "likes only": ["log_likes"]}
    B = {}
    for p, d in rb.targets().items():
        B[p] = {}
        for k, mdl in models.items():
            pr = np.full(len(d), np.median(ys)) if mdl is None else mdl.predict(d[cols.get(k, fe.C2)])
            s = fe.score(pr, d.y.values, d.brand.values); B[p][k] = {"MAE": s["MAE"], "factor": s["factor"], "CI": s.get("CI")}
    fb = json.loads((ROOT / "data/processed/fallback_estimator_numbers.json").read_text())["E177"]["Brand24 likes + comments"]
    B["facebook"]["export reactions (another time)"] = {"MAE": fb.get("MAE"), "factor": fb["factor"]}
    OUT.write_text(json.dumps({"E226": {"in platform (CatBoost, author-safe folds)": A, "out of platform (ridge from X + YouTube)": B}}, indent=1))
    print(json.dumps(B, indent=1))


def dodge(vals, gap=.045):
    """Label y positions (log10) pushed apart by at least gap, keeping order."""
    o = np.argsort(vals); y = np.log10(np.array(vals, float)); out = y.copy()
    for i in range(1, len(o)):
        out[o[i]] = max(y[o[i]], out[o[i - 1]] + gap)
    return 10 ** out


def plot():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in Path("/usr/share/fonts/truetype/liberation").glob("LiberationSans-*.ttf"): fm.fontManager.addfont(str(f))
    plt.rcParams.update({"font.family": "Liberation Sans" if any(f.name == "Liberation Sans" for f in fm.fontManager.ttflist) else "DejaVu Sans"})
    d = json.loads(OUT.read_text())["E226"]; A, B = d["in platform (CatBoost, author-safe folds)"], d["out of platform (ridge from X + YouTube)"]
    KA = list(STEPS_A); KB = ["median", "comments only", "likes only", "likes + comments", "likes + comments, fresh source"]
    for lang in ["pl", "en"]:
        pl = lang == "pl"; dec = lambda v: ("×" + f"{v:.1f}").replace(".", "," if pl else ".")
        T = {"pl": ("Co daje każda informacja: lajki zmniejszają błąd najbardziej",
                    "A. Na tej samej platformie (CatBoost, nowe marki i autorzy)", ["mediana", "obserwujący", "+ lajki", "+ komen-\ntarze", "+ udostęp-\nnienia\n(końcowy)"],
                    "B. Poza platformą (ridge uczony tylko na X i YouTube)", ["mediana", "tylko\nkomentarze", "tylko lajki", "lajki +\nkomentarze", "+ świeże\nźródło"],
                    "typowy błąd (×), skala log", "reakcje z eksportu\n(inny moment)"),
             "en": ("What each input buys: likes cut the error most",
                    "A. Same platform (CatBoost, new brands and authors)", ["median", "followers", "+ likes", "+ comments", "+ shares\n(final model)"],
                    "B. Out of platform (ridge trained only on X and YouTube)", ["median", "comments\nonly", "likes only", "likes +\ncomments", "+ fresh\nsource"],
                    "typical error (×), log scale", "export reactions\n(another time)")}[lang]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True, gridspec_kw={"wspace": .08}); fig.patch.set_facecolor(SURF)
        for a, D, K, lab, ttl in [(ax[0], A, KA, T[2], T[1]), (ax[1], B, KB, T[4], T[3])]:
            a.set_facecolor(SURF); x = np.arange(len(K))
            V = {p: [v[k]["factor"] for k in K] for p, v in D.items()}
            ends, starts = dodge([V[p][-1] for p in V]), dodge([V[p][0] for p in V])
            for i, (p, vals) in enumerate(V.items()):
                a.plot(x, vals, "-o", color=COL[p], lw=2, ms=6, zorder=3, markeredgecolor=SURF, markeredgewidth=1.5)
                a.text(x[-1] + .12, ends[i], f"{LAB[p]} {dec(vals[-1])}", color=COL[p], fontsize=9, va="center", fontweight="bold")
                a.text(x[0] - .12, starts[i], dec(vals[0]), color=COL[p], fontsize=8, va="center", ha="right")
            a.set_xticks(x); a.set_xticklabels(lab, fontsize=8, color=INK); a.set_title(ttl, fontsize=10, loc="left", color=INK)
            a.set_xlim(-.6, len(K) - .2 + .9); a.grid(axis="y", color="#ebeae6", lw=.7); a.tick_params(length=0, labelsize=8, colors=MUTED)
            [a.spines[s].set_visible(False) for s in a.spines]
        st = B["facebook"].get("export reactions (another time)")
        if st:
            ax[1].plot([3], [st["factor"]], "D", color=COL["facebook"], mfc=SURF, ms=7, mew=2, zorder=4)
            ax[1].annotate(f"{T[6]} {dec(st['factor'])}", (3, st["factor"]), (3.25, st["factor"] * 1.5), fontsize=8, color=COL["facebook"], ha="left",
                           arrowprops=dict(arrowstyle="-", color=COL["facebook"], lw=.8))
        ax[0].set_yscale("log"); ax[0].minorticks_off(); tk = [1.5, 2, 3, 5, 10, 20]
        ax[0].set_yticks(tk); ax[0].set_yticklabels([dec(t).replace(",0", "").replace(".0", "") for t in tk]); ax[0].set_ylabel(T[5], fontsize=9, color=MUTED)
        fig.text(.125, 1.02, T[0], fontsize=13.5, fontweight="bold", color=INK)
        fig.savefig(ROOT / "figures" / f"preprint_ladder_{lang}.png", dpi=170, bbox_inches="tight", facecolor=SURF); plt.close(fig)


if __name__ == "__main__":
    plot() if "--plot" in sys.argv else run()
