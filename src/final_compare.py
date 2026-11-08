"""Preprint final figure: our estimate vs Brand24's reach column, on the platforms where reach is not the view count
(X, Facebook videos, Instagram reels). Ours: X = out-of-fold CatBoost from src/results_a.py (author-safe folds);
FB / IG = X + YouTube ridge on likes + comments (src/results_b.py). Same posts for both; truth = export views (X) or our
same-moment read (FB / IG). Reports MAE, factor 10^MAE, median bias, share within x2, share over-estimated.
Writes data/processed/final_compare_numbers.json and figures/final_compare_{en,pl}.png.
    python src/final_compare.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, fb_extend as fe, ig_transfer as ig, validation as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
L2 = np.log10(2)
P = ["twitter", "facebook", "instagram"]
LAB = {"twitter": "X", "facebook": "Facebook", "instagram": "Instagram"}
SUB = {"pl": {"twitter": "uczony na X (nowe marki)", "facebook": "bez danych z FB", "instagram": "bez danych z IG"},
       "en": {"twitter": "trained on X (new brands)", "facebook": "no FB data", "instagram": "no IG data"}}
INK, MUTED, SURF, OURS, B24 = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#a8a6a0"


def stats(y, p, g):
    e = y - p; mae, a, b = va.mae_ci(e, g)
    return {"posts": int(len(y)), "MAE": round(mae, 3), "CI": [round(a, 3), round(b, 3)], "factor": round(10 ** mae, 2),
            "median bias (+ = under)": round(float(np.median(e)), 3), "within x2": round(float((np.abs(e) < L2).mean()), 3),
            "over-estimated": round(float((e < 0).mean()), 3)}


def frames():
    raw = eu.load(ROOT).astype({"mention_id": str}); reach = raw.set_index("mention_id").b24_estimated_reach
    f, meta = em.load(ROOT); mid = raw.loc[meta.index].mention_id.values
    pr = pd.read_csv(ROOT / "data/processed/results_preds.csv"); x = pr[pr.platform == "twitter"].reset_index(drop=True)
    m = (meta.platform == "twitter").values
    x["mention_id"], x["brand"] = mid[m], meta.brand[m].values
    m2 = meta.platform.isin(["twitter", "youtube"]).values
    src = fe.ridge().fit(f[m2][fe.C2].fillna(0), meta.y[m2])
    fb = fe.fb_frame(ROOT); fb = fb[fb.status == "ok"].dropna(subset=["y"]).reset_index(drop=True); fb["pred"] = src.predict(fb[fe.C2])
    r = ig.frame(); r = r[(r.status == "ok") & r.views.notna() & r.likes.notna()].copy()
    r["y"] = np.log10(r.views.clip(lower=1)); r["log_likes"] = np.log1p(r.likes); r["log_comments"] = np.log1p(r.comments.fillna(0)); r["pred"] = src.predict(r[fe.C2])
    out = {}
    for p, d in {"twitter": x, "facebook": fb, "instagram": r}.items():
        d = d.copy(); d["mention_id"] = d.mention_id.astype(str); d["reach"] = reach.reindex(d.mention_id).values
        d = d[d.reach.notna()]; d["b24"] = np.log10(d.reach.clip(lower=1)); out[p] = d
    return out


def main():
    D = frames(); res = {}
    for p, d in D.items():
        res[p] = {"ours": stats(d.y.values, d.pred.values, d.brand.values), "Brand24 reach": stats(d.y.values, d.b24.values, d.brand.values)}
    (ROOT / "data/processed/final_compare_numbers.json").write_text(json.dumps({"final compare": res}, indent=1))
    for f in Path("/usr/share/fonts/truetype/liberation").glob("LiberationSans-*.ttf"): fm.fontManager.addfont(str(f))
    plt.rcParams.update({"font.family": "Liberation Sans" if any(f.name == "Liberation Sans" for f in fm.fontManager.ttflist) else "DejaVu Sans"})
    for lang in ["en", "pl"]:
        pl = lang == "pl"; dec = lambda v, k=1: f"{v:.{k}f}".replace(".", "," if pl else ".")
        T = ("Nasz model myli się kilka razy mniej niż zasięg w eksporcie" if pl else "Our model errs several times less than the export's reach",
             "Te same posty. X: model uczony na X, oceniany na nowych markach. Facebook i Instagram: model uczony tylko na X i YouTube." if pl else
             "Same posts. X: model trained on X, scored on new brands. Facebook and Instagram: model trained only on X and YouTube.",
             "nasz model" if pl else "our model", "zasięg Brand24" if pl else "Brand24 reach", "typowy błąd (×)" if pl else "typical error (×)",
             "w granicach ×2" if pl else "within ×2")
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"wspace": .35}); fig.patch.set_facecolor(SURF)
        yy = np.arange(len(P))[::-1]
        for a, key, fmt in [(ax[0], "factor", lambda v: "×" + dec(v)), (ax[1], "within x2", lambda v: f"{v:.0%}")]:
            a.set_facecolor(SURF)
            for k, (who, c) in enumerate([("ours", OURS), ("Brand24 reach", B24)]):
                v = [res[p][who][key] for p in P]; pos = yy + (.18 if k == 0 else -.18)
                a.barh(pos, v, height=.34, color=c, zorder=2)
                for yv, vv in zip(pos, v): a.text(vv, yv, " " + fmt(vv), va="center", fontsize=9, color=INK, fontweight="bold" if k == 0 else "normal")
            a.set_yticks(yy); a.set_yticklabels([] if a is not ax[0] else [LAB[p] for p in P], fontsize=10, color=INK, fontweight="bold")
            if a is ax[0]:
                for yv, p in zip(yy, P):
                    a.text(-.03, yv - .32, SUB[lang][p], transform=a.get_yaxis_transform(), ha="right", va="center", fontsize=7.5, color=MUTED)
            a.tick_params(length=0, labelsize=8, colors=MUTED); [a.spines[s].set_visible(False) for s in a.spines]
            a.grid(axis="x", color="#ebeae6", lw=.7, zorder=0); a.set_title(T[4] if key == "factor" else T[5], fontsize=10, color=MUTED, loc="left")
        ax[0].set_xscale("log"); ax[0].minorticks_off(); ax[0].set_xlim(1, 60); ax[0].set_xticks([1, 2, 5, 10, 20, 50]); ax[0].set_xticklabels(["1", "2", "5", "10", "20", "50"])
        ax[1].set_xlim(0, 1); ax[1].set_xticks([0, .25, .5, .75]); ax[1].set_xticklabels(["0%", "25%", "50%", "75%"])
        h = [plt.Rectangle((0, 0), 1, 1, color=OURS), plt.Rectangle((0, 0), 1, 1, color=B24)]
        fig.legend(h, [T[2], T[3]], loc="lower center", bbox_to_anchor=(.5, -.1), ncol=2, frameon=False, fontsize=9)
        fig.text(.125, 1.15, T[0], fontsize=13.5, fontweight="bold", color=INK); fig.text(.125, 1.07, T[1], fontsize=9.5, color=MUTED)
        fig.savefig(ROOT / "figures" / f"final_compare_{lang}.png", dpi=170, bbox_inches="tight", facecolor=SURF); plt.close(fig)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
