"""Figure 2 for the M1 card (right-skewed distributions) + extra-feature test. Run: python src/m1_figures2.py [pl|en]"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
lang = sys.argv[1] if len(sys.argv) > 1 else "en"
T = {
    "pl": dict(xlab_a="log10(wyświetlenia lub reach Brand24)", ylab_a="gęstość",
               title_a="A. Rozkład w skali log: zakres 7 rzędów wielkości", xlab_b="% najpopularniejszych postów",
               ylab_b="% łącznych wyświetleń", title_b="B. Koncentracja: mało postów, dużo wyświetleń",
               fb="Facebook (reach Brand24)", ig="Instagram (reach Brand24)"),
    "en": dict(xlab_a="log10(views or Brand24 reach)", ylab_a="density",
               title_a="A. Log-scale distribution: 7 orders of magnitude", xlab_b="% of most popular posts",
               ylab_b="% of total views", title_b="B. Concentration: few posts, most views",
               fb="Facebook (Brand24 reach)", ig="Instagram (Brand24 reach)"),
}[lang]

d = pd.read_csv(ROOT / "data" / "raw" / "reach_samples_students.csv", parse_dates=["created_at", "metrics_updated_at"])
d["eng"] = d[["likes", "comments", "shares"]].fillna(0).sum(axis=1)
names = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X", "facebook": T["fb"], "instagram": T["ig"]}
col = {"tiktok": "#3f2a6b", "youtube": "#2a9d8f", "twitter": "#e4572e", "facebook": "#9a9aa8", "instagram": "#c9a227"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
bins = np.linspace(0, 8, 33); out = {}
for p in names:
    v = d[p == d.platform]
    s = (v.view_count if p in ("tiktok", "youtube", "twitter") else v.b24_estimated_reach).dropna()
    s = s[s > 0]
    ax[0].hist(np.log10(s), bins=bins, histtype="step" if p in ("facebook", "instagram") else "stepfilled",
               alpha=1 if p in ("facebook", "instagram") else .35, color=col[p], lw=1.6, ls="--" if p in ("facebook", "instagram") else "-", label=names[p], density=True)
    x = np.sort(s.values)[::-1]; c = np.cumsum(x) / x.sum(); k = len(x)
    out[p] = dict(median=float(np.median(s)), mean=float(s.mean()), top1_share=round(float(c[max(int(.01 * k) - 1, 0)]) * 100, 1), top10_share=round(float(c[int(.1 * k) - 1]) * 100, 1))
    sh = np.arange(1, k + 1) / k
    ax[1].plot(sh * 100, c * 100, color=col[p], lw=1.8, ls="--" if p in ("facebook", "instagram") else "-")
ax[0].set_xlabel(T["xlab_a"]); ax[0].set_ylabel(T["ylab_a"]); ax[0].legend(frameon=False, fontsize=7.5)
ax[0].set_title(T["title_a"], loc="left", fontweight="bold", fontsize=10)
ax[1].set_xlim(0, 100); ax[1].set_xlabel(T["xlab_b"]); ax[1].set_ylabel(T["ylab_b"])
ax[1].set_title(T["title_b"], loc="left", fontweight="bold", fontsize=10)
plt.tight_layout(); plt.savefig(ROOT / "figures" / f"m1_distributions_{lang}.png", dpi=200)

# extra features: do they improve the model (log10 views, 5-fold CV)
ext = {}
for p in ["tiktok", "youtube", "twitter"]:
    g = d[(d.platform == p) & (d.view_count > 0) & (d.author_followers > 0)].copy()
    g["age_d"] = np.log10((g.metrics_updated_at - g.created_at).dt.total_seconds().div(86400).clip(lower=.01))
    g["hour"] = g.created_at.dt.hour; g["dow"] = g.created_at.dt.dayofweek
    g["en"] = (g.language == "en").astype(int); g["len"] = np.log10(g.content.fillna("").str.len() + 1)
    g["sent"] = g.sentiment.map({"positive": 1, "neutral": 0, "negative": -1}).fillna(0)
    y = np.log10(g.view_count.values)
    base = [np.log10(g.author_followers), np.log10(g.eng + 1), np.log10(g.likes.fillna(0) + 1), np.log10(g.comments.fillna(0) + 1)]
    sets = {"base": base, "+ metrics age": base + [g.age_d], "+ hour, weekday": base + [g.hour, g.dow],
            "+ language": base + [g.en], "+ content length": base + [g.len], "+ sentiment": base + [g.sent],
            "all": base + [g.age_d, g.hour, g.dow, g.en, g.len, g.sent]}
    ext[p] = {}
    for k, cols in sets.items():
        X = np.c_[tuple(np.asarray(c, float) for c in cols)]; e = []
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            m = GradientBoostingRegressor(random_state=0).fit(X[tr], y[tr]); e.append(np.abs(y[te] - m.predict(X[te])).mean())
        ext[p][k] = round(float(np.mean(e)), 3)
out["extra_features_mae"] = ext
json.dump(out, open(ROOT / "data" / "processed" / "m1_numbers2.json", "w"), indent=1)
print(json.dumps(out, indent=1))
