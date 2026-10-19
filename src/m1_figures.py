"""Wykresy i liczby do karty M1 (docs/M1_karta_eksploracji). Uruchom: python src/m1_figures.py"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")

d = pd.read_csv("reach_samples_students.csv")
d["eng"] = d[["likes", "comments", "shares"]].fillna(0).sum(axis=1)
names = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X", "facebook": "Facebook", "instagram": "Instagram"}
order = ["tiktok", "youtube", "twitter", "facebook", "instagram"]
C = {"gt": "#3f2a6b", "base": "#9a9aa8", "gbm": "#2a9d8f", "b24": "#e4572e"}

cov = {p: d[d.platform == p].view_count.notna().mean() * 100 for p in order}
res, out = {}, {}
for p in ["tiktok", "youtube", "twitter"]:
    g = d[(d.platform == p) & (d.view_count > 0) & (d.author_followers > 0)]
    y = np.log10(g.view_count.values)
    X = np.c_[np.log10(g.author_followers), np.log10(g.eng + 1), np.log10(g.likes.fillna(0) + 1), np.log10(g.comments.fillna(0) + 1)]
    base, mod = [], []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        c = np.median(y[tr] - X[tr, 0])
        base.append(np.abs(y[te] - (X[te, 0] + c)).mean())
        m = GradientBoostingRegressor(random_state=0).fit(X[tr], y[tr])
        mod.append(np.abs(y[te] - m.predict(X[te])).mean())
    res[p] = (np.mean(base), np.mean(mod))
    out[p] = dict(n=len(g), mae_const=round(np.mean(base), 2), mae_gbm=round(np.mean(mod), 2),
                  rho_foll=round(spearmanr(g.view_count, g.author_followers)[0], 2),
                  rho_eng=round(spearmanr(g.view_count, g.eng)[0], 2))
x = d[(d.platform == "twitter") & (d.view_count > 0) & (d.b24_estimated_reach > 0)]
b24_x = np.abs(np.log10(x.view_count) - np.log10(x.b24_estimated_reach)).mean()
out["twitter"]["mae_b24_rule"] = round(b24_x, 2); out["twitter"]["n_b24"] = len(x)
out["coverage"] = {k: round(v, 1) for k, v in cov.items()}
json.dump(out, open("data/processed/m1_numbers.json", "w"), indent=1)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
a = ax[0]
a.barh([names[p] for p in order][::-1], [cov[p] for p in order][::-1], color=[C["gt"] if cov[p] > 50 else "#d9d4e6" for p in order][::-1])
for i, p in enumerate(order[::-1]): a.text(cov[p] + 1.5, i, f"{cov[p]:.0f}%", va="center")
a.set_xlim(0, 118); a.set_xlabel("% wzmianek z znanymi wyświetleniami (view_count)")
a.set_title("A. Gdzie mamy dane odniesienia", loc="left", fontweight="bold")
a = ax[1]; w = 0.26; ps = ["tiktok", "youtube", "twitter"]; xs = np.arange(3)
a.bar(xs - w, [res[p][0] for p in ps], w, color=C["base"], label="stała × obserwujący (proxy reguły)")
a.bar(xs, [res[p][1] for p in ps], w, color=C["gbm"], label="model: obserwujący + reakcje")
a.bar(xs[2] + w, b24_x, w, color=C["b24"], label="faktyczna reguła Brand24 (tylko X)")
for xx, v in [(xs[0]-w,res['tiktok'][0]),(xs[1]-w,res['youtube'][0]),(xs[2]-w,res['twitter'][0]),(xs[0],res['tiktok'][1]),(xs[1],res['youtube'][1]),(xs[2],res['twitter'][1]),(xs[2]+w,b24_x)]:
    a.text(xx, v + .02, f"{v:.2f}", ha="center", fontsize=8)
a.set_xticks(xs); a.set_xticklabels([names[p] for p in ps]); a.set_ylabel("średni błąd bezwzględny, log10 (5-fold CV)")
a.set_ylim(0, 1.3); a.legend(frameon=False, fontsize=8, loc="upper left")
a.set_title("B. Błąd oszacowania wyświetleń", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/m1_dane_i_blad.png", dpi=200)
print(json.dumps(out, indent=1))
