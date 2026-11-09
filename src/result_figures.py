"""Figures for the model results (no preprint text). Run: python src/result_figures.py [pl|en]

Fig 1: CQR coverage (50/90%) and width by platform.
Fig 2: FB coverage by follower tier and by view quartile.
Fig 3: concentration of views in the top 1/5/10% of posts.
Fig 4: calibration budget - MAE vs number of target posts.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
lang = sys.argv[1] if len(sys.argv) > 1 else "en"
P = ["tiktok", "youtube", "twitter"]
NAME = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X"}
COL = {"tiktok": "#2a9d8f", "youtube": "#e4572e", "twitter": "#3f2a6b"}
T = {
 "en": dict(f1_title="CQR interval coverage and width", f1_y="coverage", f1_x="platform", f1_leg="nominal 90% / 50%",
            f1_note="width in log10 above the 90% bar", f2_title="Coverage by follower tier",
            f2_x="follower tier", f2_y="coverage (90% band)", f2_q="view quartile", f3_title="Concentration of views: the top posts' share",
            f3_y="share of all views", f3_x="top share of posts", f4_title="Calibration budget: MAE vs target posts",
            f4_x="number of target posts", f4_y="MAE, log10", f4_leg="X target, shift calibrated"),
 "pl": dict(f1_title="Pokrycie i szerokość przedziałów CQR", f1_y="pokrycie", f1_x="platforma", f1_leg="nominał 90% / 50%",
            f1_note="szerokość w log10 nad słupkiem 90%", f2_title="Pokrycie wg tieru followersów",
            f2_x="tier followersów", f2_y="pokrycie (pasmo 90%)", f2_q="kwartyl wyświetleń", f3_title="Koncentracja wyświetleń: udział topowych postów",
            f3_y="udział wszystkich wyświetleń", f3_x="top % postów", f4_title="Budżet kalibracji: MAE vs liczba postów celu",
            f4_x="liczba postów celu", f4_y="MAE, log10", f4_leg="cel X, kalibracja przesunięciem")}[lang]

def save(fig, name):
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"{name}_{lang}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

# Fig 1
c90 = json.load(open(ROOT / "data" / "processed" / "cqr_numbers.json"))
c50 = json.load(open(ROOT / "data" / "processed" / "cqr_50_numbers.json"))
fig, ax = plt.subplots(figsize=(6.4, 3.8))
x = np.arange(len(P)); w = .35
v3 = json.load(open(ROOT / "data" / "processed" / "final_v3_x_numbers.json"))
cov90 = [(v3["coverage"]["90%"] if p == "twitter" else c90[p]["cqr"]["coverage"]) for p in P]
cov50 = [(v3["coverage"]["50%"] if p == "twitter" else c50[p]["50%"]["coverage"]) for p in P]
w90 = [(v3["width"]["90%"] if p == "twitter" else c90[p]["cqr"]["median_width"]) for p in P]
w50 = [(v3["width"]["50%"] if p == "twitter" else c50[p]["50%"]["median_width"]) for p in P]
ax.bar(x - w/2, cov90, w, color=[COL[p] for p in P], label="90%")
ax.bar(x + w/2, cov50, w, color=[COL[p] for p in P], alpha=.45, label="50%")
ax.axhline(.9, color="black", ls="--", lw=1); ax.axhline(.5, color="grey", ls=":", lw=1)
for i, p in enumerate(P):
    ax.text(i - w/2, cov90[i] + .01, f"w={w90[i]:.2f}", ha="center", fontsize=8)
    ax.text(i + w/2, cov50[i] + .01, f"w={w50[i]:.2f}", ha="center", fontsize=8)
ax.set_xticks(x, [NAME[p] for p in P]); ax.set_ylim(0, 1.05)
ax.set_xlabel(T["f1_x"]); ax.set_ylabel(T["f1_y"]); ax.set_title(T["f1_title"])
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
save(fig, "cqr_bands")

# Fig 2
fb = json.load(open(ROOT / "data" / "processed" / "fb_cqr_tiers_numbers.json"))
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6))
for ax, key, title in [(axs[0], "by_followers", T["f2_x"]), (axs[1], "by_views", T["f2_q"])]:
    ks = list(fb[key].keys()); vals = [fb[key][k]["coverage"] for k in ks]
    ax.plot(range(len(ks)), vals, "o-", color="#6a4c93")
    ax.axhline(.9, color="black", ls="--", lw=1)
    ax.set_xticks(range(len(ks)), ks, rotation=30, ha="right")
    ax.set_ylim(0.6, 1.02); ax.set_ylabel(T["f2_y"]); ax.set_xlabel(title)
axs[0].set_title(T["f2_title"])
axs[1].set_title("Coverage by view quartile" if lang == "en" else "Pokrycie wg kwartyla wyświetleń")
fig.suptitle(("Facebook: 90% CQR coverage (" + str(fb["posts"]) + " posts)") if lang == "en" else ("Facebook: pokrycie CQR 90% (" + str(fb["posts"]) + " postów)"), fontsize=11, y=1.02)
save(fig, "fb_coverage")

# Fig 3
rs = json.load(open(ROOT / "data" / "processed" / "reach_shape_numbers.json"))["platforms"]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
labels = ["top 1%", "top 5%", "top 10%"]; w = .25
for i, p in enumerate(P):
    vals = [rs[p]["share_top_1pct"], rs[p]["share_top_5pct"], rs[p]["share_top_10pct"]]
    ax.bar(np.arange(3) + (i - 1) * w, vals, w, color=COL[p], label=NAME[p])
ax.set_xticks(range(3), labels); ax.set_ylim(0, 1.05)
ax.set_xlabel(T["f3_x"]); ax.set_ylabel(T["f3_y"]); ax.set_title(T["f3_title"])
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
save(fig, "reach_concentration")

# Fig 4
cc = json.load(open(ROOT / "data" / "processed" / "calibration_curve_numbers.json"))
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for p in P:
    ks = sorted(cc[p], key=int); xs = [int(k) for k in ks]; ys = [cc[p][k]["MAE"] for k in ks]
    ax.plot(xs, ys, "o-", color=COL[p], label=NAME[p])
ax.set_xlabel(T["f4_x"]); ax.set_ylabel(T["f4_y"]); ax.set_title(T["f4_title"])
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
save(fig, "calibration_curve")
print("figures written")


# Fig 5: FB rule vs model by tier
fr = json.load(open(ROOT / "data" / "processed" / "fb_rule_vs_model_numbers.json"))
tiers = list(fr.keys())
fig, ax = plt.subplots(figsize=(6.6, 3.6))
x = np.arange(len(tiers)); w = .38
ax.bar(x - w/2, [fr[t]["rule_MAE"] for t in tiers], w, color="#9a9aa8", label="reguła Brand24" if lang == "pl" else "Brand24 rule")
ax.bar(x + w/2, [fr[t]["model_MAE"] for t in tiers], w, color="#6a4c93", label="model reakcji" if lang == "pl" else "reaction model")
for i, t in enumerate(tiers):
    ax.text(i - w/2, fr[t]["rule_MAE"] + .02, f'n={fr[t]["n"]}', ha="center", fontsize=7)
ax.set_xticks(x, tiers, rotation=20, ha="right"); ax.set_ylim(0, 1.2)
ax.set_xlabel("tier followersów" if lang == "pl" else "follower tier"); ax.set_ylabel(T["f4_y"]); ax.set_title("Facebook: reguła vs model, po tierach" if lang == "pl" else "Facebook: rule vs model, by follower tier")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
save(fig, "fb_rule_vs_model")
print("figures written")
