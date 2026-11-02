"""Five explanatory figures for the project overview (PL): figures/story_{a..e}_pl.png. Run from src/: python story_figures.py"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import eda_utils as eu, modeling as mo, extra_models as em

ROOT = Path(__file__).resolve().parent.parent; FIG = ROOT / "figures"; PL = ["tiktok", "youtube", "twitter"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
f, meta = em.load(ROOT); raw = eu.load(ROOT).astype({"mention_id": str}); NM = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X", "facebook": "Facebook", "instagram": "Instagram"}

# A: views vs reactions
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
for a, p in zip(ax, PL):
    m = (meta.platform == p).values; x = np.log10(1 + np.expm1(f[m].log_likes.fillna(0)) + np.expm1(f[m].log_comments.fillna(0)))
    a.scatter(x, meta.y[m], s=5, alpha=.25, color=eu.PCOL[p]); r = np.corrcoef(x, meta.y[m])[0, 1]
    a.set_title(f"{NM[p]} (korelacja {r:.2f})".replace(".", ",")); a.set_xlabel("log10(1 + lajki + komentarze)"); a.set_ylabel("log10(wyświetlenia)")
plt.tight_layout(); fig.savefig(FIG / "story_a_pl.png", dpi=170, bbox_inches="tight"); plt.close()

# B: out-of-fold prediction vs truth
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4)); mk = em.models()["extra trees"]
for a, p in zip(ax, PL):
    m = (meta.platform == p).values; M = meta[m].reset_index(drop=True); X = f[m][em.ENG].reset_index(drop=True)
    pr = em.cv_predict(mk, X, M.y, M.brand); mae = np.abs(M.y - pr).mean(); lim = [M.y.min(), M.y.max()]
    a.scatter(pr, M.y, s=5, alpha=.25, color=eu.PCOL[p]); a.plot(lim, lim, color="black", lw=1); a.fill_between(lim, [l - .3 for l in lim], [l + .3 for l in lim], color="grey", alpha=.15)
    a.set_title(f"{NM[p]}: błąd typowo ×{10 ** mae:.1f}".replace(".", ",")); a.set_xlabel("przewidywane log10(wyświetlenia)"); a.set_ylabel("prawdziwe log10(wyświetlenia)")
plt.tight_layout(); fig.savefig(FIG / "story_b_pl.png", dpi=170, bbox_inches="tight"); plt.close()

# C: X, Brand24 reach vs our model
m = (meta.platform == "twitter").values; M = meta[m].reset_index(drop=True); X = f[m][em.ENG].reset_index(drop=True); pr = em.cv_predict(mk, X, M.y, M.brand)
b24 = np.log10(M[eu.REACH].where(M[eu.REACH] > 0)); ok = b24.notna()
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True); lim = [0, M.y.max()]
for a, v, t, c in [(ax[0], b24[ok], f"Zasięg Brand24 (błąd ×{10 ** np.abs(M.y[ok] - b24[ok]).mean():.0f})", "#9a9aa8"), (ax[1], pd.Series(pr)[ok], f"Nasz model z reakcji (błąd ×{10 ** np.abs(M.y[ok] - pr[ok]).mean():.1f})".replace(".", ","), eu.PCOL["twitter"])]:
    a.scatter(v, M.y[ok], s=6, alpha=.3, color=c); a.plot(lim, lim, color="black", lw=1); a.set_title(t); a.set_xlabel("oszacowanie log10"); a.set_ylabel("prawdziwe log10(wyświetlenia) na X")
plt.tight_layout(); fig.savefig(FIG / "story_c_pl.png", dpi=170, bbox_inches="tight"); plt.close()

# D: label shares per platform
lab = pd.read_csv(ROOT / "data" / "processed" / "mention_labels.csv", dtype={"mention_id": str}).merge(raw[["mention_id", "platform"]], on="mention_id")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for a, col, order, cols, ttl in [(ax[0], "type", ["organic", "promo", "spam", "unclear"], ["#4c9f70", "#e0a030", "#b5403a", "#b8b8c4"], "Typ wzmianki (nasza etykieta)"), (ax[1], "sentiment", ["positive", "neutral", "negative"], ["#4c9f70", "#b8b8c4", "#b5403a"], "Sentyment (nasza etykieta)")]:
    t = pd.crosstab(lab.platform, lab[col], normalize="index")[order].reindex(list(NM)) * 100; left = np.zeros(len(t))
    for o, c in zip(order, cols): a.barh([NM[p] for p in t.index][::-1], t[o][::-1], left=left[::-1], color=c, label=o); left += t[o].values
    a.set_xlabel("udział wzmianek (%)"); a.set_ylabel("platforma"); a.set_title(ttl); a.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=4, frameon=False)
plt.tight_layout(); fig.savefig(FIG / "story_d_pl.png", dpi=170, bbox_inches="tight"); plt.close()

# E: missing values heatmap
cols = ["view_count", "author_followers", "likes", "comments", "shares", "sentiment", "content"]; names = ["wyświetlenia", "followersi", "lajki", "komentarze", "udostępnienia", "sentyment Brand24", "treść"]
miss = pd.DataFrame({NM[p]: raw[raw.platform == p][cols].isna().mean() * 100 for p in NM}).T; miss.columns = names
fig, ax = plt.subplots(figsize=(9, 3.8)); im = ax.imshow(miss.values, cmap="Purples", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=25, ha="right"); ax.set_yticks(range(len(miss))); ax.set_yticklabels(miss.index); ax.set_xlabel("kolumna"); ax.set_ylabel("platforma")
for i in range(miss.shape[0]):
    for j in range(miss.shape[1]): ax.text(j, i, f"{miss.values[i, j]:.0f}", ha="center", va="center", color="white" if miss.values[i, j] > 55 else "black", fontsize=10)
ax.set_title("Braki danych (% pustych pól)"); fig.colorbar(im, ax=ax, label="% braków"); plt.tight_layout(); fig.savefig(FIG / "story_e_pl.png", dpi=170, bbox_inches="tight"); plt.close()
print("ok")
