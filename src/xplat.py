"""Cross-platform transfer of engagement (notebooks/models/06_engagement_transfer.ipynb).
Link 1 (followers + text -> likes, comments, shares) can be validated on all five platforms, including Facebook and Instagram.
Link 2 (engagement -> views) can be validated only where views exist (TikTok, YouTube, X): see transfer.py."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
import eda_utils as eu, extra_features as ef

SEED = 42; ALL = ["tiktok", "youtube", "twitter", "facebook", "instagram"]
FEATS = ["log_followers", "na_followers", "chars", "words", "n_urls", "n_hashtags", "n_mentions", "n_emoji", "caps_share", "digit_share", "n_questions", "n_exclaim", "promo_words",
         "non_ascii_share", "vader_compound", "vader_neg", "vader_pos", "english", "hour", "weekday"]
TARGETS = {"likes": "likes", "comments": "comments", "shares": "shares"}


def load(root):
    """Features that exist on every platform, and the engagement targets (log1p) with NaN where the platform does not report them."""
    d = eu.load(root).reset_index(drop=True); t = ef.full_text(d)
    X = pd.concat([ef.text_stats(t), ef.vader(t)], axis=1)
    X["log_followers"] = np.log10(d.author_followers.where(d.author_followers > 0)); X["na_followers"] = d.author_followers.isna().astype(int)
    X["english"] = (d.language == "en").astype(int); X["hour"] = d.created_at.dt.hour; X["weekday"] = d.created_at.dt.weekday
    Y = pd.DataFrame({k: np.log1p(d[c]) for k, c in TARGETS.items()})
    return X[FEATS], Y, d[["platform", "brand"]]


def model():
    return ExtraTreesRegressor(200, min_samples_leaf=5, n_jobs=-1, random_state=SEED)


def usable(Y, meta, target, plat):
    """Rows of a platform where the target is reported, and not constant (YouTube shares are all zero, Instagram shares missing)."""
    m = ((meta.platform == plat) & Y[target].notna()).values
    return m if m.sum() > 200 and Y.loc[m, target].nunique() > 2 else None


def matrix(X, Y, meta, target):
    """Rows = platform the model is trained on, columns = platform it is tested on. Diagonal = brand-grouped 5-fold CV inside the platform.
    Returns MAE, Spearman, and skill = 1 - MAE / MAE(test-platform median)."""
    mae = pd.DataFrame(np.nan, index=ALL, columns=ALL); rho = mae.copy(); skill = mae.copy(); masks = {p: usable(Y, meta, target, p) for p in ALL}
    for te_p in ALL:
        if masks[te_p] is None: continue
        mt = masks[te_p]; Xt, yt = X[mt].fillna(-1), Y.loc[mt, target]; base = np.abs(yt - yt.median()).mean()
        for tr_p in ALL:
            if masks[tr_p] is None: continue
            if tr_p == te_p:
                pred = np.full(len(yt), np.nan); g = meta.brand[mt].values
                for tr, te in GroupKFold(5).split(Xt, yt, g): pred[te] = model().fit(Xt.iloc[tr], yt.iloc[tr]).predict(Xt.iloc[te])
            else:
                ms = masks[tr_p]; pred = model().fit(X[ms].fillna(-1), Y.loc[ms, target]).predict(Xt)
            mae.loc[tr_p, te_p] = np.abs(yt - pred).mean(); rho.loc[tr_p, te_p] = spearmanr(yt, pred)[0]; skill.loc[tr_p, te_p] = 1 - mae.loc[tr_p, te_p] / base
    return mae, rho, skill


def plot_matrix(rho, skill, target, path=None, lang="en"):
    NM = {p: eu.NAMES[p] for p in ALL}; fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for a, M, ttl, vmin, vmax in [(ax[0], rho, "Rank correlation (Spearman)" if lang == "en" else "Korelacja rang (Spearman)", 0, 1), (ax[1], skill, "Skill vs the platform median (1 = perfect, <0 = worse than median)" if lang == "en" else "Zysk względem mediany platformy (1 = idealnie, <0 = gorzej niż mediana)", -1, 1)]:
        im = a.imshow(M.values.astype(float), cmap="Greens" if vmin == 0 else "RdYlGn", vmin=vmin, vmax=vmax)
        a.set_xticks(range(5)); a.set_xticklabels([NM[p] for p in ALL], rotation=25, ha="right"); a.set_yticks(range(5)); a.set_yticklabels([NM[p] for p in ALL])
        a.set_xlabel("tested on" if lang == "en" else "testowane na"); a.set_ylabel("trained on" if lang == "en" else "uczone na"); a.set_title(ttl, fontsize=10, fontweight="bold")
        for i in range(5):
            for j in range(5):
                v = M.values[i, j]; a.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=9, fontweight="bold" if i == j else "normal")
        fig.colorbar(im, ax=a, fraction=.046)
    tt = {"likes": ("likes", "lajków"), "comments": ("comments", "komentarzy"), "shares": ("shares", "udostępnień")}[target]
    fig.suptitle(f"Predicting {tt[0]} from followers and text across platforms (diagonal = within platform)" if lang == "en" else f"Przewidywanie {tt[1]} z followersów i tekstu między platformami (przekątna = w obrębie platformy)", fontweight="bold")
    plt.tight_layout()
    if path: fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()


def level_shift(d, plats=("tiktok", "youtube", "twitter")):
    """Link 2 (engagement -> views), trained on two platforms and tested on the third: raw error, level bias, and the error left once the bias is removed (an oracle calibration that is impossible on Facebook and Instagram).
    Also the platform's own log-log line: views ~ intercept + slope * log10(1 + likes + comments)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    d = d[d.platform.isin(plats) & d.y.notna()]; F = ["log_likes", "log_comments"]; rows = []
    for p in plats:
        tr, te = d[d.platform != p], d[d.platform == p]; m = HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, random_state=SEED).fit(tr[F].fillna(0), tr.y); r = te.y - m.predict(te[F].fillna(0))
        s = np.log10(1 + np.expm1(te.log_likes.fillna(0)) + np.expm1(te.log_comments.fillna(0))).values.reshape(-1, 1); lin = LinearRegression().fit(s, te.y)
        rows.append({"held-out platform": eu.NAMES[p], "n": len(te), "transfer MAE": np.abs(r).mean(), "bias (true - predicted)": r.mean(), "MAE after removing bias": np.abs(r - r.mean()).mean(), "own slope": lin.coef_[0], "own intercept": lin.intercept_})
    return pd.DataFrame(rows)
