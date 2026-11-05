"""Facebook / Instagram before the hand check, and a residual atlas (notebooks/models/32_fbig_more.ipynb, E81-E83).
E81 how FB / IG posts look next to TikTok / YouTube / X (reaction structure, adversarial AUC: which source they resemble).
E82 pre-registered estimates for the hand-check sample (data/manual/fbig_sample.csv), written before any view is seen, and an
    evaluation that scores them once the views are filled in (MAE, bias, rank, calibration with n posts, Brand24 reach).
E83 residual atlas on TikTok / YouTube / X: which post traits get more (or fewer) views than reactions suggest.
Outputs carry mention ids and statistics only, never author names or URLs."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, CatBoostRegressor
import matplotlib.pyplot as plt
import eda_utils as eu, extra_models as em, reaction_tiers as rt, history as H
from validation import SEED

SRC = ["tiktok", "youtube", "twitter"]; TGT = ["facebook", "instagram"]
NAME = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X", "facebook": "Facebook", "instagram": "Instagram"}
B24_K = {"facebook": 100, "instagram": 20}


def feats(d, likes=None, comments=None, shares=None):
    """ENG columns built from raw rows with the transforms of modeling.load_features (natural log1p; log10 followers)."""
    L = d.likes if likes is None else likes; C = d.comments if comments is None else comments; S = d.shares if shares is None else shares
    f = pd.DataFrame(index=d.index)
    f["log_followers"] = np.log10(d.author_followers.where(d.author_followers > 0))
    for c, v in [("likes", L), ("comments", C), ("shares", S)]:
        v = pd.to_numeric(v, errors="coerce"); f[f"log_{c}"] = np.log1p(v); f[f"na_{c}"] = v.isna().astype(int)
    f["na_followers"] = d.author_followers.isna().astype(int)
    return f[em.ENG]


def video_kind(d):
    return d.url.fillna("").str.extract(r"/(reels?|videos?|shorts|p)/", expand=False).fillna("other")


# ---------- E81 ----------
def structure(raw_all):
    """Per platform (and FB / IG videos and Reels only): reaction structure known on every platform."""
    d = raw_all.assign(kind=video_kind(raw_all)); rows = []
    for (p, sub), q in [((p, "all posts"), q) for p, q in d.groupby("platform")] + [((p, "videos / Reels"), q[q.kind.isin(["reel", "reels", "video", "videos"])]) for p, q in d[d.platform.isin(TGT)].groupby("platform")]:
        L, C, S = q.likes.fillna(0), q.comments.fillna(0), q.shares
        rows.append({"platform": p, "posts": sub, "n": len(q), "median likes": float(L.median()), "share 0 reactions": float(((L == 0) & (C == 0)).mean()),
                     "median comments per like": float((C / L.where(L > 0)).median()), "median shares per like": float((S / L.where(L > 0)).median()),
                     "followers known": float(q.author_followers.notna().mean()), "median followers": float(q.author_followers.median()),
                     "median likes per 1k followers": float((1000 * L / q.author_followers.where(q.author_followers > 0)).median()),
                     "English": float((q.language == "en").mean()), "median text length": float((q.content.fillna(q.title).fillna("")).str.len().median())})
    return pd.DataFrame(rows)


def adversarial(raw_all, video_only=True):
    """AUC of a classifier telling a target platform from a source platform on columns every platform has (log likes,
    log comments, comments per like, text length). 0.5 = indistinguishable. With followers as well where FB has them."""
    d = raw_all.copy()
    if video_only: d = d[~d.platform.isin(TGT) | video_kind(d).isin(["reel", "reels", "video", "videos"])]
    X = pd.DataFrame({"ll": np.log1p(d.likes.fillna(0)), "lc": np.log1p(d.comments.fillna(0)), "cpl": np.log1p(d.comments.fillna(0)) - np.log1p(d.likes.fillna(0)),
                      "tl": np.log10(d.content.fillna(d.title).fillna("").str.len() + 1), "lf": np.log10(d.author_followers.where(d.author_followers > 0))}, index=d.index)
    rows = []
    for t in TGT:
        for s in SRC:
            for cols in [["ll", "lc", "cpl"], ["ll", "lc", "cpl", "tl"], ["ll", "lc", "cpl", "lf"]]:
                if t == "instagram" and "lf" in cols: continue
                m = d.platform.isin([t, s]).values; Z = X[m][cols]; y = (d.platform[m] == t).astype(int).values
                if "lf" in cols: k = Z.lf.notna().values; Z, y = Z[k], y[k]
                pr = np.zeros(len(y))
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Z, y):
                    pr[te] = CatBoostClassifier(iterations=200, depth=4, verbose=0, random_seed=SEED, allow_writing_files=False, thread_count=4).fit(Z.iloc[tr], y[tr]).predict_proba(Z.iloc[te])[:, 1]
                rows.append({"target": t, "source": s, "columns": "+".join(cols), "posts": len(y), "AUC": roc_auc_score(y, pr)})
    return pd.DataFrame(rows)


# ---------- E82 ----------
def models():
    """name -> (source platforms, columns, factory). All trained on posts with views of the source platforms only."""
    C2 = ["log_likes", "log_comments"]
    cb = lambda: CatBoostRegressor(loss_function="MAE", iterations=300, verbose=0, random_seed=SEED, allow_writing_files=False, thread_count=4)
    M = {f"ridge likes+comments, {NAME[s]}": ([s], C2, rt.LOPO) for s in SRC}
    M["ridge likes+comments, all three"] = (SRC, C2, rt.LOPO)
    M["CatBoost likes+comments, all three"] = (SRC, C2, cb)
    M["CatBoost reactions+followers, all three (FB only)"] = (SRC, em.ENG, cb)
    return M


def preregister(root, f, meta):
    """Estimates for every row of the hand-check sample from the reactions recorded by Brand24 (before the check)."""
    s = pd.read_csv(Path(root) / "data" / "manual" / "fbig_sample.csv", dtype={"mention_id": str})
    raw = eu.load(root).astype({"mention_id": str}).set_index("mention_id").loc[s.mention_id]
    X = feats(raw.reset_index()); out = s[["mention_id", "platform", "kind", "tier"]].copy()
    out["brand24 reach"] = np.log10(raw.b24_estimated_reach.clip(lower=1).values)
    for k, (src, cols, mk) in models().items():
        m = meta.platform.isin(src).values; mod = mk().fit(f[m][cols].fillna(0), meta.y[m]); v = mod.predict(X[cols].fillna(0))
        if "FB only" in k: v = np.where(out.platform == "facebook", v, np.nan)
        out[k] = v
    return out


def predict_now(root, f, meta, path):
    """Same models, applied to the reactions seen at the hand check (likes_now ...); reactions missing at the check fall back to Brand24's."""
    s = pd.read_csv(path, dtype={"mention_id": str}); raw = eu.load(root).astype({"mention_id": str}).set_index("mention_id").loc[s.mention_id].reset_index()
    nz = lambda a, b: pd.to_numeric(s[a], errors="coerce").fillna(raw[b])
    X = feats(raw, nz("likes_now", "likes"), nz("comments_now", "comments"), pd.to_numeric(s.shares_now, errors="coerce").where(s.platform == "facebook").fillna(raw.shares))
    out = s[["mention_id", "platform"]].copy(); out["views"] = pd.to_numeric(s.views, errors="coerce")
    out["checked_at"] = pd.to_datetime(s.checked_at, errors="coerce"); out["age_days"] = (out.checked_at - pd.to_datetime(raw.created_at)).dt.days
    out["brand24 reach"] = np.log10(raw.b24_estimated_reach.clip(lower=1).values)
    for k, (src, cols, mk) in models().items():
        m = meta.platform.isin(src).values; v = mk().fit(f[m][cols].fillna(0), meta.y[m]).predict(X[cols].fillna(0))
        out[k] = np.where(out.platform == "facebook", v, np.nan) if "FB only" in k else v
    return out


def evaluate(P, ns=(5, 10, 25, 50), n_draw=300):
    """MAE, bias (true - estimate), Spearman per platform and model on rows with views > 0; and the MAE after a shift fitted
    on n random posts (median error), scored on the other posts."""
    rng = np.random.default_rng(SEED); M = [c for c in P.columns if c not in ("mention_id", "platform", "views", "checked_at", "age_days")]
    P = P[P.views > 0]; rows, cal = [], []
    for p, q in P.groupby("platform"):
        y = np.log10(q.views.values)
        for k in M:
            v = q[k].values; ok = ~np.isnan(v)
            if ok.sum() < 5: continue
            e = y[ok] - v[ok]
            rows.append({"platform": p, "model": k, "posts": int(ok.sum()), "MAE": float(np.abs(e).mean()), "bias (true - estimate)": float(e.mean()), "typical factor": float(10 ** np.abs(e).mean()),
                         "Spearman": float(spearmanr(y[ok], v[ok])[0])})
            for n in ns:
                if ok.sum() < n + 5: continue
                r = []
                for _ in range(n_draw):
                    i = rng.permutation(ok.sum()); a, b = i[:n], i[n:]; r.append(np.abs(e[b] - np.median(e[a])).mean())
                cal.append({"platform": p, "model": k, "calibration posts": n, "MAE after shift": float(np.mean(r)), "p10": float(np.percentile(r, 10)), "p90": float(np.percentile(r, 90))})
    return pd.DataFrame(rows), pd.DataFrame(cal)


def fake_fill(root, P0, path, seed=SEED):
    """Dry run only: a copy of the sample with invented views (ridge estimate + shift + noise), to test the pipeline. Never a result."""
    s = pd.read_csv(Path(root) / "data" / "manual" / "fbig_sample.csv", dtype={"mention_id": str}).drop(columns="url")
    rng = np.random.default_rng(seed); v = P0["ridge likes+comments, all three"].values + rng.normal(.3, .5, len(s))
    s["views"] = np.round(10 ** v).astype(int).astype(object); s["checked_at"] = "2026-11-06 10:00"; s["views_visible"] = "yes"
    s.loc[rng.random(len(s)) < .1, "views"] = ""; s.to_csv(path, index=False); return path


def implied_vpl(root, P0):
    """Views per like implied by the pre-registered estimates on the sample (likes >= 1), against Brand24's constants (100 FB, 20 IG)."""
    raw = eu.load(root).astype({"mention_id": str}).set_index("mention_id").loc[P0.mention_id]; L = raw.likes.values; rows = []
    for p, q in P0.assign(likes=L).groupby("platform"):
        q = q[q.likes >= 1]
        for k in [c for c in q.columns if c not in ("mention_id", "platform", "kind", "tier", "likes")]:
            v = q[k].dropna()
            if len(v): rows.append({"platform": p, "estimate": k, "posts": len(v), "median views per like": float(np.median(10 ** v / q.likes[v.index]))})
        rows.append({"platform": p, "estimate": "Brand24 constant", "posts": len(q), "median views per like": float(B24_K[p])})
    return pd.DataFrame(rows)


# ---------- E83 ----------
def traits(raw, labels):
    c = raw.content.fillna(raw.title).fillna("").astype(str); n = c.str.len(); t = raw.created_at
    lang = raw.language.where(raw.language.isin(raw.language.value_counts().index[:5]), "other").fillna("other")
    ctry = raw.country.where(raw.country.isin(raw.country.value_counts().index[:6]), "other").fillna("other")
    return pd.DataFrame({
        "language": lang, "country": ctry, "weekday (UTC)": np.where(t.dt.dayofweek >= 5, "weekend", "weekday"),
        "hour (UTC)": pd.cut(t.dt.hour, [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]).astype(str),
        "Brand24 sentiment": raw.sentiment.fillna("none").astype(str), "our type label": labels.reindex(raw.index).fillna("none").values,
        "question mark": np.where(c.str.contains(r"\?"), "yes", "no"), "emoji": np.where(c.str.contains(r"[\U0001F300-\U0001FAFF☀-➿]"), "yes", "no"),
        "hashtags": pd.cut(c.str.count(r"#\w+"), [-1, 0, 3, 1e9], labels=["0", "1-3", "4+"]).astype(str),
        "text length": pd.cut(n, [-1, 80, 300, 1e9], labels=["< 80", "80-300", "> 300"]).astype(str),
        "link": np.where(c.str.contains("http"), "yes", "no"), "reply (@ first)": np.where(c.str.match(r"^\s*@"), "yes", "no"),
        "all caps words": np.where(c.str.count(r"\b[A-Z]{4,}\b") >= 2, "2+", "0-1")}, index=raw.index)


def atlas(d, T, n_boot=400, min_n=30):
    """Mean over-performance (true - final model, log10) per trait group and platform, minus the platform mean, with a
    brand-bootstrap 95% CI; and the raw difference in mean log views."""
    rng = np.random.default_rng(SEED); rows = []
    for p in SRC:
        z = d[d.platform == p]; t = T.loc[z.index]; r0 = z.r - z.r.mean(); y0 = z.y - z.y.mean()
        br = z.brand.values; ub = np.unique(br); ix = {b: np.where(br == b)[0] for b in ub}
        draws = [np.concatenate([ix[b] for b in rng.choice(ub, len(ub))]) for _ in range(n_boot)]
        for col in T.columns:
            for g in t[col].unique():
                s = (t[col] == g).values
                if s.sum() < min_n or s.sum() > len(z) - min_n: continue
                B = [r0.values[i][s[i]].mean() - r0.values[i][~s[i]].mean() for i in draws if s[i].any() and (~s[i]).any()]
                rows.append({"platform": p, "trait": col, "group": str(g), "posts": int(s.sum()), "share": float(s.mean()),
                             "over-performance vs rest": float(r0[s].mean() - r0[~s].mean()), "CI low": float(np.percentile(B, 2.5)), "CI high": float(np.percentile(B, 97.5)),
                             "raw log views vs rest": float(y0[s].mean() - y0[~s].mean())})
    return pd.DataFrame(rows)


def plot_atlas(A, path=None, lang="en", top=14):
    t = {"en": ("Who gets more views than their reactions suggest", "over-performance vs other posts (log10; 0.1 ≈ ×1.26)"),
         "pl": ("Kto ma więcej wyświetleń, niż sugerują reakcje", "nadwyżka względem pozostałych postów (log10; 0,1 ≈ ×1,26)")}[lang]
    q = A[(A["CI low"] > 0) | (A["CI high"] < 0)].copy(); q["abs"] = q["over-performance vs rest"].abs()
    fig, ax = plt.subplots(1, 3, figsize=(13, 5.2))
    for a, p in zip(ax, SRC):
        z = q[q.platform == p].nlargest(top, "abs").sort_values("over-performance vs rest")
        lab = [f"{r.trait}: {r.group} ({r.posts})" for r in z.itertuples()]
        a.errorbar(z["over-performance vs rest"], range(len(z)), xerr=[z["over-performance vs rest"] - z["CI low"], z["CI high"] - z["over-performance vs rest"]],
                   fmt="o", color="#2a6fdb", ecolor="#9bb8ea", capsize=2)
        a.set_yticks(range(len(z))); a.set_yticklabels(lab, fontsize=7.5); a.axvline(0, color="grey", lw=.8); a.set_title(NAME[p])
    ax[1].set_xlabel(t[1]); fig.suptitle(t[0], y=1.02); fig.tight_layout()
    if path: fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig
