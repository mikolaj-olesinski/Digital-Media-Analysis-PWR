"""Hand-checked view counts of Facebook / Instagram videos and Reels (plan A, B14).
`sample()` draws the posts to check (stratified by reactions) into data/manual/fbig_sample.csv (git-ignored: it holds
post links). A person opens each public link, fills views, likes, comments and shares seen at the same moment, and the
date of the check. `evaluate()` then scores the out-of-platform models on these posts: the first measured error of the
transfer to FB / IG in the project. Only numbers are recorded, no names, no screenshots, no automated scraping."""
import numpy as np, pandas as pd
from pathlib import Path
import eda_utils as eu, reaction_tiers as rt

COLS_FILL = ["checked_at", "views", "likes_now", "comments_now", "shares_now", "views_visible", "note"]


def sample(root, n=150, seed=42):
    d = eu.load(root); d = d[d.platform.isin(["facebook", "instagram"])].copy()
    d["kind"] = d.url.fillna("").str.extract(r"/(reels?|videos?)/", expand=False)
    d = d[d.kind.notna()]
    react = d.likes.fillna(0) + d.comments.fillna(0) + d.shares.fillna(0)
    d["tier"] = pd.cut(react, rt.EDGES, labels=rt.TIERS).astype(str)
    out = []
    for p, q in d.groupby("platform"):
        share = q.tier.value_counts(normalize=True); k = (share * n).round().astype(int).clip(lower=min(10, n // 10))
        for t, kk in k.items():
            z = q[q.tier == t]; out.append(z.sample(min(kk, len(z)), random_state=seed))
    s = pd.concat(out)[["mention_id", "platform", "kind", "tier", "url", "created_at", "metrics_updated_at", "author_followers", "likes", "comments", "shares"]]
    s = s.sample(frac=1, random_state=seed).reset_index(drop=True)
    for c in COLS_FILL: s[c] = ""
    path = Path(root) / "data" / "manual" / "fbig_sample.csv"; s.to_csv(path, index=False)
    return s.drop(columns="url"), path


def evaluate(root, f, meta):
    """Score the transfer models on the filled rows. Uses the reactions seen at the check (likes_now, comments_now)."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    s = pd.read_csv(Path(root) / "data" / "manual" / "fbig_sample.csv", dtype={"mention_id": str})
    s = s[pd.to_numeric(s.views, errors="coerce") > 0].copy()
    X = pd.DataFrame({"log_likes": np.log1p(pd.to_numeric(s.likes_now, errors="coerce").fillna(0)), "log_comments": np.log1p(pd.to_numeric(s.comments_now, errors="coerce").fillna(0))})
    y = np.log10(pd.to_numeric(s.views)); C = ["log_likes", "log_comments"]
    rows = []
    for src in ["tiktok", "youtube", "twitter", "all three"]:
        m = (meta.platform == src).values if src != "all three" else np.ones(len(meta), bool)
        mod = make_pipeline(StandardScaler(), Ridge()).fit(f[m][C].fillna(0), meta.y[m])
        for p in ["facebook", "instagram"]:
            k = (s.platform == p).values
            if k.sum() == 0: continue
            e = y[k].values - mod.predict(X[k])
            rows.append({"target": p, "trained on": src, "posts": int(k.sum()), "MAE": float(np.abs(e).mean()), "bias (true - estimate)": float(e.mean()), "typical factor": float(10 ** np.abs(e).mean())})
    return pd.DataFrame(rows)
