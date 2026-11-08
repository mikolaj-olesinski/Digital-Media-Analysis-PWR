"""E218: does adding the X posts that had no export views change the X model?
Same-moment re-read of X (E204). "New" posts = re-read posts outside the export view sample (em.load, view_count > 0).
In-platform: rt.FINAL (CatBoost, MAE) on re-read log likes / comments / shares + export log10 followers + log10 age at
the re-read, 5-fold brand CV over all re-read X posts; trained with vs without the new posts (same folds, same test
posts), scored on all, new and old posts. Transfer: ridge on C2, X re-read (with / without new) and export X, alone and
with export YouTube, scored on FB / IG (results_b.targets()).
Writes data/processed/x_reread_numbers.json.
    .venv/bin/python src/x_reread.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, fb_extend as fe, reaction_tiers as rt, results_b as rb, validation as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C2 = fe.C2
N_PAIR = 300
FEAT = ["log_likes", "log_comments", "log_shares", "log_followers", "log_age"]


def reread(plat, drop_nan=True):
    """Same-moment re-read (E204): status ok, views present, dedup by mention_id (last row), NaN reactions = missing."""
    f = {"youtube": "youtube", "twitter": "x", "tiktok": "tiktok"}[plat]
    d = pd.read_csv(ROOT / f"data/processed/{f}_full_fetched.csv", dtype={"mention_id": str}).drop_duplicates("mention_id", keep="last")
    d = d[(d.status == "ok") & d.views.notna()].copy()
    if drop_nan: d = d.dropna(subset=["likes", "comments"])
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id"); r = raw.loc[d.mention_id]
    d["platform"] = plat; d["brand"] = r.brand.fillna("unknown").values
    d["followers"] = pd.to_numeric(r.author_followers, errors="coerce").values
    d["created_at"] = pd.to_datetime(r.created_at.values, errors="coerce")
    d["y"] = np.log10(d.views.clip(lower=1)); d["log_likes"] = np.log1p(d.likes.fillna(0)); d["log_comments"] = np.log1p(d.comments.fillna(0))
    return d.reset_index(drop=True)


def stats(y, p, g):
    e = y - p; mae, lo, hi = va.mae_ci(e, g)
    return {"n": int(len(y)), "MAE": round(mae, 3), "factor": round(10 ** mae, 2), "CI": [round(lo, 3), round(hi, 3)], "median bias (+ = under)": round(float(np.median(e)), 3)}


def paired(y, pb, pn, g, seed=42):
    """MAE(new) - MAE(base) on the same posts, 95% CI from resampling brands. Negative = new is better."""
    d = pd.DataFrame({"g": g, "e": np.abs(y - pn) - np.abs(y - pb)}).groupby("g").e.agg(["sum", "count"])
    rng = np.random.default_rng(seed); idx = np.arange(len(d))
    bs = [d.iloc[s]["sum"].sum() / d.iloc[s]["count"].sum() for s in (rng.choice(idx, len(idx)) for _ in range(N_PAIR))]
    return {"diff": round(float(d["sum"].sum() / d["count"].sum()), 3), "CI": [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])]}


def main():
    f, meta = em.load(ROOT); mid = eu.load(ROOT).astype({"mention_id": str}).loc[meta.index].mention_id.values
    exp_x = set(mid[meta.platform.values == "twitter"])
    x = reread("twitter")
    x["new"] = ~x.mention_id.isin(exp_x)
    x["log_shares"] = np.log1p(x.shares)                                          # NaN stays missing for CatBoost
    x["log_followers"] = np.log10(x.followers.where(x.followers > 0))
    age = (pd.to_datetime(x.fetched_at, utc=True).dt.tz_convert(None) - x.created_at).dt.total_seconds() / 86400
    x["log_age"] = np.log10(age.clip(lower=1 / 24))
    y, g, new = x.y.values, x.brand.values, x.new.values
    P = {"with new posts": np.zeros(len(x)), "without new posts": np.zeros(len(x))}
    for tr, te in GroupKFold(5).split(x, y, g):
        P["with new posts"][te] = rt.FINAL().fit(x.iloc[tr][FEAT], y[tr]).predict(x.iloc[te][FEAT])
        t2 = tr[~new[tr]]
        P["without new posts"][te] = rt.FINAL().fit(x.iloc[t2][FEAT], y[t2]).predict(x.iloc[te][FEAT])
    out = {"posts": int(len(x)), "new posts (no export views)": int(new.sum()), "old posts": int((~new).sum()), "brands": int(len(set(g))),
           "median log10 views new / old": [round(float(np.median(y[new])), 3), round(float(np.median(y[~new])), 3)],
           "zero-like share new / old": [round(float((x.likes[new] == 0).mean()), 3), round(float((x.likes[~new] == 0).mean()), 3)]}
    ins = {}
    for part, m in (("all re-read X (primary)", np.ones(len(x), bool)), ("new posts", new), ("old posts", ~new)):
        ins[part] = {k: stats(y[m], p[m], g[m]) for k, p in P.items()}
        ins[part]["paired diff with - without"] = paired(y[m], P["without new posts"][m], P["with new posts"][m], g[m])
    out["in-platform CatBoost, brand CV"] = ins
    # transfer to FB / IG
    yt_m = meta.platform.values == "youtube"; xe_m = meta.platform.values == "twitter"
    ex = lambda m: pd.DataFrame({"log_likes": f[m].log_likes.fillna(0).values, "log_comments": f[m].log_comments.fillna(0).values, "y": meta.y.values[m]})
    rr = lambda m: x[m][C2 + ["y"]]
    S = {"X export": ex(xe_m), "X re-read (all)": rr(np.ones(len(x), bool)), "X re-read (old only)": rr(~new),
         "X export + YouTube export": pd.concat([ex(xe_m), ex(yt_m)]), "X re-read (all) + YouTube export": pd.concat([rr(np.ones(len(x), bool)), ex(yt_m)]),
         "X re-read (old only) + YouTube export": pd.concat([rr(~new), ex(yt_m)])}
    tr_out = {}
    for tname, d in rb.targets().items():
        yy, gg = d.y.values, d.brand.values; pr = {k: fe.ridge().fit(s[C2], s.y).predict(d[C2]) for k, s in S.items()}
        r = {k: stats(yy, p, gg) for k, p in pr.items()}
        r["paired: X re-read (all) - X export"] = paired(yy, pr["X export"], pr["X re-read (all)"], gg)
        r["paired: X re-read (all) - X re-read (old only)"] = paired(yy, pr["X re-read (old only)"], pr["X re-read (all)"], gg)
        r["paired: X re-read (all) + YT - X export + YT"] = paired(yy, pr["X export + YouTube export"], pr["X re-read (all) + YouTube export"], gg)
        tr_out[tname] = r
    out["transfer ridge C2 -> FB / IG"] = tr_out
    (ROOT / "data/processed/x_reread_numbers.json").write_text(json.dumps({"E218": out}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
