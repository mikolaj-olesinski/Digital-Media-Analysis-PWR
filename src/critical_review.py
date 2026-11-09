"""Critical review batch (E178-E188): validity threats and business uses not covered before. Statistics only.
  E178 FB rests on few brands: per-brand MAE, leave-one-brand-out range, MAE without the 3 largest brands.
  E179 age in the noise ceiling: E56 pairs also matched on check age; a fixed 30-day target (cross-sectional b, E101).
  E180 models for totals: MAE (final), RMSE, tail-weighted MAE, quantile 0.7; post MAE, top-1% MAE, brand and brand-month totals.
  E181 X missing views (E105): inverse-probability weights from p(view observed | inputs); naive vs weighted MAE estimate,
       level of predictions on the posts without views.
  E182 subgroups: language (FB 402 and TT/YT/X), Brand24 sentiment (negative / neutral / positive): MAE and bias, brand CI.
  E183 creator score: authors with >= 4 posts; mean residual in the first vs second half of their posts (stability).
  E184 anomalies: large positive / negative residuals vs our promo / spam labels and duplicates (lift over the base rate).
  E185 early virality alert from the first check (history file): precision / recall for the top 1% and 5% of final views.
  E186 monthly ranking within an industry: Kendall of predicted vs true brand-month totals (>= 3 brands), model vs rule.
  E187 = sentiment part of E182.
  E188 earned media value: brand totals x an assumed CPM; share of brands within +-50% for model vs rule.
    python src/critical_review.py [names...]     (default: all)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau, spearmanr
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, fb_extend as fe, history as H, validation as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLATS = ["tiktok", "youtube", "twitter"]
SEED = 42
OUT = ROOT / "data" / "processed" / "critical_review_numbers.json"


def cb(loss="MAE", it=300):
    return CatBoostRegressor(loss_function=loss, iterations=it, random_seed=SEED, verbose=0, allow_writing_files=False, thread_count=4)


def oof(X, y, g, model=cb, w=None):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = model(); m.fit(X.iloc[tr], y[tr], sample_weight=None if w is None else w[tr]); p[te] = m.predict(X.iloc[te])
    return p


def boot_ci(v, g, n=400, f=np.mean):
    rng = np.random.default_rng(SEED); ub = np.unique(g); ix = {b: np.where(g == b)[0] for b in ub}
    s = [f(v[np.concatenate([ix[b] for b in rng.choice(ub, len(ub))])]) for _ in range(n)]
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def base():
    f, meta = em.load(ROOT); raw = eu.load(ROOT).loc[f.index]
    P = np.zeros(len(f))
    for p in PLATS:
        m = (meta.platform == p).values; P[m] = oof(f[m][em.ENG].fillna(0).reset_index(drop=True), meta.y[m].values, meta.brand[m].values)
    return f, meta, raw, P


# ---------- E178 ----------
def e178(ctx):
    d = fe.fb_frame(ROOT); d = d[d.status == "ok"].dropna(subset=["y"]).reset_index(drop=True)
    pred = fe.train_source(ROOT).predict(d[fe.C2].fillna(0)); e = np.abs(d.y.values - pred); g = d.brand.values
    per = pd.DataFrame({"b": g, "e": e}).groupby("b").e.agg(["size", "mean"]).sort_values("size", ascending=False)
    loo = [float(e[g != b].mean()) for b in per.index]; top3 = per.index[:3]
    return {"brands": int(len(per)), "posts": int(len(d)), "MAE": float(e.mean()), "share of posts in top 3 brands": float(per["size"][:3].sum() / len(d)),
            "MAE without top 3 brands": float(e[~np.isin(g, top3)].mean()), "leave-one-brand-out MAE min / max": [min(loo), max(loo)],
            "per-brand MAE (brands with >= 5 posts) p10 / median / p90": np.percentile(per["mean"][per["size"] >= 5], [10, 50, 90]).round(3).tolist(),
            "brands with >= 5 posts": int((per["size"] >= 5).sum())}


# ---------- E179 ----------
B_CROSS = {"tiktok": .043, "youtube": .164, "twitter": .037}


def e179(ctx):
    f, meta, raw, P = ctx; out = {}
    age = np.log10(f.metrics_age_days.clip(lower=0) + 1)
    for p in PLATS:
        m = (meta.platform == p).values; y = meta.y[m].values; g = meta.brand[m].values
        X4 = f[m][["log_followers", "log_likes", "log_comments", "log_shares"]].fillna(-1).values; a = age[m].values
        r = {}
        for name, X in [("inputs only (E56)", X4), ("inputs + check age", np.c_[X4, a])]:
            d, j = NearestNeighbors(n_neighbors=6, metric="chebyshev").fit(X).kneighbors(X); pr = []
            for i in range(len(X)):
                for dd, jj in zip(d[i, 1:], j[i, 1:]):
                    if dd <= .05 and g[jj] != g[i] and i < jj: pr.append((i, jj)); break
            pr = np.array(pr); dy = np.abs(y[pr[:, 0]] - y[pr[:, 1]]); used = np.unique(pr.ravel())
            r[name] = {"pairs": int(len(pr)), "lower bound on MAE": float(dy.mean() / 2), "final model MAE on these posts": float(np.abs(y[used] - P[m][used]).mean()),
                       "x-factor between matched posts": float(10 ** dy.mean())}
        # fixed horizon: views at 30 days with the cross-sectional growth exponent
        ad = f.metrics_age_days[m].clip(lower=.1).values; y30 = y + B_CROSS[p] * (np.log10(30) - np.log10(ad))
        X = f[m][em.ENG].fillna(0).reset_index(drop=True)
        p30 = oof(X, y30, g); pa = oof(pd.concat([X, pd.Series(np.log10(ad), name="log_age")], axis=1), y, g)
        r["target views at check (final)"] = float(np.abs(y - P[m]).mean()); r["target views at check, + log age"] = float(np.abs(y - pa).mean())
        r["target views at 30 days"] = float(np.abs(y30 - p30).mean()); r["check age days p10 / median / p90"] = np.percentile(ad, [10, 50, 90]).round(1).tolist()
        out[p] = r
    return out


# ---------- E180 + E186 + E188 ----------
def _totals(df, key, col):
    t = df.groupby(key).agg(n=("y", "size"), true=("v", "sum"), pred=(col, "sum"))
    return t[t.n >= 5]


def e180(ctx):
    f, meta, raw, P = ctx; ind = pd.read_csv(ROOT / "data" / "processed" / "brand_industry.csv").set_index("brand").industry
    out = {"models": {}, "monthly industry ranking": {}, "earned media value": {}}; frames = []
    for p in PLATS:
        m = (meta.platform == p).values; X = f[m][em.ENG].fillna(0).reset_index(drop=True); y = meta.y[m].values; g = meta.brand[m].values
        w = 10 ** (0.5 * (y - y.mean()))   # tail weight grows with views (square root of the view ratio)
        preds = {"MAE (final)": P[m], "RMSE": oof(X, y, g, lambda: cb("RMSE")), "MAE, tail-weighted": oof(X, y, g, w=w),
                 "quantile 0.7": oof(X, y, g, lambda: cb("Quantile:alpha=0.7"))}
        rule = np.log10(meta[eu.REACH][m].clip(lower=1).values)
        df = pd.DataFrame({"platform": p, "brand": g, "y": y, "v": 10 ** y, "month": raw.created_at[m].dt.strftime("%Y-%m").values, "rule": 10 ** rule,
                           **{k: 10 ** v for k, v in preds.items()}})
        top = y >= np.quantile(y, .99); res = {}
        for k, v in list(preds.items()) + [("Brand24 rule", rule)]:
            bt = _totals(df, "brand", "rule" if k == "Brand24 rule" else k); bm = _totals(df, ["brand", "month"], "rule" if k == "Brand24 rule" else k)
            lr = np.log10(bt.pred / bt.true); lm = np.log10(bm.pred / bm.true)
            res[k] = {"post MAE": float(np.abs(y - v).mean()), "top 1% MAE": float(np.abs(y[top] - v[top]).mean()), "top 1% bias (pred - true)": float((v[top] - y[top]).mean()),
                      "brand total |log ratio| median": float(np.median(np.abs(lr))), "brand totals off > x2": float((np.abs(lr) > np.log10(2)).mean()),
                      "brand-month total |log ratio| median": float(np.median(np.abs(lm))), "brand-month off > x2": float((np.abs(lm) > np.log10(2)).mean()), "brands": int(len(bt)), "brand-months": int(len(bm))}
        out["models"][p] = res; frames.append(df)
    D = pd.concat(frames); D["industry"] = D.brand.map(ind).fillna("other")
    for p in PLATS:
        z = D[D.platform == p]; r = {}
        for k in ["MAE (final)", "MAE, tail-weighted", "rule"]:
            taus = []
            for (i, mo), q in z.groupby(["industry", "month"]):
                t = q.groupby("brand").agg(n=("v", "size"), true=("v", "sum"), pred=(k, "sum")); t = t[t.n >= 3]
                if len(t) >= 3: taus.append(kendalltau(t.true, t.pred)[0])
            r[k] = {"industry-months": len(taus), "mean Kendall": float(np.nanmean(taus)) if taus else None, "share with Kendall >= 0.6": float(np.mean(np.array(taus) >= .6)) if taus else None}
        out["monthly industry ranking"][p] = r
        # earned media value at an assumed CPM (USD per 1000 views); the CPM cancels in ratios, shown for scale
        CPM = 5.0; bt = z.groupby("brand").agg(n=("v", "size"), true=("v", "sum"), model=("MAE (final)", "sum"), rule=("rule", "sum")); bt = bt[bt.n >= 5]
        out["earned media value"][p] = {"assumed CPM USD": CPM, "brands": int(len(bt)), "median true EMV USD": float(np.median(bt.true) * CPM / 1000),
                                        **{f"{k}: share of brands within +-50%": float((np.abs(bt[k] / bt.true - 1) <= .5).mean()) for k in ["model", "rule"]},
                                        **{f"{k}: median abs error USD": float(np.median(np.abs(bt[k] - bt.true)) * CPM / 1000) for k in ["model", "rule"]}}
    return out


# ---------- E181 ----------
def e181(ctx):
    d = eu.load(ROOT); x = d[d.platform == "twitter"].copy()
    for c in ["author_followers", "likes", "comments", "shares", "view_count"]: x[c] = pd.to_numeric(x[c], errors="coerce")
    X = pd.DataFrame({"log_followers": np.log10(x.author_followers.where(x.author_followers > 0)), **{f"log_{c}": np.log1p(x[c]) for c in ["likes", "comments", "shares"]},
                      **{f"na_{c}": x[c].isna().astype(int) for c in ["likes", "comments", "shares"]}, "na_followers": x.author_followers.isna().astype(int),
                      "age": (x.metrics_updated_at - x.created_at).dt.days, "year": x.created_at.dt.year}).fillna(-1).reset_index(drop=True)
    obs = (x.view_count > 0).values; g = x.brand.values
    po = np.zeros(len(X))
    for tr, te in GroupKFold(5).split(X, obs, g):
        po[te] = CatBoostClassifier(iterations=300, depth=4, verbose=0, random_seed=SEED, allow_writing_files=False, thread_count=4).fit(X.iloc[tr], obs[tr]).predict_proba(X.iloc[te])[:, 1]
    w = 1 / np.clip(po, .05, 1); Xo, yo, go, wo = X[obs].reset_index(drop=True), np.log10(x.view_count[obs].values), g[obs], w[obs]
    cols = [c for c in X.columns if c not in ("age", "year")] + ["age"]
    pn = oof(Xo[cols], yo, go); pw = oof(Xo[cols], yo, go, w=wo)
    full_n = cb().fit(Xo[cols], yo); full_w = cb().fit(Xo[cols], yo, sample_weight=wo); Xm = X[~obs][cols]
    e = np.abs(yo - pn); ew = np.abs(yo - pw)
    return {"posts": int(len(X)), "observed share": float(obs.mean()), "selection AUC": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(obs, po)),
            "effective sample share of weights": float(wo.sum() ** 2 / (wo ** 2).sum() / len(wo)),
            "naive model: MAE (unweighted)": float(e.mean()), "naive model: MAE weighted to all posts": float(np.average(e, weights=wo)),
            "weighted model: MAE (unweighted)": float(ew.mean()), "weighted model: MAE weighted to all posts": float(np.average(ew, weights=wo)),
            "mean predicted log10 views on posts without views: naive / weighted": [float(full_n.predict(Xm).mean()), float(full_w.predict(Xm).mean())],
            "mean log10 views observed": float(yo.mean())}


# ---------- E182 + E187 ----------
def _grp(e, b, g, mask):
    return {"posts": int(mask.sum()), "MAE": float(np.abs(e[mask]).mean()), "MAE CI": boot_ci(np.abs(e[mask]), g[mask]), "bias (true - pred)": float(e[mask].mean()),
            "bias CI": boot_ci(e[mask], g[mask])} if mask.sum() >= 30 else {"posts": int(mask.sum())}


def e182(ctx):
    f, meta, raw, P = ctx; out = {"three platforms": {}}
    for p in PLATS:
        m = (meta.platform == p).values; e = meta.y[m].values - P[m]; g = meta.brand[m].values; lang = raw.language[m].fillna("unknown").values; s = raw.sentiment[m].fillna("unknown").values
        out["three platforms"][p] = {"English": _grp(e, None, g, lang == "en"), "other language": _grp(e, None, g, lang != "en"),
                                     **{f"sentiment {k}": _grp(e, None, g, s == k) for k in ["negative", "neutral", "positive"]}}
    d = fe.fb_frame(ROOT); d = d[d.status == "ok"].dropna(subset=["y"]).reset_index(drop=True)
    r = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id").loc[d.mention_id]
    e = d.y.values - fe.train_source(ROOT).predict(d[fe.C2].fillna(0)); g = d.brand.values; lang = r.language.fillna("unknown").values; s = r.sentiment.fillna("unknown").values
    out["facebook"] = {"English": _grp(e, None, g, lang == "en"), "other language": _grp(e, None, g, lang != "en"),
                       "languages": pd.Series(lang).value_counts().head(6).to_dict(), **{f"sentiment {k}": _grp(e, None, g, s == k) for k in ["negative", "neutral", "positive"]}}
    return out


# ---------- E183 ----------
def e183(ctx):
    f, meta, raw, P = ctx; out = {}
    for p in PLATS:
        m = (meta.platform == p).values
        z = pd.DataFrame({"a": raw.author_name[m].values, "t": raw.created_at[m].values, "r": meta.y[m].values - P[m], "txt": raw.content[m].fillna("").values}).dropna(subset=["a"])
        z = z.drop_duplicates(["a", "txt"]); rows = []
        for a, q in z.sort_values("t").groupby("a"):
            if len(q) >= 4: h = len(q) // 2; rows.append((q.r.iloc[:h].mean(), q.r.iloc[h:].mean(), len(q)))
        if len(rows) < 10: out[p] = {"authors": len(rows)}; continue
        A = np.array(rows); hi1, hi2 = A[:, 0] > np.median(A[:, 0]), A[:, 1] > np.median(A[:, 1])
        top1 = A[:, 0] >= np.quantile(A[:, 0], .8); top2 = A[:, 1] >= np.quantile(A[:, 1], .8)
        out[p] = {"authors with >= 4 posts": int(len(A)), "posts covered": int(A[:, 2].sum()), "Spearman first vs second half": float(spearmanr(A[:, 0], A[:, 1])[0]),
                  "same side of the median": float((hi1 == hi2).mean()), "top-20% list precision in the second half": float(top2[top1].mean()),
                  "mean residual of first-half top 20% in second half": float(A[top1, 1].mean())}
    return out


# ---------- E184 ----------
def e184(ctx):
    f, meta, raw, P = ctx
    lab = pd.read_csv(ROOT / "data" / "processed" / "mention_labels.csv", dtype={"mention_id": str}).set_index("mention_id").type
    typ = lab.reindex(raw.mention_id.astype(str)).values; txt = raw.content.fillna(raw.title).fillna("").str.lower().str.replace(r"\s+", " ", regex=True)
    dup = txt.duplicated(keep=False).values & (txt.str.len() > 20).values; out = {}
    for p in PLATS:
        m = (meta.platform == p).values; r = meta.y[m].values - P[m]; t = typ[m]; dd = dup[m]
        hi, lo = r >= np.quantile(r, .95), r <= np.quantile(r, .05); res = {}
        for name, s in [("all posts", np.ones(len(r), bool)), ("over-performers (top 5% residual)", hi), ("under-performers (bottom 5% residual)", lo)]:
            res[name] = {"posts": int(s.sum()), "promo": float((t[s] == "promo").mean()), "spam": float((t[s] == "spam").mean()), "duplicate text": float(dd[s].mean()),
                         "median reactions": float(np.median(np.expm1(f.log_likes[m].fillna(0).values[s])))}
        out[p] = res
    return out


# ---------- E185 ----------
def e185(ctx):
    s = H.snapshots(H.load(ROOT)); pr = H.pairs(s); P = H.forecast_cv(pr); out = {}
    for plat, q in P.groupby("platform"):
        r = {}
        for k, z in q.groupby("model"):
            y, pred = z.y.values, z.pred.values; res = {"posts": int(len(z))}
            for top in (.01, .05):
                tgt = y >= np.quantile(y, 1 - top); n = max(1, int(round(top * len(y))))
                for budget, nb in [("alerts = target size", n), ("alerts = 3x target size", 3 * n)]:
                    sel = np.argsort(-pred)[:nb]; hit = tgt[sel].sum()
                    res[f"top {int(top*100)}%, {budget}"] = {"precision": float(hit / nb), "recall": float(hit / tgt.sum())}
            r[k] = res
        out[plat] = r
    return out


ALL = {"E178": e178, "E179": e179, "E180": e180, "E181": e181, "E182": e182, "E183": e183, "E184": e184, "E185": e185}


def main(names):
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    ctx = base() if any(n in names for n in ["E179", "E180", "E182", "E183", "E184"]) else None
    for n in names:
        out[n] = ALL[n](ctx); OUT.write_text(json.dumps(out, indent=1, default=float)); print(n, json.dumps(out[n], default=float)[:1500], flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(ALL))
