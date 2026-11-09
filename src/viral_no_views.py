"""E232: can we flag viral / outlier posts without views, from reactions, followers and post fields?
A. Viral (top 10% of views within the platform) from fields only, no views as input:
   - same platform-out: train on the other two of TikTok / YouTube / X, test on the held-out one;
   - FB and IG: train on X + YouTube (never on the target), test on the 402 FB videos and 642 IG reels.
   Scores: ridge estimate of views (likes + comments), CatBoost classifier on likes + comments (+ followers where both sides
   have it), raw likes. AUC, precision in the top 10% of the score, base rate 10%.
B. Over-performer (views far above what reactions suggest) on TikTok / YouTube / X, same platform, 5-fold CV grouped by
   E55 components: label = top 10% of the out-of-fold residual of ridge on likes + comments. Predicted from all
   non-view fields (followers, reactions, text length, type, sentiment, language, month, age) vs from reactions only.
Writes data/processed/viral_no_views_numbers.json.
    .venv/bin/python -W ignore src/viral_no_views.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, fb_extend as fe, ig_transfer as ig, limits as lm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C2 = ["log_likes", "log_comments"]; Q = .9
kw = dict(random_seed=42, iterations=300, verbose=0, allow_writing_files=False, thread_count=4)


def prec(lab, s, k=.1):
    n = max(int(round(k * len(s))), 1); o = np.argsort(-s)[:n]; return round(float(lab[o].mean()), 3)


def score(lab, s):
    return {"AUC": round(float(roc_auc_score(lab, s)), 3), "precision top 10%": prec(lab, s)}


def top_label(y, p):
    return y.groupby(p).transform(lambda v: v >= v.quantile(Q)).astype(int).values


def methods(tr, te, ytr_lab, lab):
    X, T = tr[C2].fillna(0), te[C2].fillna(0)
    out = {"ridge estimate": score(lab, fe.ridge().fit(X, tr.y).predict(T)),
           "classifier (likes + comments)": score(lab, CatBoostClassifier(**kw).fit(X, ytr_lab).predict_proba(T)[:, 1]),
           "raw likes": score(lab, te.log_likes.fillna(0).values)}
    return out


def block_a():
    f, meta = em.load(ROOT); f = f.copy(); f["platform"] = meta.platform.values; f["y"] = meta.y.values
    f["lab"] = top_label(f.y, f.platform); out = {"base rate": round(float(f.lab.mean()), 3)}
    for p in ["tiktok", "youtube", "twitter"]:
        tr, te = f[f.platform != p], f[f.platform == p]
        out[p] = {"posts": int(len(te)), **methods(tr, te, tr.lab, te.lab.values)}
    s = f[f.platform.isin(["twitter", "youtube"])]
    fb = fe.fb_frame(ROOT); fb = fb[fb.status == "ok"].dropna(subset=["y"]).copy(); fb["lab"] = (fb.y >= fb.y.quantile(Q)).astype(int)
    out["facebook"] = {"posts": int(len(fb)), **methods(s, fb, s.lab, fb.lab.values)}
    r = ig.frame(); r = r[(r.status == "ok") & r.views.notna() & r.likes.notna()].copy()
    r["y"] = np.log10(r.views.clip(lower=1)); r["log_likes"] = np.log1p(r.likes); r["log_comments"] = np.log1p(r.comments.fillna(0))
    r["lab"] = (r.y >= r.y.quantile(Q)).astype(int)
    out["instagram"] = {"posts": int(len(r)), **methods(s, r, s.lab, r.lab.values)}
    return out


def block_b():
    f, meta = em.load(ROOT); raw = eu.load(ROOT).loc[f.index]
    comp = lm.components(meta.reset_index(drop=True), raw.reset_index(drop=True)); out = {}
    allf = [c for c in f.columns]
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values; X = f[m].reset_index(drop=True); y = meta.y[m].reset_index(drop=True); g = pd.Series(comp[m]).values
        res = np.zeros(len(X))
        for tr, te in GroupKFold(5).split(X, y, g):
            res[te] = y.iloc[te].values - fe.ridge().fit(X.iloc[tr][C2].fillna(0), y.iloc[tr]).predict(X.iloc[te][C2].fillna(0))
        over = (res >= np.quantile(res, Q)).astype(int); under = (res <= np.quantile(res, 1 - Q)).astype(int); r = {"posts": int(len(X))}
        for name, lab in (("over-performer (top 10% residual)", over), ("under-performer (bottom 10% residual)", under)):
            r[name] = {}
            for fname, cols in (("reactions only", C2 + ["na_likes", "na_comments"]), ("all non-view fields", allf)):
                pp = np.zeros(len(X))
                for tr, te in GroupKFold(5).split(X, lab, g):
                    pp[te] = CatBoostClassifier(**kw).fit(X.iloc[tr][cols], lab[tr]).predict_proba(X.iloc[te][cols])[:, 1]
                r[name][fname] = score(lab, pp)
        out[p] = r
    return out


def block_c():
    """B redone against the FINAL model (results_preds.csv residual, as in E184) and with / without the age of the check."""
    f, meta = em.load(ROOT); raw = eu.load(ROOT).loc[f.index]
    comp = lm.components(meta.reset_index(drop=True), raw.reset_index(drop=True)); d = pd.read_csv(ROOT / "data/processed/results_preds.csv"); out = {}
    sets = {"fields of the final model (followers, reactions, flags)": em.ENG,
            "post fields only (text length, type, sentiment, language, month)": ["english", "text_len", "type_code", "sentiment_code", "month"],
            "post fields + age of the check": ["english", "text_len", "type_code", "sentiment_code", "month", "metrics_age_days"],
            "all, without age": [c for c in f.columns if c != "metrics_age_days"], "all, with age": list(f.columns)}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values; g0 = d[d.platform == p]; X = f[m].reset_index(drop=True); g = pd.Series(comp[m]).values
        res = (g0.y - g0.pred).values; r = {"posts": int(len(X))}
        for name, lab in (("over-performer (top 10% residual of the final model)", (res >= np.quantile(res, Q)).astype(int)),
                          ("under-performer (bottom 10%)", (res <= np.quantile(res, 1 - Q)).astype(int))):
            r[name] = {}
            for fname, cols in sets.items():
                pp = np.zeros(len(X))
                for tr, te in GroupKFold(5).split(X, lab, g):
                    pp[te] = CatBoostClassifier(**kw).fit(X.iloc[tr][cols], lab[tr]).predict_proba(X.iloc[te][cols])[:, 1]
                r[name][fname] = score(lab, pp)
        out[p] = r
    return out


def main():
    out = {"A: viral (top 10% of views) without views": block_a(), "B: over / under-performers vs ridge on likes + comments (inflated: age, nonlinearity)": block_b(),
           "C: over / under-performers vs the final model": block_c()}
    (ROOT / "data/processed/viral_no_views_numbers.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
