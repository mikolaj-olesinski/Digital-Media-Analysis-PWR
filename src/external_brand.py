"""External brand data as model columns (E170 daily Wikipedia attention, E171 Wikidata company facts).
Data from src/fetch_wikimedia_more.py; only brand article titles were sent out (user consent, 2026-11-06).
  E170  brand attention at post time: log10 Wikipedia views on the post day and the mean of the 7 days before,
        each minus the brand's median over the 60 days before (a news spike of the brand), plus the 7-day peak.
        Only past and same-day pageviews are used, so the columns exist at prediction time.
  E171  company facts: log10 employees, log10 revenue (USD), brand age in years at the post, missing = -1.
Evaluated (a) TikTok / YouTube / X: final CatBoost + columns, same brand folds, paired brand-bootstrap gain;
(b) Facebook, 402 fetched posts: ridge on likes + comments (+ columns) trained on X + YouTube, and in-platform brand CV.
Writes data/processed/external_brand_numbers.json. Statistics only.
    .venv/bin/python src/external_brand.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, history as H, author_next as an, fb_extend as fe, validation as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLATS = ["tiktok", "youtube", "twitter"]


def attention(brand, date, daily):
    """Per post: attention columns from the brand's daily pageviews (past and same day only)."""
    S = {b: g.set_index("date").views.sort_index() for b, g in daily.groupby("brand")}
    rows = []
    for b, t in zip(brand, date):
        s = S.get(b)
        if s is None or pd.isna(t): rows.append((np.nan, np.nan, np.nan)); continue
        d = pd.Timestamp(t).normalize(); base = s[(s.index >= d - pd.Timedelta(days=60)) & (s.index < d)]
        if len(base) < 30 or d not in s.index: rows.append((np.nan, np.nan, np.nan)); continue
        lb = np.log10(base.median() + 1); wk = s[(s.index >= d - pd.Timedelta(days=7)) & (s.index <= d)]
        rows.append((np.log10(s[d] + 1) - lb, np.log10(wk.mean() + 1) - lb, np.log10(wk.max() + 1) - lb))
    return pd.DataFrame(rows, columns=["wiki_day_vs_base", "wiki_week_vs_base", "wiki_peak_vs_base"])


def facts(brand, date, wd):
    w = wd.set_index("brand")
    out = pd.DataFrame({"log_employees": np.log10(brand.map(w.employees)), "log_revenue_usd": np.log10(brand.map(w.revenue_usd)),
                        "brand_age_years": pd.to_datetime(date).dt.year.values - brand.map(w.inception_year).values})
    return out.reset_index(drop=True)


def gains(P, name):
    return {p: dict(zip(["gain", "CI low", "CI high"], map(float, H.gain_ci(P, p, "final model", name)))) for p in PLATS}


def main():
    daily = pd.read_csv(ROOT / "data" / "processed" / "brand_wiki_daily.csv", parse_dates=["date"])
    wd = pd.read_csv(ROOT / "data" / "processed" / "brand_wikidata.csv")
    f, meta = em.load(ROOT); raw = eu.load(ROOT).loc[f.index]
    A = attention(meta.brand.values, raw.created_at.values, daily).set_index(f.index)
    F = facts(meta.brand.reset_index(drop=True), raw.created_at.reset_index(drop=True), wd).set_index(f.index)
    out = {"coverage": {"attention": A.notna().all(axis=1).groupby(meta.platform.values).mean().round(3).to_dict(),
                        "employees": F.log_employees.notna().groupby(meta.platform.values).mean().round(3).to_dict(),
                        "inception": F.brand_age_years.notna().groupby(meta.platform.values).mean().round(3).to_dict()},
           "attention_quantiles": A.wiki_day_vs_base.quantile([.1, .5, .9, .99]).round(3).to_dict(), "three_platforms": {}, "facebook": {}}
    P0 = None
    for name, X in [("+ Wikipedia attention (E170)", A), ("+ Wikidata facts (E171)", F), ("+ both", pd.concat([A, F], axis=1))]:
        P = an.with_columns(f, meta, X.fillna(-1), name); S = H.summarise(P); P0 = P if P0 is None else P0
        out["three_platforms"][name] = {"MAE": S.pivot(index="model", columns="platform", values="MAE").round(4).to_dict(), "gain": gains(P, name)}
        print(name, json.dumps(out["three_platforms"][name]["gain"]), flush=True)
    # residual check: do the columns relate to the final model's error at all?
    base = P0[P0.model == "final model"].set_index("idx")
    r = base.y - base.pred; out["spearman_residual"] = {}
    for c in list(A) + list(F):
        v = pd.concat([A, F], axis=1)[c].reindex(base.index); ok = v.notna()
        out["spearman_residual"][c] = {p: round(float(pd.Series(r[ok & (base.platform == p)]).corr(v[ok & (base.platform == p)], method="spearman")), 3) for p in PLATS}
    # Facebook, 402 fetched posts
    d = fe.fb_frame(ROOT); d = d[d.status == "ok"].dropna(subset=["y"]).reset_index(drop=True)
    FA = attention(d.brand.values, d.created_at.values, daily); FF = facts(d.brand, d.created_at, wd); FX = pd.concat([FA, FF], axis=1)
    m = meta.platform.isin(["twitter", "youtube"]).values
    SA = pd.concat([A, F], axis=1)[m].reset_index(drop=True)
    src = pd.concat([f[m][fe.C2].fillna(0).reset_index(drop=True), SA.fillna(-1)], axis=1)
    for name, cols in [("likes + comments", fe.C2), ("+ attention", fe.C2 + list(A)), ("+ facts", fe.C2 + list(F)), ("+ both", fe.C2 + list(A) + list(F))]:
        tgt = pd.concat([d[fe.C2].fillna(0), FX.fillna(-1)], axis=1)
        p = fe.ridge().fit(src[cols], meta.y[m].values).predict(tgt[cols])
        oof = np.zeros(len(d))
        for tr, te in GroupKFold(5).split(tgt, d.y, d.brand):
            oof[te] = fe.ridge().fit(tgt.iloc[tr][cols], d.y.iloc[tr]).predict(tgt.iloc[te][cols])
        out["facebook"][name] = {"transfer X+YouTube": fe.score(p, d.y.values, d.brand.values), "in-platform brand CV": fe.score(oof, d.y.values, d.brand.values)}
        print("FB", name, out["facebook"][name]["transfer X+YouTube"]["MAE"], out["facebook"][name]["in-platform brand CV"]["MAE"], flush=True)
    out["facebook_coverage"] = {"attention": float(FA.notna().all(axis=1).mean().round(3)), "employees": float(FF.log_employees.notna().mean().round(3))}
    (ROOT / "data" / "processed" / "external_brand_numbers.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
