"""Country, local time, holidays, industry, where promo over-performs, and brand popularity on Wikipedia
(notebooks/models/33_context_more.ipynb, E84-E86).
E84 country, local time (main time zone of the post's country), local holidays and industry (E69 brand communities) as columns.
E85 where exactly promo over-performs: promo minus organic over-performance by subgroup, and the label's gain per subgroup.
E86 monthly English Wikipedia pageviews of the brand (src/fetch_wikipedia.py) as columns and against brand-level over-performance.
Outputs carry statistics only."""
import numpy as np, pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo
import holidays
from scipy.stats import spearmanr
import extra_models as em, reaction_tiers as rt, history as H, account_clusters as ac, author_next as an
from validation import SEED

PLATS = ["tiktok", "youtube", "twitter"]
TZ = {"US": "America/New_York", "IN": "Asia/Kolkata", "GB": "Europe/London", "AU": "Australia/Sydney", "CA": "America/Toronto", "PH": "Asia/Manila",
      "ID": "Asia/Jakarta", "PK": "Asia/Karachi", "NG": "Africa/Lagos", "DE": "Europe/Berlin", "KE": "Africa/Nairobi", "VN": "Asia/Ho_Chi_Minh",
      "PL": "Europe/Warsaw", "FR": "Europe/Paris", "MX": "America/Mexico_City", "ZA": "Africa/Johannesburg", "MY": "Asia/Kuala_Lumpur",
      "SG": "Asia/Singapore", "BR": "America/Sao_Paulo", "TH": "Asia/Bangkok", "JP": "Asia/Tokyo", "IT": "Europe/Rome", "NZ": "Pacific/Auckland",
      "NP": "Asia/Kathmandu", "KR": "Asia/Seoul", "ES": "Europe/Madrid", "NL": "Europe/Amsterdam", "AE": "Asia/Dubai", "IE": "Europe/Dublin", "BD": "Asia/Dhaka"}
TOP = ["US", "IN", "GB", "AU", "CA", "PH", "ID", "PK", "NG", "DE"]


def local_context(raw):
    """Per post: one-hot of the 10 most common countries (+ other / unknown), local hour, local weekend, local public holiday
    (holidays package, offline). Posts without a known time zone get local columns = -1."""
    cc = raw.country.fillna("unknown").astype(str); t = raw.created_at
    out = pd.DataFrame(index=raw.index)
    for c in TOP: out[f"country_{c}"] = (cc == c).astype(int)
    out["country_other"] = (~cc.isin(TOP + ["unknown"])).astype(int); out["country_unknown"] = (cc == "unknown").astype(int)
    lh, lw, lhol = np.full(len(raw), -1.0), np.full(len(raw), -1.0), np.full(len(raw), -1.0); cache = {}
    for i, (c, ts) in enumerate(zip(cc.values, t.values)):
        if c not in TZ or pd.isna(ts): continue
        loc = pd.Timestamp(ts).tz_localize("UTC").tz_convert(ZoneInfo(TZ[c])); lh[i] = loc.hour; lw[i] = float(loc.dayofweek >= 5)
        if c not in cache:
            try: cache[c] = holidays.country_holidays(c, years=[2024, 2025, 2026])
            except Exception: cache[c] = {}
        lhol[i] = float(loc.date() in cache[c])
    out["local_hour"], out["local_weekend"], out["local_holiday"] = lh, lw, lhol
    return out


def industry(meta_brand, lab, k=9):
    c = meta_brand.map(lab).fillna(-1).astype(int)
    return pd.DataFrame({f"industry_{i}": (c == i).astype(int) for i in range(k)}, index=meta_brand.index)


def variants(f, meta, groups):
    """Final model vs + each group of columns, and + all of them (same brand folds)."""
    P = []
    for name, X in groups.items():
        q = an.with_columns(f, meta, X, name); P.append(q if not P else q[q.model != "final model"])
    return pd.concat(P, ignore_index=True)


# ---------- E85 ----------
def promo_subgroups(d, f, meta, raw, lab_type, kind, lab_ind, n_boot=400, min_n=20):
    """Promo minus organic mean over-performance within subgroups, brand-bootstrap 95% CI."""
    rng = np.random.default_rng(SEED)
    q = d.assign(type=lab_type.reindex(d.index).values, kind=kind.reindex(d.index).values, tier=ac.tier(d.followers).values,
                 react=pd.cut(rt.reactions(f).loc[d.index], rt.EDGES, labels=rt.TIERS).astype(str).values,
                 us=np.where(raw.country.reindex(d.index) == "US", "US", np.where(raw.country.reindex(d.index).isna(), "unknown", "not US")),
                 english=np.where(raw.language.reindex(d.index) == "en", "English", "other language"), industry=meta.brand.map(lab_ind).fillna(-1).astype(int).astype(str).values)
    q = q[q.type.isin(["promo", "organic"])]; rows = []
    for p in PLATS:
        z = q[q.platform == p]
        for col in ["all", "tier", "react", "kind", "us", "english", "industry"]:
            groups = [("all posts", np.ones(len(z), bool))] if col == "all" else [(g, (z[col] == g).values) for g in sorted(z[col].unique())]
            for g, s in groups:
                w = z[s]; pm, om = (w.type == "promo").values, (w.type == "organic").values
                if pm.sum() < min_n or om.sum() < min_n: continue
                br = w.brand.values; ub = np.unique(br); ix = {b: np.where(br == b)[0] for b in ub}; B = []
                for _ in range(n_boot):
                    i = np.concatenate([ix[b] for b in rng.choice(ub, len(ub))])
                    if pm[i].any() and om[i].any(): B.append(w.r.values[i][pm[i]].mean() - w.r.values[i][om[i]].mean())
                rows.append({"platform": p, "split": col, "group": g, "promo posts": int(pm.sum()), "organic posts": int(om.sum()), "promo share": float(pm.mean() / (pm.mean() + om.mean())),
                             "promo − organic": float(w.r[pm].mean() - w.r[om].mean()), "CI low": float(np.percentile(B, 2.5)), "CI high": float(np.percentile(B, 97.5))})
    return pd.DataFrame(rows)


def label_gain_by(P, key, name):
    """Gain of `name` over the final model within groups of `key` (array aligned to P's final-model rows per platform)."""
    a = P[P.model == "final model"].reset_index(drop=True); b = P[P.model == name].reset_index(drop=True)
    g = pd.DataFrame({"platform": a.platform, "key": key, "gain": np.abs(a.y - a.pred) - np.abs(b.y - b.pred)})
    return g.groupby(["platform", "key"]).gain.agg(["size", "mean"]).reset_index().rename(columns={"size": "posts", "mean": "gain"})


# ---------- E86 ----------
def wiki_columns(root, raw, meta):
    """log10 monthly pageviews of the brand's article in the post's month, and the brand's median over 2024-01..2026-10.
    Brands without a matched article (name check failed) get -1."""
    path = Path(root) / "data" / "processed" / "brand_wikipedia.csv"
    if not path.exists(): return None
    w = pd.read_csv(path); w = w[w.name_match.astype(str) == "True"].dropna(subset=["views"])
    med = np.log10(w.groupby("brand").views.median() + 1); mon = np.log10(w.set_index(["brand", "month"]).views + 1)
    key = list(zip(meta.brand, raw.created_at.dt.strftime("%Y-%m")))
    out = pd.DataFrame({"wiki_month": [mon.get(k, np.nan) for k in key], "wiki_median": meta.brand.map(med).values}, index=meta.index)
    out["wiki_month_vs_median"] = out.wiki_month - out.wiki_median
    return out.fillna(-1)


def wiki_vs_brand(d, W, min_posts=5):
    rows = []
    for p in PLATS:
        z = d[d.platform == p].assign(w=W.wiki_median.reindex(d.index).values)
        g = z[z.w >= 0].groupby("brand").agg(n=("r", "size"), r=("r", "mean"), y=("y", "mean"), w=("w", "first")); g = g[g.n >= min_posts]
        rows.append({"platform": p, "brands": len(g), "Spearman wiki vs brand over-performance": spearmanr(g.w, g.r)[0], "Spearman wiki vs brand mean log views": spearmanr(g.w, g.y)[0]})
    return pd.DataFrame(rows)
