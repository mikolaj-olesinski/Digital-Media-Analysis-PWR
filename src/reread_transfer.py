"""E210: does same-moment re-read source data (views and reactions from one request) improve the transfer to FB / IG?
Ridge on log likes + log comments (fe.ridge, fe.C2). Base: Brand24 export X + YouTube. Primary (fixed in the card):
re-read X + YouTube, rows with missing likes or comments dropped. Other variants: fill-0, + TikTok, export + re-read
stacked. Also the E201 31-combination grid with re-read TikTok / YouTube / X plus FB / IG as platforms.
Targets: results_b.targets() (FB 402 videos, IG reels with visible likes).
Writes data/processed/reread_transfer_numbers.json.
    .venv/bin/python src/reread_transfer.py"""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, fb_extend as fe, results_b as rb, train_combos as tc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
L2 = np.log10(2)
P = ["tiktok", "youtube", "twitter", "facebook", "instagram"]


# ---------- re-read loader (same copy in every WP-A script via import) ----------
def reread(fill0=False, plats=("youtube", "x", "tiktok")):
    """Same-moment re-read rows: status ok, views present, one row per mention_id (last kept).
    Missing likes / comments are dropped unless fill0 (sensitivity). Brand and created_at from the raw export."""
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    out = []
    for p in plats:
        d = pd.read_csv(ROOT / f"data/processed/{p}_full_fetched.csv", dtype={"mention_id": str})
        d = d.drop_duplicates("mention_id", keep="last")
        for c in ["views", "likes", "comments"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d[(d.status == "ok") & d.views.notna()]
        d = d[d.mention_id.isin(raw.index)]
        if fill0:
            d = d.assign(likes=d.likes.fillna(0), comments=d.comments.fillna(0))
        else:
            d = d.dropna(subset=["likes", "comments"])
        r = raw.loc[d.mention_id]
        out.append(pd.DataFrame({
            "mention_id": d.mention_id.values, "platform": "twitter" if p == "x" else p,
            "brand": r.brand.fillna("unknown").values, "y": np.log10(d.views.clip(lower=1)).values,
            "log_likes": np.log1p(d.likes).values, "log_comments": np.log1p(d.comments).values,
            "created_at": pd.to_datetime(r.created_at.values, errors="coerce"),
            "fetched_at": pd.to_datetime(d.fetched_at, errors="coerce", utc=True, format="mixed").dt.tz_convert(None).values}))
    return pd.concat(out, ignore_index=True)


def export_source(plats=("twitter", "youtube")):
    f, meta = em.load(ROOT); m = meta.platform.isin(plats).values
    raw = eu.load(ROOT).loc[meta.index[m]]
    return pd.DataFrame({"mention_id": raw.mention_id.astype(str).values, "platform": meta.platform[m].values,
                         "brand": meta.brand[m].fillna("unknown").values, "y": meta.y[m].values,
                         "log_likes": f[m].log_likes.fillna(0).values, "log_comments": f[m].log_comments.fillna(0).values})


# ---------- scoring ----------
def paired_diff(e_new, e_base, g, B=2000, seed=42):
    """MAE(new) - MAE(base) with a paired brand bootstrap 95% CI (negative = new is better)."""
    a, b = np.abs(e_new), np.abs(e_base); u = np.unique(g); idx = {k: np.where(g == k)[0] for k in u}
    rng = np.random.default_rng(seed); dd = []
    for _ in range(B):
        i = np.concatenate([idx[k] for k in rng.choice(u, len(u))]); dd.append(a[i].mean() - b[i].mean())
    return {"diff": round(float(a.mean() - b.mean()), 3), "CI": [round(float(np.percentile(dd, 2.5)), 3), round(float(np.percentile(dd, 97.5)), 3)]}


def full_score(pred, y, g):
    s = fe.score(pred, y, g); e = y - pred
    qt, qp = y >= np.quantile(y, .8), pred >= np.quantile(pred, .8)
    s.update({"median bias (+ = under)": round(float(np.median(e)), 3), "within x2": round(float((np.abs(e) < L2).mean()), 3),
              "top 20% by true y, mean bias": round(float(e[qt].mean()), 3),
              "top 20% by predicted y, mean bias": round(float(e[qp].mean()), 3)})
    br = pd.DataFrame({"g": g, "t": 10 ** y, "p": 10 ** pred}).groupby("g").agg(n=("t", "size"), t=("t", "sum"), p=("p", "sum"))
    br = br[br.n >= 5]; r = np.log10(br.p / br.t)
    s["brand totals (>=5 posts)"] = {"brands": int(len(br)), "median log10 ratio": round(float(r.median()), 3),
                                     "share off >x2": round(float((r.abs() > L2).mean()), 3),
                                     "Kendall": round(float(kendalltau(br.t, br.p)[0]), 3)}
    return s


def fit(src):
    return fe.ridge().fit(src[fe.C2], src.y)


def coefs(m):
    sc, rg = m[0], m[-1]; b = rg.coef_ / sc.scale_
    return {"log_likes": round(float(b[0]), 3), "log_comments": round(float(b[1]), 3), "intercept": round(float(rg.intercept_ - (b * sc.mean_).sum()), 3)}


def score_variants(variants, T, base="export X + YT (base)"):
    out, preds = {}, {}
    for name, src in variants.items():
        m = fit(src); out[name] = {"rows": int(len(src)), "coefficients (raw scale)": coefs(m)}
        for p, d in T.items():
            preds[(name, p)] = m.predict(d[fe.C2]); out[name][p] = full_score(preds[(name, p)], d.y.values, d.brand.values)
    for name in variants:
        if name == base: continue
        for p, d in T.items():
            y, g = d.y.values, d.brand.values
            out[name][p]["paired diff vs base"] = paired_diff(y - preds[(name, p)], y - preds[(base, p)], g)
    return out


def grid(d):
    res = {}
    for k in range(1, 5):
        for c in itertools.combinations(P, k):
            m = tc.fit(d, c)
            res[" + ".join(c)] = {q: tc.stats(d[d.platform == q].y.values, m.predict(d[d.platform == q][fe.C2])) for q in P if q not in c}
    return res


def summarise_grid(res):
    """Per target: median, min and max factor over all training sets that exclude it."""
    s = {}
    for q in P:
        v = [r[q]["MAE"] for r in res.values() if q in r]
        s[q] = {"sets": len(v), "median MAE": round(float(np.median(v)), 3), "median factor": round(float(10 ** np.median(v)), 2),
                "factor range": [round(float(10 ** min(v)), 2), round(float(10 ** max(v)), 2)]}
    return s


def main():
    T = rb.targets()
    ex = export_source(); rr = reread(); rr0 = reread(fill0=True)
    xy = rr[rr.platform.isin(["twitter", "youtube"])]
    variants = {"export X + YT (base)": ex,
                "re-read X + YT, drop missing (primary)": xy,
                "re-read X + YT, fill 0": rr0[rr0.platform.isin(["twitter", "youtube"])],
                "re-read X + YT + TikTok, drop missing": rr,
                "export + re-read X + YT stacked": pd.concat([ex, xy], ignore_index=True)}
    out = {"targets": {p: {"posts": int(len(d)), "brands": int(d.brand.nunique())} for p, d in T.items()},
           "re-read rows (drop missing)": rr.platform.value_counts().to_dict(),
           "re-read rows (fill 0)": rr0.platform.value_counts().to_dict(),
           "re-read X posts not in the export view set": int((~xy[xy.platform == "twitter"].mention_id.isin(ex.mention_id)).sum()),
           "variants": score_variants(variants, T)}
    # E201 grid: export vs re-read TikTok / YouTube / X, the same FB / IG targets
    fbig = pd.concat([d.assign(platform=p) for p, d in T.items()], ignore_index=True)[["platform", "brand", "y"] + fe.C2]
    ex3 = tc.data(); ex3 = ex3[ex3.platform.isin(["tiktok", "youtube", "twitter"])][["platform", "brand", "y"] + fe.C2]
    g_ex = grid(pd.concat([ex3, fbig], ignore_index=True)); g_rr = grid(pd.concat([rr[["platform", "brand", "y"] + fe.C2], fbig], ignore_index=True))
    out["grid (31 training sets)"] = {"export TikTok/YouTube/X": {"summary": summarise_grid(g_ex), "combinations": g_ex},
                                      "re-read TikTok/YouTube/X": {"summary": summarise_grid(g_rr), "combinations": g_rr}}
    (ROOT / "data/processed/reread_transfer_numbers.json").write_text(json.dumps(out, indent=1, default=str))
    for n, v in out["variants"].items():
        print(n, v["rows"], v["coefficients (raw scale)"])
        for p in T: print("  ", p, {k: v[p][k] for k in v[p] if k in ("MAE", "CI", "factor", "median bias (+ = under)", "top 20% by true y, mean bias", "top 20% by predicted y, mean bias", "paired diff vs base")})
    print(json.dumps({k: v["summary"] for k, v in out["grid (31 training sets)"].items()}, indent=1))


if __name__ == "__main__":
    main()
