"""Time-varying context, past-only (E172-E175). Every column for a post at time t uses only other posts whose view check
happened before t (metrics_updated_at < t), so it exists at prediction time and never sees the post's own views.
  E172 brand momentum   the brand's earlier posts on the same platform: mean views per reaction (log10 views - log10
                        reactions) over the last 30 days and over all the past, and their counts. Leak-free re-test of
                        E169, whose brand prior was fitted on the training rows' own views.
  E173 brand buzz       number of mentions of the brand in the whole Brand24 file (all five platforms, no views needed)
                        in the 7 days before t, and its ratio to the brand's 90-day rate.
  E174 platform time    the platform's mean views per reaction over the previous 7 and 30 days, and the platform's own
                        English Wikipedia article attention (post day and 7-day mean vs the 60-day median).
  E175 text cues        regex flags the earlier text columns lacked: giveaway / contest, call to action (link in bio,
                        subscribe, follow), sponsorship disclosure (#ad, sponsored), number of @mentions, URLs, question,
                        other brand names from the file in the text, YouTube title length and clickbait cues.
Evaluated with the final CatBoost + columns, same brand folds, paired brand-bootstrap gains; plus a temporal check (train
before 2026-06-01, test after). FB: buzz and platform attention on the 402 fetched posts with the X + YouTube ridge.
Writes data/processed/time_momentum_numbers.json. Statistics only.
    python src/time_momentum.py"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, history as H, author_next as an, fb_extend as fe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLATS = ["tiktok", "youtube", "twitter"]
DAY = np.timedelta64(1, "D")


def vpr(f):
    return np.log10(np.expm1(f.log_likes.fillna(0)) + np.expm1(f.log_comments.fillna(0)) + 1)


def brand_momentum(meta, raw, f):
    """Columns per post from the same brand's earlier posts on the same platform (check time before t)."""
    r = (meta.y - vpr(f)).values; t = raw.created_at.values; c = raw.metrics_updated_at.values
    out = np.full((len(meta), 4), np.nan)
    for (p, b), ix in meta.groupby(["platform", "brand"]).indices.items():
        for i in ix:
            ok = ix[(c[ix] < t[i]) & (ix != i)]
            if len(ok): out[i, 0] = r[ok].mean(); out[i, 1] = len(ok)
            rec = ok[c[ok] >= t[i] - 30 * DAY]
            if len(rec): out[i, 2] = r[rec].mean(); out[i, 3] = len(rec)
    return pd.DataFrame(out, columns=["brand_vpr_past", "brand_n_past", "brand_vpr_30d", "brand_n_30d"], index=meta.index)


def buzz(brand, t, allraw):
    """Mentions of the brand in the whole file in [t-7d, t) and the ratio to its 90-day rate."""
    T = {b: np.sort(g.values) for b, g in allraw.groupby("brand").created_at}
    n7, n90 = [], []
    for b, ti in zip(brand, t):
        s = T.get(b)
        if s is None or pd.isna(ti): n7.append(np.nan); n90.append(np.nan); continue
        n7.append(np.searchsorted(s, ti) - np.searchsorted(s, ti - 7 * DAY)); n90.append(np.searchsorted(s, ti) - np.searchsorted(s, ti - 90 * DAY))
    n7, n90 = np.array(n7, float), np.array(n90, float)
    return pd.DataFrame({"buzz_7d": np.log10(n7 + 1), "buzz_7d_vs_90d": np.log10((n7 + 1) / (n90 / 90 * 7 + 1))})


def platform_time(meta, raw, f, pw):
    r = (meta.y - vpr(f)).values; t = raw.created_at.values; c = raw.metrics_updated_at.values; out = np.full((len(meta), 4), np.nan)
    S = {p: g.set_index("date").views.sort_index() for p, g in pw.groupby("platform")}
    for p, ix in meta.groupby("platform").indices.items():
        o = np.argsort(c[ix]); cs = c[ix][o]; rs = r[ix][o]; cum = np.r_[0, np.cumsum(rs)]
        for i in ix:
            for k, w in [(0, 7), (1, 30)]:
                a, e = np.searchsorted(cs, t[i] - w * DAY), np.searchsorted(cs, t[i])
                if e - a >= 5: out[i, k] = (cum[e] - cum[a]) / (e - a)
            s = S[p]; d = pd.Timestamp(t[i]).normalize(); base = s[(s.index >= d - pd.Timedelta(days=60)) & (s.index < d)]
            if len(base) >= 30 and d in s.index:
                lb = np.log10(base.median()); out[i, 2] = np.log10(s[d]) - lb; out[i, 3] = np.log10(s[(s.index > d - pd.Timedelta(days=7)) & (s.index <= d)].mean()) - lb
    return pd.DataFrame(out, columns=["plat_vpr_7d", "plat_vpr_30d", "plat_wiki_day", "plat_wiki_week"], index=meta.index)


def text_cues(raw, brands):
    tx = (raw.title.fillna("") + " " + raw.content.fillna("")).str.lower(); title = raw.title.fillna("")
    pat = {"giveaway": r"giveaway|contest|win a|chance to win|sweepstake|raffle", "call_to_action": r"link in (my )?bio|subscribe|follow (us|me)|click (the )?link|shop now|swipe up|use code",
           "disclosure": r"#ad\b|#sponsored|sponsored|paid partnership|#partner|#gifted|affiliate", "news_repost": r"breaking|^rt @|via @|according to|reports?\b"}
    out = pd.DataFrame({k: tx.str.contains(v, regex=True).astype(int) for k, v in pat.items()}, index=raw.index)
    out["n_at_mentions"] = tx.str.count(r"@\w+"); out["has_url"] = tx.str.contains(r"https?://|www\.").astype(int); out["question"] = tx.str.contains(r"\?").astype(int)
    rx = re.compile(r"\b(" + "|".join(re.escape(b.lower()) for b in sorted(brands, key=len, reverse=True) if len(b) >= 4) + r")\b")
    out["n_brands_named"] = tx.map(lambda s: len(set(rx.findall(s))))
    out["title_len"] = title.str.len(); out["title_caps_share"] = title.map(lambda s: sum(ch.isupper() for ch in s) / max(1, sum(ch.isalpha() for ch in s)))
    out["title_clickbait"] = title.str.lower().str.contains(r"you won'?t believe|shocking|insane|!{2,}|\btop \d+|\bwhy\b|\bhow to\b|\bvs\.?\b").astype(int)
    return out


def temporal(f, meta, raw, X, name, cut="2026-06-01", n_boot=200):
    rows = {}
    for p in PLATS:
        m = (meta.platform == p).values; tr = m & (raw.created_at < cut).values; te = m & (raw.created_at >= cut).values
        B = f[em.ENG].fillna(0); BX = pd.concat([B, X.fillna(-1)], axis=1); y = meta.y.values
        mk = lambda: CatBoostRegressor(loss_function="MAE", iterations=300, random_seed=42, verbose=0, allow_writing_files=False, thread_count=4)
        pb = mk().fit(B[tr], y[tr]).predict(B[te]); pp = mk().fit(BX[tr], y[tr]).predict(BX[te])
        a, b = np.abs(y[te] - pb), np.abs(y[te] - pp); g = meta.brand.values[te]; ub = np.unique(g); ix = {k: np.where(g == k)[0] for k in ub}
        rng = np.random.default_rng(42); bs = [np.mean(a[i] - b[i]) for i in (np.concatenate([ix[k] for k in rng.choice(ub, len(ub))]) for _ in range(n_boot))]
        rows[p] = {"base": float(a.mean()), "plus": float(b.mean()), "gain": float(a.mean() - b.mean()), "CI": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "test posts": int(te.sum())}
    return rows


def main():
    f, meta = em.load(ROOT); allraw = eu.load(ROOT); raw = allraw.loc[f.index]
    pw = pd.read_csv(ROOT / "data" / "processed" / "platform_wiki_daily.csv", parse_dates=["date"])
    G = {"+ brand momentum (E172)": brand_momentum(meta, raw, f), "+ brand buzz (E173)": buzz(meta.brand.values, raw.created_at.values, allraw).set_index(f.index),
         "+ platform time (E174)": platform_time(meta, raw, f, pw), "+ text cues (E175)": text_cues(raw, allraw.brand.unique())}
    G["+ all four"] = pd.concat(list(G.values()), axis=1)
    out = {"coverage": {k: X.notna().all(axis=1).groupby(meta.platform.values).mean().round(3).to_dict() for k, X in G.items()},
           "text_cue_rates": G["+ text cues (E175)"].astype(bool).groupby(meta.platform.values).mean().round(3).to_dict("index"), "brand_cv": {}, "temporal": {}}
    for name, X in G.items():
        P = an.with_columns(f, meta, X.fillna(-1), name); S = H.summarise(P)
        out["brand_cv"][name] = {"MAE": S.pivot(index="model", columns="platform", values="MAE").round(4).to_dict(),
                                 "gain": {p: dict(zip(["gain", "CI low", "CI high"], map(float, H.gain_ci(P, p, "final model", name)))) for p in PLATS}}
        out["temporal"][name] = temporal(f, meta, raw, X, name)
        print(name, {p: round(v["gain"], 4) for p, v in out["brand_cv"][name]["gain"].items()}, "temporal", {p: round(v["gain"], 4) for p, v in out["temporal"][name].items()}, flush=True)
    # Facebook: buzz and platform Wikipedia attention exist without FB views
    d = fe.fb_frame(ROOT); d = d[d.status == "ok"].dropna(subset=["y"]).reset_index(drop=True)
    m = meta.platform.isin(["twitter", "youtube"]).values
    srcX = pd.concat([f[m][fe.C2].fillna(0).reset_index(drop=True), G["+ brand buzz (E173)"][m].reset_index(drop=True)], axis=1).fillna(-1)
    tgX = pd.concat([d[fe.C2].fillna(0), buzz(d.brand.values, d.created_at.values, allraw)], axis=1).fillna(-1)
    out["facebook"] = {}
    for name, cols in [("likes + comments", fe.C2), ("+ brand buzz", fe.C2 + ["buzz_7d", "buzz_7d_vs_90d"])]:
        out["facebook"][name] = fe.score(fe.ridge().fit(srcX[cols], meta.y[m].values).predict(tgX[cols]), d.y.values, d.brand.values)
        print("FB", name, out["facebook"][name]["MAE"], flush=True)
    (ROOT / "data" / "processed" / "time_momentum_numbers.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
