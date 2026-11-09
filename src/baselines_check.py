"""E227: red-team baselines for the preprint. Likes multiplier (views = k x likes) and median vs our models,
same platform (TT / YT / X, E55 component folds) and out of platform (FB / IG / Stack Exchange, source export X + YT).
Skill vs median, brand-bootstrap CIs, paired transfer - own-platform diff, per-brand MAE, without the 3 largest brands.
Writes data/processed/baselines_check_numbers.json.
    .venv/bin/python src/baselines_check.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, fb_extend as fe, limits as lm, results_b as rb  # noqa: E402
import reread_transfer as rt, stackexchange_transfer as se, train_combos as tc  # noqa: E402
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
L2 = np.log10(2)


def boot(e, g, B=2000, seed=42):
    a = np.abs(e); u = np.unique(g); idx = {k: np.where(g == k)[0] for k in u}; rng = np.random.default_rng(seed)
    m = [a[np.concatenate([idx[k] for k in rng.choice(u, len(u))])].mean() for _ in range(B)]
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def sc(y, p, g, base=None):
    e = y - p; mae = float(np.abs(e).mean())
    s = {"MAE": round(mae, 3), "factor": round(10 ** mae, 2), "CI": boot(e, g), "within x2": round(float((np.abs(e) < L2).mean()), 3)}
    if base is not None: s["skill vs median"] = round(1 - mae / base, 3)
    return s


def kfit(ll, y):
    """Likes multiplier: k = median(y - log10 likes) over posts with >= 1 like. ll = log1p(likes)."""
    likes = np.expm1(ll); m = likes >= 1
    return float(np.median(y[m] - np.log10(likes[m])))


def kpred(ll, k):
    return np.log10(np.maximum(np.expm1(np.nan_to_num(ll)), 1)) + k


def same_platform():
    f, meta = em.load(ROOT); raw = eu.load(ROOT).loc[f.index]
    comp = lm.components(meta.reset_index(drop=True), raw.reset_index(drop=True))
    cb = pd.read_csv(ROOT / "data/processed/results_preds.csv"); out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values; y = meta.y[m].values; ll = f[m].log_likes.values; g = meta.brand[m].fillna("unknown").values
        c = pd.Series(comp[m]).values; pm, pk = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in GroupKFold(5).split(y, y, c):
            pm[te] = np.median(y[tr]); pk[te] = kpred(ll[te], kfit(np.nan_to_num(ll[tr]), y[tr]))
        pc = cb[cb.platform == p].pred.values; assert np.allclose(cb[cb.platform == p].y.values, y)
        b = float(np.abs(y - pm).mean())
        out[p] = {"posts": int(len(y)), "median": sc(y, pm, g, b), "likes multiplier": sc(y, pk, g, b), "CatBoost (final)": sc(y, pc, g, b),
                  "CatBoost - multiplier": rt.paired_diff(y - pc, y - pk, g)}
    return out


def out_of_platform():
    src = rt.export_source(); model = fe.ridge().fit(src[fe.C2], src.y)
    k = kfit(src.log_likes.values, src.y.values); smed = float(src.y.median())
    T = dict(rb.targets()); s = se.frame(); T["stackexchange"] = s[["y", "brand"] + fe.C2]
    out = {"likes multiplier k (log10 views per like)": round(k, 3)}
    for p, d in T.items():
        d = d.reset_index(drop=True); y, g = d.y.values, d.brand.values
        pr, pk = model.predict(d[fe.C2]), kpred(d.log_likes.values, k)
        po = np.zeros(len(y))
        for tr, te in GroupKFold(min(5, len(set(g)))).split(y, y, g):
            po[te] = fe.ridge().fit(d.iloc[tr][fe.C2], y[tr]).predict(d.iloc[te][fe.C2])
        pmo = np.zeros(len(y))
        for tr, te in GroupKFold(min(5, len(set(g)))).split(y, y, g): pmo[te] = np.median(y[tr])
        b = float(np.abs(y - pmo).mean())
        r = {"posts": int(len(y)), "brands": int(len(set(g))), "sd of y": round(float(y.std()), 3),
             "target median, brand CV (uses target labels)": sc(y, pmo, g, b),
             "source median (X + YT)": sc(y, np.full(len(y), smed), g, b),
             "ridge X + YT (ours)": sc(y, pr, g, b), "likes multiplier X + YT": sc(y, pk, g, b),
             "own-platform ridge, brand CV": sc(y, po, g, b),
             "ours - own platform": rt.paired_diff(y - pr, y - po, g),
             "ours - likes multiplier": rt.paired_diff(y - pr, y - pk, g)}
        be = pd.DataFrame({"g": g, "e": np.abs(y - pr)}).groupby("g").e.agg(["mean", "size"])
        top3 = be.sort_values("size", ascending=False).index[:3]; keep = ~np.isin(g, top3)
        r["per brand (ours)"] = {"median brand MAE": round(float(be["mean"].median()), 3), "median brand factor": round(10 ** float(be["mean"].median()), 2),
                                 "brand factor range": [round(10 ** float(be["mean"].min()), 2), round(10 ** float(be["mean"].max()), 2)],
                                 "share of posts in 3 largest brands": round(float((~keep).mean()), 3),
                                 "without 3 largest brands": sc(y[keep], pr[keep], g[keep])}
        wb = [spearmanr(y[g == b], pr[g == b])[0] for b in np.unique(g) if (g == b).sum() >= 10]
        r["Spearman"] = {"all posts": round(float(spearmanr(y, pr)[0]), 3), "within brand, mean (brands >= 10 posts)": round(float(np.nanmean(wb)), 3), "brands": len(wb)}
        out[p] = r
    return out


def combos_ci():
    """Brand-bootstrap CI for the 'other four platforms' row of the E201 table."""
    D = tc.data(); out = {}
    for p in tc.P:
        m = tc.fit(D, [q for q in tc.P if q != p]); t = D[D.platform == p]
        out[p] = sc(t.y.values, m.predict(t[fe.C2]), t.brand.fillna("unknown").values)
    return out


def main():
    out = {"same platform": same_platform(), "out of platform": out_of_platform(), "combos: other four platforms": combos_ci()}
    (ROOT / "data/processed/baselines_check_numbers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
