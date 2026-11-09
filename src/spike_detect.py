"""E235: can we find posts that suddenly get far more views than the account normally gets (a small account with ~500 views
that has 50k), using only followers and reactions (no views as input)?
Spike A (follower baseline): baseline = out-of-fold linear fit of log10 views on log10 followers per platform (what an account
of this size usually gets). Spike = true views >= 30x (or 100x) the baseline. Scores, all without views:
  - lift = ridge estimate from likes + comments (out of fold) minus the baseline;
  - like rate = log10 likes - log10 followers;
  - estimate alone (ignores followers; plain virality).
AUC and precision in the top 1% / 5% of the score, overall and for accounts with <= 1000 followers.
Spike B (own baseline): posts whose author has >= 2 earlier posts in our data: spike = views >= 10x the median of the author's
earlier posts; how many exist (the check is not possible where there are no earlier posts).
Folds: E55 components, 5-fold. Writes data/processed/spike_detect_numbers.json.
    .venv/bin/python -W ignore src/spike_detect.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, fb_extend as fe, limits as lm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def prec(lab, s, k):
    n = max(int(round(k * len(s))), 1); o = np.argsort(-s)[:n]; return round(float(lab[o].mean()), 3)


def sc(lab, s):
    return {"AUC": round(float(roc_auc_score(lab, s)), 3), "precision top 1%": prec(lab, s, .01), "precision top 5%": prec(lab, s, .05)}


def main():
    f, meta = em.load(ROOT); raw = eu.load(ROOT).loc[f.index]
    comp = lm.components(meta.reset_index(drop=True), raw.reset_index(drop=True)); out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values; y = meta.y[m].reset_index(drop=True).values; fol = meta.author_followers[m].reset_index(drop=True)
        lf = np.log10(fol.where(fol > 0)).values; ok = ~np.isnan(lf); g = pd.Series(comp[m]).values
        likes = f[m].log_likes.fillna(0).reset_index(drop=True).values; C2 = f[m][fe.C2].fillna(0).reset_index(drop=True)
        base = np.full(len(y), np.nan); est = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(y, y, g):
            t = tr[ok[tr]]; base[te] = LinearRegression().fit(lf[t].reshape(-1, 1), y[t]).predict(np.nan_to_num(lf[te], nan=np.nanmedian(lf)).reshape(-1, 1))
            est[te] = fe.ridge().fit(C2.iloc[tr], y[tr]).predict(C2.iloc[te])
        res = y - base; small = ok & (10 ** lf <= 1000)
        r = {"posts": int(len(y)), "accounts <= 1000 followers": int(small.sum()), "median views, <= 1000 followers": int(10 ** np.median(y[small]))}
        for name, thr in (("spike >= 30x the account's usual", np.log10(30)), ("spike >= 100x", 2.0)):
            lab = (res >= thr) & ok; lab = lab.astype(int)
            rr = {"share of posts": round(float(lab[ok].mean()), 4), "posts": int(lab.sum())}
            scores = {"lift = estimate from reactions - account baseline": est - base, "like rate (log likes - log followers)": likes - np.nan_to_num(lf, nan=np.nanmedian(lf)),
                      "estimate alone (ignores followers)": est}
            for sn, s in scores.items():
                if lab[ok].sum() >= 5: rr[sn] = {"all": sc(lab[ok], s[ok])}
                if lab[small].sum() >= 5: rr.setdefault(sn, {})["accounts <= 1000"] = sc(lab[small], s[small])
            r[name] = rr
        ex = small & (res >= 2.0) & (10 ** y >= 20000)
        r["examples: <= 1000 followers and >= 20k views"] = {"count": int(ex.sum()), "median followers": int(10 ** np.median(lf[ex])) if ex.any() else None,
                                                             "median likes": int(np.expm1(np.median(likes[ex]))) if ex.any() else None, "median baseline views": int(10 ** np.median(base[ex])) if ex.any() else None,
                                                             "median views": int(10 ** np.median(y[ex])) if ex.any() else None,
                                                             "reaction lift flags it (top 5% of lift)": round(float((est - base)[ex].__ge__(np.quantile((est - base)[ok], .95)).mean()), 3) if ex.any() else None}
        # B: own baseline
        d = pd.DataFrame({"a": meta.author[m].values, "t": raw.created_at[m].values, "y": y, "lift": est - base}).dropna(subset=["a"]).sort_values("t")
        v = 10 ** d.y.values; keep = []; lab = []
        for a, q in d.groupby("a"):
            vv = 10 ** q.y.values
            for i in range(2, len(q)): keep.append(q.index[i]); lab.append(int(vv[i] >= 10 * np.median(vv[:i])))
        r["B: own baseline (author has >= 2 earlier posts)"] = {"posts with a baseline": len(keep), "spikes >= 10x": int(sum(lab))}
        if sum(lab) >= 5:
            s = d.loc[keep, "lift"].values; r["B: own baseline (author has >= 2 earlier posts)"]["lift"] = sc(np.array(lab), s)
        out[p] = r
    (ROOT / "data/processed/spike_detect_numbers.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
