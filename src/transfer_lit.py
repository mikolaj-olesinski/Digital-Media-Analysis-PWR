"""Transfer methods suggested by the literature scan (notebooks/models/36_transfer_lit.ipynb, E89-E92).
Leave one platform out on TikTok / YouTube / X, baseline ridge on log likes + log comments trained on the two other platforms.
E89 few-shot recalibration with k target posts (drawn by brand, at most 2 per brand): median shift; empirical-Bayes shrunk
    shift; intercept + slope with the slope shrunk to 1; biased ridge pulled to the source weights; few-shot conformal interval.
E90 no target labels: anchor regression with the platform as anchor (gamma curve), linear group DRO, stabilised weights.
E91 no target labels: input subset chosen by source-to-source transfer (train on one source, score on the other).
E92 brand effect carried over from the brand's posts on the source platforms (target-fold brands kept in training).
Outputs carry statistics only."""
import itertools
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
import extra_models as em, reaction_tiers as rt, platform_map as pm
from validation import SEED

PLATS = ["tiktok", "youtube", "twitter"]
C2 = ["log_likes", "log_comments"]


def base(f, meta):
    M = meta.reset_index(drop=True); X = f[pm.COLS].reset_index(drop=True)
    Z = pm._struct(X)   # inputs (NaN -> -1) + comments per like, shares per like, likes per follower
    return M, Z


def _std(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_fit(X, y, w=None, alpha=1.0):
    return Ridge(alpha=alpha).fit(X, y, sample_weight=w)


def _folds(M, t):
    te_all = (M.platform == t).values; fid = np.full(len(M), -1)
    for k, (_, te) in enumerate(GroupKFold(5).split(np.zeros(te_all.sum()), M.y.values[te_all], M.brand.values[te_all])): fid[np.where(te_all)[0][te]] = k
    return te_all, fid


# ---------- E89 ----------
def _draw(brand, k, rng, per_brand=2):
    """Indices of about k posts from random brands, at most per_brand per brand."""
    order = rng.permutation(np.unique(brand)); pick = []
    for b in order:
        i = np.where(brand == b)[0]; pick.extend(rng.permutation(i)[:per_brand])
        if len(pick) >= k: break
    return np.array(pick[:k])


def fewshot(f, meta, ks=(5, 10, 25, 50, 100), n_rep=200):
    """Zero-shot ridge (all source posts, target brands removed as E13 is not needed here: the k labels are disjoint from the
    scored posts by brand). Variants fitted on the k posts; scored on target posts of other brands."""
    M, Z = base(f, meta); y = M.y.values; plat = M.platform.values; rng = np.random.default_rng(SEED); rows = []
    for t in PLATS:
        s = plat != t; tg = ~s; srcs = [p for p in PLATS if p != t]
        Xs, Xt = _std(Z.loc[s, C2].values, Z.loc[tg, C2].values); m = ridge_fit(Xs, y[s]); ft = m.predict(Xt); yt = y[tg]; bt = M.brand.values[tg]
        # between-platform offset variance from source-to-source transfer; within-platform residual variance
        bias = []
        for a, b in [(srcs[0], srcs[1]), (srcs[1], srcs[0])]:
            ia, ib = plat == a, plat == b; A, B = _std(Z.loc[ia, C2].values, Z.loc[ib, C2].values); bias.append(np.mean(y[ib] - ridge_fit(A, y[ia]).predict(B)))
        tau2 = float(np.mean(np.square(bias)))
        allb = []   # all six ordered platform pairs: what FB / IG would use (here it includes the target, so optimistic)
        for a, b in itertools.permutations(PLATS, 2):
            ia, ib = plat == a, plat == b; A, B = _std(Z.loc[ia, C2].values, Z.loc[ib, C2].values); allb.append(np.mean(y[ib] - ridge_fit(A, y[ia]).predict(B)))
        tau2_all = float(np.mean(np.square(allb))); sig2 = float(np.var(y[s] - m.predict(Xs)))
        Xt1 = np.c_[np.ones(len(Xt)), Xt]; ws = np.r_[m.intercept_, m.coef_]
        rows.append({"platform": t, "k": 0, "method": "zero-shot ridge", "MAE": float(np.abs(yt - ft).mean()), "tau2": tau2, "tau2 all pairs": tau2_all, "sigma2": sig2})
        for k in ks:
            acc = {n: [] for n in ["median shift", "mean shift", "shrunk mean shift (EB)", "shrunk median shift (EB)", "shrunk median shift (EB, tau2 from all six pairs)", "intercept + slope (slope shrunk to 1)", "biased ridge to source weights"]}
            cov, wid, cov0, wid0 = [], [], [], []
            q0 = np.quantile(np.abs(y[s] - m.predict(Xs)), .9)   # exchangeable split-conformal width from the sources (zero-shot)
            for _ in range(n_rep):
                c = _draw(bt, k, rng); rest = ~np.isin(bt, bt[c]); r = yt[c] - ft[c]; e = yt[rest] - ft[rest]
                lam = k / (k + sig2 / max(tau2, 1e-6))
                acc["median shift"].append(np.abs(e - np.median(r)).mean()); acc["mean shift"].append(np.abs(e - r.mean()).mean())
                acc["shrunk mean shift (EB)"].append(np.abs(e - lam * r.mean()).mean()); acc["shrunk median shift (EB)"].append(np.abs(e - lam * np.median(r)).mean())
                lam6 = k / (k + sig2 / tau2_all); acc["shrunk median shift (EB, tau2 from all six pairs)"].append(np.abs(e - lam6 * np.median(r)).mean())
                # y = a + b f, penalty L * (b - 1)^2, L by leave-one-out on the k posts
                fc = ft[c]; best = None
                for L in [.1, 1, 10, 100]:
                    loo = []
                    for j in range(len(c)):
                        o = np.arange(len(c)) != j; a_, b_ = _slope(fc[o], yt[c][o], L * k); loo.append(abs(yt[c][j] - a_ - b_ * fc[j]))
                    if best is None or np.mean(loo) < best[0]: best = (np.mean(loo), L)
                a_, b_ = _slope(fc, yt[c], best[1] * k); acc["intercept + slope (slope shrunk to 1)"].append(np.abs(yt[rest] - a_ - b_ * ft[rest]).mean())
                # biased ridge on [1, x]: min ||y - X w||^2 + L ||w - w_s||^2
                Xc = Xt1[c]; best = None
                for L in [.1, 1, 10, 100]:
                    loo = []
                    for j in range(len(c)):
                        o = np.arange(len(c)) != j; w_ = _biased(Xc[o], yt[c][o], ws, L); loo.append(abs(yt[c][j] - Xc[j] @ w_))
                    if best is None or np.mean(loo) < best[0]: best = (np.mean(loo), L)
                w_ = _biased(Xc, yt[c], ws, best[1]); acc["biased ridge to source weights"].append(np.abs(yt[rest] - Xt1[rest] @ w_).mean())
                # few-shot split conformal 90% after the median shift
                qa = np.quantile(np.abs(r - np.median(r)), min(1, np.ceil((k + 1) * .9) / k)); cov.append(np.mean(np.abs(e - np.median(r)) <= qa)); wid.append(qa)
                cov0.append(np.mean(np.abs(e) <= q0)); wid0.append(q0)
            for n, v in acc.items(): rows.append({"platform": t, "k": k, "method": n, "MAE": float(np.mean(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))})
            rows.append({"platform": t, "k": k, "method": "90% interval: few-shot conformal", "coverage": float(np.mean(cov)), "half-width (×)": float(10 ** np.median(wid))})
            rows.append({"platform": t, "k": k, "method": "90% interval: source conformal, no shift", "coverage": float(np.mean(cov0)), "half-width (×)": float(10 ** np.median(wid0))})
    return pd.DataFrame(rows)


def _slope(f, y, L):
    """min sum (y - a - b f)^2 + L (b - 1)^2, closed form."""
    fm, ym = f.mean(), y.mean(); fc, yc = f - fm, y - ym
    b = (fc @ yc + L) / (fc @ fc + L); return ym - b * fm, b


def _biased(X, y, w0, L):
    return w0 + np.linalg.solve(X.T @ X + L * np.eye(X.shape[1]), X.T @ (y - X @ w0))


# ---------- E90 ----------
def anchor_dro(f, meta, gammas=(0, .5, 1, 2, 5, 10, 100), clips=None):
    """Zero-shot on the target (target-fold brands removed from the sources, as E13). Anchor regression: transform sources by
    W = I - (1 - sqrt(gamma)) P_A with A = source platform, then ridge. Group DRO: weighted ridge with group weights updated by
    exponentiated gradient on group mean squared error. Stabilised importance weights: E88 classifier weights flattened (w^0.25,
    w^0.5) or truncated at the 95th percentile."""
    M, Z = base(f, meta); y = M.y.values; plat = M.platform.values; out = []
    for t in PLATS:
        te_all, fid = _folds(M, t); srcs = [p for p in PLATS if p != t]
        pt = pm._clf().fit(Z, te_all.astype(int)).predict_proba(Z)[:, 1]; w_raw = np.clip(pt / (1 - pt) * ((~te_all).sum() / te_all.sum()), .05, 20)
        names = [f"anchor gamma={g}" for g in gammas] + ["group DRO (squared loss)", "weights: classifier (E88)", "weights: flattened ^0.5", "weights: flattened ^0.25", "weights: truncated at p95"]
        P = {n: np.full(len(M), np.nan) for n in names}
        for k in range(5):
            te = fid == k; tr = (~te_all) & ~M.brand.isin(set(M.brand[te])).values
            Xtr, Xte = _std(Z.loc[tr, C2].values, Z.loc[te, C2].values); ytr = y[tr]; g = plat[tr]
            for gm in gammas:
                Xa, ya = Xtr.copy(), ytr.copy(); X1 = np.c_[np.ones(len(Xa)), Xa]; Xa1 = X1.copy()
                for p in srcs:
                    i = g == p; Xa1[i] = X1[i] - (1 - np.sqrt(gm)) * X1[i].mean(0); ya[i] = ytr[i] - (1 - np.sqrt(gm)) * ytr[i].mean()
                w_ = np.linalg.lstsq(Xa1.T @ Xa1 + 1e-6 * np.eye(3), Xa1.T @ ya, rcond=None)[0]; P[f"anchor gamma={gm}"][te] = np.c_[np.ones(te.sum()), Xte] @ w_
            q = np.ones(len(srcs)) / len(srcs)
            for _ in range(100):
                sw = np.array([q[srcs.index(p)] / (g == p).mean() for p in g]); m = ridge_fit(Xtr, ytr, sw)
                L = np.array([np.mean((ytr[g == p] - m.predict(Xtr[g == p])) ** 2) for p in srcs]); q = q * np.exp(.5 * L); q /= q.sum()
            P["group DRO (squared loss)"][te] = m.predict(Xte)
            wtr = w_raw[tr]
            for n, w in [("weights: classifier (E88)", wtr), ("weights: flattened ^0.5", wtr ** .5), ("weights: flattened ^0.25", wtr ** .25), ("weights: truncated at p95", np.minimum(wtr, np.quantile(wtr, .95)))]:
                P[n][te] = ridge_fit(Xtr, ytr, w).predict(Xte)
        for n, v in P.items():
            out.append(pd.DataFrame({"platform": t, "model": n, "brand": M.brand.values[te_all], "y": y[te_all], "pred": v[te_all]}))
    return pd.concat(out, ignore_index=True)


# ---------- E91 ----------
FEATS = ["log_likes", "log_comments", "log_shares", "log_followers", "cpl", "lpf"]


def subset_search(f, meta):
    """For every subset of FEATS: source-to-source score (train on one source platform, MAE on the other, both directions,
    averaged) and target MAE of pooled ridge on both sources. The chosen subset minimises the source-to-source score."""
    M, Z = base(f, meta); y = M.y.values; plat = M.platform.values; rows = []
    for t in PLATS:
        srcs = [p for p in PLATS if p != t]; tg = plat == t; s = ~tg
        for r in range(1, len(FEATS) + 1):
            for sub in itertools.combinations(FEATS, r):
                sub = list(sub); ss = []
                for a, b in [(srcs[0], srcs[1]), (srcs[1], srcs[0])]:
                    ia, ib = plat == a, plat == b; A, B = _std(Z.loc[ia, sub].values, Z.loc[ib, sub].values); ss.append(np.abs(y[ib] - ridge_fit(A, y[ia]).predict(B)).mean())
                A, B = _std(Z.loc[s, sub].values, Z.loc[tg, sub].values); e = y[tg] - ridge_fit(A, y[s]).predict(B)
                rows.append({"platform": t, "subset": "+".join(sub), "n features": len(sub), "source-to-source MAE": float(np.mean(ss)), "target MAE": float(np.abs(e).mean()), "target bias": float(e.mean())})
    return pd.DataFrame(rows)


# ---------- E92 ----------
def brand_carry(f, meta, kshr=(1, 3, 10)):
    """Brands kept in the sources (a brand's posts on other platforms are known). Ridge on the sources; brand effect = mean
    residual of the brand's source posts, shrunk by n / (n + k); added to the target post's prediction."""
    M, Z = base(f, meta); y = M.y.values; plat = M.platform.values; out = []
    for t in PLATS:
        s = plat != t; tg = ~s; A, B = _std(Z.loc[s, C2].values, Z.loc[tg, C2].values); m = ridge_fit(A, y[s])
        res = pd.Series(y[s] - m.predict(A)).groupby(M.brand.values[s]).agg(["sum", "count"]); pb = m.predict(B); bt = M.brand.values[tg]
        out.append(pd.DataFrame({"platform": t, "model": "ridge, brands kept", "brand": bt, "y": y[tg], "pred": pb}))
        sm, n = res["sum"].reindex(bt).fillna(0).values, res["count"].reindex(bt).fillna(0).values
        for kk in kshr:
            out.append(pd.DataFrame({"platform": t, "model": f"+ brand effect from sources, shrink k={kk}", "brand": bt, "y": y[tg], "pred": pb + sm / (n + kk)}))
    return pd.concat(out, ignore_index=True)
