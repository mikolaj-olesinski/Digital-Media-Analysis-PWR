"""Shared validation: brand-grouped and temporal splits, MAE on log10, brand-cluster bootstrap CI, split-conformal intervals with coverage."""
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

SEED, N_BOOT = 42, 300


def grouped_cv_predict(model_fn, X, y, groups, n_splits=5):
    """Out-of-fold predictions; a group (brand) is never in train and test at once."""
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        pred[te] = model_fn().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
    return pred


def temporal_split(dates, test_frac=.3):
    """Boolean masks (train, test): the latest test_frac of the rows by date is the test set."""
    cut = dates.sort_values().iloc[int(len(dates) * (1 - test_frac))]
    return (dates < cut).values, (dates >= cut).values


def mae_ci(err, groups, n_boot=N_BOOT, seed=SEED):
    """Mean absolute error and its 95% CI from resampling whole groups (brands) with replacement."""
    rng = np.random.default_rng(seed)
    g = pd.DataFrame({"e": np.abs(err), "g": groups}).groupby("g").e.agg(["sum", "count"]); idx = np.arange(len(g))
    boots = [g.iloc[s]["sum"].sum() / g.iloc[s]["count"].sum() for s in (rng.choice(idx, len(idx)) for _ in range(n_boot))]
    return float(np.abs(err).mean()), *np.percentile(boots, [2.5, 97.5])


def conformal_cv(model_fn, X, y, groups, alpha=.1, n_splits=5, seed=SEED):
    """Split-conformal intervals inside grouped CV: in every fold a quarter of the training brands calibrate the residual quantile.
    Returns point prediction, lower and upper bound; coverage should be about 1 - alpha."""
    rng = np.random.default_rng(seed); pred, lo, hi = (np.full(len(y), np.nan) for _ in range(3))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        brands = pd.unique(groups.iloc[tr]); cal_b = rng.choice(brands, max(1, len(brands) // 4), replace=False)
        cal = tr[groups.iloc[tr].isin(cal_b).values]; fit = np.setdiff1d(tr, cal)
        m = model_fn().fit(X.iloc[fit], y.iloc[fit]); r = np.abs(y.iloc[cal] - m.predict(X.iloc[cal]))
        q = np.quantile(r, min(1, np.ceil((len(r) + 1) * (1 - alpha)) / len(r)))
        pred[te] = m.predict(X.iloc[te]); lo[te], hi[te] = pred[te] - q, pred[te] + q
    return pred, lo, hi


def coverage(y, lo, hi):
    ok = ~np.isnan(lo); return float(((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])).mean()), float((hi[ok] - lo[ok]).mean())
