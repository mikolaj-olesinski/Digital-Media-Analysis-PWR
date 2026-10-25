"""Leave-one-platform-out transfer of a views model (notebooks/models/02_transfer_and_intervals.ipynb).
Only features present on every platform are used: log1p(likes), log1p(comments). Train on two platforms with views, test on the third."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import eda_utils as eu
from validation import grouped_cv_predict, mae_ci, coverage

SEED = 42
FEATURES = ["log_likes", "log_comments"]
KNOWN = ["tiktok", "youtube", "twitter"]
MODELS = {"log-log regression": LinearRegression, "gradient model": lambda: HistGradientBoostingRegressor(max_iter=150, learning_rate=.05, random_state=SEED)}


def load(root):
    d = eu.load(root)
    d["log_likes"], d["log_comments"] = np.log1p(d.likes.fillna(0)), np.log1p(d.comments.fillna(0))  # a missing count is treated as 0 (missing goes with small posts)
    d["y"] = np.log10(d.view_count.where(d.view_count > 0))
    return d


def lopo(d, model_name):
    """Per held-out platform: transfer MAE (+CI), bias, in-platform grouped-CV MAE with the same features, and the signed residuals."""
    rows, resid = [], {}
    for h in KNOWN:
        tr, te = d[d.platform.isin(KNOWN) & (d.platform != h) & d.y.notna()], d[(d.platform == h) & d.y.notna()]
        m = MODELS[model_name]().fit(tr[FEATURES], tr.y); e = te.y - m.predict(te[FEATURES]); resid[h] = e
        own = grouped_cv_predict(MODELS[model_name], te[FEATURES], te.y, te.brand)
        mae, lo, hi = mae_ci(e.values, te.brand.values)
        rows.append({"held-out platform": eu.NAMES[h], "transfer MAE": mae, "CI low": lo, "CI high": hi, "bias (mean error)": e.mean(),
                     "in-platform MAE": np.abs(te.y - own).mean(), "rows": len(te)})
    return pd.DataFrame(rows).set_index("held-out platform"), resid


def cross_platform_intervals(resid, alpha=.1):
    """For each held-out platform the interval half-width comes from the transfer residuals of the OTHER platforms; check its coverage on the held-out one."""
    rows = []
    for h in KNOWN:
        others = np.abs(np.concatenate([resid[o].values for o in KNOWN if o != h])); q = np.quantile(others, 1 - alpha)
        e = resid[h].values; rows.append({"held-out platform": eu.NAMES[h], "half-width (log10)": q, "coverage": (np.abs(e) <= q).mean(), "target": 1 - alpha})
    return pd.DataFrame(rows).set_index("held-out platform")


def predict_unknown(d, model_name, resid, alpha=.1):
    """Fit on all three platforms with views; predict Facebook/Instagram with a +- band from pooled transfer residuals."""
    tr = d[d.platform.isin(KNOWN) & d.y.notna()]; m = MODELS[model_name]().fit(tr[FEATURES], tr.y)
    q = np.quantile(np.abs(np.concatenate([r.values for r in resid.values()])), 1 - alpha)
    out = d[d.platform.isin(["facebook", "instagram"])].copy(); out["pred"] = m.predict(out[FEATURES]); out["lo"], out["hi"] = out.pred - q, out.pred + q
    out["b24"] = np.log10(out[eu.REACH].where(out[eu.REACH] > 0)); return out, q


def plot_transfer(tab):
    fig, ax = plt.subplots(figsize=(7, 3.4)); x = np.arange(len(tab)); w = .38
    ax.bar(x - w / 2, tab["in-platform MAE"], w, color=eu.ALL_COL, label="trained on the same platform")
    ax.bar(x + w / 2, tab["transfer MAE"], w, color=[eu.PCOL[p] for p in KNOWN], label="trained on the other two",
           yerr=[tab["transfer MAE"] - tab["CI low"], tab["CI high"] - tab["transfer MAE"]], ecolor="black", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(tab.index); ax.set_xlabel("held-out platform"); ax.set_ylabel("MAE of log10(views)"); ax.set_title("Transfer across platforms", fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False); plt.tight_layout(); plt.show()


def plot_unknown(out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for a, p in zip(ax, ["facebook", "instagram"]):
        o = out[(out.platform == p) & out.b24.notna()].sort_values("pred"); k = np.linspace(0, len(o) - 1, min(len(o), 400)).astype(int); o = o.iloc[k]
        a.fill_between(range(len(o)), o.lo, o.hi, color=eu.PCOL[p], alpha=.25, label="90% band of our estimate")
        a.plot(range(len(o)), o.pred, color=eu.PCOL[p], lw=1.5, label="our estimate"); a.scatter(range(len(o)), o.b24, s=4, color="black", label="Brand24 reach")
        a.set_xlabel("mentions sorted by our estimate"); a.set_title(eu.NAMES[p], fontweight="bold")
    ax[0].set_ylabel("log10(views or reach)"); ax[1].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False); plt.tight_layout(); plt.show()
