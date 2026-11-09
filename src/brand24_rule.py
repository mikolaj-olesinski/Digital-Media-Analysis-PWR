"""Brand24's reach rule reconstructed from the data (notebooks/models/10_brand24_rule.ipynb). Unconfirmed by Brand24: this is what the numbers look like, not their documentation."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
import eda_utils as eu, new_ideas as ni
from validation import mae_ci

SEED = 42; ORDER = ["tiktok", "youtube", "twitter", "facebook", "instagram"]; KNOWN = ["tiktok", "youtube", "twitter"]


def load(root):
    d = eu.load(root).copy(); d["E"] = d.likes.fillna(0) + d.comments.fillna(0) + d.shares.fillna(0); d["F"] = d.author_followers; d["R"] = d[eu.REACH]; return d


def step_constant(F, platform):
    """Constant per follower, read from the rows without reactions: it falls in steps as the account grows, with different steps on X and Facebook."""
    return np.select([F < 1e4, F < 5e4, F < 1e5, F < 1e6], [.10, .09, .07, .05] if platform == "twitter" else [.10, .06, .05, .04], .02)


def per_reaction(d):
    """Most common reach per reaction, by number of reactions and platform."""
    q = d[d.E > 0].assign(k=lambda t: t.R / t.E, bin=lambda t: pd.cut(t.E, [0, 10, 100, np.inf], labels=["1-10", "11-100", ">100"])); return q.groupby(["platform", "bin"], observed=True).k.agg(n="size", mode=lambda s: s.round(-1).mode().iloc[0] if len(s) else np.nan, median="median").round(1).reset_index()


def per_follower(d):
    q = d[(d.E == 0) & (d.F > 0) & (d.R > 0)].assign(c=lambda t: t.R / t.F, bin=lambda t: pd.cut(t.F, [0, 1e4, 5e4, 1e5, 1e6, np.inf], labels=["<10k", "10k-50k", "50k-100k", "100k-1M", ">1M"])); return q.groupby(["platform", "bin"], observed=True).c.agg(n="size", mode=lambda s: s.round(2).mode().iloc[0], median="median").round(3).reset_index()


def rule_fit(d):
    """Share of rows the reconstructed rule reproduces within 2% (or within 1, for small values)."""
    rows = []
    K = {"twitter": lambda E: np.where(E <= 10, 700, 300) * E, "facebook": lambda E: 100 * E, "instagram": lambda E: 20 * E}
    for p, k in K.items():
        q = d[d.platform == p]; F = q.F.fillna(0).values; E = q.E.values; a = np.floor(step_constant(F, p) * F) if p != "instagram" else np.zeros(len(q)); pr = np.maximum(k(E), a); rows.append({"platform": eu.NAMES[p], "rows": len(q), "followers part wins": float((a > k(E)).mean()), "matched by the rule": float((np.abs(pr - q.R.values) <= np.maximum(.02 * q.R.values, 1)).mean()), "with reactions": float((q.E > 0).mean())})
    return pd.DataFrame(rows)


def real_vs_rule(d):
    """Real views per reaction on platforms with views against the reach per reaction Brand24 uses (median, posts with at least one reaction)."""
    rows = []
    for p in ORDER:
        q = d[(d.platform == p) & (d.E > 0)]; v = q[q.view_count > 0]; x = (v.view_count / v.E) if len(v) else pd.Series(dtype=float)
        rows.append({"platform": eu.NAMES[p], "posts": len(q), "reach per reaction": float((q.R / q.E).median()), "views per reaction": float(x.median()) if len(x) else np.nan, "low": float(x.quantile(.25)) if len(x) else np.nan, "high": float(x.quantile(.75)) if len(x) else np.nan})
    return pd.DataFrame(rows)


def plot_real_vs_rule(t, path=None, lang="en"):
    pl = lang == "pl"; fig, ax = plt.subplots(figsize=(8, 5.4)); x = np.arange(len(t)); w = .38; col = [eu.PCOL[p] for p in ORDER]
    ax.bar(x - w / 2, t["views per reaction"].fillna(0), w, color=col, yerr=[(t["views per reaction"] - t.low).fillna(0), (t.high - t["views per reaction"]).fillna(0)], ecolor="black", capsize=3, label="prawdziwe wyświetlenia na reakcję" if pl else "real views per reaction"); ax.bar(x + w / 2, t["reach per reaction"], w, color="#b9b9b4", label="zasięg Brand24 na reakcję" if pl else "Brand24 reach per reaction")
    for i, r in t.iterrows():
        ax.text(i + w / 2, r["reach per reaction"] * 1.08, f"{r['reach per reaction']:.0f}", ha="center", fontsize=9)
        ax.text(i - w / 2, 1.3 if np.isnan(r["views per reaction"]) else r.high * 1.08, ("brak" if pl else "none") if np.isnan(r["views per reaction"]) else f"{r['views per reaction']:.0f}", ha="center", fontsize=9)
    ax.set_yscale("log"); ax.set_ylim(1, 3000); ax.set_xticks(x); ax.set_xticklabels(t.platform); ax.set_xlabel("platforma" if pl else "platform"); ax.set_ylabel("na jedną reakcję (mediana, skala log)" if pl else "per reaction (median, log scale)"); ax.set_title("Ile wyświetleń przypada na jedną reakcję" if pl else "Views per reaction", fontweight="bold"); ax.legend(loc="upper center", bbox_to_anchor=(.5, -.17), frameon=False, ncol=2); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()


def corrected_rule(d):
    """Keep the form of the rule, but learn the two constants from the training brands: views = max(k x reactions, c x followers). Compared with Brand24 reach and one median."""
    rows = []
    for p in KNOWN:
        q = d[(d.platform == p) & (d.view_count > 0)].reset_index(drop=True); y = np.log10(q.view_count).values; g = q.brand.values; pr = np.full(len(q), np.nan); ks, cs = [], []
        for tr, te in GroupKFold(5).split(q, groups=g):
            t = q.iloc[tr]; k = (t.view_count / t.E)[t.E > 0].median(); c = (t.view_count / t.F)[(t.E == 0) & (t.F > 0)].median(); c = c if np.isfinite(c) else (t.view_count / t.F)[t.F > 0].median(); fb = t.view_count[t.E == 0].median(); fb = fb if np.isfinite(fb) else t.view_count.median(); s = q.iloc[te]; v = np.maximum(k * s.E, c * s.F.fillna(0)); v = np.where(v > 0, v, fb); pr[te] = np.log10(np.maximum(v, 1)); ks.append(k); cs.append(c)
        v = mae_ci(y - pr, g); a = {"platform": eu.NAMES[p], "k per reaction": float(np.median(ks)), "c per follower": float(np.median(cs)), "corrected rule MAE": v[0], "low": v[1], "high": v[2]}
        if p == "twitter":
            ok = (q.R > 0).values; b = np.log10(q.R.where(q.R > 0)).values; w = mae_ci(y[ok] - b[ok], g[ok]); dd = ni.paired_diff(y[ok], b[ok], pr[ok], g[ok]); a |= {"Brand24 MAE (same rows)": w[0], "corrected rule MAE (same rows)": float(np.abs(y[ok] - pr[ok]).mean()), "gain": dd[0], "gain low": dd[1], "gain high": dd[2]}
        rows.append(a)
    return pd.DataFrame(rows)


def x_error_split(d):
    """Error of Brand24 reach against views on X, separately for posts with and without reactions, with the real and assumed ratios."""
    q = d[(d.platform == "twitter") & (d.view_count > 0) & (d.R > 0)]; rows = []
    for name, m in [("with reactions", q.E > 0), ("without reactions", q.E == 0)]:
        t = q[m]; e = np.log10(t.view_count) - np.log10(t.R); v = mae_ci(e.values, t.brand.values); rows.append({"posts": name, "n": len(t), "MAE": v[0], "low": v[1], "high": v[2], "bias": float(e.mean())})
    return pd.DataFrame(rows)
