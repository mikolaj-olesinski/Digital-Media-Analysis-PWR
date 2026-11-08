"""E212: undo regression to the mean in the transfer ridge. (a) 2SLS: export log likes / comments instrument the
re-read ones (posts with both reads), y = re-read views; (b) reliability correction of the likes slope; (c) variance
matching with the factor from brand-grouped CV on the source. Base: E210 primary (ridge on re-read X + YouTube).
Scored on FB / IG and leave-one-platform-out among re-read TikTok / YouTube / X.
Writes data/processed/deshrink_numbers.json.
    .venv/bin/python src/deshrink.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb_extend as fe, results_b as rb  # noqa: E402
from reread_transfer import reread, export_source, paired_diff, full_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C2 = fe.C2


class Lin:
    def __init__(self, b, a): self.b, self.a = np.asarray(b, float), float(a)
    def predict(self, X): return np.asarray(X[C2], float) @ self.b + self.a


def ols(X, y):
    A = np.column_stack([np.asarray(X, float), np.ones(len(X))]); c = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)[0]
    return c[:-1], c[-1]


def tsls(Xend, Z, y):
    A = np.column_stack([np.asarray(Z, float), np.ones(len(Z))])
    Xhat = A @ np.linalg.lstsq(A, np.asarray(Xend, float), rcond=None)[0]
    return ols(Xhat, y)


def reliability(Xrr, Xex, y):
    """Divide each OLS slope by λ = cov(export, re-read) / var(re-read) of that regressor; intercept refitted at the means."""
    b, _ = ols(Xrr, y); lam = np.array([np.cov(Xex[:, j], Xrr[:, j])[0, 1] / np.var(Xrr[:, j], ddof=1) for j in range(Xrr.shape[1])])
    bc = b / lam; return bc, float(np.mean(y) - Xrr.mean(0) @ bc), lam


def vm_factor(src):
    """sd(y) / sd(out-of-fold ridge prediction), brand-grouped 5-fold CV on the source."""
    p = np.zeros(len(src))
    for tr, te in GroupKFold(5).split(src, src.y, src.brand):
        p[te] = fe.ridge().fit(src.iloc[tr][C2], src.y.iloc[tr]).predict(src.iloc[te][C2])
    return float(src.y.std() / np.std(p))


def methods(src, pairs):
    """Fitted predictors on one training set. pairs: posts with both reads in the training platforms."""
    base = fe.ridge().fit(src[C2], src.y)
    Xrr = pairs[[c + "_rr" for c in C2]].values; Xex = pairs[C2].values; y = pairs.y_rr.values
    b_ols, a_ols = ols(Xrr, y); b_iv, a_iv = tsls(Xrr, Xex, y); b_rel, a_rel, lam = reliability(Xrr, Xex, y)
    k = vm_factor(src)
    M = {"ridge, re-read (base)": base.predict,
         "OLS on paired posts": Lin(b_ols, a_ols).predict,
         "2SLS on paired posts (primary)": Lin(b_iv, a_iv).predict,
         "reliability-corrected OLS": Lin(b_rel, a_rel).predict,
         "variance matching": lambda X, base=base, k=k: (lambda p: p.mean() + (p - p.mean()) * k)(base.predict(X))}
    info = {"paired posts": int(len(pairs)), "OLS slopes (likes, comments)": [round(float(v), 3) for v in b_ols],
            "2SLS slopes": [round(float(v), 3) for v in b_iv], "reliability λ": [round(float(v), 3) for v in lam],
            "reliability-corrected slopes": [round(float(v), 3) for v in b_rel],
            "2SLS / OLS likes slope": round(float(b_iv[0] / b_ols[0]), 3), "variance-matching factor": round(k, 3)}
    return M, info


def evaluate(M, d):
    y, g = d.y.values, d.brand.values; P = {n: f(d[C2]) for n, f in M.items()}; out = {}
    for n, p in P.items():
        s = full_score(p, y, g)
        if n != "ridge, re-read (base)": s["paired diff vs base"] = paired_diff(y - p, y - P["ridge, re-read (base)"], g)
        out[n] = s
    return out


def main():
    T = rb.targets(); rr = reread(); ex = export_source(("tiktok", "twitter", "youtube"))
    pr = ex.merge(rr[["mention_id", "y"] + C2], on="mention_id", suffixes=("", "_rr")).rename(columns={"y_rr": "y_rr"})
    out = {}
    M, info = methods(rr[rr.platform.isin(["twitter", "youtube"])], pr[pr.platform.isin(["twitter", "youtube"])])
    out["X + YT re-read -> FB / IG"] = {"fit": info, **{p: evaluate(M, d) for p, d in T.items()}}
    for tgt in ["tiktok", "youtube", "twitter"]:
        tr = [p for p in ["tiktok", "youtube", "twitter"] if p != tgt]
        M, info = methods(rr[rr.platform.isin(tr)], pr[pr.platform.isin(tr)])
        out[f"leave-one-platform-out, target {tgt} (re-read)"] = {"fit": info, tgt: evaluate(M, rr[rr.platform == tgt])}
    (ROOT / "data/processed/deshrink_numbers.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print("==", k, v["fit"])
        for p, r in v.items():
            if p == "fit": continue
            for n, s in r.items():
                print(f"  {p:9s} {n:32s} MAE {s['MAE']} {s['CI']} x{s['factor']} topP {s['top 20% by predicted y, mean bias']} topT {s['top 20% by true y, mean bias']} "
                      f"tot {s['brand totals (>=5 posts)']['median log10 ratio']} K {s['brand totals (>=5 posts)']['Kendall']} {s.get('paired diff vs base', '')}")


if __name__ == "__main__":
    main()
