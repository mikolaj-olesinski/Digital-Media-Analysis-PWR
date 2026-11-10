"""E236: other linear models than ridge on likes + comments, out of platform.
Leave one export platform out (train on the other two) and FB / IG from X + YouTube. Models on log likes + log comments:
OLS (no penalty), ridge (alpha 1, as published), lasso, elastic net, Huber (outlier-robust loss), median (quantile 0.5) regression.
Typical error x10^MAE and paired brand-bootstrap difference vs ridge. Writes data/processed/linear_penalties_numbers.json.
    .venv/bin/python -W ignore src/linear_penalties.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, QuantileRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em, fb_extend as fe, ig_transfer as ig, reread_transfer as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]; C2 = fe.C2
M = {"OLS (no penalty)": lambda: make_pipeline(StandardScaler(), LinearRegression()),
     "ridge (published, alpha 1)": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
     "lasso (alpha 0.01)": lambda: make_pipeline(StandardScaler(), Lasso(alpha=.01)),
     "elastic net (alpha 0.01, l1 0.5)": lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=.01, l1_ratio=.5)),
     "Huber (robust loss)": lambda: make_pipeline(StandardScaler(), HuberRegressor(max_iter=500)),
     "median regression (quantile 0.5)": lambda: make_pipeline(StandardScaler(), QuantileRegressor(quantile=.5, alpha=0, solver="highs"))}


def run(tr, te, ytr, yte, brand):
    out = {}; base = None
    for n, mk in M.items():
        p = mk().fit(tr[C2].fillna(0), ytr).predict(te[C2].fillna(0)); e = yte - p
        r = {"factor": round(float(10 ** np.abs(e).mean()), 2), "MAE": round(float(np.abs(e).mean()), 3)}
        if n.startswith("ridge"): base = e
        out[n] = (r, e)
    for n, (r, e) in out.items():
        if not n.startswith("ridge"): r["vs ridge"] = rt.paired_diff(e, base, brand)
    return {n: r for n, (r, e) in out.items()}


def main():
    f, meta = em.load(ROOT); f = f.copy(); f["platform"] = meta.platform.values; f["y"] = meta.y.values; f["brand"] = meta.brand.fillna("unknown").values; res = {}
    for p in ["tiktok", "youtube", "twitter"]:
        tr, te = f[f.platform != p], f[f.platform == p]; res[p] = run(tr, te, tr.y, te.y.values, te.brand.values)
    s = f[f.platform.isin(["twitter", "youtube"])]
    fb = fe.fb_frame(ROOT); fb = fb[fb.status == "ok"].dropna(subset=["y"]).copy(); res["facebook"] = run(s, fb, s.y, fb.y.values, fb.brand.values)
    r = ig.frame(); r = r[(r.status == "ok") & r.views.notna() & r.likes.notna()].copy()
    r["y"] = np.log10(r.views.clip(lower=1)); r["log_likes"] = np.log1p(r.likes); r["log_comments"] = np.log1p(r.comments.fillna(0))
    res["instagram"] = run(s, r, s.y, r.y.values, r.brand.fillna("unknown").values)
    (ROOT / "data/processed/linear_penalties_numbers.json").write_text(json.dumps(res, indent=1))
    for p, d in res.items(): print(p, {n: (v["factor"], v.get("vs ridge", {}).get("diff")) for n, v in d.items()})


def in_platform():
    """Same nested brand-grouped protocol as the other rows of the in-platform family table (src/tuning.py, 8 fields):
    lasso (L1) and elastic net (L1 + L2) next to ridge (L2). Writes data/processed/linear_penalties_inplatform.json."""
    import tuning as tu
    from scipy.stats import loguniform, uniform
    fam = tu.families
    tu.families = lambda: {"ridge": fam()["ridge"],
                           "lasso": (make_pipeline(StandardScaler(), Lasso(max_iter=5000)), {"lasso__alpha": loguniform(1e-4, 1)}),
                           "elastic net": (make_pipeline(StandardScaler(), ElasticNet(max_iter=5000)), {"elasticnet__alpha": loguniform(1e-4, 1), "elasticnet__l1_ratio": uniform(.1, .8)})}
    f, meta = em.load(ROOT); out = {}
    for plat in ["tiktok", "youtube", "twitter"]:
        t, _, _ = tu.nested(f, meta, plat)
        out[plat] = {r["model"]: {"tuned factor": round(float(10 ** r["tuned MAE"]), 2), "tuned MAE": round(float(r["tuned MAE"]), 3)} for _, r in t.iterrows()}
    (ROOT / "data/processed/linear_penalties_inplatform.json").write_text(json.dumps(out, indent=1)); print(out)


if __name__ == "__main__":
    main()
    in_platform()
