"""E213: Results B re-scored with the same-moment re-read source (E210 primary: ridge on log likes + log comments,
re-read X + YouTube, rows with missing likes / comments dropped), next to the export base. Outputs as results_b.py
plus a 90% CQR band with the E145 protocol (20 splits, 25% of target brands for calibration; FB by follower tier).
Writes data/processed/results_b2_numbers.json (results_b_numbers.json is not touched).
    .venv/bin/python src/results_b2.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, fb_extend as fe, results_b as rb  # noqa: E402
from reread_transfer import reread, export_source, paired_diff  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
L2 = np.log10(2)
SEED, REPS = 42, 20


def qm(a):
    return LGBMRegressor(objective="quantile", alpha=a, n_estimators=300, learning_rate=.05, num_leaves=31,
                         subsample=.9, colsample_bytree=.9, random_state=SEED, verbose=-1)


def point(src, d):
    m = fe.ridge().fit(src[fe.C2], src.y); mc = fe.ridge().fit(src[["log_comments"]], src.y)
    y, g = d.y.values, d.brand.values; pr = m.predict(d[fe.C2]); e = y - pr
    s = fe.score(pr, y, g)
    s.update({"within x2": round(float((np.abs(e) < L2).mean()), 3), "under-predicted": round(float((e > 0).mean()), 3),
              "median bias (+ = under)": round(float(np.median(e)), 3)})
    rng = np.random.default_rng(42); c = []
    for _ in range(200):
        i = rng.permutation(len(e)); c.append(np.abs(e[i[25:]] - np.median(e[i[:25]])).mean())
    s["25-post median shift"] = {"MAE": round(float(np.mean(c)), 3), "factor": round(float(10 ** np.mean(c)), 2)}
    s["comments only"] = fe.score(mc.predict(d[["log_comments"]]), y, g)
    return s, pr


def cqr(src, d, tiers=None):
    lo, hi = qm(.05).fit(src[fe.C2], src.y), qm(.95).fit(src[fe.C2], src.y)
    L, H = lo.predict(d[fe.C2]), hi.predict(d[fe.C2]); y, g = d.y.values, d.brand.values
    rng = np.random.default_rng(SEED); brands = np.array(sorted(set(g)))
    keys = ["all"] + (list(tiers.cat.categories) if tiers is not None else []); cov = {k: [] for k in keys}; wid = []
    for _ in range(REPS):
        cal = np.isin(g, rng.choice(brands, max(1, int(.25 * len(brands))), replace=False)); te = ~cal
        q = float(np.quantile(np.maximum(L[cal] - y[cal], y[cal] - H[cal]), .9))
        inside = (y >= L - q) & (y <= H + q); wid.append(float(np.mean((H + q - L + q)[te])))
        cov["all"].append(inside[te].mean())
        for k in keys[1:]:
            kk = te & (tiers == k).values
            if kk.sum(): cov[k].append(inside[kk].mean())
    out = {"coverage": {k: round(float(np.mean(v)), 3) for k, v in cov.items() if v}, "mean width (log10)": round(float(np.mean(wid)), 3)}
    if tiers is not None: out["posts per tier"] = tiers.value_counts().to_dict()
    return out


def main():
    T = rb.targets(); rr = reread()
    S = {"export X + YT (base)": export_source(), "re-read X + YT (E210 primary)": rr[rr.platform.isin(["twitter", "youtube"])]}
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    fb = fe.fb_frame(ROOT); fb = fb[fb.status == "ok"].dropna(subset=["y"]).reset_index(drop=True)
    fol = pd.to_numeric(raw.loc[fb.mention_id, "author_followers"].values, errors="coerce")
    tiers = pd.Series(pd.cut(fol, [0, 1e4, 1e6, 1e9], labels=["<10k", "10k-1M", ">1M"], right=False))
    assert np.allclose(fb.y.values, T["facebook"].y.values)
    out = {}
    for p, d in T.items():
        y, g = d.y.values, d.brand.values; own = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(d, y, g):
            own[te] = fe.ridge().fit(d.iloc[tr][fe.C2], y[tr]).predict(d.iloc[te][fe.C2])
        mo = float(np.abs(y - own).mean())
        r, P = {"posts": int(len(y)), "brands": int(len(set(g))), "own platform ridge, brand CV": {"MAE": round(mo, 3), "factor": round(10 ** mo, 2)}}, {}
        for n, s in S.items():
            r[n], P[n] = point(s, d)
            r[n]["CQR 90%"] = cqr(s, d, tiers if p == "facebook" else None)
        r["paired diff re-read - export"] = paired_diff(y - P["re-read X + YT (E210 primary)"], y - P["export X + YT (base)"], g)
        out[p] = r
    (ROOT / "data/processed/results_b2_numbers.json").write_text(json.dumps({"results B2": out}, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
