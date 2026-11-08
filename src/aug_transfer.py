"""E215: domain randomization and mixup for nonlinear transfer models (CatBoost rt.FINAL, MLP 64-64) on log likes + log
comments. Leave-one-platform-out over TikTok / YouTube / X (export) and X + YouTube -> FB / IG (results_b.targets).
(a) trend-consistent randomization, (b) like-propensity randomization, (c) cross-platform mixup; base: ridge and the
plain models. Writes data/processed/aug_transfer_numbers.json.
    python -W ignore src/aug_transfer.py"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em, fb_extend as fe, reaction_tiers as rt, results_b as rb, wpb_common as w  # noqa: E402

K = 10
PRIMARY = "CatBoost + (a) trend"


def aug_trend(s, rng):
    beta = Ridge().fit(s[w.C2], s.y).coef_
    X, Y = [s[w.C2].values], [s.y.values]
    for _ in range(K):
        for p in s.platform.unique():
            m = (s.platform == p).values; d = rng.normal(0, 0.5, 2)
            X.append(s[w.C2].values[m] + d); Y.append(s.y.values[m] + beta @ d)
    return np.vstack(X), np.concatenate(Y)


def aug_prop(s, rng):
    X, Y = [s[w.C2].values], [s.y.values]
    for _ in range(K):
        for p in s.platform.unique():
            m = (s.platform == p).values; a = rng.uniform(0.8, 1.2); b = rng.normal(0, 0.3, 2)
            X.append(a * s[w.C2].values[m] + b); Y.append(s.y.values[m])
    return np.vstack(X), np.concatenate(Y)


def aug_mixup(s, rng):
    ps = s.platform.unique(); assert len(ps) == 2
    i0, i1 = np.where(s.platform == ps[0])[0], np.where(s.platform == ps[1])[0]
    n = len(s); a, b = rng.choice(i0, n), rng.choice(i1, n); lam = rng.beta(0.4, 0.4, n)[:, None]
    Xs = lam * s[w.C2].values[a] + (1 - lam) * s[w.C2].values[b]; ys = lam[:, 0] * s.y.values[a] + (1 - lam[:, 0]) * s.y.values[b]
    return np.vstack([s[w.C2].values, Xs]), np.concatenate([s.y.values, ys])


AUG = {"plain": None, "(a) trend": aug_trend, "(b) propensity": aug_prop, "(c) mixup": aug_mixup}


def catboost():
    return rt.FINAL()


class MLP3:
    """Mean of three MLP 64-64 seeds on standardised inputs."""
    def fit(self, X, y):
        self.m = [make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), early_stopping=True, max_iter=500, random_state=s)).fit(X, y) for s in range(3)]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.m], axis=0)


MODELS = {"CatBoost": catboost, "MLP": MLP3}


def run_source(s, targets):
    """s: training rows (platform, y, C2). targets: {name: frame with y, brand, C2}. Returns predictions per model."""
    r = fe.ridge().fit(s[w.C2], s.y); preds = {"ridge": {t: r.predict(d[w.C2]) for t, d in targets.items()}}
    for an, fn in AUG.items():
        if fn is None:
            X, y = s[w.C2].values, s.y.values
        else:
            X, y = fn(s, np.random.default_rng(w.SEED))
        for mn, mk in MODELS.items():
            m = mk().fit(X, y)
            preds[f"{mn} + {an}" if an != "plain" else mn] = {t: m.predict(d[w.C2].values) for t, d in targets.items()}
        print(list(targets), an, "done", flush=True)
    return preds


def main():
    f, meta = em.load(w.ROOT)
    D = pd.DataFrame({"platform": meta.platform.values, "brand": meta.brand.values, "y": meta.y.values,
                      "log_likes": f.log_likes.fillna(0).values, "log_comments": f.log_comments.fillna(0).values})
    res = {"note": "MAE of log10 views. diff vs ridge = MAE(model) - MAE(ridge), positive = worse than ridge (reported as -gain); gain vs plain = MAE(plain) - MAE(variant). Brand bootstrap 300 draws."}
    allp = {}
    for t in w.PLAT3:
        s = D[D.platform != t].reset_index(drop=True); tg = {t: D[D.platform == t].reset_index(drop=True)}
        allp[t] = (tg[t], run_source(s, tg))
    s = D[D.platform.isin(["twitter", "youtube"])].reset_index(drop=True); tg = rb.targets()
    p = run_source(s, tg)
    for t in tg:
        allp[t] = (tg[t], {k: {t: v[t]} for k, v in p.items()})
    within = {}
    for t, (d, p) in allp.items():
        y, g = d.y.values, d.brand.values; out = {}
        for k, v in p.items():
            sc = w.score(y, v[t], g)
            if k != "ridge":
                pg = w.paired_gain(y, p["ridge"][t], v[t], g)
                sc["diff vs ridge (+ = worse)"] = {"diff": -pg["gain"], "CI": [-pg["CI"][1], -pg["CI"][0]]}
                base = k.split(" + ")[0]
                if k != base:
                    sc["gain vs plain " + base] = w.paired_gain(y, p[base][t], v[t], g)
            out[k] = sc
        res[t] = out
        within[t] = {k: round(out[k]["MAE"] - out["ridge"]["MAE"], 4) for k in out if k != "ridge"}
    res["MAE minus ridge, all models"] = within
    n_bad = sum(within[t][PRIMARY] >= 0.03 for t in within)
    res["verdict"] = {"primary": PRIMARY, "targets where primary is >= 0.03 worse than ridge": int(n_bad),
                      "refuted (>= 3 of 5)": bool(n_bad >= 3)}
    (w.ROOT / "data/processed/aug_transfer_numbers.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"MAE": {t: {k: res[t][k]["MAE"] for k in res[t]} for t in allp}, "verdict": res["verdict"]}, indent=1))


if __name__ == "__main__":
    main()
