"""Which posts to hand-check (E97): with a budget of k Facebook posts, which selection removes the level error best?

The current hand-check sample is stratified by reactions. Here the same budget is spent by five strategies and the
MAE on the remaining FB posts is compared: random, stratified by reactions, extremes of the estimate, brand-diverse,
and high-leverage (far from the source platforms in feature space). The calibration is one median shift.

    .venv/bin/python src/active_check.py

Writes data/processed/active_check_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import extra_models as em  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
C2 = ["log_likes", "log_comments"]


def frame(root, path, sample_only=False):
    d = pd.read_csv(root / path, dtype={"mention_id": str})
    d = d[d.status == "ok"].copy()
    if sample_only:
        s = pd.read_csv(root / "data" / "manual" / "fbig_sample.csv", dtype={"mention_id": str})
        keep = set(s.loc[(s.platform == "facebook") & (pd.to_numeric(s.views, errors="coerce") > 0), "mention_id"])
        d = d[d.mention_id.isin(keep)]
    raw = eu.load(root).astype({"mention_id": str}).set_index("mention_id").loc[d.mention_id]
    d = d.assign(brand=raw.brand.values, reactions=pd.to_numeric(d.reactions, errors="coerce").fillna(0),
                 log_likes=np.log1p(pd.to_numeric(d.reactions, errors="coerce").fillna(0)),
                 log_comments=np.log1p(pd.to_numeric(d.comments, errors="coerce").fillna(0)),
                 y=np.log10(pd.to_numeric(d.views, errors="coerce"))).dropna(subset=["y"]).reset_index(drop=True)
    return d


def model(root):
    f, meta = em.load(root)
    m = meta.platform.isin(["twitter", "youtube"]).values
    return make_pipeline(StandardScaler(), Ridge()).fit(f[m][C2].fillna(0), meta.y[m]), f, meta


def pick(strategy, d, pred, k, rng):
    n = len(d)
    if strategy == "random":
        return rng.choice(n, k, replace=False)
    if strategy == "reactions":
        tiers = pd.qcut(d.reactions.rank(method="first"), 5, labels=False)
        out = []
        for t in range(5):
            idx = np.where(tiers == t)[0]
            out += list(rng.choice(idx, max(1, k // 5), replace=k // 5 > len(idx)))
        return np.array(out[:k])
    if strategy == "extremes":
        lo, hi = k // 2, k - k // 2
        return np.r_[np.argsort(pred)[:lo], np.argsort(pred)[-hi:]]
    if strategy == "brands":
        brands = d.brand.drop_duplicates().values
        chosen = rng.choice(brands, min(k, len(brands)), replace=False)
        return np.array([rng.choice(np.where(d.brand.values == b)[0]) for b in chosen])[:k]
    if strategy == "leverage":
        z = (d[C2] - d[C2].mean()) / d[C2].std()
        return np.argsort(-(z ** 2).sum(axis=1).values)[:k]
    raise ValueError(strategy)


def main():
    out = {}
    for name, path, sample_only in [("sample_130", "data/processed/fb_fetched.csv", True),
                                    ("all_402", "data/processed/fb_all_fetched.csv", False)]:
        d = frame(ROOT, path, sample_only)
        mod, f, meta = model(ROOT)
        pred = mod.predict(d[C2].fillna(0))
        out[name] = {"posts": int(len(d)), "zero_shot_MAE": round(float(np.abs(d.y.values - pred).mean()), 3),
                     "strategies": {}}
        rng = np.random.default_rng(SEED)
        for k in (5, 10, 25, 50):
            out[name]["strategies"][str(k)] = {}
            for strat in ["random", "reactions", "extremes", "brands", "leverage"]:
                vals = []
                for _ in range(300):
                    i = pick(strat, d, pred, k, rng)
                    cal, rest = i, np.setdiff1d(np.arange(len(d)), i)
                    shift = np.median(d.y.values[cal] - pred[cal])
                    vals.append(np.abs(d.y.values[rest] - (pred[rest] + shift)).mean())
                out[name]["strategies"][str(k)][strat] = round(float(np.mean(vals)), 3)
        print(name, json.dumps(out[name]["strategies"], indent=1))
    (ROOT / "data" / "processed" / "active_check_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
