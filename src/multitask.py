"""Multi-task across platforms (E113): a shared trunk with one head per platform, trained jointly, against a
per-platform MLP with the same capacity and against CatBoost. Does sharing help when the platforms have different
levels?

    .venv/bin/python src/multitask.py

Writes data/processed/multitask_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
torch.manual_seed(SEED)
PLAT = ["tiktok", "youtube", "twitter"]


class Shared(nn.Module):
    def __init__(self, d, heads=3, h=32):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.emb = nn.Embedding(heads, 3)
        self.heads = nn.ModuleList([nn.Linear(h + 3, 1) for _ in range(heads)])

    def forward(self, x, p):
        z = torch.cat([self.trunk(x), self.emb(p)], 1)
        return torch.cat([head(z) for head in self.heads], 1)


class Single(nn.Module):
    def __init__(self, d, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x)


def train_shared(X, y, p, epochs=150):
    m = Shared(X.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xt, yt, pt = map(torch.tensor, (X.astype(np.float32), y.astype(np.float32), p.astype(np.int64)))
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        out = m(Xt, pt).gather(1, pt[:, None]).squeeze(1)
        loss = (out - yt).abs().mean(); loss.backward(); opt.step()
    return m


def train_single(X, y, epochs=150):
    m = Single(X.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xt, yt = map(torch.tensor, (X.astype(np.float32), y.astype(np.float32)))
    for _ in range(epochs):
        m.train(); opt.zero_grad(); out = m(Xt).squeeze(1)
        loss = (out - yt).abs().mean(); loss.backward(); opt.step()
    return m


def main():
    d = eu.load(ROOT)
    d = d[d.platform.isin(PLAT) & (pd.to_numeric(d.view_count, errors="coerce") > 0)].copy()
    d["y"] = np.log10(d.view_count)
    d["log_likes"] = np.log1p(d.likes.fillna(0)); d["log_comments"] = np.log1p(d.comments.fillna(0))
    d["log_followers"] = np.log10(d.author_followers.where(d.author_followers > 0)).fillna(0)
    d["na_followers"] = d.author_followers.isna().astype(float)
    F = ["log_likes", "log_comments", "log_followers", "na_followers"]
    d["pcode"] = d.platform.map({p: i for i, p in enumerate(PLAT)})
    X = d[F].fillna(0).values; y = d.y.values; p = d.pcode.values; brands = d.brand.values
    preds = {name: np.zeros(len(d)) for name in ["multi", "single", "cat"]}
    for tr, te in GroupKFold(5).split(X, y, brands):
        m = train_shared(X[tr], y[tr], p[tr])
        with torch.no_grad():
            out = m(torch.tensor(X[te].astype(np.float32)), torch.tensor(p[te].astype(np.int64)))
        preds["multi"][te] = out.gather(1, torch.tensor(p[te][:, None])).squeeze(1).numpy()
        for pl in np.unique(p):
            k = (p[tr] == pl)
            ms = train_single(X[tr][k], y[tr][k])
            with torch.no_grad():
                preds["single"][te[p[te] == pl]] = ms(torch.tensor(X[te][p[te] == pl].astype(np.float32))).squeeze(1).numpy()
        mc = CatBoostRegressor(loss_function="MAE", iterations=300, verbose=0, random_seed=SEED, allow_writing_files=False, thread_count=4)
        mc.fit(X[tr], y[tr]); preds["cat"][te] = mc.predict(X[te])
    out = {}
    for pl in PLAT:
        k = d.platform.values == pl
        out[pl] = {name: round(float(np.abs(preds[name][k] - y[k]).mean()), 3) for name in preds}
        out[pl]["posts"] = int(k.sum())
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "multitask_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
