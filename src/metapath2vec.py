"""Metapath2vec on the author-brand-hashtag graph (E127): metapath-guided random walks (A-B-A, B-A-B, A-H-A) plus
skip-gram with negative sampling, then the author/brand/hashtag embeddings as CatBoost features. This is the
heterogeneous-graph method from the literature, beyond the hand-built features (E03/E06/E12/E29), GraphSAGE (E65) and
spectral embeddings (E122).

    .venv/bin/python src/metapath2vec.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import extra_models as em  # noqa: E402
import reaction_tiers as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
DIM = 16
WALKS = 3
LENGTH = 15
WINDOW = 3
NEG = 5
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)


def build(raw):
    a = raw.author_name.fillna("NA").astype(str).values
    b = raw.brand.fillna("NA").astype(str).values
    h = raw.content.fillna("").map(lambda s: [x.lower() for x in re.findall(r"#\w+", s)]).values
    adj = {("A", "B"): {}, ("B", "A"): {}, ("A", "H"): {}, ("H", "A"): {}, ("B", "H"): {}, ("H", "B"): {}}
    def add(t1, t2, x, y):
        adj[(t1, t2)].setdefault(x, []).append(y); adj[(t2, t1)].setdefault(y, []).append(x)
    for i in range(len(raw)):
        add("A", "B", a[i], b[i])
        for tag in h[i]:
            add("A", "H", a[i], tag); add("B", "H", b[i], tag)
    nodes = {}
    for (t1, t2), d in adj.items():
        for k in d: nodes[(t1, k)] = None
    idx = {k: i for i, k in enumerate(nodes)}
    return a, b, h, adj, idx


def walks(adj, idx):
    pats = [("A", "B", "A"), ("B", "A", "B"), ("A", "H", "A")]
    seqs = []
    for pat in pats:
        starts = [k[1] for k in idx if k[0] == pat[0]]
        for s in starts:
            for _ in range(WALKS):
                cur_t, cur = pat[0], s; seq = [idx[(cur_t, cur)]]
                for step in range(1, LENGTH):
                    nt = pat[1] if step % 2 == 1 else pat[0]
                    cand = adj[(cur_t, nt)].get(cur)
                    if not cand: break
                    cur = cand[rng.integers(len(cand))]; cur_t = nt; seq.append(idx[(cur_t, cur)])
                if len(seq) > 2: seqs.append(seq)
    return seqs


def train_sgns(seqs, V):
    pairs = []
    for s in seqs:
        for i, c in enumerate(s):
            for j in range(max(0, i - WINDOW), min(len(s), i + WINDOW + 1)):
                if j != i: pairs.append((c, s[j]))
    P = torch.tensor(pairs, dtype=torch.long)
    emb = nn.Embedding(V, DIM); ctx = nn.Embedding(V, DIM)
    nn.init.uniform_(emb.weight, -0.1, 0.1); nn.init.uniform_(ctx.weight, -0.1, 0.1)
    opt = torch.optim.Adam(list(emb.parameters()) + list(ctx.parameters()), lr=0.05)
    for _ in range(3):
        perm = torch.randperm(len(P))
        for k in range(0, len(P), 4096):
            batch = P[perm[k:k + 4096]]
            pos = (emb(batch[:, 0]) * ctx(batch[:, 1])).sum(1)
            neg = torch.randint(0, V, (len(batch), NEG))
            negs = torch.bmm(ctx(neg), emb(batch[:, 0]).unsqueeze(2)).squeeze(2)
            loss = (-torch.log(torch.sigmoid(pos) + 1e-9) - torch.log(torch.sigmoid(-negs) + 1e-9).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return emb.weight.detach().numpy()


def main():
    raw = eu.load(ROOT)
    a, b, h, adj, idx = build(raw)
    seqs = walks(adj, idx)
    E = train_sgns(seqs, len(idx))
    ea = np.stack([E[idx[("A", x)]] for x in a]); eb = np.stack([E[idx[("B", x)]] for x in b])
    eh = np.stack([np.mean([E[idx[("H", t)]] for t in tags], axis=0) if tags else np.zeros(DIM) for tags in h])
    G = pd.DataFrame(np.concatenate([ea, eb, eh], axis=1), columns=[f"g{i}" for i in range(3 * DIM)])
    f, meta = em.load(ROOT)
    Gf = G.loc[f.index]
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        X2 = pd.concat([X, Gf[m].reset_index(drop=True)], axis=1)
        row = {}
        for name, Z in [("base", X), ("+metapath", X2)]:
            pred = np.full(len(y), np.nan)
            for tr, te in GroupKFold(5).split(Z, y, g):
                pred[te] = rt.FINAL().fit(Z.iloc[tr], y.iloc[tr]).predict(Z.iloc[te])
            row[name] = round(float(np.abs(y.values - pred).mean()), 3)
        out[p] = row
        print(p, row)
    (ROOT / "data" / "processed" / "metapath_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
