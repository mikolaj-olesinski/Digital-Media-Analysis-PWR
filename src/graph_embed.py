"""Graph embeddings (E122): spectral embeddings of the author-brand bipartite graph as features. Earlier graph features
(hand-built) gained at most 0.009 (E12/E29); this tests whether low-rank structural embeddings do better.

    .venv/bin/python src/graph_embed.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import extra_models as em  # noqa: E402
import reaction_tiers as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIM = 8


def main():
    raw = eu.load(ROOT)
    f, meta = em.load(ROOT)
    authors = raw.author_name.fillna("NA").astype(str)
    brands = raw.brand.fillna("NA").astype(str)
    ua = {a: i for i, a in enumerate(pd.unique(authors))}
    ub = {b: i + len(ua) for i, b in enumerate(pd.unique(brands))}
    n = len(ua) + len(ub)
    rows = np.array([ua[a] for a in authors]); cols = np.array([ub[b] for b in brands])
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)); A = A + A.T
    deg = np.asarray(A.sum(1)).ravel(); dinv = 1 / np.sqrt(np.maximum(deg, 1))
    An = csr_matrix(A.multiply(dinv[:, None]).multiply(dinv[None, :]))
    U, S, _ = svds(An, k=DIM)
    order = np.argsort(-S); U = U[:, order] * np.sqrt(S[order])
    emb = pd.DataFrame(U, index=list(ua) + list(ub))
    ea = emb.loc[authors.values].reset_index(drop=True); eb = emb.loc[brands.values].reset_index(drop=True)
    ea.columns = [f"ae{i}" for i in range(DIM)]; eb.columns = [f"be{i}" for i in range(DIM)]
    E = pd.concat([ea, eb], axis=1)
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        X2 = pd.concat([X, E.loc[f.index][m].reset_index(drop=True)], axis=1)
        row = {}
        for name, Z in [("base", X), ("+graph", X2)]:
            pred = np.full(len(y), np.nan)
            for tr, te in GroupKFold(5).split(Z, y, g):
                pred[te] = rt.FINAL().fit(Z.iloc[tr], y.iloc[tr]).predict(Z.iloc[te])
            row[name] = round(float(np.abs(y.values - pred).mean()), 3)
        out[p] = row
        print(p, row)
    (ROOT / "data" / "processed" / "graph_embed_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
