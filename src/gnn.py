"""GraphSAGE on the text-similarity graph (notebooks/models/28_gnn.ipynb, E65).
Graph per platform: each post linked to its 10 nearest neighbours by e5 embedding (cosine), all posts of the platform
(transductive: text only, no labels). Node features: final-model inputs + embedding PCA 16, standardised. Loss only on
training-fold nodes. Compared with the same network without the graph (MLP) and with the final CatBoost."""
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
import extra_models as em, reaction_tiers as rt

PLATS = ["tiktok", "youtube", "twitter"]
CFG = dict(hidden=64, dropout=.2, lr=3e-3, weight_decay=1e-4, epochs=400, k=10, seeds=(0, 1, 2))


def knn_adj(E, k):
    import torch
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(E); _, j = nn.kneighbors(E)
    rows = np.repeat(np.arange(len(E)), k); cols = j[:, 1:].ravel()
    r = np.r_[rows, cols]; c = np.r_[cols, rows]                       # symmetric
    A = torch.sparse_coo_tensor(np.vstack([r, c]), np.ones(len(r), np.float32), (len(E), len(E))).coalesce()
    deg = torch.sparse.sum(A, 1).to_dense().clamp(min=1)
    return torch.sparse_coo_tensor(A.indices(), A.values() / deg[A.indices()[0]], A.shape).coalesce()   # row-normalised mean


class Net:
    def __init__(self, d, graph, cfg, seed):
        import torch
        torch.manual_seed(seed); h = cfg["hidden"]; self.graph = graph
        L = torch.nn.Linear
        self.l1s, self.l1n, self.l2s, self.l2n, self.out = L(d, h), L(d, h), L(h, h), L(h, h), L(h, 1)
        self.params = [p for m in (self.l1s, self.l1n, self.l2s, self.l2n, self.out) for p in m.parameters()]
        self.drop = torch.nn.Dropout(cfg["dropout"])

    def __call__(self, X, A, train=True):
        import torch
        self.drop.train(train); r = torch.relu
        h = r(self.l1s(X) + (self.l1n(torch.sparse.mm(A, X)) if self.graph else 0)); h = self.drop(h)
        h = r(self.l2s(h) + (self.l2n(torch.sparse.mm(A, h)) if self.graph else 0)); h = self.drop(h)
        return self.out(h).squeeze(-1)


def fit_predict(X, y, A, tr, te, graph, cfg=CFG):
    import torch
    torch.set_num_threads(4)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    ym, ys = y[tr].mean(), y[tr].std(); yt = torch.tensor((y - ym) / ys, dtype=torch.float32); trt = torch.tensor(tr)
    preds = []
    for s in cfg["seeds"]:
        net = Net(X.shape[1], graph, cfg, s); opt = torch.optim.AdamW(net.params, cfg["lr"], weight_decay=cfg["weight_decay"])
        for _ in range(cfg["epochs"]):
            loss = torch.nn.functional.l1_loss(net(Xt, A)[trt], yt[trt]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): preds.append(net(Xt, A, False).numpy()[te] * ys + ym)
    return np.mean(preds, axis=0)


def run(f, meta, E, pe):
    out = []
    for p in PLATS:
        m = (meta.platform == p).values; M = meta[m].reset_index(drop=True); y = M.y.values
        Xc = f[m][em.ENG].fillna(0).reset_index(drop=True)
        X = np.hstack([Xc.values, pe.loc[f.index[m]].values]).astype(np.float32)
        A = knn_adj(E.loc[f.index[m]].values, CFG["k"])
        pr = {k: np.full(len(y), np.nan) for k in ["CatBoost (final model)", "MLP (same features, no graph)", "GraphSAGE (text graph)"]}
        for tr, te in GroupKFold(5).split(X, y, M.brand):
            pr["CatBoost (final model)"][te] = rt.FINAL().fit(Xc.iloc[tr], y[tr]).predict(Xc.iloc[te])
            pr["MLP (same features, no graph)"][te] = fit_predict(X, y, A, tr, te, False)
            pr["GraphSAGE (text graph)"][te] = fit_predict(X, y, A, tr, te, True)
        pr["mean CatBoost + GraphSAGE"] = (pr["CatBoost (final model)"] + pr["GraphSAGE (text graph)"]) / 2
        pr["mean CatBoost + MLP (control)"] = (pr["CatBoost (final model)"] + pr["MLP (same features, no graph)"]) / 2
        out += [pd.DataFrame({"platform": p, "model": k, "brand": M.brand.values, "y": y, "pred": v}) for k, v in pr.items()]
    return pd.concat(out, ignore_index=True)
