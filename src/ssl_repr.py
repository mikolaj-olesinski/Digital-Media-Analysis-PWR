"""E216: self-supervised representations (denoising autoencoder with swap noise, SCARF contrastive) learned without views.
In-platform: rt.FINAL on ENG vs ENG + 8-dim code (brand-grouped 5-fold CV, TikTok / YouTube / X).
Transfer: ridge on log likes + log comments vs + code, trained on export X + YouTube (re-read source as sensitivity),
scored on FB / IG. Transductive encoder (all rows, no views) and strict encoder (refit inside each fold / without FB-IG).
Writes data/processed/ssl_repr_numbers.json.
    python -W ignore src/ssl_repr.py"""
import json, pickle, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em, fb_extend as fe, ig_transfer as ig, reaction_tiers as rt, wpb_common as w  # noqa: E402

torch.set_num_threads(4)
EPOCHS, BATCH, CODE = 50, 256, 8
RICH = em.ENG + ["english", "text_len"]


# ---------- encoders ----------
def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, o))


def fit_encoder(X, method, seed=w.SEED):
    """X: standardised float array (rows used for fitting). Returns the encoder (input -> 8-dim code)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    T = torch.tensor(X, dtype=torch.float32); n, d = T.shape
    enc = mlp(d, CODE)
    head = mlp(CODE, d) if method == "DAE" else nn.Sequential(nn.Linear(CODE, 32), nn.ReLU(), nn.Linear(32, 16))
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)
    for _ in range(EPOCHS):
        for b in torch.tensor(rng.permutation(n)).split(BATCH):
            x = T[b]; rate = 0.2 if method == "DAE" else 0.6
            mask = torch.rand_like(x) < rate
            other = T[torch.randint(0, n, (len(b), d)), torch.arange(d)]   # swap noise: value from a random row, same column
            xc = torch.where(mask, other, x)
            if method == "DAE":
                loss = ((head(enc(xc)) - x) ** 2).mean()
            else:
                z1 = nn.functional.normalize(head(enc(x)), dim=1); z2 = nn.functional.normalize(head(enc(xc)), dim=1)
                logits = z1 @ z2.T / 0.5
                loss = nn.functional.cross_entropy(logits, torch.arange(len(b)))
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval()
    return enc


def encode(enc, X):
    with torch.no_grad():
        return enc(torch.tensor(X, dtype=torch.float32)).numpy()


# ---------- data ----------
def build():
    F, M = w.raw_eng(); F = F.fillna(0)
    E = pickle.load(open(w.ROOT / "data/processed/embeddings_e5_cache.pkl", "rb")).values
    pos = pd.Series(np.arange(len(M)), index=M.mention_id.values)
    rr = w.reread()
    R = F.iloc[pos.loc[rr.mention_id].values].copy().reset_index(drop=True)
    R["log_likes"], R["log_comments"], R["na_likes"], R["na_comments"] = rr.log_likes.values, rr.log_comments.values, 0, 0
    rows = pd.concat([F.reset_index(drop=True), R], ignore_index=True)           # export rows 0..12499, then re-read rows
    rows_pos = np.concatenate([np.arange(len(F)), pos.loc[rr.mention_id].values])  # raw position of every row
    return F, M, E, pos, rr, rows, rows_pos


def targets_with_id():
    fb = fe.fb_frame(w.ROOT); fb = fb[fb.status == "ok"].dropna(subset=["y"]).reset_index(drop=True)
    r = ig.frame(); r = r[(r.status == "ok") & r.views.notna() & r.likes.notna()].copy()
    r["y"] = np.log10(r.views.clip(lower=1)); r["log_likes"] = np.log1p(r.likes); r["log_comments"] = np.log1p(r.comments.fillna(0))
    cols = ["mention_id", "y", "brand"] + w.C2
    return {"facebook": fb[cols].astype({"mention_id": str}), "instagram": r[cols].astype({"mention_id": str}).reset_index(drop=True)}


def inputs(rows, rows_pos, E, fit_idx, cols):
    """Standardised encoder inputs for all rows; PCA(16) of e5 and the scaler fitted on fit_idx only."""
    pca = PCA(16, random_state=w.SEED).fit(E[np.unique(rows_pos[fit_idx])])
    Z = np.column_stack([rows[cols].values, pca.transform(E[rows_pos])])
    sc = StandardScaler().fit(Z[fit_idx]); return sc.transform(Z), pca, sc


# ---------- experiments ----------
def in_platform(F, M, E, rows, rows_pos, f, meta):
    allidx = np.arange(len(rows))
    Zt, _, _ = inputs(rows, rows_pos, E, allidx, RICH)
    trans = {m: encode(fit_encoder(Zt, m), Zt[:len(F)]) for m in ["DAE", "SCARF"]}      # codes of the 12,500 export rows
    print("transductive encoders done", flush=True)
    out = {}
    for plat in w.PLAT3:
        mk = (meta.platform == plat).values; ML = meta[mk]; P = ML.index.values          # raw positions
        X = f[mk][em.ENG].fillna(0).reset_index(drop=True); y, g = ML.y.values, ML.brand.values
        Zraw = Zt[P]                                                                       # raw rich inputs (transductive PCA)
        names = ["base ENG", "reference ENG + raw rich inputs", "DAE transductive (primary)", "SCARF transductive", "DAE strict", "SCARF strict"]
        preds = {k: np.zeros(len(y)) for k in names}
        for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
            def fitpred(A):
                return rt.FINAL().fit(A[tr], y[tr]).predict(A[te])
            preds["base ENG"][te] = fitpred(X.values)
            preds["reference ENG + raw rich inputs"][te] = fitpred(np.column_stack([X.values, Zraw[:, len(em.ENG):]]))
            for m in ["DAE", "SCARF"]:
                preds[f"{m} transductive" + (" (primary)" if m == "DAE" else "")][te] = fitpred(np.column_stack([X.values, trans[m][P]]))
            fit_idx = allidx[~np.isin(rows_pos, P[te])]
            Zs, _, _ = inputs(rows, rows_pos, E, fit_idx, RICH)
            for m in ["DAE", "SCARF"]:
                code = encode(fit_encoder(Zs[fit_idx], m), Zs[:len(F)])
                preds[f"{m} strict"][te] = fitpred(np.column_stack([X.values, code[P]]))
            print(plat, "fold", k, flush=True)
        o = {}
        for kk, v in preds.items():
            s = w.score(y, v, g)
            if kk != "base ENG":
                s["gain vs base"] = w.paired_gain(y, preds["base ENG"], v, g)
            o[kk] = s
        out[plat] = o
        print(plat, {kk: o[kk]["MAE"] for kk in o}, flush=True)
    return out


def transfer(F, M, E, pos, rr, rows, rows_pos):
    tg = targets_with_id()
    T = pd.concat([d.assign(platform=p) for p, d in tg.items()], ignore_index=True)
    trows = rows[w.C2].copy(); trows_pos = rows_pos.copy()
    rows_plat = M.platform.values[rows_pos]
    allrows = pd.concat([trows, T[w.C2]], ignore_index=True); allpos = np.concatenate([trows_pos, pos.loc[T.mention_id].values])
    n0 = len(trows)
    vc = M.view_count.values[rows_pos]
    src_ex = np.where((np.arange(n0) < len(F)) & np.isin(rows_plat, ["twitter", "youtube"]) & (vc > 0))[0]
    src_rr = np.where((np.arange(n0) >= len(F)) & np.isin(rows_plat, ["twitter", "youtube"]))[0]
    y_ex = np.log10(M.view_count.values[rows_pos[src_ex]]); y_rr = rr.y.values[src_rr - len(F)]
    tidx = np.arange(n0, len(allrows))
    variants = {"transductive": np.arange(len(allrows)),
                "strict (no FB / IG rows)": np.where(np.isin(np.concatenate([rows_plat, T.platform.values]), ["tiktok", "youtube", "twitter"]) & (np.arange(len(allrows)) < n0))[0]}
    out = {}
    for sname, sidx, sy in [("export X + YouTube source (primary)", src_ex, y_ex), ("re-read X + YouTube source (sensitivity)", src_rr, y_rr)]:
        base = fe.ridge().fit(allrows.iloc[sidx][w.C2], sy); preds = {"base C2": base.predict(T[w.C2])}
        for vname, fit_idx in variants.items():
            Z, _, _ = inputs(allrows, allpos, E, fit_idx, w.C2)
            if vname == "transductive":
                preds["reference C2 + raw e5 PCA16"] = fe.ridge().fit(Z[sidx], sy).predict(Z[tidx])
            for m in ["DAE", "SCARF"]:
                code = encode(fit_encoder(Z[fit_idx], m), Z)
                A = np.column_stack([allrows[w.C2].values, code])
                lab = f"{m} {vname}" + (" (primary)" if m == "DAE" and vname == "transductive" else "")
                preds[lab] = fe.ridge().fit(A[sidx], sy).predict(A[tidx])
        o = {"source rows": int(len(sidx))}
        for p in tg:
            mm = (T.platform == p).values; y, g = T.y.values[mm], T.brand.values[mm]; op = {}
            for k, v in preds.items():
                s = w.score(y, v[mm], g)
                if k != "base C2":
                    s["gain vs base"] = w.paired_gain(y, preds["base C2"][mm], v[mm], g)
                op[k] = s
            o[p] = op
        out[sname] = o
        print(sname, {p: {k: o[p][k]["MAE"] for k in o[p]} for p in tg}, flush=True)
    return out


def main():
    t0 = time.time()
    F, M, E, pos, rr, rows, rows_pos = build()
    f, meta = em.load(w.ROOT)
    res = {"note": "MAE of log10 views; gain = MAE(base) - MAE(variant), + = better, brand bootstrap 300 draws. Encoder MLP 64-64 -> 8, 50 epochs; DAE swap noise 0.2; SCARF corruption 0.6, tau 0.5.",
           "encoder rows": {"export": int(len(F)), "re-read": int(len(rr))}}
    res["transfer"] = transfer(F, M, E, pos, rr, rows, rows_pos)
    res["in-platform"] = in_platform(F, M, E, rows, rows_pos, f, meta)
    ok_in = {p: res["in-platform"][p]["DAE transductive (primary)"]["gain vs base"] for p in w.PLAT3}
    ok_tr = {p: res["transfer"]["export X + YouTube source (primary)"][p]["DAE transductive (primary)"]["gain vs base"] for p in ["facebook", "instagram"]}
    res["verdict"] = {"primary in-platform gains": ok_in, "primary transfer gains": ok_tr,
                      "supported": bool(any(v["gain"] >= 0.01 and v["CI"][0] > 0 for v in ok_in.values()) or any(v["gain"] >= 0.02 and v["CI"][0] > 0 for v in ok_tr.values()))}
    res["runtime seconds"] = round(time.time() - t0)
    (w.ROOT / "data/processed/ssl_repr_numbers.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["verdict"], indent=1))


if __name__ == "__main__":
    main()
