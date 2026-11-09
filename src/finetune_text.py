"""Fine-tuned multilingual transformer on TikTok / YouTube / X posts (notebooks/models/23_finetune_text.ipynb, E53).
Variant "row as text": platform, followers, reactions and text in one string -> log10(views).
Variant "text correction": platform + text -> residual of the final CatBoost model (inner out-of-fold, no leakage).
CPU only; weights read from the local Hugging Face cache (HF_HOME, HF_HUB_OFFLINE=1)."""
import os, time, pickle
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
import extra_models as em, reaction_tiers as rt

PLATS = ["tiktok", "youtube", "twitter"]
NAME = "intfloat/multilingual-e5-small"
CFG = dict(epochs=3, batch=32, max_len=128, lr=5e-5, lr_head=1e-3, seed=0)
THREADS = int(os.environ.get("OMP_NUM_THREADS", 4))


def _n(x):
    return "unknown" if pd.isna(x) else f"{int(x):,}"


def texts(f, meta, raw):
    """The two inputs for every post. Numbers are written as in the data (raw counts), links removed."""
    body = (raw.title.fillna("") + " " + raw.content.fillna("")).str.replace(r"https?://\S+", "", regex=True).str.strip().str.slice(0, 600)
    plat = meta.platform.map({"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X"})
    row = ("query: platform " + plat + ". followers " + raw.author_followers.map(_n) + ". likes " + raw.likes.map(_n) +
           ". comments " + raw.comments.map(_n) + ". shares " + raw.shares.map(_n) + ". text: " + body)
    return row.values, ("query: platform " + plat + ". " + body).values


def folds(meta):
    """One brand-grouped fold id per post over the three platforms together."""
    fid = np.zeros(len(meta), int)
    for k, (_, te) in enumerate(GroupKFold(5).split(meta, meta.y, meta.brand)):
        fid[te] = k
    return fid


def catboost_oof(f, meta, fid):
    """Final model per platform on the joint folds; also inner out-of-fold residuals for each outer training fold."""
    pred = np.full(len(meta), np.nan); inner = {}
    for p in PLATS:
        m = np.where((meta.platform == p).values)[0]; X = f.iloc[m][em.ENG].fillna(0).reset_index(drop=True); y = meta.y.values[m]
        for k in range(5):
            tr, te = np.where(fid[m] != k)[0], np.where(fid[m] == k)[0]
            pred[m[te]] = rt.FINAL().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            r = np.full(len(tr), np.nan); Xt, yt, gt = X.iloc[tr].reset_index(drop=True), y[tr], meta.brand.values[m[tr]]
            for itr, ite in GroupKFold(5).split(Xt, yt, gt):
                r[ite] = yt[ite] - rt.FINAL().fit(Xt.iloc[itr], yt[itr]).predict(Xt.iloc[ite])
            inner.setdefault(k, {}).update(dict(zip(m[tr], r)))
    return pred, inner


class Reg:
    def __init__(self, cfg=CFG):
        import torch
        from transformers import AutoTokenizer, AutoModel
        torch.manual_seed(cfg["seed"]); torch.set_num_threads(THREADS)
        self.tok = AutoTokenizer.from_pretrained(NAME); self.enc = AutoModel.from_pretrained(NAME); self.cfg = cfg
        for p in self.enc.embeddings.word_embeddings.parameters(): p.requires_grad = False
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)

    def _emb(self, batch):
        import torch
        b = self.tok(list(batch), truncation=True, max_length=self.cfg["max_len"], padding=True, return_tensors="pt")
        h = self.enc(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1)

    def fit(self, X, y, log=None):
        import torch
        c = self.cfg; self.mu, self.sd = float(np.mean(y)), float(np.std(y)) or 1.0; z = torch.tensor((y - self.mu) / self.sd, dtype=torch.float32)
        enc_p = [p for p in self.enc.parameters() if p.requires_grad]
        opt = torch.optim.AdamW([{"params": enc_p, "lr": c["lr"]}, {"params": self.head.parameters(), "lr": c["lr_head"]}], weight_decay=.01)
        steps = c["epochs"] * int(np.ceil(len(X) / c["batch"])); sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
        rng = np.random.default_rng(c["seed"]); self.enc.train(); t0 = time.time(); s = 0
        for ep in range(c["epochs"]):
            order = rng.permutation(len(X)); tot = 0.0
            for i in range(0, len(X), c["batch"]):
                idx = order[i:i + c["batch"]]
                loss = torch.nn.functional.l1_loss(self.head(self._emb(X[idx])).squeeze(-1), z[idx])
                loss.backward(); torch.nn.utils.clip_grad_norm_(enc_p + list(self.head.parameters()), 1.0); opt.step(); sch.step(); opt.zero_grad()
                tot += float(loss.detach()) * len(idx); s += 1
            if log: log(f"epoch {ep + 1}: train L1 {tot / len(X):.3f} (std units), {time.time() - t0:.0f} s")
        return self

    def predict(self, X):
        import torch
        self.enc.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                out.append(self.head(self._emb(X[i:i + 64])).squeeze(-1).numpy())
        return np.concatenate(out) * self.sd + self.mu


def run(f, meta, raw, cache, log=print):
    """Out-of-fold predictions for both variants; cached per fold so a restarted job continues."""
    store = pickle.load(open(cache, "rb")) if os.path.exists(cache) else {}
    fid = folds(meta); row, txt = texts(f, meta, raw)
    if "catboost" not in store:
        store["catboost"], store["inner"] = catboost_oof(f, meta, fid); store["fid"] = fid; pickle.dump(store, open(cache, "wb"))
    y = meta.y.values
    for var in ("row as text", "text correction"):
        pred = store.get(var, np.full(len(meta), np.nan))
        for k in range(5):
            te = np.where(fid == k)[0]
            if not np.isnan(pred[te]).any(): continue
            tr = np.where(fid != k)[0]; t = time.time()
            if var == "row as text":
                pred[te] = Reg().fit(row[tr], y[tr], log).predict(row[te])
            else:
                r = np.array([store["inner"][k][i] for i in tr])
                pred[te] = Reg().fit(txt[tr], r, log).predict(txt[te])
            store[var] = pred; store.setdefault("seconds", {})[f"{var} fold {k}"] = time.time() - t
            pickle.dump(store, open(cache, "wb")); log(f"{var} fold {k} done in {time.time() - t:.0f} s")
    return store


def table(store, meta):
    cb = store["catboost"]; rows = {"CatBoost (final model)": cb, "row as text": store["row as text"],
                                    "mean CatBoost + row as text": (cb + store["row as text"]) / 2,
                                    "CatBoost + text correction": cb + store["text correction"],
                                    "CatBoost + half text correction": cb + store["text correction"] / 2}
    return pd.concat([pd.DataFrame({"platform": meta.platform.values, "model": k, "brand": meta.brand.values, "y": meta.y.values, "pred": v}) for k, v in rows.items()], ignore_index=True)
