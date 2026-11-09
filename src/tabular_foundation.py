"""Tabular foundation models against and with the final model (notebooks/models/19_tabular_foundation.ipynb, E47).
TabPFN v2 regressor (weights tabpfn-v2-regressor.ckpt, Apache-2.0 + attribution "Built with PriorLabs-TabPFN") and
TabICL v2 (BSD-3). CPU only; the weights are read from a local cache (TABPFN_MODEL_CACHE_DIR, HF_HOME), data never leave
the machine. Settings fixed before this run: TabPFN median output, n_estimators 8, seeds 0/1/2 averaged."""
import os, time
import numpy as np, pandas as pd
from catboost import CatBoostRegressor
import extra_models as em, reaction_tiers as rt

PLATS = ["tiktok", "youtube", "twitter"]
SEEDS = (0, 1, 2); N_EST = 8
THREADS = int(os.environ.get("OMP_NUM_THREADS", 4))
SINGLE = ["CatBoost (final model)", "TabPFN v2", "TabICL"]
MEANS = {"mean CatBoost + TabPFN": ["CatBoost (final model)", "TabPFN v2"],
         "mean CatBoost + TabICL": ["CatBoost (final model)", "TabICL"],
         "mean of all three": SINGLE}


def catboost(seed):
    return lambda: CatBoostRegressor(loss_function="MAE", random_seed=seed, iterations=300, verbose=0, allow_writing_files=False, thread_count=THREADS)


def tabpfn(seed):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    class Median(TabPFNRegressor):
        def predict(self, X, **k):
            return super().predict(X, output_type="median")

    def make():
        base = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", n_estimators=N_EST, random_state=seed, ignore_pretraining_limits=True)
        return Median(**base.get_params())
    return make


def tabicl(seed):
    from tabicl import TabICLRegressor
    return lambda: TabICLRegressor(device="cpu", n_estimators=N_EST, random_state=seed)


MAKERS = {"CatBoost (final model)": catboost, "TabPFN v2": tabpfn, "TabICL": tabicl}


def run(f, meta):
    """Long table of out-of-fold predictions (seed-averaged) plus run time per model and platform."""
    import torch
    torch.set_num_threads(THREADS)
    out, times = [], []
    for p in PLATS:
        m = (meta.platform == p).values; M = meta[m].reset_index(drop=True)
        X = f[m][em.ENG].fillna(0).reset_index(drop=True)
        common = dict(platform=p, brand=M.brand.values, y=M.y.values, tier=rt.tier(f[m]),
                      zero=((f[m].log_likes.fillna(0) == 0) & (f[m].log_comments.fillna(0) == 0)).values)
        preds = {}
        for k, mk in MAKERS.items():
            t = time.time(); preds[k] = np.mean([em.cv_predict(mk(s), X, M.y, M.brand) for s in SEEDS], axis=0)
            times.append({"platform": p, "model": k, "seconds": time.time() - t, "seeds": len(SEEDS)})
        for k, parts in MEANS.items():
            preds[k] = np.mean([preds[q] for q in parts], axis=0)
        out += [pd.DataFrame({**common, "model": k, "pred": v}) for k, v in preds.items()]
    return pd.concat(out, ignore_index=True), pd.DataFrame(times)


def plot(G, path=None, lang="en"):
    """Paired gain over the final model with 95% brand bootstrap CI, per platform."""
    import matplotlib.pyplot as plt
    import eda_utils as eu
    pl = lang == "pl"
    names = {"TabPFN v2": "TabPFN v2", "TabICL": "TabICL", "mean CatBoost + TabPFN": "średnia CatBoost + TabPFN" if pl else "mean CatBoost + TabPFN",
             "mean CatBoost + TabICL": "średnia CatBoost + TabICL" if pl else "mean CatBoost + TabICL", "mean of all three": "średnia trzech" if pl else "mean of all three"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for x, p in zip(ax, PLATS):
        r = G[(G.platform == p) & (G.subset == "all posts")].reset_index(drop=True); i = np.arange(len(r))[::-1]
        x.errorbar(r.gain, i, xerr=[r.gain - r["CI low"], r["CI high"] - r.gain], fmt="o", color=eu.PCOL[p], capsize=3)
        for j, g, hi in zip(i, r.gain, r["CI high"]):
            x.text(hi + .002, j, f"{g:+.3f}", fontsize=9, va="center")
        x.axvline(0, color="black", lw=.8); x.axvline(.01, color="grey", lw=.6, ls=":")
        x.set_yticks(i); x.set_yticklabels([names[k] for k in r.model]); x.set_title(eu.NAMES[p], fontweight="bold")
        x.set_xlabel("spadek MAE względem modelu końcowego (>0 = lepiej)" if pl else "fall in MAE vs the final model (>0 = better)")
    fig.suptitle("Modele fundamentowe dla tabel: około +0,01 na YouTube, do +0,005 na TikToku i X" if pl else "Tabular foundation models: about +0.01 on YouTube, up to +0.005 on TikTok and X", fontweight="bold"); plt.tight_layout()
    if path: fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()
