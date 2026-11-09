"""YouTube zeros as missing (E116): 23% of YouTube posts have 0 likes and 0 comments, and they hold a third of the
error (E27). Are those zeros real, or "not collected"? Test four treatments against the current baseline.

    .venv/bin/python src/youtube_zeros.py

Writes data/processed/youtube_zeros_numbers.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em  # noqa: E402
import reaction_tiers as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLS = em.ENG


def mae(pred, y):
    return float(np.abs(y - pred).mean())


def main():
    f, meta = em.load(ROOT)
    m = (meta.platform == "youtube").values
    X0, y, g = f[m][COLS].copy(), meta.y[m].values, meta.brand[m].values
    likes = np.expm1(X0.log_likes.fillna(0)); comments = np.expm1(X0.log_comments.fillna(0))
    zero = ((likes == 0) & (comments == 0)).values
    variants = {}
    variants["base (0 is 0)"] = X0.copy()
    nanv = X0.copy()
    nanv.loc[zero, ["log_likes", "log_comments"]] = np.nan
    variants["0 -> NaN"] = nanv
    nanf = nanv.copy()
    nanf.loc[zero, ["na_likes", "na_comments"]] = 1
    variants["0 -> NaN + flag"] = nanf
    small = X0.copy()
    small.loc[zero, ["log_likes", "log_comments"]] = np.log1p(0.5)
    variants["0 -> 0.5"] = small
    out = {"posts": int(len(y)), "zero_share": round(float(zero.mean()), 3), "variants": {}}
    for name, X in variants.items():
        oof = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, g):
            oof[te] = rt.FINAL().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        out["variants"][name] = {"MAE_all": round(mae(y, oof), 3),
                                 "MAE_zero_reaction": round(mae(y[zero], oof[zero]), 3),
                                 "MAE_nonzero": round(mae(y[~zero], oof[~zero]), 3)}
    # drop zero-reaction posts from training, evaluate on all
    Xd, yd, gd = X0[~zero], y[~zero], g[~zero]
    oof = np.full(len(y), np.nan)
    # brand-grouped by the full test set; train only on non-zero posts of the training brands
    for tr, te in GroupKFold(5).split(X0, y, g):
        tr_brands = set(g[tr])
        keep = np.array([b in tr_brands for b in gd])
        oof[te] = rt.FINAL().fit(Xd[keep], yd[keep]).predict(X0.iloc[te])
    out["variants"]["drop zero posts from training"] = {"MAE_all": round(mae(y, oof), 3),
                                                        "MAE_zero_reaction": round(mae(y[zero], oof[zero]), 3),
                                                        "MAE_nonzero": round(mae(y[~zero], oof[~zero]), 3)}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "youtube_zeros_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
