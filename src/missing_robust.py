"""Robustness to missing reactions (E138): the client's data is often incomplete. Corrupt the test set (set likes and
comments to missing, with the flag) at 10/25/50%, and report the MAE against the clean case.

    .venv/bin/python src/missing_robust.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em  # noqa: E402
import reaction_tiers as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0]


def main():
    f, meta = em.load(ROOT)
    rng = np.random.default_rng(SEED)
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        res = {lvl: [] for lvl in LEVELS}
        for tr, te in GroupKFold(5).split(X, y, g):
            mod = rt.FINAL().fit(X.iloc[tr], y.iloc[tr])
            base = X.iloc[te].copy()
            yt = y.iloc[te].values
            for lvl in LEVELS:
                Xc = base.copy()
                if lvl > 0:
                    mask = rng.random(len(Xc)) < lvl
                    Xc.loc[mask, ["log_likes", "log_comments", "na_likes", "na_comments"]] = [np.nan, np.nan, 1, 1]
                res[lvl].append(np.abs(yt - mod.predict(Xc)).mean())
        out[p] = {f"{int(l*100)}%": round(float(np.mean(v)), 3) for l, v in res.items()}
        print(p, out[p])
    (ROOT / "data" / "processed" / "missing_robust_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
