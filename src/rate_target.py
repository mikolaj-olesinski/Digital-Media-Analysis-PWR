"""Target definition (E139): views grow with the age of the check. Does modelling views per day (rate) and multiplying
back by age beat modelling views directly? Also test adding log(age) as a feature.

    .venv/bin/python src/rate_target.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402
import extra_models as em  # noqa: E402
import reaction_tiers as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = eu.load(ROOT)
    f, meta = em.load(ROOT)
    age = ((raw.metrics_updated_at - raw.created_at).dt.total_seconds() / 86400).loc[f.index].reset_index(drop=True).clip(0.1, 3650)
    log_age = np.log10(age)
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        la = log_age[m].reset_index(drop=True)
        res = {}
        for name, Xv, target, back in [
            ("direct", X, y, None),
            ("direct+age", X.assign(log_age=la), y, None),
            ("rate", X.assign(log_age=la), y - la, la),
        ]:
            pred = np.full(len(y), np.nan)
            for tr, te in GroupKFold(5).split(Xv, target, g):
                pv = rt.FINAL().fit(Xv.iloc[tr], target.iloc[tr]).predict(Xv.iloc[te])
                pred[te] = pv if back is None else pv + la.iloc[te].values
            res[name] = round(float(np.abs(y.values - pred).mean()), 3)
        out[p] = res
        print(p, res)
    (ROOT / "data" / "processed" / "rate_target_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
