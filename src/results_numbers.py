"""Collects the numbers shown on the results slides (pitchdeck/results_build.py) into data/processed/results_numbers.json.
Run: python src/results_numbers.py   (about 3 minutes: it reruns the model ladder and the transfer test)"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import eda_utils as eu, modeling as mo, transfer as tf

df = eu.load(ROOT)
out = {"views_known_pct": {p: round(df[df.platform == p].view_count.notna().mean() * 100) for p in eu.PLATFORMS}}
f, meta = mo.load_features(ROOT)
res = __import__("pandas").concat([mo.ladder(f, meta, p) for p in mo.PLATS], ignore_index=True)
out["ladder"] = {p: {r.model: [round(r.MAE, 3), round(r["CI low"], 3), round(r["CI high"], 3)] for _, r in res[res.platform == eu.NAMES[p]].iterrows()} for p in mo.PLATS}
imp = __import__("pandas").concat([mo.importance(f, meta, p) for p in mo.PLATS], axis=1)
out["importance"] = {eu.NAMES[p]: imp[eu.NAMES[p]].round(3).to_dict() for p in mo.PLATS}
d = tf.load(ROOT); tab, resid = tf.lopo(d, "log-log regression")
out["transfer"] = {k: {c: round(float(v), 3) for c, v in r.items()} for k, r in tab.iterrows()}
out["cross_platform_coverage"] = tf.cross_platform_intervals(resid)["coverage"].round(3).to_dict()
est, q = tf.predict_unknown(d, "log-log regression", resid); est = est[est.b24.notna()].assign(inside=lambda x: (x.b24 >= x.lo) & (x.b24 <= x.hi))
out["unknown"] = {"half_width_log10": round(float(q), 3), **{eu.NAMES[p]: {"our_median": round(float(g.pred.median()), 2), "brand24_median": round(float(g.b24.median()), 2),
                  "brand24_inside_pct": round(float(g.inside.mean() * 100)), "n": len(g)} for p, g in est.groupby("platform")}}
json.dump(out, open(ROOT / "data" / "processed" / "results_numbers.json", "w"), indent=1); print(json.dumps(out, indent=1)[:1500])
