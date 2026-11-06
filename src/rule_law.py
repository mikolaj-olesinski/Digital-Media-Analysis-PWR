"""Reverse-engineering Brand24's reach rule (E168): the constants k (per reaction) and c(followers) in
reach = max(k * reactions, c(followers) * followers), estimated per platform and per bin, with the share of rows the
reconstructed rule reproduces within 1%.

    .venv/bin/python src/rule_law.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAT = ["tiktok", "youtube", "twitter", "facebook", "instagram"]
BINS = [0, 1_000, 10_000, 100_000, 1_000_000, np.inf]


def main():
    d = eu.load(ROOT)
    d = d[d[eu.REACH] > 0].copy()
    d["reactions"] = d[["likes", "comments", "shares"]].fillna(0).sum(axis=1)
    out = {}
    for p in PLAT:
        q = d[d.platform == p].copy()
        if q.empty:
            continue
        reacted = q[(q.reactions >= 1) & (q[eu.REACH] > 0)].copy()
        k = (reacted[eu.REACH] / reacted.reactions)
        by_react = {}
        for lo, hi, name in [(1, 10, "1-10"), (11, 100, "11-100"), (101, np.inf, "101+")]:
            s = k[(reacted.reactions >= lo) & (reacted.reactions <= hi)]
            if len(s):
                by_react[name] = {"n": int(len(s)), "median_k": round(float(s.median()), 1)}
        # follower ratio, only rows without reactions (the c(followers) branch)
        no_re = q[(q.reactions == 0) & (q.author_followers > 0)].copy()
        no_re["ratio"] = no_re[eu.REACH] / no_re.author_followers
        steps = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            s = no_re[(no_re.author_followers > lo) & (no_re.author_followers <= hi)].ratio
            if len(s) >= 10:
                steps.append({"followers": f"{int(lo)}-{int(hi) if np.isfinite(hi) else 'inf'}",
                              "n": int(len(s)), "median_ratio": round(float(s.median()), 3)})
        # reproduction with the per-platform median k and the per-bin c
        kmed = float(k.median()) if len(k) else np.nan
        c_bin = {s["followers"]: s["median_ratio"] for s in steps}
        pred = []
        for _, r in q.iterrows():
            if r.reactions >= 1 and np.isfinite(kmed):
                kv = kmed * r.reactions
            else:
                kv = 0.0
            if r.author_followers > 0 and steps:
                for s in steps:
                    lo, hi = s["followers"].split("-")
                    if float(r.author_followers) > float(lo) and (hi == "inf" or float(r.author_followers) <= float(hi)):
                        kv = max(kv, s["median_ratio"] * r.author_followers)
                        break
            pred.append(kv)
        pred = np.array(pred)
        ok = np.abs(pred - q[eu.REACH].values) <= 0.01 * q[eu.REACH].values
        out[p] = {"rows": int(len(q)), "median_k": None if not np.isfinite(kmed) else kmed,
                  "k_by_reactions": by_react, "steps_without_reactions": steps,
                  "reproduced_within_1pct": round(float(ok.mean()), 3)}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "rule_law_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
