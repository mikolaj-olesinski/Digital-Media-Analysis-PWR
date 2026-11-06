"""New columns (E121): recency, sequence and text-surface features that were not in the final model. Do they add
anything on top of followers + reactions + flags?

    .venv/bin/python src/new_columns.py
"""
import json
import re
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
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def build(raw):
    t = raw.content.fillna("") + " " + raw.title.fillna("")
    d = pd.DataFrame(index=raw.index)
    d["n_url"] = t.str.count(r"https?://").astype(float)
    d["n_hashtag"] = t.str.count(r"#\w+").astype(float)
    d["n_mention"] = t.str.count(r"@\w+").astype(float)
    d["n_emoji"] = t.map(lambda s: len(EMOJI.findall(s))).astype(float)
    d["has_q"] = t.str.contains(r"\?").astype(float)
    d["has_excl"] = t.str.contains(r"!").astype(float)
    d["caps_share"] = t.map(lambda s: (sum(c.isupper() for c in s) / max(len(s), 1))).astype(float)
    d["digit_share"] = t.map(lambda s: (sum(c.isdigit() for c in s) / max(len(s), 1))).astype(float)
    d["hour"] = raw.created_at.dt.hour.astype(float); d["weekday"] = raw.created_at.dt.weekday.astype(float)
    # recency and sequence within brand x platform
    key = raw.brand.fillna("NA") + "|" + raw.platform
    order = raw.assign(k=key).sort_values(["k", "created_at"])
    d.loc[order.index, "brand_seq_frac"] = order.groupby("k").cumcount().values / np.maximum(order.groupby("k").k.transform("size").values - 1, 1)
    d.loc[order.index, "brand_gap_days"] = np.log1p(order.groupby("k").created_at.diff().dt.total_seconds().div(86400).values)
    # activity in the last 30 days
    for name, col in [("brand_30d", "k"), ("author_30d", "author")]:
        s = pd.Series(0.0, index=raw.index)
        for _, q in raw.assign(g=key if col == "k" else raw.author_name.fillna("NA")).groupby("g"):
            dates = q.created_at.sort_values()
            for i, dt in dates.items():
                s.loc[i] = float(((dates >= dt - pd.Timedelta(days=30)) & (dates <= dt)).sum())
        d[name] = s.values
    return d


def main():
    raw = eu.load(ROOT)
    f, meta = em.load(ROOT)
    new = build(raw)
    cols = ["n_url", "n_hashtag", "n_mention", "n_emoji", "has_q", "has_excl", "caps_share", "digit_share",
            "hour", "weekday", "brand_seq_frac", "brand_gap_days", "brand_30d", "author_30d"]
    Xn = new.loc[f.index, cols].fillna(0)
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values
        X, y, g = f[m][em.ENG].reset_index(drop=True), meta.y[m].reset_index(drop=True), meta.brand[m].reset_index(drop=True)
        X2 = pd.concat([X, Xn[m].reset_index(drop=True)], axis=1)
        row = {}
        for name, Z in [("base", X), ("+new", X2)]:
            pred = np.full(len(y), np.nan)
            for tr, te in GroupKFold(5).split(Z, y, g):
                pred[te] = rt.FINAL().fit(Z.iloc[tr], y.iloc[tr]).predict(Z.iloc[te])
            row[name] = round(float(np.abs(y.values - pred).mean()), 3)
        out[p] = row
        print(p, row)
    (ROOT / "data" / "processed" / "new_columns_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
