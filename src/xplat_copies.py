"""Cross-platform copies (E103): is the same text posted on more than one platform in the Brand24 sample, so that a
platform multiplier could be estimated from it? Normalise the text, require > 30 characters, count texts on >= 2
platforms.

    .venv/bin/python src/xplat_copies.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = eu.load(ROOT)
    t = (d.title.fillna("") + " " + d.content.fillna("")).str.lower()
    t = t.str.replace(r"https?://\S+", "", regex=True).str.replace(r"[^a-z0-9ąćęłńóśźż ]", " ", regex=True).str.split().str.join(" ")
    d = d.assign(norm=t.where(t.str.len() > 30))
    g = d.dropna(subset=["norm"]).groupby("norm").platform.nunique()
    multi = g[g >= 2]
    out = {"texts": int(len(g)), "texts_on_2plus_platforms": int(len(multi)),
           "share": round(float(len(multi) / len(g)), 5),
           "rows_in_those_groups": int(d.norm.isin(multi.index).sum())}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "xplat_copies_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
