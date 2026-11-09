"""Monthly English Wikipedia pageviews per brand (E86), a public measure of brand popularity outside the Brand24 file.
Only brand names leave the machine (consent from the user on 2026-11-05); no author names, URLs or post texts.
Step 1: Wikipedia search for '<brand> company' -> top article, checked by name overlap (flag `name_match`).
Step 2: Wikimedia REST pageviews (user agents, all access), monthly, 2024-01 to 2026-10.
Output: data/processed/brand_wikipedia.csv (brand, title, name_match, month, views). Run: .venv/bin/python src/fetch_wikipedia.py"""
import re, time, json, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd
import eda_utils as eu

UA = {"User-Agent": "pwr-digital-media-student-project/0.1 (research; brand popularity)"}
ROOT = Path(__file__).resolve().parents[1]


def get(url):
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r: return json.load(r)
        except Exception as e:
            if getattr(e, "code", None) == 404: return None
            time.sleep(10 * (k + 1))
    return None


norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower().replace("&", "and"))


def search(brand):
    q = urllib.parse.quote(f"{brand} company")
    j = get(f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={q}&format=json&srlimit=3")
    hits = [h["title"] for h in (j or {}).get("query", {}).get("search", [])]
    for t in hits:   # prefer a hit whose title contains the brand name
        base = norm(t.split(" (")[0]); b = norm(brand)
        if b and not t.startswith("List of") and (base == b or (base.startswith(b) and len(b) >= 4) or (b.startswith(base) and len(base) >= 4)): return t, True
    return (hits[0], False) if hits else (None, False)


def pageviews(title):
    t = urllib.parse.quote(title.replace(" ", "_"), safe="")
    j = get(f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{t}/monthly/20240101/20261031")
    return [(i["timestamp"][:4] + "-" + i["timestamp"][4:6], i["views"]) for i in (j or {}).get("items", [])]


if __name__ == "__main__":
    brands = sorted(eu.load(ROOT).brand.unique()); rows = []
    for i, b in enumerate(brands):
        t, ok = search(b); time.sleep(1)
        pv = pageviews(t) if t else []; time.sleep(1)
        rows += [{"brand": b, "title": t, "name_match": ok, "month": m, "views": v} for m, v in pv] or [{"brand": b, "title": t, "name_match": ok, "month": None, "views": None}]
        if i % 50 == 0: print(i, b, "->", t, ok, len(pv), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "data" / "processed" / "brand_wikipedia.csv", index=False); print("done", len(brands))
