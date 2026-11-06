"""Daily Wikipedia pageviews and Wikidata facts per brand (E170, E171), external brand data beyond E86's monthly views.
Only Wikipedia article titles of brands leave the machine (same consent as E86, 2026-11-05/06); no author names, URLs
or post texts. Uses the articles matched in data/processed/brand_wikipedia.csv minus the hand-rejected ones (WIKI_BAD).
Outputs:
  data/processed/brand_wiki_daily.csv     brand, date, views (daily, user agents, 2024-01-01..2026-10-31)
  data/processed/brand_wikidata.csv       brand, qid, employees, revenue_usd, inception_year, official follower counts
                                          per platform (P8687 with P553 qualifier), country of origin
Run: .venv/bin/python src/fetch_wikimedia_more.py [daily|wikidata|both]"""
import sys, time, json, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
UA = {"User-Agent": "pwr-digital-media-student-project/0.1 (research; brand popularity)"}
# Wikidata items of social platforms used as P553 qualifiers
PLATFORM_QID = {"Q918": "x", "Q866": "youtube", "Q48938223": "tiktok", "Q355": "facebook", "Q209330": "instagram"}


def get(url):
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r: return json.load(r)
        except Exception as e:
            if getattr(e, "code", None) == 404: return None
            time.sleep(10 * (k + 1))
    return None


def articles():
    from context_more import WIKI_BAD
    w = pd.read_csv(ROOT / "data" / "processed" / "brand_wikipedia.csv").drop_duplicates("brand")
    w = w[(w.name_match.astype(str) == "True") & ~w.brand.isin(WIKI_BAD) & w.title.notna()]
    return dict(zip(w.brand, w.title))


def daily():
    rows = []
    for i, (b, t) in enumerate(articles().items()):
        q = urllib.parse.quote(t.replace(" ", "_"), safe="")
        j = get(f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{q}/daily/20240101/20261031")
        rows += [{"brand": b, "date": f"{x['timestamp'][:4]}-{x['timestamp'][4:6]}-{x['timestamp'][6:8]}", "views": x["views"]} for x in (j or {}).get("items", [])]
        if i % 50 == 0: print("daily", i, b, len(rows), flush=True)
        time.sleep(1)
    pd.DataFrame(rows).to_csv(ROOT / "data" / "processed" / "brand_wiki_daily.csv", index=False); print("daily done", len(rows))


def _amount(claims, pid):
    """Latest value of a quantity property (by P585 point in time if present)."""
    best = None
    for c in claims.get(pid, []):
        try: v = float(c["mainsnak"]["datavalue"]["value"]["amount"])
        except (KeyError, TypeError): continue
        t = c.get("qualifiers", {}).get("P585", [{}])[0].get("datavalue", {}).get("value", {}).get("time", "")
        if best is None or t > best[0]: best = (t, v, c)
    return best


def wikidata():
    art = articles(); titles = list(art.values()); qid = {}
    for i in range(0, len(titles), 50):
        j = get("https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&ppprop=wikibase_item&redirects=1&format=json&titles=" + urllib.parse.quote("|".join(titles[i:i + 50])))
        q = (j or {}).get("query", {}); norm = {x["from"]: x["to"] for x in q.get("normalized", []) + q.get("redirects", [])}
        found = {p["title"]: p.get("pageprops", {}).get("wikibase_item") for p in q.get("pages", {}).values()}
        for t in titles[i:i + 50]:
            tt = norm.get(norm.get(t, t), norm.get(t, t)); qid[t] = found.get(tt)
        time.sleep(1)
    ents = {}; ids = sorted({v for v in qid.values() if v})
    for i in range(0, len(ids), 50):
        j = get("https://www.wikidata.org/w/api.php?action=wbgetentities&props=claims&format=json&ids=" + "|".join(ids[i:i + 50]))
        ents.update((j or {}).get("entities", {})); time.sleep(1)
    rows = []
    for b, t in art.items():
        e = ents.get(qid.get(t) or "", {}); cl = e.get("claims", {}); r = {"brand": b, "qid": qid.get(t)}
        emp = _amount(cl, "P1128"); r["employees"] = emp[1] if emp else None
        rev = _amount(cl, "P2139")   # revenue; keep only USD (Q4917) to stay comparable
        r["revenue_usd"] = rev[1] if rev and rev[2]["mainsnak"]["datavalue"]["value"].get("unit", "").endswith("/Q4917") else None
        inc = cl.get("P571", [{}])[0].get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("time")
        r["inception_year"] = int(inc[1:5]) if inc and inc[1:5].isdigit() else None
        co = cl.get("P495", []) or cl.get("P17", [])
        r["country_qid"] = co[0]["mainsnak"].get("datavalue", {}).get("value", {}).get("id") if co else None
        for c in cl.get("P8687", []):   # social media followers, qualifier P553 = platform
            try: v = float(c["mainsnak"]["datavalue"]["value"]["amount"])
            except (KeyError, TypeError): continue
            for qq in c.get("qualifiers", {}).get("P553", []):
                p = PLATFORM_QID.get(qq.get("datavalue", {}).get("value", {}).get("id"))
                if p: r[f"followers_{p}"] = max(v, r.get(f"followers_{p}") or 0)
        rows.append(r)
    pd.DataFrame(rows).to_csv(ROOT / "data" / "processed" / "brand_wikidata.csv", index=False); print("wikidata done", len(rows), "with qid", sum(1 for r in rows if r["qid"]))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("wikidata", "both"): wikidata()
    if what in ("daily", "both"): daily()
