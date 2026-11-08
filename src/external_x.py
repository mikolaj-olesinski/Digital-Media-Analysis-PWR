"""External X sample via the pay-per-use API (E198), with a hard spending guard.

Reads a bearer token from X_BEARER_TOKEN (or .tools/x_token.txt), runs recent search for brand mentions, expands the
author (followers) and saves the posts locally. The 7-day window of recent search is the limit without Enterprise.

Cost model (X 2026): $0.005 per post read + $0.010 per user read, deduplicated within 24 h. The script refuses to spend
more than MAX_USD.

    X_BEARER_TOKEN=... python src/external_x.py --dry-run
    X_BEARER_TOKEN=... python src/external_x.py --brands 10
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.x.com/2/tweets/search/recent"
POST_COST, USER_COST = 0.005, 0.010


def token():
    t = os.environ.get("X_BEARER_TOKEN", "").strip()
    if not t and (ROOT / ".tools" / "x_token.txt").exists():
        t = (ROOT / ".tools" / "x_token.txt").read_text().strip()
    if not t:
        raise SystemExit("Missing bearer token: set X_BEARER_TOKEN or .tools/x_token.txt")
    return t


def get(url, params, tok):
    import urllib.parse, urllib.request
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params),
                                 headers={"Authorization": f"Bearer {tok}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brands", type=int, default=10)
    ap.add_argument("--max-posts", type=int, default=100)
    ap.add_argument("--max-usd", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    import eda_utils as eu
    brands = list(eu.load(ROOT).brand.value_counts().head(a.brands).index)
    per = max(10, a.max_posts // max(1, len(brands)))
    plan = [{"query": f'"{b}" -is:retweet lang:en', "max_results": min(100, max(10, per))} for b in brands]
    est_posts = sum(p["max_results"] for p in plan)
    est = est_posts * POST_COST + est_posts * USER_COST
    print(json.dumps({"brands": brands, "plan": plan, "estimated_posts": est_posts,
                      "estimated_usd": round(est, 2), "cap_usd": a.max_usd}, indent=1))
    if a.dry_run:
        return
    if est > a.max_usd:
        raise SystemExit(f"Refusing: estimated ${est:.2f} > cap ${a.max_usd:.2f}")
    tok = token()
    rows = []
    for p in plan:
        try:
            r = get(API, {"query": p["query"], "max_results": p["max_results"],
                          "tweet.fields": "created_at,public_metrics,lang",
                          "expansions": "author_id", "user.fields": "public_metrics"}, tok)
        except Exception as e:  # noqa: BLE001
            print("error", p["query"], e); continue
        users = {u["id"]: u for u in (r.get("includes") or {}).get("users", [])}
        for t in r.get("data", []):
            u = users.get(t.get("author_id"), {})
            m = t.get("public_metrics", {})
            rows.append({"id": t.get("id"), "brand_query": p["query"].strip('"').split('"')[0],
                         "created_at": t.get("created_at"), "lang": t.get("lang"),
                         "view_count": m.get("impression_count"), "like_count": m.get("like_count"),
                         "reply_count": m.get("reply_count"), "retweet_count": m.get("retweet_count"),
                         "quote_count": m.get("quote_count"), "bookmark_count": m.get("bookmark_count"),
                         "author_followers": (u.get("public_metrics") or {}).get("followers_count")})
        print(p["query"], len(r.get("data", [])), flush=True)
    if not rows:
        raise SystemExit("no rows")
    d = pd.DataFrame(rows).drop_duplicates(subset=["id"], keep="first")
    d.to_csv(ROOT / "data" / "processed" / "external_x_fetched.csv", index=False)
    out = {"rows": int(len(d)), "posts_returned": int(len(rows)),
           "cost_usd": round(len(rows) * (POST_COST + USER_COST), 2),
           "median_views": float(d.view_count.median()) if "view_count" in d and d.view_count.notna().any() else None,
           "share_followers": round(float(d.author_followers.notna().mean()), 3)}
    print(json.dumps(out, indent=1))
    (ROOT / "data" / "processed" / "external_x_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
