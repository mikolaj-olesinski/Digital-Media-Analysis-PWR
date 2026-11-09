"""E225: collect Stack Exchange questions that mention our brands (public official API, no key, CC BY-SA).
One request per (brand, site) pair, questions created 2026-04-01 .. 2026-08-31, written to disk after every request
(resumable: pairs already in data/external/stackexchange/requests.jsonl are skipped). Respects `backoff`, stops
cleanly when quota_remaining <= RESERVE. Owner fields other than reputation are dropped before anything is stored.
Tidy file: data/processed/stackexchange_posts.csv (brand name must appear in title or body, whole-word, case-insensitive).
    python src/stackexchange_fetch.py [--max-requests N]"""
import argparse, gzip, json, re, sys, time, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/stackexchange"
API = "https://api.stackexchange.com/2.3/"
FROM, TO = int(datetime(2026, 4, 1, tzinfo=timezone.utc).timestamp()), int(datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp())
RESERVE = 12
FIELDS = ("question.question_id;question.creation_date;question.title;question.body;question.tags;question.view_count;"
          "question.score;question.up_vote_count;question.down_vote_count;question.answer_count;question.comment_count;"
          "question.owner;shallow_user.reputation;question.closed_date;.items;.has_more;.quota_remaining;.backoff")

# brand -> sites where the brand plausibly is a topic (names that are common words are left out on purpose)
TECH = ["Stripe", "Twilio", "Salesforce", "HubSpot", "Adobe", "Autodesk", "Microsoft", "Apple", "PayPal", "Mollie", "Coinbase",
        "Bluesky", "Reddit", "Twitch", "YouTube", "Facebook", "Instagram", "TikTok", "LinkedIn", "Pinterest", "Snapchat",
        "Tesla", "Uber", "Lyft", "Airbnb", "Etsy", "eBay", "Samsung", "AMD", "Dell", "Lenovo", "Asus", "Acer", "Sony",
        "Motorola", "TeamViewer", "Anydesk", "Acronis", "Semrush", "Freshworks", "MailerLite", "Netflix", "Booking.com",
        "Expedia", "Trivago", "Tripadvisor", "Starlink", "DoorDash", "Hootsuite", "Grubhub", "Wayfair", "Newegg", "Amazon"]
SITES = {
    "stackoverflow": [b for b in TECH if b not in ("Amazon", "Apple", "Tesla")] + ["Apple"],
    "superuser": ["Microsoft", "Dell", "HP", "Lenovo", "Asus", "Acer", "AMD", "TeamViewer", "Anydesk", "Acronis", "Samsung", "Sony", "Apple",
                  "Newegg", "Xfinity", "Spectrum", "Starlink", "Netflix", "YouTube", "Adobe", "Motorola", "Amazon"],
    "android": ["Samsung", "Motorola", "Google Fi", "Sony", "Verizon", "T-Mobile", "AT&T", "Snapchat", "TikTok", "Instagram", "Facebook",
                "Netflix", "Hulu", "Disney+", "HBO Max", "Peacock", "YouTube", "Twitch", "Uber", "Lyft", "Mint Mobile", "Boost Mobile",
                "Cricket Wireless", "Tracfone", "Straight Talk", "Amazon", "LinkedIn", "Pinterest", "Reddit", "Spotify"],
    "apple": ["Apple TV+", "Netflix", "YouTube", "Instagram", "Facebook", "Microsoft", "Samsung", "Verizon", "T-Mobile", "AT&T", "Sony", "Amazon", "Disney+", "Hulu"],
    "gaming": ["Twitch", "Sony", "Microsoft", "Netflix", "GameStop", "Amazon", "Samsung", "Asus", "Lenovo", "Reddit", "YouTube"],
    "travel": ["Airbnb", "Booking.com", "Expedia", "Kayak", "Trivago", "Tripadvisor", "Priceline", "Orbitz", "Hilton", "Marriott", "Hyatt", "IHG",
               "Wyndham", "Delta", "JetBlue", "Southwest", "Allegiant", "Frontier", "Hertz", "Avis", "Alamo", "Uber", "Lyft", "Amazon", "BWH Hotels"],
    "mechanics": ["Ford", "Toyota", "Honda", "Kia", "Hyundai", "Subaru", "Nissan", "Mazda", "Audi", "BMW", "Volkswagen", "Mercedes-Benz", "Tesla",
                  "Chevrolet", "Jeep", "Dodge", "Buick", "GMC", "Lexus", "Acura", "Cadillac", "Chrysler", "Bosch", "AutoZone", "Advance Auto Parts"],
    "money": ["Chase", "Capital One", "Wells Fargo", "Bank of America", "Citibank", "PayPal", "Coinbase", "eToro", "Geico", "State Farm", "Progressive",
              "USAA", "Truist", "TD Bank", "U.S. Bank", "PNC Bank", "Regions Bank", "Fifth Third Bank", "Huntington Bank", "Citizens Bank", "Barclays",
              "Prudential", "MetLife", "Aetna", "Cigna", "Humana", "Liberty Mutual", "Nationwide", "John Hancock", "Mint Mobile", "Stripe", "Amazon", "Walmart", "Costco"],
    "diy": ["The Home Depot", "Lowe's", "Menards", "Ace Hardware", "Whirlpool", "Bosch", "Electrolux", "Haier", "Costco", "Tractor Supply Company", "Dell", "Amazon"],
    "webapps": ["Facebook", "Instagram", "LinkedIn", "Pinterest", "Reddit", "Twitch", "YouTube", "Netflix", "Hulu", "Etsy", "eBay", "Airbnb", "Amazon",
                "Hootsuite", "HubSpot", "MailerLite", "Salesforce", "Bluesky", "TikTok", "Snapchat", "Nextdoor", "Truth Social", "Grubhub", "Uber Eats", "DoorDash", "Disney+", "HBO Max"],
}


def pairs():
    brands = set(eu.load(ROOT).brand.unique())
    amb = set(pd.read_csv(ROOT / "data/processed/brand_industry.csv").query("ambiguous_name == 'yes'").brand)
    out = []
    for site, bl in SITES.items():
        for b in dict.fromkeys(bl):
            if b in brands and b not in amb: out.append((b, site))
    # one site per round first (breadth over brands), then the rest
    return sorted(out, key=lambda t: (list(SITES[t[1]]).index(t[0]) // 4, list(SITES).index(t[1])))


def get(path, **p):
    url = API + path + "?" + urllib.parse.urlencode(p)
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": "amc-research/1.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        b = r.read()
        if r.headers.get("Content-Encoding") == "gzip": b = gzip.decompress(b)
    return json.loads(b)


def scrub(it):
    o = it.pop("owner", None) or {}
    it["owner_reputation"] = o.get("reputation")
    return it


def collect(max_requests):
    RAW.mkdir(parents=True, exist_ok=True)
    log = RAW / "requests.jsonl"
    done = {(r["brand"], r["site"]) for r in map(json.loads, log.read_text().splitlines())} if log.exists() else set()
    fp = RAW / "filter.txt"
    if not fp.exists():
        j = get("filters/create", include=FIELDS, base="none", unsafe="false"); fp.write_text(j["items"][0]["filter"]); print("filter quota", j.get("quota_remaining"))
    flt = fp.read_text().strip(); n = 0
    for brand, site in pairs():
        if (brand, site) in done: continue
        if n >= max_requests: break
        j = get("search/advanced", q=brand, site=site, fromdate=FROM, todate=TO, pagesize=100, sort="creation", order="desc", filter=flt)
        n += 1
        for it in j.get("items", []): scrub(it)
        rec = {"brand": brand, "site": site, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "has_more": j.get("has_more"), "quota_remaining": j.get("quota_remaining"), "items": j.get("items", [])}
        with log.open("a") as f: f.write(json.dumps(rec) + "\n")
        print(f"{n:3d} {site:13s} {brand:20s} items={len(rec['items']):3d} more={rec['has_more']} quota={rec['quota_remaining']}", flush=True)
        if j.get("quota_remaining", 0) <= RESERVE: print("quota reserve reached, stopping"); break
        time.sleep(max(1.0, float(j.get("backoff", 0)) + 1))
    return n


def tidy():
    rows = []
    for r in map(json.loads, (RAW / "requests.jsonl").read_text().splitlines()):
        pat = re.compile(r"(?<![\w])" + re.escape(r["brand"]).replace(r"\ ", r"[\s-]?") + r"(?![\w])", re.I)
        for q in r["items"]:
            if not pat.search((q.get("title") or "") + " " + (q.get("body") or "")): continue
            rows.append({"question_id": q["question_id"], "site": r["site"], "brand": r["brand"],
                         "created_at": datetime.fromtimestamp(q["creation_date"], timezone.utc).strftime("%Y-%m-%d"),
                         "fetched_at": r["fetched_at"], "views": q.get("view_count"), "score": q.get("score"),
                         "up_votes": q.get("up_vote_count"), "answers": q.get("answer_count"), "comments": q.get("comment_count"),
                         "owner_reputation": q.get("owner_reputation"), "tags": ";".join(q.get("tags", [])), "title_len": len(q.get("title") or "")})
    d = pd.DataFrame(rows).drop_duplicates(["site", "question_id"])   # first brand found wins
    d.to_csv(ROOT / "data/processed/stackexchange_posts.csv", index=False)
    print(len(d), "questions", d.brand.nunique(), "brands", d.site.nunique(), "sites")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--max-requests", type=int, default=250); a = ap.parse_args()
    print("pairs:", len(pairs()))
    collect(a.max_requests); tidy()
