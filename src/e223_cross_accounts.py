"""E223 data step: do the owners of our FB / IG scored posts also have a YouTube or X account?

The question E221 could not answer inside the export: an account's audience profile on YouTube / X (followers, views
per like, views per follower) as a prior for its FB / IG posts. This step collects those profiles from public pages and
records only counts plus a salted hash per owner.

Identity rules (a name alone is never evidence):
- `bio_link`: the YouTube channel about section or the X bio links to the owner's FB / IG page.
- `same_handle_and_name`: the same handle, the same display name, and a cross-link or a verification badge.
- `none` otherwise. No match is accepted on a name alone.

Owner sources: `author_url` in the export for the 402 scored FB videos and, for the IG reels, the same column. The
public reel page is login-walled anonymously (checked), so IG owners exist only where the export carries the handle.

Output (both git-ignored / counts only):
- `data/manual/cross_accounts.csv` — one row per owner, keyed by a salted hash; the salt is `data/manual/.salt`.
- `data/processed/cross_accounts_numbers.json` — counts only, no names, handles or URLs.

Resumable: a row is appended after every owner, so a stopped run continues where it left off.

    .venv/bin/python src/e223_cross_accounts.py            # full run
    .venv/bin/python src/e223_cross_accounts.py --limit 3  # smoke test
"""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/manual/cross_accounts.csv"
SALT_FILE = ROOT / "data/manual/.salt"
NUMBERS = ROOT / "data/processed/cross_accounts_numbers.json"
YTDLP = str(ROOT / ".venv/bin/yt-dlp")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

SUFFIX = ("official", "channel", "global", "tv", "yt", "hq")
GENERIC = {"channel", "c", "user", "u", "people", "pages", "pg", "profile.php", "watch", "shorts", "video", "videos",
           "reel", "reels", "p", "status", "i", "web", "groups", "events", "photo", "photos", "permalink.php",
           "story.php", "hashtag", "share", "www", ""}
COLS = ["owner_hash", "owner_platform", "posts", "has_handle", "yt_evidence", "yt_followers", "x_evidence",
        "x_followers", "checked_at", "note"]
GATE_POSTS = 40
_last = [0.0]


def throttle(sleep):
    """One request per `sleep` seconds plus jitter, measured across the whole run."""
    gap = sleep + random.uniform(0, 2)
    delta = time.time() - _last[0]
    if delta < gap:
        time.sleep(gap - delta)
    _last[0] = time.time()


def norm(s):
    if not isinstance(s, str) or not s.strip():
        return None
    k = re.sub(r"[^a-z0-9]", "", s.lower())
    changed = True
    while changed and k:
        changed = False
        for suf in SUFFIX:
            if k.endswith(suf) and len(k) > len(suf):
                k = k[: -len(suf)]
                changed = True
    return k if len(k) >= 3 else None


def handle_of(url):
    if not isinstance(url, str) or "http" not in url:
        return None
    try:
        parts = [p for p in urllib.parse.urlparse(url).path.split("/") if p]
    except ValueError:
        return None
    for p in parts:
        p = p.strip().lstrip("@")
        if p.lower() in GENERIC:
            continue
        if re.fullmatch(r"UC[\w-]{20,}", p) or re.fullmatch(r"\d{6,}", p):
            return None
        return p
    return None


def salt():
    if not SALT_FILE.exists():
        SALT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SALT_FILE.write_text(hashlib.sha256(os.urandom(32)).hexdigest() + "\n")
        SALT_FILE.chmod(0o600)
    return SALT_FILE.read_text().strip()


def keyed(s, platform, ident):
    return hashlib.sha256((s + "|" + platform + "|" + ident).encode()).hexdigest()[:20]


def owners():
    """Accounts behind the scored posts: platform, handle, name, page url, post count."""
    raw = pd.read_csv(ROOT / "data/raw/reach_samples_students.csv", dtype={"mention_id": str}, low_memory=False)
    raw["mention_id"] = raw.mention_id.astype(str)
    raw = raw.drop_duplicates("mention_id").set_index("mention_id")

    fb = pd.read_csv(ROOT / "data/processed/fb_all_fetched.csv", dtype={"mention_id": str})
    fb = fb[(fb.status == "ok") & fb.views.notna()]
    pairs = [(m, "facebook") for m in fb.mention_id]

    ig_files = sorted((ROOT / "data/manual").glob("ig_browser_results_*.csv"))
    ig = pd.concat([pd.read_csv(f, dtype={"id": str}) for f in ig_files], ignore_index=True)
    ig = ig[(ig.status == "ok") & ig.views.notna()]
    batch = pd.read_csv(ROOT / "data/manual/ig_browser_batch.csv", dtype={"id": str, "mention_id": str})
    ig_ids = batch.set_index("id").reindex(ig.id).mention_id.dropna().astype(str)
    pairs += [(m, "instagram") for m in ig_ids.values]

    accounts, order = {}, []
    for mid, plat in pairs:
        if mid not in raw.index:
            continue
        row = raw.loc[mid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        url = row.author_url if isinstance(row.author_url, str) else None
        name = row.author_name if isinstance(row.author_name, str) else None
        hd = handle_of(url)
        ident = hd or norm(name)
        if not ident:
            continue
        k = (plat, ident.lower())
        if k not in accounts:
            accounts[k] = {"platform": plat, "handle": hd, "name": name, "url": url, "posts": 0}
            order.append(k)
        accounts[k]["posts"] += 1
    return [accounts[k] for k in order]


def yt_search(query, sleep):
    throttle(sleep)
    cmd = [YTDLP, "--no-warnings", "--flat-playlist", "-J", "ytsearch3:" + query]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return []
    if p.returncode != 0:
        return []
    try:
        doc = json.loads(p.stdout)
    except json.JSONDecodeError:
        return []
    found, seen = [], set()
    for e in doc.get("entries") or []:
        url = e.get("channel_url") or (f"https://www.youtube.com/channel/{e['channel_id']}" if e.get("channel_id") else None)
        if not url or url in seen:
            continue
        seen.add(url)
        found.append({"url": url, "name": e.get("channel"),
                      "handle": (e.get("uploader_id") or "").lstrip("@") or None,
                      "verified": bool(e.get("channel_is_verified"))})
    return found


def json_object(html, marker):
    """The balanced JSON object that follows `marker` in a YouTube page."""
    i = html.find(marker)
    if i == -1:
        return None
    start = html.find("{", i)
    if start == -1:
        return None
    depth = 0
    for k in range(start, min(len(html), start + 80000)):
        ch = html[k]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(html[start:k + 1])
                except json.JSONDecodeError:
                    return None
    return None


def yt_about(url, sleep):
    throttle(sleep)
    req = urllib.request.Request(url.rstrip("/") + "/about",
                                 headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            html = r.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        return {"error": f"http {e.code}"}
    except Exception as e:  # noqa: BLE001
        return {"error": type(e).__name__}
    vm = json_object(html, '"aboutChannelViewModel":') or {}
    links = []
    for item in vm.get("links") or []:
        try:
            value = item["channelExternalLinkViewModel"]["link"]["content"]
        except (KeyError, TypeError):
            continue
        if isinstance(value, str):
            links.append(value)
    text = " ".join(str(x) for x in (vm.get("description"), vm.get("content"), *links) if x)
    count = str(vm.get("subscriberCountText") or "")
    m = re.match(r"([\d.,]+)\s*([KMB]?)", count)
    followers = None
    if m:
        followers = int(float(m.group(1).replace(",", "")) * {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}[m.group(2)])
    return {"followers": followers, "text": text, "verified": bool(vm.get("isVerified")), "sub": count}


def similar(a, b):
    """Loose name similarity: one normalised name contains the other."""
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def plausible(owner, cand):
    """A candidate worth opening: near name, same handle, or one handle inside the other."""
    if owner["handle"] and cand["handle"] and owner["handle"].lower() == cand["handle"].lower():
        return True
    if similar(owner["name"], cand["name"]):
        return True
    if owner["handle"] and cand["handle"]:
        a, b = owner["handle"].lower(), cand["handle"].lower()
        if min(len(a), len(b)) >= 4 and (a in b or b in a):
            return True
    return False


def links_back(text, owner):
    """Does the channel text link to this owner's FB / IG page?"""
    if not text or not owner.get("handle"):
        return False
    low = re.sub(r"\s+", "", text.lower())
    host = "facebook.com" if owner["platform"] == "facebook" else "instagram.com"
    hd = owner["handle"].lower()
    return (host + "/" + hd) in low or (host + "/@" + hd) in low


def best_youtube(owner, sleep, cache):
    """Strongest YouTube evidence for one owner."""
    query = owner["name"] if isinstance(owner["name"], str) and owner["name"].strip() else owner["handle"]
    if not query:
        return "none", None, "no query"
    best, best_rank, best_followers, note = "none", 0, None, ""
    opened = 0
    for cand in yt_search(query, sleep)[:3]:
        if opened >= 2:
            break
        same_name = bool(owner["name"]) and norm(cand["name"]) == norm(owner["name"])
        same_handle = bool(owner["handle"]) and (cand["handle"] or "").lower() == owner["handle"].lower()
        if not plausible(owner, cand):
            continue
        opened += 1
        if cand["url"] not in cache:
            cache[cand["url"]] = yt_about(cand["url"], sleep)
        about = cache[cand["url"]]
        if about.get("error"):
            note = about["error"]
            continue
        verified = bool(cand.get("verified")) or bool(about.get("verified"))
        if links_back(about.get("text"), owner):
            rank, ev = 2, "bio_link"
        elif same_handle and same_name and verified:
            rank, ev = 1, "same_handle_and_name"
        else:
            rank, ev = 0, "none"
        if rank > best_rank:
            best, best_rank, best_followers = ev, rank, about.get("followers")
        if rank == 2:
            break
    return best, best_followers, note


def x_probe(handle, sleep):
    """Public X lookup. FxTwitter serves statuses only; its user endpoint answers 404, the syndication timeline 429."""
    throttle(sleep)
    req = urllib.request.Request("https://api.fxtwitter.com/" + handle, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            doc = json.load(r)
    except urllib.error.HTTPError as e:
        return {"status": "blocked" if e.code == 429 else "absent", "evidence": "unavailable" if e.code == 429 else "none"}
    except Exception as e:  # noqa: BLE001
        return {"status": type(e).__name__, "evidence": "unavailable"}
    user = doc.get("user") or {}
    return {"status": "ok", "evidence": "none", "followers": user.get("followers"), "text": user.get("description") or ""}


def load_done():
    if not OUT.exists():
        return set()
    d = pd.read_csv(OUT, dtype=str)
    return set(d.owner_hash) if "owner_hash" in d.columns else set()


def append(row):
    new = not OUT.exists()
    pd.DataFrame([row], columns=COLS).to_csv(OUT, mode="a", header=new, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=4.0)
    ap.add_argument("--only-platform", default="", help="facebook | instagram; drops those rows and rechecks them")
    args = ap.parse_args()

    s = salt()
    if args.only_platform and OUT.exists():
        keep = pd.read_csv(OUT, dtype=str)
        keep = keep[keep.owner_platform != args.only_platform]
        keep.to_csv(OUT, index=False)
    accts = [a for a in owners() if not args.only_platform or a["platform"] == args.only_platform]
    done = load_done()
    todo = [a for a in accts if keyed(s, a["platform"], (a["handle"] or norm(a["name"])).lower()) not in done]
    print(f"owners: {len(accts)} ({sum(1 for a in accts if a['platform'] == 'facebook')} FB, "
          f"{sum(1 for a in accts if a['platform'] == 'instagram')} IG); to check: {len(todo)}", flush=True)
    if args.limit:
        todo = todo[: args.limit]

    cache = {}
    hits = [0, 0]
    x_mode = None
    for i, a in enumerate(todo, 1):
        row = {c: None for c in COLS}
        row["owner_hash"] = keyed(s, a["platform"], (a["handle"] or norm(a["name"])).lower())
        row["owner_platform"] = a["platform"]
        row["posts"] = a["posts"]
        row["has_handle"] = bool(a["handle"])
        row["checked_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        note = ""
        if not a["handle"]:
            # numeric FB page id: no handle to verify a match against, so the search cannot be evidence-checked
            row["yt_evidence"], row["x_evidence"] = "none", "none"
            row["note"] = "no handle (numeric page id)"
            append(row)
            continue
        try:
            ev, followers, note = best_youtube(a, args.sleep, cache)
            row["yt_evidence"], row["yt_followers"] = ev, followers
            if x_mode is None and a["handle"] and re.fullmatch(r"[A-Za-z0-9_]{1,15}", a["handle"]):
                probe = x_probe(a["handle"], args.sleep)
                x_mode = probe["status"]
                row["x_evidence"] = "unavailable" if probe["status"] in ("absent", "blocked") else probe["evidence"]
                row["x_followers"] = probe.get("followers")
            elif x_mode == "ok":
                probe = x_probe(a["handle"], args.sleep)
                row["x_evidence"], row["x_followers"] = probe["evidence"], probe.get("followers")
            else:
                # FxTwitter serves statuses, not profiles: its user endpoint answers 404, the syndication timeline 429
                row["x_evidence"] = "unavailable"
        except KeyboardInterrupt:
            print("interrupted", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            note = f"{type(e).__name__}: {e}"[:100]
            row["yt_evidence"] = row["yt_evidence"] or "none"
        row["note"] = note
        append(row)
        if row["yt_evidence"] in ("bio_link", "same_handle_and_name"):
            hits[i % 2] += 1
        if i % 10 == 0:
            print(f"{i}/{len(todo)} checked; YouTube matches so far: {sum(hits)}", flush=True)

    if todo:
        refresh_numbers(s)


def refresh_numbers(s):
    if not OUT.exists():
        return
    d = pd.read_csv(OUT)
    scored = {"facebook": 402, "instagram": 696}
    matched = d[d.yt_evidence.isin(["bio_link", "same_handle_and_name"])]
    x_matched = d[d.x_evidence.isin(["bio_link", "same_handle_and_name"])]
    covered = int(matched.posts.sum())
    doc = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scored_posts": scored,
        "scored_total": sum(scored.values()),
        "owners_checked": int(len(d)),
        "owners": {p: int((d.owner_platform == p).sum()) for p in ("facebook", "instagram")},
        "owners_with_handle": int(d.has_handle.sum()),
        "matched_youtube": int(len(matched)),
        "matched_youtube_bio_link": int((d.yt_evidence == "bio_link").sum()),
        "matched_x": int(len(x_matched)),
        "x_lookup": "unavailable" if (d.x_evidence == "unavailable").any() else "attempted",
        "facebook_owners_linkable": bool((d.has_handle & (d.owner_platform == "facebook")).any()),
        "gate_scope": "IG owners only; FB author URLs in the export are numeric page ids, so their bios cannot be "
                      "matched against a link target (a public watch page does expose the page URL: recovery pending)",
        "gate_provisional": True,
        "covered_posts": covered,
        "covered_share": round(covered / sum(scored.values()), 4),
        "gate_posts": GATE_POSTS,
        "gate_passed": covered >= GATE_POSTS,
        "owners_with_handle_by_platform": {p: int(((d.owner_platform == p) & d.has_handle).sum()) for p in ("facebook", "instagram")},
    }
    NUMBERS.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"numbers: {doc['matched_youtube']} YouTube, {doc['matched_x']} X, {covered} scored posts covered", flush=True)


if __name__ == "__main__":
    sys.exit(main())
