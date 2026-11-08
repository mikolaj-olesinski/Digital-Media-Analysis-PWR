"""E223 data step: find the same account on YouTube / X for FB and IG owners, with strong evidence only.

Owners: ok Instagram reels (ig_browser_results_* merged with ig_browser_batch) and the 402 Facebook videos with views,
taking the handle from the export author_url. For each distinct owner: check youtube.com/@handle and the FxTwitter user
endpoint, accept a match only when the other platform's bio/links point back to the owner's own platform URL
(match_evidence = bio_link); same_handle_and_name additionally requires the display name to be identical. Never guess
from the name. Only counts and salted hashes are written.

    python src/cross_accounts.py
"""
import glob
import hashlib
import json
import random
import re
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "manual" / "cross_accounts_deepseek.csv"
SALT = ROOT / "data" / "manual" / ".salt"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"
COLS = ["owner_hash", "platform", "posts", "yt_match", "yt_evidence", "yt_followers",
        "x_match", "x_evidence", "x_followers", "collected_at"]


def salt():
    if not SALT.exists():
        SALT.write_text(secrets.token_hex(16))
        SALT.chmod(0o600)
    return SALT.read_text().strip()


def hkey(platform, handle):
    return hashlib.sha256(f"{salt()}:{platform}:{handle}".encode()).hexdigest()[:16]


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def owners():
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    out = []
    # Instagram
    batch = pd.read_csv(ROOT / "data" / "manual" / "ig_browser_batch.csv", dtype={"id": str, "mention_id": str})
    frames = [pd.read_csv(f, dtype={"id": str}) for f in glob.glob(str(ROOT / "data" / "manual" / "ig_browser_results_*.csv"))]
    ig = pd.concat(frames).drop_duplicates("id")
    ig = ig[ig.status == "ok"].merge(batch, on="id", how="left")
    ig["handle"] = ig.mention_id.map(raw.author_url).map(lambda u: _handle(u, "instagram"))
    ig = ig[ig.handle.notna()]
    for h, g in ig.groupby("handle"):
        out.append({"platform": "instagram", "handle": h, "posts": len(g),
                    "name": raw.author_name.get(g.mention_id.iloc[0])})
    # Facebook
    fb = pd.read_csv(ROOT / "data" / "processed" / "fb_all_fetched.csv", dtype={"mention_id": str})
    fb = fb[(fb.status == "ok") & (fb.views > 0)]
    fb["handle"] = fb.mention_id.map(raw.author_url).map(lambda u: _handle(u, "facebook"))
    fb = fb[fb.handle.notna()]
    for h, g in fb.groupby("handle"):
        out.append({"platform": "facebook", "handle": h, "posts": len(g),
                    "name": raw.author_name.get(g.mention_id.iloc[0])})
    return pd.DataFrame(out)


def _handle(u, plat):
    if not isinstance(u, str):
        return None
    m = re.search(rf"{plat}\.com/([^/?#]+)", u)
    if not m:
        return None
    h = m.group(1).lower()
    return None if h in {"p", "reel", "reels", "tv", "explore", "watch", "video", "videos", "share",
                         "permalink.php", "profile.php"} else h


def links_in(html):
    return [u for u in re.findall(r'https?://[^"\\\s]+', html) if "instagram.com" in u or "facebook.com" in u
            or "youtube.com" in u or "x.com" in u or "twitter.com" in u]


def check_yt(handle, want_platform, want_handle):
    try:
        html = get(f"https://www.youtube.com/@{handle}")
    except urllib.error.HTTPError as e:
        return ("none", "", None) if e.code in (404, 410) else ("blocked", f"http {e.code}", None)
    except Exception:  # noqa: BLE001
        return ("none", "", None)
    title = (re.search(r'"title":"([^"]{1,80})"', html) or [None, ""])[1]
    links = links_in(html)
    linked = any(want_platform in u and want_handle in u.lower() for u in links)
    subs = re.search(r'"subscriberCountText":\{"simpleText":"([^"]+)"', html)
    return (("bio_link" if linked else "none"), ("link" if linked else ""), subs.group(1) if subs else None)


def check_x(handle, want_platform, want_handle):
    try:
        j = json.loads(get(f"https://api.fxtwitter.com/{handle}"))
    except urllib.error.HTTPError as e:
        return ("none", "", None) if e.code == 404 else ("blocked", f"http {e.code}", None)
    except Exception:  # noqa: BLE001
        return ("none", "", None)
    u = j.get("user") or {}
    text = " ".join(str(u.get(k, "")) for k in ("description", "website", "location"))
    linked = want_platform in text and want_handle in text.lower()
    return (("bio_link" if linked else "none"), ("link" if linked else ""), u.get("followers"))


def main():
    ow = owners()
    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT, dtype={"owner_hash": str}).owner_hash)
    print("owners", len(ow), "todo", int((~ow.apply(lambda r: hkey(r.platform, r.handle) in done, axis=1)).sum()), flush=True)
    blocked = 0
    for i, r in enumerate(ow.itertuples(), 1):
        key = hkey(r.platform, r.handle)
        if key in done:
            continue
        yt = check_yt(r.handle, r.platform, r.handle)
        time.sleep(2.5 + random.random() * 2)
        x = check_x(r.handle, r.platform, r.handle)
        time.sleep(2.5 + random.random() * 2)
        if "blocked" in (yt[0], x[0]):
            blocked += 1
        row = {"owner_hash": key, "platform": r.platform, "posts": int(r.posts),
               "yt_match": yt[0], "yt_evidence": yt[1], "yt_followers": yt[2],
               "x_match": x[0], "x_evidence": x[1], "x_followers": x[2],
               "collected_at": pd.Timestamp.utcnow().isoformat(timespec="seconds")}
        pd.DataFrame([row], columns=COLS).to_csv(OUT, mode="a", header=not OUT.exists() or OUT.stat().st_size == 0, index=False)
        if i % 20 == 0:
            print("done", i, "blocked", blocked, flush=True)
        if blocked >= 10:
            print("stopping: 10 blocked", flush=True)
            return
    print("finished", flush=True)


if __name__ == "__main__":
    main()
