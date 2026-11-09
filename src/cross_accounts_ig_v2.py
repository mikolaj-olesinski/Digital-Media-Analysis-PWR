"""E223 second attempt, login-free: IG owners only, reading the YouTube /about tab and the FxTwitter user bio.

Same evidence rule as before (a bio link back to the owner's own Instagram URL). Writes data/manual/cross_accounts_ig_v2.csv
with salted hashes; counts only in the committed JSON.

    python src/cross_accounts_ig_v2.py
"""
import glob, hashlib, json, random, re, sys, time, urllib.error, urllib.request
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "manual" / "cross_accounts_ig_v2.csv"
SALT = ROOT / "data" / "manual" / ".salt"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"
COLS = ["owner_hash", "posts", "yt_match", "yt_link", "x_match", "x_link", "checked_at"]


def salt():
    if not SALT.exists():
        SALT.write_text(__import__("secrets").token_hex(16)); SALT.chmod(0o600)
    return SALT.read_text().strip()


def hkey(h):
    return hashlib.sha256(f"{salt()}:instagram:{h}".encode()).hexdigest()[:16]


def get(u):
    req = urllib.request.Request(u, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "ignore")


def main():
    raw = eu.load(ROOT).astype({"mention_id": str}).set_index("mention_id")
    b = pd.read_csv(ROOT/"data"/"manual"/"ig_browser_batch.csv", dtype={"id": str, "mention_id": str})
    fr = [pd.read_csv(f, dtype={"id": str}) for f in glob.glob(str(ROOT/"data"/"manual"/"ig_browser_results_*.csv"))]
    ig = pd.concat(fr).drop_duplicates("id"); ig = ig[ig.status == "ok"].merge(b, on="id", how="left")
    ig["handle"] = ig.mention_id.map(raw.author_url).map(
        lambda u: (re.search(r"instagram\.com/([^/?#]+)", u).group(1).lower() if isinstance(u, str) and re.search(r"instagram\.com/([^/?#]+)", u) else None))
    ig = ig[ig.handle.notna()]
    owners = ig.groupby("handle").size().reset_index(name="posts")
    done = set(pd.read_csv(OUT, dtype={"owner_hash": str}).owner_hash) if OUT.exists() else set()
    print("owners", len(owners), "todo", int(sum(hkey(h) not in done for h in owners.handle)), flush=True)
    for i, r in enumerate(owners.itertuples(), 1):
        k = hkey(r.handle)
        if k in done: continue
        ytm = "none"; ytl = ""; xm = "none"; xl = ""
        for suffix in ("/about", ""):
            try:
                html = get(f"https://www.youtube.com/@{r.handle}{suffix}")
                if f"instagram.com/{r.handle}" in html.lower():
                    ytm, ytl = "bio_link", "link"; break
            except urllib.error.HTTPError as e:
                if e.code in (404, 410): break
            except Exception: pass
        time.sleep(2.5 + random.random()*2)
        try:
            j = json.loads(get(f"https://api.fxtwitter.com/{r.handle}"))
            u = j.get("user") or {}
            text = " ".join(str(u.get(k, "")) for k in ("description", "website", "location")).lower()
            if f"instagram.com/{r.handle}" in text:
                xm, xl = "bio_link", "link"
        except Exception: pass
        pd.DataFrame([{"owner_hash": k, "posts": int(r.posts), "yt_match": ytm, "yt_link": ytl,
                       "x_match": xm, "x_link": xl,
                       "checked_at": pd.Timestamp.utcnow().isoformat(timespec="seconds")}], columns=COLS).to_csv(
            OUT, mode="a", header=not OUT.exists() or OUT.stat().st_size == 0, index=False)
        if i % 10 == 0: print("done", i, flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
