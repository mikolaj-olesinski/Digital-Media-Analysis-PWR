"""E221: with normalised names and profile-URL handles, do authors appear on more than one platform?
Keys: normalised author_name (missing names dropped first; lowercase; only a-z0-9 kept; suffixes official / tv / yt / hq /
channel / global dropped; keys under 3 characters ignored) and the handle from author_url (first non-generic path
segment, @ removed, normalised the same way). Labelled posts: TikTok / YouTube / X with views (export or same-moment
re-read), FB 402 videos, IG reels with views and likes. Gate: >= 2% of labelled posts matched -> E70-style test.
Writes counts only (no names, handles or URLs) to data/processed/cross_author_norm_numbers.json.
    .venv/bin/python src/cross_author_norm.py"""
import json, re, sys
from pathlib import Path
from urllib.parse import urlparse
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, fb_extend as fe, ig_transfer as ig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUFFIX = ("official", "channel", "global", "tv", "yt", "hq")
GENERIC = {"channel", "c", "user", "u", "people", "pages", "pg", "profile.php", "watch", "shorts", "video", "videos", "reel", "reels",
           "p", "status", "i", "web", "groups", "events", "photo", "photos", "permalink.php", "story.php", "hashtag", "share", "www", ""}
PLAT = ["tiktok", "youtube", "twitter", "facebook", "instagram"]
GATE = 0.02


def norm(s):
    if not isinstance(s, str) or not s.strip(): return None
    k = re.sub(r"[^a-z0-9]", "", s.lower())
    changed = True
    while changed and k:
        changed = False
        for suf in SUFFIX:
            if k.endswith(suf) and len(k) > len(suf): k = k[: -len(suf)]; changed = True
    return k if len(k) >= 3 else None


def handle(u):
    if not isinstance(u, str) or not u.strip(): return None
    try: parts = [p for p in urlparse(u.strip()).path.split("/")]
    except ValueError: return None
    for p in parts:
        p = p.strip().lstrip("@")
        if p.lower() in GENERIC: continue
        if re.fullmatch(r"UC[\w-]{20,}", p): return None        # YouTube channel id: not a name
        if re.fullmatch(r"\d{6,}", p): return None              # numeric profile id
        return norm(p)
    return None


def labelled(raw):
    """mention_id -> True for posts with a view label on any source."""
    ids = set(raw[(raw.platform.isin(["tiktok", "youtube", "twitter"])) & (raw.view_count > 0)].mention_id)
    for f in ("youtube", "x", "tiktok"):
        d = pd.read_csv(ROOT / f"data/processed/{f}_full_fetched.csv", dtype={"mention_id": str}).drop_duplicates("mention_id", keep="last")
        ids |= set(d[(d.status == "ok") & d.views.notna()].mention_id)
    fb = fe.fb_frame(ROOT); ids |= set(fb[(fb.status == "ok") & fb.y.notna()].mention_id)
    r = ig.frame(); ids |= set(r[(r.status == "ok") & r.views.notna() & r.likes.notna()].mention_id.astype(str))
    return ids


def main():
    raw = eu.load(ROOT).astype({"mention_id": str})
    raw["k_name"] = raw.author_name.map(norm); raw["k_handle"] = raw.author_url.map(handle)
    lab = labelled(raw); raw["labelled"] = raw.mention_id.isin(lab)
    out = {"posts": int(len(raw)), "labelled posts": raw[raw.labelled].platform.value_counts().reindex(PLAT).fillna(0).astype(int).to_dict(),
           "share with a name key": raw.groupby("platform").k_name.apply(lambda s: round(float(s.notna().mean()), 3)).reindex(PLAT).to_dict(),
           "share with a handle key": raw.groupby("platform").k_handle.apply(lambda s: round(float(s.notna().mean()), 3)).reindex(PLAT).to_dict()}
    res = {}
    for kind, cols in (("name", ["k_name"]), ("handle", ["k_handle"]), ("name or handle", ["k_name", "k_handle"])):
        long = pd.concat([raw[["mention_id", "platform", "labelled", c]].rename(columns={c: "key"}) for c in cols]).dropna(subset=["key"])
        nplat = long.groupby("key").platform.nunique()
        multi = set(nplat[nplat >= 2].index)
        hit = set(long[long.key.isin(multi)].mention_id)
        m = raw.mention_id.isin(hit)
        # posts whose author key has a *labelled* post on another platform (what a prior would need)
        labp = long[long.labelled].groupby("key").platform.agg(set)
        def other_lab(r):
            for c in cols:
                k = r[c]
                if isinstance(k, str) and k in labp.index and (labp[k] - {r.platform}): return True
            return False
        ol = raw[raw.labelled].apply(other_lab, axis=1)
        res[kind] = {"keys on 2+ platforms": int(len(multi)),
                     "keys on 3+ platforms": int((nplat >= 3).sum()),
                     "posts (all) whose key is on another platform": raw[m].platform.value_counts().reindex(PLAT).fillna(0).astype(int).to_dict(),
                     "labelled posts whose key is on another platform": raw[m & raw.labelled].platform.value_counts().reindex(PLAT).fillna(0).astype(int).to_dict(),
                     "labelled posts with a labelled post of the same key on another platform": raw[raw.labelled][ol.values].platform.value_counts().reindex(PLAT).fillna(0).astype(int).to_dict(),
                     "share of labelled posts matched": round(float((m & raw.labelled).sum() / raw.labelled.sum()), 4),
                     "share of labelled posts with a labelled cross-platform post": round(float(ol.sum() / raw.labelled.sum()), 4)}
    out["matching"] = res
    best = res["name or handle"]["share of labelled posts with a labelled cross-platform post"]
    out["gate"] = {"threshold": GATE, "share (name or handle, labelled on another platform)": best, "passed": bool(best >= GATE)}
    if best >= GATE: out["gate"]["note"] = "gate passed: run the E70-style shrunk prior test (not implemented in this script)"
    (ROOT / "data/processed/cross_author_norm_numbers.json").write_text(json.dumps({"E221": out}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
