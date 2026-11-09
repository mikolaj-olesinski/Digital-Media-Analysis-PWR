"""E230: out-of-platform transfer with more fields than likes + comments.
A. Leave-one-platform-out over the export (TikTok / YouTube / X): train on the other two, test on the held-out one.
   Field sets: C2 (log likes, log comments); C3 (+ log shares); ENG (8 fields: followers, likes, comments, shares
   + 4 missing flags). Missing handling: 'impute' (reactions missing = 0 as in the published ridge, followers = training median, + flags) for ridge, native NaN for CatBoost,
   'drop' (only columns with < 20% missing in both source and target).
B. FB (X + YouTube source): C2 vs C2 + shares (FB re-read has shares; X / YouTube shares from the export).
C. IG (X + YouTube source): C2 vs C2 + followers (owner profiles, E229), ridge and CatBoost.
Paired brand-bootstrap diffs vs ridge on C2. Writes data/processed/transfer_fields_numbers.json.
    .venv/bin/python -W ignore src/transfer_fields.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em, fb_extend as fe, ig_transfer as ig, reread_transfer as rt, reaction_tiers as rti  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C2 = ["log_likes", "log_comments"]; C3 = C2 + ["log_shares"]
ENG = ["log_followers", "log_likes", "log_comments", "log_shares", "na_likes", "na_comments", "na_shares", "na_followers"]


def sc(y, p):
    return {"MAE": round(float(np.abs(y - p).mean()), 3), "factor": round(float(10 ** np.abs(y - p).mean()), 2)}


def fit_pred(kind, tr, te, cols, ytr):
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    if kind == "ridge":
        z = [c for c in cols if c in ("log_likes", "log_comments", "log_shares")]   # as in the published ridge: missing reaction = 0
        Xtr[z], Xte[z] = Xtr[z].fillna(0), Xte[z].fillna(0)
        med = Xtr.median(); Xtr, Xte = Xtr.fillna(med).fillna(0), Xte.fillna(med).fillna(0)  # followers: training median
        return fe.ridge().fit(Xtr, ytr).predict(Xte)
    return rti.FINAL().fit(Xtr, ytr).predict(Xte)


def block_a():
    f, meta = em.load(ROOT); f = f.copy(); f["platform"] = meta.platform.values; f["y"] = meta.y.values
    f["brand"] = meta.brand.fillna("unknown").values; out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        tr, te = f[f.platform != p], f[f.platform == p]
        miss = {c: [round(float(tr[c].isna().mean()), 2), round(float(te[c].isna().mean()), 2)] for c in ["log_followers", "log_likes", "log_comments", "log_shares"]}
        keep = [c for c, (a, b) in miss.items() if a < .2 and b < .2]
        sets = {"C2 likes + comments": C2, "C3 + shares": C3, "ENG 8 fields (impute / native NaN)": ENG, f"only columns with < 20% missing on both sides: {keep}": keep}
        base = None; r = {"posts": int(len(te)), "missing share (source, target)": miss}
        for name, cols in sets.items():
            for kind in ("ridge", "CatBoost"):
                pr = fit_pred(kind, tr, te, cols, tr.y); s = sc(te.y.values, pr)
                if base is None: base = te.y.values - pr
                else: s["vs ridge C2"] = rt.paired_diff(te.y.values - pr, base, te.brand.values)
                r[f"{kind} | {name}"] = s
        out[p] = r
    return out


def source():
    f, meta = em.load(ROOT); m = meta.platform.isin(["twitter", "youtube"]).values
    s = f[m].copy(); s["y"] = meta.y[m].values; return s


def block_b():
    s = source(); fb = fe.fb_frame(ROOT); fb = fb[fb.status == "ok"].dropna(subset=["y"]).copy()
    fb["log_shares"] = np.log1p(pd.to_numeric(fb.shares, errors="coerce")); out = {"posts": int(len(fb)), "FB shares missing": round(float(fb.log_shares.isna().mean()), 3)}
    base = None
    for name, cols in {"C2": C2, "C3 + shares": C3}.items():
        for kind in ("ridge", "CatBoost"):
            pr = fit_pred(kind, s, fb, cols, s.y); st = sc(fb.y.values, pr)
            if base is None: base = fb.y.values - pr
            else: st["vs ridge C2"] = rt.paired_diff(fb.y.values - pr, base, fb.brand.values)
            out[f"{kind} | {name}"] = st
    return out


def block_c():
    s = source(); s = s[s.log_followers.notna()]
    r = ig.frame(); r = r[(r.status == "ok") & r.views.notna() & r.likes.notna()].copy()
    o = pd.read_csv(ROOT / "data/manual/ig_work/ig_reel_owners.csv")[["id", "owner"]]
    pf = pd.read_csv(ROOT / "data/manual/ig_owner_profiles.csv"); pf = pf[pf.status == "ok"][["owner", "followers"]]
    d = r.merge(o, on="id", how="left").merge(pf, on="owner", how="left"); d = d[d.followers.notna()].copy()
    d["y"] = np.log10(d.views.clip(lower=1)); d["log_likes"] = np.log1p(d.likes); d["log_comments"] = np.log1p(d.comments.fillna(0))
    d["log_followers"] = np.log1p(d.followers); g = d.brand.fillna("unknown").values; out = {"reels": int(len(d))}; base = None
    for name, cols in {"C2": C2, "C2 + followers": C2 + ["log_followers"]}.items():
        for kind in ("ridge", "CatBoost"):
            pr = fit_pred(kind, s, d, cols, s.y); st = sc(d.y.values, pr)
            if base is None: base = d.y.values - pr
            else: st["vs ridge C2"] = rt.paired_diff(d.y.values - pr, base, g)
            out[f"{kind} | {name}"] = st
    return out


def main():
    out = {"A: leave one export platform out": block_a(), "B: Facebook from X + YT": block_b(), "C: Instagram from X + YT": block_c()}
    (ROOT / "data/processed/transfer_fields_numbers.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print("==", k)
        for p, r in v.items():
            if isinstance(r, dict) and "factor" in r: print(" ", p, r["factor"], r.get("vs ridge C2", ""))
            elif isinstance(r, dict):
                print(" ", p)
                for n, s in r.items():
                    if isinstance(s, dict) and "factor" in s: print("    ", n, s["factor"], s.get("vs ridge C2", ""))


if __name__ == "__main__":
    main()
