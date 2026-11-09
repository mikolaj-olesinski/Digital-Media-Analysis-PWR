"""E199: predict a view range (bin) instead of a number.
For several bin schemes, compares
  (a) the regression model (CatBoost MAE, the final features), whose point estimate is cut into bins, and
  (b) a direct CatBoost multiclass classifier on the same features,
against a majority-class baseline. Metrics: accuracy, macro F1, weighted F1, share within one bin, and for the
classifier the calibration of its top probability (ECE).
In-platform: TikTok, YouTube, X with brand-grouped 5-fold CV.
Transfer: trained on X + YouTube (likes + comments only), tested on Facebook (402 fetched) and Instagram (hand batch).
Writes data/processed/view_bins_numbers.json. Statistics only.
    python src/view_bins.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_models as em, fb_extend as fe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SCHEMES = {  # edges in log10 views
    "decades (<10, 10-100, ..., 1M+)": [1, 2, 3, 4, 5, 6],
    "4 user ranges (<100, 100-1k, 1k-10k, 10k+)": [2, 3, 4],
    "3 ranges (<1k, 1k-10k, 10k+)": [3, 4],
    "half decades (x3.2 wide)": list(np.arange(0.5, 7.0, 0.5)),
}


def cut(v, edges):
    return np.digitize(v, edges)


def cls():
    return CatBoostClassifier(loss_function="MultiClass", iterations=300, random_seed=SEED, verbose=0, allow_writing_files=False, thread_count=4)


def reg():
    return CatBoostRegressor(loss_function="MAE", iterations=300, random_seed=SEED, verbose=0, allow_writing_files=False, thread_count=4)


def ece(conf, hit, k=10):
    b = np.minimum((conf * k).astype(int), k - 1)
    return float(sum(abs(conf[b == i].mean() - hit[b == i].mean()) * (b == i).mean() for i in range(k) if (b == i).any()))


def metrics(t, p, conf=None):
    o = {"accuracy": float(accuracy_score(t, p)), "macro F1": float(f1_score(t, p, average="macro", zero_division=0)),
         "weighted F1": float(f1_score(t, p, average="weighted", zero_division=0)), "within one bin": float((np.abs(t - p) <= 1).mean())}
    if conf is not None:
        o["mean top probability"] = float(conf.mean()); o["ECE"] = ece(conf, (t == p).astype(float))
    return o


def proba_top(m, X):
    pr = m.predict_proba(X); c = np.asarray(m.classes_).astype(int)
    return c[pr.argmax(1)], pr.max(1)


def in_platform(f, meta):
    out = {}
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values; cols = em.ENG + (["metrics_age_days"] if p == "twitter" else [])
        X = f[m][cols].reset_index(drop=True); y = meta.y[m].values; g = meta.brand[m].values
        point = np.full(len(y), np.nan); res = {s: {"clf": np.zeros(len(y), int), "conf": np.zeros(len(y)), "maj": np.zeros(len(y), int)} for s in SCHEMES}
        for tr, te in GroupKFold(5).split(X, y, g):
            point[te] = reg().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            for s, e in SCHEMES.items():
                t = cut(y, e); mc = cls().fit(X.iloc[tr], t[tr]); a, c = proba_top(mc, X.iloc[te])
                res[s]["clf"][te] = a; res[s]["conf"][te] = c; res[s]["maj"][te] = np.bincount(t[tr]).argmax()
        out[p] = {}
        for s, e in SCHEMES.items():
            t = cut(y, e)
            out[p][s] = {"n": int(len(y)), "classes present": int(len(np.unique(t))), "majority": metrics(t, res[s]["maj"]),
                         "regression cut into bins": metrics(t, cut(point, e)), "classifier": metrics(t, res[s]["clf"], res[s]["conf"])}
        print(p, {s: (round(v["regression cut into bins"]["accuracy"], 3), round(v["classifier"]["accuracy"], 3), round(v["classifier"]["macro F1"], 3)) for s, v in out[p].items()}, flush=True)
    return out


def ig_frame():
    b = pd.read_csv(ROOT / "data/manual/ig_browser_batch.csv", dtype={"mention_id": str})
    rs = sorted((ROOT / "data/manual").glob("ig_browser_results_*.csv"))
    d = pd.concat([pd.read_csv(r) for r in rs]).merge(b, on="id")
    d = d[(d.status == "ok") & d.likes.notna() & d.views.notna()]
    return pd.DataFrame({"log_likes": np.log1p(d.likes.values), "log_comments": np.log1p(d.comments.fillna(0).values), "y": np.log10(d.views.clip(lower=1).values)})


def transfer(f, meta):
    m = meta.platform.isin(["twitter", "youtube"]).values; X = f[m][fe.C2].fillna(0); y = meta.y[m].values
    rid = fe.ridge().fit(X, y)
    fb = fe.fb_frame(ROOT); fb = fb[fb.status == "ok"].dropna(subset=["y"])
    tests = {"facebook": (fb[fe.C2].fillna(0), fb.y.values), "instagram": None}
    ig = ig_frame(); tests["instagram"] = (ig[fe.C2], ig.y.values)
    out = {}
    for name, (Xt, yt) in tests.items():
        out[name] = {}
        for s, e in SCHEMES.items():
            t = cut(yt, e); mc = cls().fit(X, cut(y, e)); a, c = proba_top(mc, Xt)
            out[name][s] = {"n": int(len(yt)), "majority (of X+YT)": metrics(t, np.full(len(t), np.bincount(cut(y, e)).argmax())),
                            "ridge cut into bins": metrics(t, cut(rid.predict(Xt), e)), "classifier": metrics(t, a, c)}
        print(name, {s: (round(v["ridge cut into bins"]["accuracy"], 3), round(v["classifier"]["accuracy"], 3)) for s, v in out[name].items()}, flush=True)
    return out


def main():
    f, meta = em.load(ROOT)
    out = {"E199": {"in platform": in_platform(f, meta), "transfer from X + YouTube": transfer(f, meta)}}
    (ROOT / "data" / "processed" / "view_bins_numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
