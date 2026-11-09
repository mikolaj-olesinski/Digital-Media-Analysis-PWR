"""E224: is a mention really about its brand? A views-free off-brand detector for all 12,500 mentions, a blind
300-post validation sample with three AI label passes, and the effect of the suspicious posts on the view model.

Phase 1 (no view, reaction or reach column is read): four detectors scored so that HIGH = suspicious
  lex   - the normalised brand name, or its joined handle / hashtag form, is absent from title + content
  emb   - cosine to the brand's leave-self-out e5 centroid, as a within-brand robust z-score
  graph - TF-IDF and e5 10-NN text graphs + Louvain: smoothed same-brand share of the neighbours and of the
          community against the brand's platform base rate (log lift, sign flipped); the two graphs are averaged
          as percentile ranks
  clf   - one-vs-rest logistic regression on TF-IDF predicting the brand: 1 - out-of-fold probability of the assigned
          brand (low probability = suspicious); brands with at least MIN_POSTS posts on the platform
  combined - mean of the percentile ranks of the available detectors over all 12,500 posts

Phase 1b: a 300-post sample stratified by combined-score decile, brands with an ambiguous name weighted x3, spread over
all five platforms. `offbrand_sample.csv` carries no scores (blind labelling); `offbrand_scores.csv` carries no text.

Phase 2 (`--stage eval`, after the three blind passes and the adjudication exist): Fleiss kappa, AUC per detector,
precision in the top 5% / 10%, off-brand share per platform and per brand, design-weighted population estimate, the
error of the final model on the top-5% posts against the rest, and the change in brand view totals when they are
dropped. All labels are AI labels and await a team member's check.

    python -W ignore src/offbrand.py            # scores + sample + phase-1 numbers (Kueue job amc-offbrand)
    python -W ignore src/offbrand.py --stage eval
"""
import argparse, json, re, sys, unicodedata
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu, extra_models as em, graph_lab as gl, label_utils as lu  # noqa: E402
import validation as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
MIN_POSTS = 5                     # a brand needs this many posts on a platform to be a classifier class
EMB_MIN_POSTS = 5                 # a brand needs this many posts for a leave-self-out centroid
K = 10                            # neighbours in the text graphs
M_SMOOTH = 2.0                    # smoothing of the same-brand share towards the base rate
ABBREV = {"AB InBev", "PepsiCo"}  # brands whose product names the lexical check cannot see (documented)
DET = ["lex", "emb", "graph_tfidf", "graph_e5", "graph", "clf"]          # raw scores, high = suspicious
DET_PCT = ["lex_pct", "emb_pct", "graph_tfidf_pct", "graph_e5_pct", "graph_pct", "clf_pct", "combined"]  # ranked
DROP = {"the", "and", "of", "&"}
DECILES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# posts sampled per combined-score decile (over-samples the suspicious end, keeps every decile)
N_PER_DECILE = {1: 15, 2: 15, 3: 15, 4: 15, 5: 15, 6: 25, 7: 25, 8: 25, 9: 50, 10: 100}
N_SAMPLE = sum(N_PER_DECILE.values())
AMB_WEIGHT = 3.0
BRAND_CAP_DECILE = 4
BRAND_CAP_TOTAL = 10
PASSES = ["A", "B", "C"]
SHARDS = 3
PASS_DIR = ROOT / "data" / "manual" / "offbrand_passes"


# ---------- text helpers ----------
def norm(s):
    """Lowercase, drop accents and apostrophes, keep letters and digits, everything else becomes a space."""
    s = str(s).replace("'", "").replace("\u2019", "").lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def tokens(s):
    return [t for t in norm(s).split() if t not in DROP]


def brand_forms(brand):
    """Full token sequence and joined forms (with and without 'the' / 'and' / 'of') of a brand name."""
    all_t = norm(brand).split()
    core = [t for t in all_t if t not in DROP]
    return tuple(core), "".join(core), "".join(all_t)


def brand_in_text(text, brand):
    """Does the brand name appear as a contiguous token sequence, as a joined handle / hashtag form, or as the
    prefix of a hashtag ("#TheNorthFace" -> the joined form)? Short keys are only matched exactly."""
    tt = tokens(text); st = set(tt)
    full, joined, joined_all = brand_forms(brand)
    if not full:
        return False
    if joined in st or joined_all in st:
        return True
    n = len(full)
    if any(tuple(tt[i:i + n]) == full for i in range(len(tt) - n + 1)):
        return True
    return any(len(joined) >= 5 and t.startswith(joined) for t in st)


def mask_text(t, limit=600, brands=()):
    """URLs and @handles out (the sample is shared), email addresses too; the text is cut at `limit` characters.
    An "@ Name" form is masked only when Name is not one of the brands (a brand name stays, it is needed to label)."""
    t = str(t)
    t = re.sub(r"https?://\S+|www\.\S+", "[url]", t)
    t = re.sub(r"\b(?:[a-z0-9-]+\.)+(?:com|co|ly|be|me|net|org|tv|gg|io|app|link|to|us)/\S*", "[url]", t, flags=re.I)
    t = re.sub(r"\b\S+@\S+\.\w+\b", "[email]", t)
    t = re.sub(r"(?<![\w])@[A-Za-z0-9_$][^\s@]{0,40}", "[handle]", t)
    low = {str(b).lower() for b in brands}
    t = re.sub(r"@[^\w@]{0,4}([A-Za-z0-9_][\w.]{1,24})",
               lambda m: m.group(0) if m.group(1).lower() in low else "[handle]", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit].strip()


# ---------- detector 1: lexical ----------
def lexical_score(raw, text):
    """1.0 when the brand name (or its handle / hashtag form) is absent from the post, 0.0 when it is present."""
    return pd.Series([0.0 if brand_in_text(t, b) else 1.0 for t, b in zip(text, raw.brand)], index=raw.index, name="lex")


# ---------- detector 2: embedding, leave-self-out centroid ----------
def embedding_scores(raw, E):
    """Cosine of each post to the mean e5 vector of the other posts of its brand, as a within-brand robust z-score
    (median and MAD of the brand's own cosines). High z = the post sits away from its brand's texts."""
    Z = np.asarray(E, dtype=float)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    cos = pd.Series(np.nan, index=raw.index, name="emb_cos"); z = pd.Series(np.nan, index=raw.index, name="emb")
    pos = {i: k for k, i in enumerate(raw.index)}
    for b, idx in raw.groupby("brand").groups.items():
        rows = np.array([pos[i] for i in idx]); n = len(rows)
        if n < EMB_MIN_POSTS:
            continue
        s = Z[rows]
        c = (s.sum(0) - s) / (n - 1)                       # leave-self-out centroid of every row at once
        c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-9)
        ci = np.einsum("ij,ij->i", s, c)
        cos.loc[idx] = ci
        scale = max(1.4826 * np.median(np.abs(ci - np.median(ci))), 0.01)
        z.loc[idx] = (np.median(ci) - ci) / scale
    return z, cos


# ---------- detector 3: text graphs + Louvain ----------
def _graph_homophily(A, comm, brand, p):
    """Per post: smoothed same-brand share among its kNN neighbours and among its community, over the base rate."""
    b = np.asarray(brand); n = len(b)
    nb_same = np.zeros(n); nb_n = np.zeros(n); c_same = np.zeros(n); c_n = np.zeros(n)
    for i in range(n):
        nb = np.nonzero(A[i])[0]
        nb_same[i], nb_n[i] = (b[nb] == b[i]).sum(), len(nb)
    order = np.argsort(comm, kind="stable"); cs = comm[order]
    bounds = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1], True])
    for j in range(len(bounds) - 1):
        rows = order[bounds[j]:bounds[j + 1]]
        for i in rows:
            same = (b[rows] == b[i]).sum() - 1
            c_same[i], c_n[i] = same, len(rows) - 1
    h_nb = (nb_same + M_SMOOTH * p) / (nb_n + M_SMOOTH)
    h_c = (c_same + M_SMOOTH * p) / (c_n + M_SMOOTH)
    return 0.5 * np.log(h_nb / p) + 0.5 * np.log(h_c / p)


def graph_scores(raw, E, text):
    """TF-IDF and e5 10-NN graphs per platform; log lift of the same-brand neighbour and community share over the
    brand's base rate on that platform; sign flipped (low lift = suspicious)."""
    out = pd.DataFrame(index=raw.index, columns=["graph_tfidf", "graph_e5"], dtype=float)
    stats = {}
    for p in eu.PLATFORMS:
        m = (raw.platform == p).values; idx = raw.index[m]
        t = text[m].values
        Z = np.asarray(E)[m]
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
        S2 = Z @ Z.T; np.fill_diagonal(S2, 0.0)
        S1 = em.text_similarity(t)
        base = raw.loc[idx, "brand"].value_counts(normalize=True)
        b = raw.loc[idx, "brand"].values; pb = pd.Series(b).map(base).values
        for name, S in [("graph_tfidf", S1), ("graph_e5", S2)]:
            A, _ = gl.knn_graph(S, K)
            comm, _ = gl.louvain(A)
            lift = _graph_homophily(A, comm, b, pb)
            out.loc[idx, name] = -lift
            stats[name + " " + eu.NAMES[p]] = {"communities": int(comm.max() + 1), "median log lift": round(float(np.median(lift)), 3)}
    return out, stats


# ---------- detector 4: one-vs-rest brand classifier ----------
def classifier_scores(raw, text):
    """Out-of-fold one-vs-rest P(brand | text) from TF-IDF logistic regressions, per platform. Brands with fewer than
    MIN_POSTS posts on the platform are not classes and stay NaN (they are left out of the rank average)."""
    out = pd.Series(np.nan, index=raw.index, name="clf"); info = {}
    for p in eu.PLATFORMS:
        m = (raw.platform == p).values; idx = raw.index[m]
        b = raw.loc[idx, "brand"].values
        cnt = pd.Series(b).value_counts(); classes = list(cnt[cnt >= MIN_POSTS].index)
        v = TfidfVectorizer(min_df=2, max_features=20000, sublinear_tf=True)
        X = v.fit_transform([s if s.strip() else "emptytext" for s in text[m]])
        rng = np.random.default_rng(SEED)
        fold = np.empty(len(b), int)
        for br in classes:                                  # balanced folds inside every brand
            r = np.flatnonzero(b == br); fold[rng.permutation(r)] = np.arange(len(r)) % 5
        fold[~np.isin(b, classes)] = rng.integers(0, 5, int((~np.isin(b, classes)).sum()))
        P = np.zeros((len(b), len(classes)))
        for f in range(5):
            tr, te = fold != f, fold == f
            for j, br in enumerate(classes):
                y = (b == br).astype(int)
                if y[tr].sum() < 1:
                    continue
                lr = LogisticRegression(solver="liblinear", C=1.0, max_iter=200).fit(X[tr], y[tr])
                P[te, j] = lr.predict_proba(X[te])[:, 1]
        known = np.isin(b, classes); cidx = {c: j for j, c in enumerate(classes)}
        vals = np.array([P[i, cidx[b[i]]] if known[i] else np.nan for i in range(len(b))])
        out.loc[idx] = vals
        info[eu.NAMES[p]] = {"brands as classes": len(classes), "posts scored": int(known.sum()),
                             "median P(own brand)": round(float(np.nanmedian(vals)), 4), "coverage": round(float(known.mean()), 3)}
    return out, info


# ---------- combining ----------
def combine(scores, platform):
    """Percentile rank of every detector, then their mean over all 12,500 posts (missing detectors skipped).
    The lexical flag and the within-brand embedding z are ranked over all posts. The classifier probability and the
    graph log lift are ranked WITHIN a platform: the number of brand classes (11 on Instagram, 240 on TikTok) and the
    base rate (0.09 on Facebook, 0.004 on TikTok) move their scale, so only the within-platform order is comparable."""
    s = scores.copy()
    for c in ["lex", "emb"]:
        s[c + "_pct"] = s[c].rank(pct=True)
    for c in ["graph_tfidf", "graph_e5", "clf"]:
        s[c + "_pct"] = s.groupby(platform.loc[s.index].values)[c].rank(pct=True)
    s["graph_pct"] = s[["graph_tfidf_pct", "graph_e5_pct"]].mean(axis=1, skipna=True)
    s["combined"] = s[["lex_pct", "emb_pct", "graph_pct", "clf_pct"]].mean(axis=1, skipna=True)
    s["combined_rank"] = s["combined"].rank(pct=True)
    s["decile"] = np.clip(np.ceil(s["combined_rank"] * 10), 1, 10).astype(int)
    return s


# ---------- phase 1b: validation sample ----------
def build_sample(raw, scores, amb):
    """300 posts: per-decile quotas, platforms allocated in proportion with a floor of 2, ambiguous-name brands
    weighted AMB_WEIGHT, at most BRAND_CAP_DECILE posts per brand and decile."""
    rng = np.random.default_rng(SEED)
    amb = set(amb); rows = []; taken = {}
    for d in DECILES:
        pool = scores.index[scores.decile == d]
        n_d = N_PER_DECILE[d]
        pl = raw.loc[pool, "platform"].value_counts().reindex(eu.PLATFORMS).fillna(0).values.astype(float)
        quota = np.maximum(2, np.floor(n_d * pl / pl.sum())).astype(int)
        while quota.sum() > n_d:                            # trim the largest allocations back to n_d
            j = int(np.argmax(quota - 2)); quota[j] -= 1
        while quota.sum() < n_d:                            # then add to the platforms with the most unused posts
            j = int(np.argmax(pl - quota)); quota[j] += 1
        picked = []
        for p, n_p in zip(eu.PLATFORMS, quota):
            if n_p <= 0:
                continue
            cand = np.array([i for i in pool if raw.at[i, "platform"] == p and i not in picked])
            w = np.array([AMB_WEIGHT if raw.at[i, "brand"] in amb else 1.0 for i in cand])
            per_brand = pd.Series([raw.at[i, "brand"] for i in picked]).value_counts() if picked else pd.Series(dtype=int)
            for _ in range(int(n_p)):
                if len(cand) == 0:
                    break
                ok_dec = np.array([per_brand.get(raw.at[i, "brand"], 0) < BRAND_CAP_DECILE for i in cand])
                ok_all = np.array([taken.get(raw.at[i, "brand"], 0) < BRAND_CAP_TOTAL for i in cand])
                ww = w * ok_dec * ok_all
                if ww.sum() <= 0:
                    ww = w * ok_all
                if ww.sum() <= 0:
                    ww = w
                j = rng.choice(len(cand), p=ww / ww.sum()); i = cand[j]
                picked.append(i); taken[raw.at[i, "brand"]] = taken.get(raw.at[i, "brand"], 0) + 1
                per_brand = pd.Series([raw.at[i2, "brand"] for i2 in picked]).value_counts()
                keep = np.arange(len(cand)) != j; cand, w = cand[keep], w[keep]
        rows += picked
    S = raw.loc[rows].copy()
    S["decile"] = scores.loc[rows, "decile"].values
    S["text"] = [mask_text(t, brands=raw.brand.unique()) for t in (S.title.fillna("") + " " + S.content.fillna("")).values]
    S = S.sample(frac=1.0, random_state=SEED).reset_index(drop=False).rename(columns={"index": "raw_index"})
    S["sample_id"] = [f"S{i:03d}" for i in range(1, len(S) + 1)]
    return S


def write_blind_passes(S, amb):
    """Three independently shuffled copies of the sample in three shards each, with no score column."""
    PASS_DIR.mkdir(parents=True, exist_ok=True)
    man = {"seed": SEED, "shards": SHARDS, "passes": {}}
    for p in PASSES:
        rng = np.random.default_rng(SEED + ord(p))
        order = rng.permutation(len(S)); man["passes"][p] = []
        for s in range(SHARDS):
            part = order[s * len(order) // SHARDS:(s + 1) * len(order) // SHARDS]
            q = S.iloc[part][["sample_id", "mention_id", "platform", "brand", "text"]]
            f = PASS_DIR / f"pass_{p}_shard{s + 1}.csv"
            q.to_csv(f, index=False)
            man["passes"][p].append(f"pass_{p}_shard{s + 1}.csv")
    json.dump(man, open(PASS_DIR / "manifest.json", "w"), indent=1)
    return man


# ---------- numbers ----------
def _r(x, k=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


def distributions(scores):
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    out = {}
    for c in DET + DET_PCT:
        s = scores[c].dropna()
        out[c] = {f"p{int(q * 100)}": _r(s.quantile(q)) for q in qs} | {"mean": _r(s.mean()), "missing share": _r(scores[c].isna().mean())}
    return out


def agreement(scores):
    """Spearman between the ranked detectors (the ones that enter the combination)."""
    out = {}
    for i, a in enumerate(DET_PCT[:-1]):
        for b in DET_PCT[i + 1:-1]:
            ok = scores[a].notna() & scores[b].notna()
            out[f"{a} vs {b}"] = {"spearman": _r(spearmanr(scores.loc[ok, a], scores.loc[ok, b])[0]), "n": int(ok.sum())}
    ok = scores[DET_PCT[:-1]].notna().all(axis=1)
    out["mean pairwise spearman"] = _r(np.mean([v["spearman"] for v in out.values() if v["spearman"] is not None]))
    out["all four on one post"] = int(ok.sum())
    # do the detectors pick the same suspicious posts? Jaccard of each top 10% with the combined top 10%
    tops = {c: set(scores.index[scores[c] >= 0.9]) for c in DET_PCT[:-1]}
    comb = set(scores.index[scores.combined >= scores.combined.quantile(0.9)])
    out["top-10% jaccard with combined"] = {c: _r(len(tops[c] & comb) / max(len(tops[c] | comb), 1)) for c in DET_PCT[:-1]}
    return out


def brand_shares(raw, scores, min_mentions=20, top=20):
    """Phase-1 view of which brands look most off-brand: mean combined percentile and share in the top decile."""
    d = pd.DataFrame({"brand": raw.brand.values, "platform": raw.platform.values, "pct": scores.combined_rank.values,
                      "top10": (scores.decile.values == 10).astype(float)})
    g = d.groupby("brand").agg(mentions=("brand", "size"), mean_percentile=("pct", "mean"), share_top_decile=("top10", "mean"))
    g = g[g.mentions >= min_mentions].sort_values("share_top_decile", ascending=False)
    return [{"brand": b, "mentions": int(r.mentions), "mean_percentile": _r(r.mean_percentile), "share_top_decile": _r(r.share_top_decile)}
            for b, r in g.head(top).iterrows()], {"brands considered": int(len(g)), "brands total": int(raw.brand.nunique())}


# ---------- phase 1 driver ----------
def run_scores():
    raw = eu.load(ROOT).astype({"mention_id": str})
    text = (raw.title.fillna("") + " " + raw.content.fillna(""))
    E = pd.read_pickle(ROOT / "data" / "processed" / "embeddings_e5_cache.pkl")
    assert len(E) == len(raw), "embedding cache is not aligned with the raw rows"
    Z = np.asarray(E, dtype=float); Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    rng = np.random.default_rng(SEED); chk = []
    for i in rng.choice(len(raw), 500, replace=False):    # alignment check: the post is closer to its brand than to a random one
        own = np.flatnonzero((raw.brand == raw.brand.iloc[i]).values); own = own[own != i]
        other = raw.brand.sample(1, random_state=int(i)).iloc[0]
        if len(own):
            c_own = Z[own].mean(0); c_own /= max(np.linalg.norm(c_own), 1e-9)
            c_oth = Z[(raw.brand == other).values].mean(0); c_oth /= max(np.linalg.norm(c_oth), 1e-9)
            chk.append(Z[i] @ c_own - Z[i] @ c_oth)
    print(f"embedding alignment check: own-brand minus random-brand cosine, median {np.median(chk):.3f}", flush=True)

    print("detector 1/4 lexical", flush=True)
    lex = lexical_score(raw, text)
    print("detector 2/4 embedding", flush=True)
    emb, cos = embedding_scores(raw, E)
    print("detector 3/4 graphs", flush=True)
    gr, gstats = graph_scores(raw, E, text)
    gr["graph"] = gr[["graph_tfidf", "graph_e5"]].mean(axis=1)
    print("detector 4/4 classifier", flush=True)
    clf, cinfo = classifier_scores(raw, text)
    print(f"classifier: median P(assigned brand) {np.nanmedian(clf):.4f}, coverage {clf.notna().mean():.3f}", flush=True)
    scores = pd.DataFrame({"mention_id": raw.mention_id.values, "lex": lex.values, "emb": emb.values,
                           "graph_tfidf": gr.graph_tfidf.values, "graph_e5": gr.graph_e5.values, "graph": gr.graph.values,
                           "clf": (1.0 - clf).values, "clf_p": clf.values}, index=raw.index)
    scores = combine(scores, raw.platform)
    scores.to_csv(ROOT / "data" / "processed" / "offbrand_scores.csv", index=False)

    amb = pd.read_csv(ROOT / "data" / "processed" / "brand_industry.csv")
    amb_yes = set(amb.loc[amb.ambiguous_name == "yes", "brand"]) & set(raw.brand)
    S = build_sample(raw, scores, amb_yes)
    S[["sample_id", "mention_id", "platform", "brand", "text"]].to_csv(ROOT / "data" / "processed" / "offbrand_sample.csv", index=False)
    man = write_blind_passes(S, amb_yes)

    # how far the sample matches the intended design
    design = {"sample": int(len(S)), "seed": SEED, "per decile": {int(k): int(v) for k, v in S.decile.value_counts().sort_index().items()},
              "per platform": {eu.NAMES[k]: int(v) for k, v in S.platform.value_counts().reindex(eu.PLATFORMS).fillna(0).items()},
              "ambiguous-name brands in sample": int(S.brand.isin(amb_yes).sum()),
              "asks per decile": N_PER_DECILE, "ambiguous weight": AMB_WEIGHT, "brand cap per decile": BRAND_CAP_DECILE,
              "brand cap overall": BRAND_CAP_TOTAL,
              "blind passes": {p: len(man["passes"][p]) for p in PASSES}}

    top, tinfo = brand_shares(raw, scores)
    num = {
        "experiment": "E224", "date": "2026-11-09",
        "data": {"mentions": int(len(raw)), "brands": int(raw.brand.nunique()), "ambiguous-name brands": int(len(amb_yes)),
                 "platforms": {eu.NAMES[p]: int((raw.platform == p).sum()) for p in eu.PLATFORMS},
                 "note": "Phase 1 reads no view, reaction or reach column."},
        "alignment check": {"metric": "cosine to own-brand centroid minus cosine to a random brand's centroid",
                            "median over 500 posts": _r(np.median(chk))},
        "detectors": {"min posts per brand (classifier)": MIN_POSTS, "min posts per brand (embedding centroid)": EMB_MIN_POSTS,
                      "k neighbours": K, "smoothing": M_SMOOTH, "classifier": cinfo, "graph communities": gstats,
                      "lexical abbreviation caveat": sorted(ABBREV)},
        "score distributions": distributions(scores),
        "spearman": agreement(scores),
        "phase1 brands": {"top by share of posts in the top decile of combined": top, **tinfo},
        "sample design": design,
        "files": {"scores": "data/processed/offbrand_scores.csv", "sample": "data/processed/offbrand_sample.csv",
                  "blind passes": "data/manual/offbrand_passes/"},
        "todo": "phase 2 (--stage eval) after the three blind passes and the adjudication",
    }
    json.dump(num, open(ROOT / "data" / "processed" / "offbrand_numbers.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"sample": design["sample"], "per decile": design["per decile"], "per platform": design["per platform"],
                      "ambiguous in sample": design["ambiguous-name brands in sample"]}, indent=1), flush=True)
    print("top brands by share in the top decile:", [(t["brand"], t["share_top_decile"]) for t in top[:8]], flush=True)


# ---------- phase 2: labels and evaluation ----------
LABEL_COLS = ["about_brand", "not_about_brand", "unclear"]


def load_passes():
    """Read the three blind label files and the adjudication; return (labels, adjudication)."""
    parts = []
    for p in PASSES:
        fs = sorted(PASS_DIR.glob(f"labels_{p}_shard*.csv"))
        if not fs:
            raise SystemExit(f"missing label files for pass {p} in {PASS_DIR}")
        q = pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in fs], ignore_index=True)
        q = q[["sample_id", "label"]].rename(columns={"label": f"label_{p}"})
        parts.append(q)
    lab = parts[0]
    for q in parts[1:]:
        lab = lab.merge(q, on="sample_id", how="outer")
    adj = PASS_DIR / "adjudication.csv"
    if adj.exists():
        a = pd.read_csv(adj, dtype={"sample_id": str})[["sample_id", "label_final"]].rename(columns={"label_final": "label_adj"})
        lab = lab.merge(a, on="sample_id", how="left")
    else:
        lab["label_adj"] = np.nan
    for c in ["label_A", "label_B", "label_C"]:
        lab[c] = lab[c].astype(str).str.strip().str.lower().replace({"nan": np.nan, "": np.nan})
        bad = set(lab[c].dropna()) - set(LABEL_COLS)
        if bad:
            raise SystemExit(f"unexpected labels in {c}: {bad}")
    lab["label_adj"] = lab["label_adj"].astype(str).str.strip().str.lower().replace({"nan": np.nan, "": np.nan})
    return lab


def evaluate():
    num = json.load(open(ROOT / "data" / "processed" / "offbrand_numbers.json"))
    raw = eu.load(ROOT).astype({"mention_id": str})
    sc = pd.read_csv(ROOT / "data" / "processed" / "offbrand_scores.csv", dtype={"mention_id": str})
    S = pd.read_csv(ROOT / "data" / "processed" / "offbrand_sample.csv", dtype={"mention_id": str, "sample_id": str})
    lab = load_passes()
    S = S.merge(lab, on="sample_id", how="left")
    if S.label_A.isna().any():
        raise SystemExit(f"{int(S.label_A.isna().sum())} sampled posts have no labels")
    dis = S[["label_A", "label_B", "label_C"]].nunique(axis=1) > 1
    una = S[["label_A", "label_B", "label_C"]].apply(lambda r: r.value_counts().index[0] if r.value_counts().iloc[0] >= 2 else None, axis=1)
    S["label_final"] = np.where(dis, S.label_adj, S.label_A)
    S["source"] = np.where(dis, "ai_adjudicated", "ai_unanimous")
    if S[dis].label_final.isna().any():
        raise SystemExit(f"{int(S[dis].label_final.isna().sum())} disputed posts have no adjudicated label")
    S[["sample_id", "mention_id", "label_A", "label_B", "label_C", "label_final", "source"]].to_csv(
        ROOT / "data" / "processed" / "offbrand_labels.csv", index=False)

    out = {"labels": {"rows": int(len(S)), "ai generated": True, "note": "AI labels, awaiting a team member's check",
                      "counts": {k: int(v) for k, v in S.label_final.value_counts().items()},
                      "share not_about_brand (sample)": _r((S.label_final == "not_about_brand").mean()),
                      "disputed": int(dis.sum()), "unanimous": int((~dis).sum()),
                      "adjudication sided with the majority":
                          _r((S[dis].label_adj == una[dis]).mean()) if dis.sum() else None}}
    full = S[["label_A", "label_B", "label_C"]].dropna()
    out["kappa"] = {"fleiss": _r(lu.fleiss_kappa(full)), "agreement all three equal": _r((full.nunique(axis=1) == 1).mean()),
                    "pairwise cohen": {f"{a}{b}": _r(lu.cohen_kappa_score(full[f"label_{a}"], full[f"label_{b}"]))
                                       for a, b in [("A", "B"), ("A", "C"), ("B", "C")]}, "n": int(len(full))}

    # per detector AUC (unclear excluded) with a post bootstrap; design-weighted AUC as a check
    S = S.drop(columns=["text"]).merge(sc, on="mention_id", how="left", validate="one_to_one")
    # inverse sampling probability per (combined-score decile x ambiguous-name flag) cell; the sample over-weights
    # both high deciles and ambiguous brands on purpose, so the population estimate needs this weight
    amb = pd.read_csv(ROOT / "data" / "processed" / "brand_industry.csv")
    amb_yes = set(amb.loc[amb.ambiguous_name == "yes", "brand"])
    S["amb"] = S.brand.isin(amb_yes)
    out["labels"]["by decile"] = {int(d): {k: int(v) for k, v in S[S.decile == d].label_final.value_counts().items()} for d in DECILES}
    pop = raw[["mention_id", "brand"]].merge(sc[["mention_id", "decile"]], on="mention_id")
    pop["amb"] = pop.brand.isin(amb_yes)
    N = pop.groupby(["decile", "amb"]).size(); n = S.groupby(["decile", "amb"]).size()
    S["w"] = [float(N.get((d, a), np.nan)) / max(int(n.get((d, a), 0)), 1) for d, a in zip(S.decile, S.amb)]
    w_dec = S.w
    keep = S[["label_final", "combined"]].notna().all(axis=1) & S.label_final.isin(["about_brand", "not_about_brand"])
    y = (S.loc[keep, "label_final"] == "not_about_brand").astype(int).values
    out["auc"] = {}
    AUC_KEYS = {"lex_pct": "lexical", "emb_pct": "embedding", "graph_tfidf_pct": "graph TF-IDF", "graph_e5_pct": "graph e5",
                "graph_pct": "graph (both)", "clf_pct": "classifier", "combined": "combined"}
    SC = {v: S[k] for k, v in AUC_KEYS.items()}
    SC["combined without lexical (post-hoc)"] = S[["emb_pct", "graph_pct", "clf_pct"]].mean(axis=1, skipna=True)
    for c in list(AUC_KEYS.values()) + ["combined without lexical (post-hoc)"]:
        ok = keep & SC[c].notna(); yy = (S.loc[ok, "label_final"] == "not_about_brand").astype(int).values; ss = SC[c][ok].values
        if yy.sum() < 5 or len(yy) - yy.sum() < 5:
            out["auc"][c] = {"n": int(len(yy)), "auc": None}; continue
        rng = np.random.default_rng(SEED); b = []
        for _ in range(1000):
            i = rng.integers(0, len(yy), len(yy))
            if 0 < yy[i].sum() < len(yy):
                b.append(roc_auc_score(yy[i], ss[i]))
        ww = w_dec[ok].values
        out["auc"][c] = {"auc": _r(roc_auc_score(yy, ss)), "low": _r(np.percentile(b, 2.5)), "high": _r(np.percentile(b, 97.5)),
                                   "n": int(len(yy)), "not_about": int(yy.sum()), "auc design-weighted": _r(_w_auc(yy, ss, ww))}

    out["precision"] = {}
    for q, qname in [(0.95, "top 5%"), (0.90, "top 10%")]:
        m = S.combined_rank >= q
        lab_here = m & S.label_final.notna()
        n_all = int(m.sum()); n_lab = int(lab_here.sum())
        defin = S[lab_here & S.label_final.isin(["about_brand", "not_about_brand"])]
        out["precision"][qname] = {
            "posts in the population": int((sc.combined_rank >= q).sum()),
            "labelled posts in the region": n_all, "labelled": n_lab, "unclear": int((S.loc[lab_here, "label_final"] == "unclear").sum()),
            "precision (unclear excluded)": _r((defin.label_final == "not_about_brand").mean()) if len(defin) else None,
            "precision (unclear as not off-brand)": _r((S.loc[lab_here, "label_final"] == "not_about_brand").mean()),
            "share of all not_about_brand labels in the region": _r(m[S.label_final == "not_about_brand"].mean())}

    # design-weighted population estimate (unclear reported separately)
    w = w_dec.values; f = S.label_final
    est = {}
    for name, sel in [("excluding unclear", f.isin(["about_brand", "not_about_brand"])), ("unclear as not off-brand", f.notna())]:
        ok = sel.values
        est[name] = {"estimate": _r(np.average((f[ok] == "not_about_brand").astype(float), weights=w[ok])), "n": int(ok.sum())}
    rng = np.random.default_rng(SEED); bs = {}
    for name, sel in [("excluding unclear", f.isin(["about_brand", "not_about_brand"])), ("unclear as not off-brand", f.notna())]:
        ok = sel.values; v = (f[ok] == "not_about_brand").astype(float).values; ww = w[ok]; e = []
        for _ in range(1000):
            i = rng.integers(0, len(v), len(v)); e.append(np.average(v[i], weights=ww[i]))
        bs[name] = [_r(np.percentile(e, 2.5)), _r(np.percentile(e, 97.5))]
    for k in est:
        est[k]["low"], est[k]["high"] = bs[k]
    est["unclear share (design-weighted)"] = _r(np.average((f == "unclear").astype(float), weights=w))
    hard = ((f == "not_about_brand") & (S.lex == 1)).values        # off-brand and the brand string is not even in the text
    ok = f.notna().values
    est["conservative lower bound: off-brand and no brand string in the text"] = {
        "estimate": _r(np.average(hard[ok].astype(float), weights=w[ok])), "n": int(ok.sum()),
        "note": "a false match that even a name-only check can see; the rest of the off-brand posts do name the keyword"}
    out["population estimate"] = est

    # per platform
    pp = {}
    for p in eu.PLATFORMS:
        m = S.platform == p; lab_p = m & f.notna()
        pp[eu.NAMES[p]] = {"labelled": int(lab_p.sum()), "not_about_brand": int((f[lab_p] == "not_about_brand").sum()),
                           "share not_about_brand": _r((f[lab_p] == "not_about_brand").mean()) if lab_p.sum() else None,
                           "unclear": int((f[lab_p] == "unclear").sum()),
                           "design-weighted share": _r(np.average((f[lab_p] == "not_about_brand").astype(float), weights=w[lab_p.values]))
                           if lab_p.sum() else None}
    out["per platform"] = pp

    # per brand (labels are thin per brand: report brands with >= 3 labelled posts)
    g = pd.DataFrame({"brand": S.brand.values, "lab": f.values}).dropna()
    g["off"] = (g.lab == "not_about_brand").astype(float); g["defin"] = g.lab.isin(["about_brand", "not_about_brand"])
    bt = g.groupby("brand").agg(labelled=("lab", "size"), not_about=("off", "sum"), definite=("defin", "sum"))
    bt = bt[bt.definite >= 5].assign(share=lambda z: z.not_about / z.definite).sort_values("share", ascending=False)
    out["per brand"] = {"brands with >= 5 labelled posts": int(len(bt)),
                        "top": [{"brand": b, "labelled": int(r.labelled), "not_about": int(r.not_about), "share": _r(r.share)} for b, r in bt.head(15).iterrows()]}

    # error of the final model on the suspicious posts (evaluation only: views are read here, not in phase 1)
    f_meta, meta = em.load(ROOT)
    pr = pd.read_csv(ROOT / "data" / "processed" / "results_preds.csv")
    rows = []
    for p in ["tiktok", "youtube", "twitter"]:
        m = (meta.platform == p).values; M = meta[m].reset_index(drop=True); q = pr[pr.platform == p].reset_index(drop=True)
        assert len(M) == len(q)
        d = pd.DataFrame({"mention_id": raw.loc[M.index, "mention_id"].values, "platform": p, "brand": M.brand.values,
                          "y": M.y.values, "pred": q.pred.values})
        rows.append(d)
    D = pd.concat(rows, ignore_index=True).merge(sc[["mention_id", "combined_rank"]], on="mention_id", how="left")
    D["e"] = D.pred - D.y                                        # + = over-estimate
    out["model error"] = {}
    for region, sel in [("top 5% of combined", D.combined_rank >= 0.95), ("rest", D.combined_rank < 0.95)]:
        d = D[sel & D.combined_rank.notna()]
        mae, lo, hi = va.mae_ci(np.abs(d.e).values, d.brand.values)
        out["model error"][region] = {"posts": int(len(d)), "MAE log10": _r(mae), "factor": _r(10 ** mae, 2), "low": _r(lo), "high": _r(hi),
                                      "median bias (+ = over)": _r(np.median(d.e)), "median views": _r(np.median(10 ** d.y), 0)}
    for p in ["tiktok", "youtube", "twitter"]:
        d = D[(D.platform == p) & (D.combined_rank >= 0.95)]; r = D[(D.platform == p) & (D.combined_rank < 0.95)]
        out["model error"][f"top 5% {eu.NAMES[p]}"] = {"posts": int(len(d)), "MAE log10": _r(np.abs(d.e).mean()),
                                                      "rest MAE log10": _r(np.abs(r.e).mean()), "median bias (+ = over)": _r(np.median(d.e))}

    D = D.merge(S[["mention_id", "label_final"]], on="mention_id", how="left")
    for lab in ["about_brand", "not_about_brand"]:
        d = D[D.label_final == lab]
        mae, lo, hi = va.mae_ci(np.abs(d.e).values, d.brand.values)
        out["model error"][f"labelled {lab}"] = {"posts": int(len(d)), "MAE log10": _r(mae), "factor": _r(10 ** mae, 2),
                                                 "low": _r(lo), "high": _r(hi), "median bias (+ = over)": _r(np.median(d.e))}

    # brand view totals if the top-5% posts are dropped (descriptive)
    D["views"] = 10 ** D.y; D["dropped"] = (D.combined_rank >= 0.95).astype(int); D["views_dropped"] = D.views * D.dropped
    gb = D.groupby("brand").agg(posts=("views", "size"), views=("views", "sum"), views_drop=("views_dropped", "sum"))
    gb["kept"] = gb.views - gb.views_drop; gb["ratio"] = gb.kept / gb.views
    big = gb[gb.posts >= 5]
    out["brand totals"] = {"brands with >= 5 posts with views": int(len(big)), "posts": int(gb.posts.sum()),
                           "median share of a brand's views kept": _r(big.ratio.median()),
                           "brands losing > 10% of views": _r((big.ratio < 0.9).mean()), "brands losing > 25% of views": _r((big.ratio < 0.75).mean()),
                           "all-brand views kept": _r(gb.kept.sum() / gb.views.sum()),
                           "largest absolute losses": [{"brand": b, "posts": int(r.posts), "share of views dropped": _r(1 - r.ratio)}
                                                       for b, r in gb.assign(loss=lambda z: z.views - z.kept).sort_values("loss", ascending=False).head(10).iterrows()]}
    num["validation"] = out
    num.pop("todo", None)
    json.dump(num, open(ROOT / "data" / "processed" / "offbrand_numbers.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"auc": out["auc"], "kappa": out["kappa"], "precision": out["precision"],
                      "population": out["population estimate"]}, indent=1), flush=True)


def _w_auc(y, s, w):
    """AUC with inverse-probability weights (weighted Mann-Whitney, ties counted as half), O(n_pos * n_neg)."""
    y = np.asarray(y); s = np.asarray(s); w = np.asarray(w, dtype=float)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    sp, sn, wp, wn = s[pos], s[neg], w[pos], w[neg]
    cmp = (sp[:, None] > sn[None, :]).astype(float) + 0.5 * (sp[:, None] == sn[None, :])
    return float(((wp[:, None] * wn[None, :]) * cmp).sum() / (wp.sum() * wn.sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all", choices=["all", "eval"])
    a = ap.parse_args()
    evaluate() if a.stage == "eval" else run_scores()
