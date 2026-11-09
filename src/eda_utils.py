"""Reusable EDA blocks for notebooks/eda/02_platforms.ipynb: one overview of all platforms and one identical report per platform.
Every platform table/figure also shows the total over all platforms for comparison."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.dates as mdates
from IPython.display import display, Markdown

REACH = "b24_estimated_reach"
PLATFORMS = ["tiktok", "youtube", "twitter", "facebook", "instagram"]
NAMES = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X (Twitter)", "facebook": "Facebook", "instagram": "Instagram"}
PCOL = {"tiktok": "#3f2a6b", "youtube": "#e4572e", "twitter": "#2a9d8f", "facebook": "#2f6db5", "instagram": "#c9a227"}
ALL_COL = "#9a9aa8"
NUM = ["author_followers", "view_count", "likes", "comments", "shares", REACH]
COLS = ["created_at", "metrics_updated_at", "brand", "language", "country", "title", "content",
        "author_followers", "view_count", "likes", "comments", "shares", "sentiment", REACH]


def load(root):
    return pd.read_csv(Path(root) / "data" / "raw" / "reach_samples_students.csv",
                       parse_dates=["created_at", "metrics_updated_at"])


def missing_zeros(g):
    """% missing and % zero per column (zeros only for numeric columns)."""
    return pd.DataFrame({"missing %": g[COLS].isna().mean() * 100,
                         "zeros %": [(g[c] == 0).mean() * 100 if c in NUM else np.nan for c in COLS]}, index=COLS)


FEATURE_SETS = {"followers": ["author_followers"], "followers + likes + comments": ["author_followers", "likes", "comments"],
                "+ shares": ["author_followers", "likes", "comments", "shares"], "+ views (target)": ["author_followers", "likes", "comments", "view_count"]}


def usable_rows(df):
    """Rows with every feature of a set present, per platform (and total)."""
    t = pd.DataFrame({n: {NAMES[p]: df[df.platform == p][c].notna().all(axis=1).sum() for p in PLATFORMS} for n, c in FEATURE_SETS.items()}).T
    t["TOTAL"] = t.sum(axis=1)
    return t.astype(int)


def gaps(g, fields=("author_followers", "view_count", "likes", "comments", "shares", "sentiment", "country")):
    """Partly missing fields (2-98%): is the gap related to Brand24 reach? A negative delta means missing = smaller posts."""
    from scipy.stats import spearmanr
    lr = np.log10(g[REACH].where(g[REACH] > 0)); out = {}
    for f in fields:
        m = g[f].isna()
        if 0.02 < m.mean() < 0.98:
            out[f] = {"missing %": m.mean() * 100, "median log10 reach, missing": lr[m].median(), "median log10 reach, present": lr[~m].median(),
                      "delta": lr[m].median() - lr[~m].median(), "rho (missing, reach)": spearmanr(m, lr, nan_policy="omit")[0],
                      "median month, missing": g.loc[m, "created_at"].median().strftime("%Y-%m"), "median month, present": g.loc[~m, "created_at"].median().strftime("%Y-%m")}
    return pd.DataFrame(out).T


def key_numbers(g):
    r = g[(g.author_followers > 0) & (g[REACH] > 0)]
    ratio = r[REACH] / r.author_followers
    top = g[REACH].dropna()
    return pd.Series({
        "mentions": len(g), "brands": g.brand.nunique(),
        "reach median": g[REACH].median(), "reach mean / median": g[REACH].mean() / g[REACH].median(),
        "reach top-1% share %": top.nlargest(max(1, len(top) // 100)).sum() / top.sum() * 100,
        "views known %": g.view_count.notna().mean() * 100,
        "reach == views %": (g[REACH] == g.view_count)[g.view_count.notna()].mean() * 100,
        "median reach / followers": ratio.median(),
        "english %": (g.language == "en").mean() * 100,
        "median metrics age, days": (g.metrics_updated_at - g.created_at).dt.days.median()})


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, fontweight="bold"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)


def _months(ax, color_by=None):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def overview(df):
    """Section 0: all platforms together."""
    display(Markdown("# All platforms (total)"))
    display(Markdown("### 0.1 · First rows (all columns)"))
    display(df.sample(3, random_state=0).assign(content=lambda x: x.content.fillna("").str.slice(0, 80),
                                                   title=lambda x: x.title.fillna("").str.slice(0, 80)).T)

    display(Markdown("### 0.2 · Missing values and zeros"))
    mz = pd.concat({NAMES[p]: missing_zeros(df[df.platform == p])["missing %"] for p in PLATFORMS} | {"TOTAL": missing_zeros(df)["missing %"]}, axis=1).T
    zz = pd.concat({NAMES[p]: missing_zeros(df[df.platform == p])["zeros %"] for p in PLATFORMS} | {"TOTAL": missing_zeros(df)["zeros %"]}, axis=1).T[NUM]
    fig, ax = plt.subplots(1, 2, figsize=(17, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    for a, t, title, cm in [(ax[0], mz, "Missing values", "Purples"), (ax[1], zz, "Zero values", "Oranges")]:
        im = a.imshow(t.values.astype(float), cmap=cm, vmin=0, vmax=100, aspect="auto")
        a.set_xticks(range(t.shape[1])); a.set_xticklabels(t.columns, rotation=45, ha="right"); a.set_yticks(range(len(t))); a.set_yticklabels(t.index)
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.values[i, j]
                if not np.isnan(v): a.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if v > 55 else "black")
        a.axhline(len(t) - 1.5, color="black", lw=1.2)
        _style(a, f"{title} by attribute (% of mentions)", "attribute", "platform"); a.grid(False)
    plt.tight_layout(); plt.show()

    display(Markdown("### 0.3 · Key numbers per platform"))
    kn = pd.DataFrame({NAMES[p]: key_numbers(df[df.platform == p]) for p in PLATFORMS} | {"TOTAL": key_numbers(df)}).T
    display(kn.round(2))

    display(Markdown("### 0.4 · Usable rows: all features of a set present"))
    display(usable_rows(df))

    display(Markdown("### 0.5 · Are partial gaps related to reach? (missing = smaller posts if delta < 0)"))
    display(pd.concat({NAMES[p]: gaps(df[df.platform == p]) for p in PLATFORMS}).round(2))

    display(Markdown("### 0.6 · Mentions over time"))
    m = df.assign(month=df.created_at.dt.to_period("M").dt.to_timestamp()).groupby(["month", "platform"]).size().unstack().reindex(columns=PLATFORMS).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 3.4)); bottom = np.zeros(len(m))
    for p in PLATFORMS:
        ax.bar(m.index, m[p], width=20, bottom=bottom, color=PCOL[p], label=NAMES[p]); bottom += m[p].values
    _months(ax); _style(ax, "Mentions per month (stacked, total = bar height)", "month of post creation", "mentions")
    ax.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False); plt.tight_layout(); plt.show()


def platform_report(df, p):
    g, c = df[df.platform == p], PCOL[p]; N = NAMES[p]
    display(Markdown(f"# {N}\n{len(g):,} mentions · {g.brand.nunique()} brands · {g.created_at.min():%Y-%m} → {g.created_at.max():%Y-%m}"))

    display(Markdown(f"### {N} · 1. First rows (all columns)"))
    display(g.head(3).assign(content=lambda x: x.content.fillna("").str.slice(0, 80), title=lambda x: x.title.fillna("").str.slice(0, 80)).T)

    display(Markdown(f"### {N} · 2. Missing values and zeros (with total over all platforms)"))
    a, t = missing_zeros(g), missing_zeros(df)
    display(pd.DataFrame({f"missing % · {N}": a["missing %"], "missing % · TOTAL": t["missing %"],
                          f"zeros % · {N}": a["zeros %"], "zeros % · TOTAL": t["zeros %"]}).T.round(1))
    gp = gaps(g)
    if len(gp):
        display(Markdown("Partial gaps vs Brand24 reach (delta < 0: missing = smaller posts)")); display(gp.round(2))
    else:
        print("no partial gaps: every field is either always present or never present")
    fig, ax = plt.subplots(figsize=(8, 4.2)); y = np.arange(len(COLS)); h = .38
    ax.barh(y - h / 2, a["missing %"], h, color=c, label=N); ax.barh(y + h / 2, t["missing %"], h, color=ALL_COL, label="All platforms")
    ax.set_yticks(y); ax.set_yticklabels(COLS); ax.invert_yaxis(); ax.set_xlim(0, 100)
    _style(ax, f"Missing values by attribute: {N} vs total", "missing values (% of mentions)", "attribute")
    ax.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False); plt.tight_layout(); plt.show()

    display(Markdown(f"### {N} · 3. Distributions (log10, positive values only)"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.4))
    for a_, col, lab in zip(ax, [REACH, "view_count", "likes"], ["Brand24 reach", "views", "likes"]):
        v, vt = g.loc[g[col] > 0, col], df.loc[df[col] > 0, col]
        if len(v):
            a_.hist(np.log10(vt), bins=40, density=True, color=ALL_COL, alpha=.55, label="All platforms")
            a_.hist(np.log10(v), bins=40, density=True, color=c, alpha=.8, label=N)
        else:
            a_.text(.5, .5, "no data on this platform", ha="center", transform=a_.transAxes)
        _style(a_, f"{lab} (n={len(v):,})", f"log10({lab})", "density")
    ax[-1].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False); plt.tight_layout(); plt.show()

    display(Markdown(f"### {N} · 4. Mentions over time"))
    m = g.set_index("created_at").resample("MS").size()
    fig, ax = plt.subplots(figsize=(7, 3)); ax.bar(m.index, m.values, width=20, color=c); _months(ax)
    _style(ax, f"{N}: mentions per month", "month of post creation", "mentions"); plt.tight_layout(); plt.show()

    display(Markdown(f"### {N} · 5. Key numbers (with total over all platforms)"))
    display(pd.DataFrame({N: key_numbers(g), "TOTAL": key_numbers(df)}).T.round(2))

    display(Markdown(f"### {N} · 6. Spearman correlations of log1p values"))
    cr = {t_: g[NUM[:-1]].apply(np.log1p).corrwith(np.log1p(g[t_]), method="spearman") for t_ in [REACH, "view_count"] if g[t_].notna().sum() > 30}
    if cr:
        x = pd.DataFrame(cr).T; x = x.drop(columns=[i for i in cr if i in x.columns]); x.index = [f"with {i}" for i in x.index]; display(x.round(2))
    else:
        print("no numeric target with enough data")

    display(Markdown(f"### {N} · 7. Text, language and sentiment"))
    def prof(d):
        txt = d.content.fillna(d.title).fillna("")
        return pd.Series({"english %": (d.language == "en").mean() * 100, "empty text %": (txt.str.len() == 0).mean() * 100,
                          "median text length": txt.str.len().median(), "duplicate text %": txt[txt.str.len() > 20].duplicated(keep=False).mean() * 100})
    display(pd.DataFrame({N: prof(g), "TOTAL": prof(df)}).T.round(1))
    sent = lambda d: d.sentiment.value_counts(normalize=True, dropna=False).mul(100)
    display(pd.DataFrame({N: sent(g), "TOTAL": sent(df)}).fillna(0).T.round(1).rename(columns=lambda c: "missing" if pd.isna(c) else c))
    display(pd.DataFrame({N: g.brand.value_counts().head(5)}).T)
