"""Helpers for notebooks/eda/03_labels.ipynb: coder agreement, adjudication and label-vs-reach views."""
from itertools import combinations
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import eda_utils as eu

TYPE_ORDER = ["organic", "promo", "spam", "unclear"]
SENT_ORDER = ["negative", "neutral", "positive"]
FIELDS = {"type": ("type", TYPE_ORDER), "sent": ("sentiment", SENT_ORDER)}   # column prefix -> (name, label order)
LABEL_COLS = ["platform", "brand", "type", "type_A", "type_B", "type_C", "type_judge", "sentiment", "sent_A", "sent_B", "sent_C", "sent_judge", "type_source"]


def load(root):
    """Raw data joined with final labels and per-coder labels (Brand24's own sentiment is renamed sentiment_b24)."""
    raw = eu.load(root).rename(columns={"sentiment": "sentiment_b24"}).astype({"mention_id": str})
    lab = pd.read_csv(root / "data" / "processed" / "mention_labels.csv", dtype={"mention_id": str})
    cod = pd.read_csv(root / "data" / "processed" / "mention_labels_coders.csv", dtype={"mention_id": str}, keep_default_na=False)
    return raw.merge(lab, on="mention_id").merge(cod, on="mention_id")


def _coders(k):
    return [f"{k}_{c}" for c in "ABC"]


def fleiss_kappa(m):
    cats = sorted(set(m.values.ravel())); n = m.shape[1]
    cnt = np.array([[(row == c).sum() for c in cats] for row in m.values])
    p_i = ((cnt ** 2).sum(1) - n) / (n * (n - 1)); p_j = cnt.sum(0) / cnt.sum()
    return (p_i.mean() - (p_j ** 2).sum()) / (1 - (p_j ** 2).sum())


def distribution(d):
    """Final label shares per platform (stacked bars) and counts with a TOTAL row."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
    for a, (col, order) in zip(ax, [("type", TYPE_ORDER), ("sentiment", SENT_ORDER)]):
        (pd.crosstab(d.platform, d[col], normalize="index")[order].reindex(eu.PLATFORMS) * 100).rename(index=eu.NAMES).plot.barh(stacked=True, ax=a, colormap="viridis")
        a.invert_yaxis(); a.set_xlabel("share of mentions (%)"); a.set_ylabel("platform"); a.set_title(f"Final {col} by platform", fontweight="bold")
        a.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, title=col)
    plt.tight_layout(); plt.show()
    return {col: pd.crosstab(d.platform, d[col], margins=True, margins_name="TOTAL").reindex(eu.PLATFORMS + ["TOTAL"])[order + ["TOTAL"]]
            for col, order in [("type", TYPE_ORDER), ("sentiment", SENT_ORDER)]}


def agreement(d):
    """Pairwise Cohen kappa, linear-weighted kappa for sentiment, and Fleiss kappa over the three coders."""
    rows = []
    for k, (name, order) in FIELDS.items():
        for a, b in combinations(_coders(k), 2):
            ok = (d[a] != "") & (d[b] != "")
            rows.append({"field": name, "pair": a[-1] + b[-1], "agreement": (d.loc[ok, a] == d.loc[ok, b]).mean(), "kappa": cohen_kappa_score(d.loc[ok, a], d.loc[ok, b]),
                         "weighted kappa": cohen_kappa_score(d.loc[ok, a], d.loc[ok, b], labels=order, weights="linear") if k == "sent" else np.nan})
        full = d[(d[_coders(k)] != "").all(axis=1)][_coders(k)]
        rows.append({"field": name, "pair": "all three", "agreement": (full.nunique(axis=1) == 1).mean(), "kappa": fleiss_kappa(full), "weighted kappa": np.nan})
    return pd.DataFrame(rows).set_index(["field", "pair"])


def unanimity(d):
    """Share of mentions where all three coders agree, per platform and in total."""
    u = pd.DataFrame({name: d[_coders(k)].nunique(axis=1).eq(1).groupby(d.platform).mean() * 100 for k, (name, _) in FIELDS.items()}).reindex(eu.PLATFORMS)
    u.loc["TOTAL"] = [d[_coders(k)].nunique(axis=1).eq(1).mean() * 100 for k in FIELDS]
    return u.rename(index=eu.NAMES).rename(columns=lambda c: f"unanimous {c} (%)")


def confusions(d):
    """Label pairs summed over all coder pairs: where do coders confuse classes?"""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, (k, (name, order)) in zip(ax, FIELDS.items()):
        fix = lambda col: d[col].where(d[col].isin(order), order[0])
        cm = sum(confusion_matrix(fix(x), fix(y), labels=order) for x, y in combinations(_coders(k), 2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=order, yticklabels=order, ax=a, cbar=False)
        a.set_title(f"{name}: label pairs over coder pairs", fontweight="bold"); a.set_xlabel("second coder"); a.set_ylabel("first coder")
    plt.tight_layout(); plt.show()


def adjudication(d):
    """For disputed rows with a 2-of-3 majority: how often does the adjudicator side with the majority?"""
    out = {}
    for k, (name, _) in FIELDS.items():
        src, judge = ("type_source", "type_judge") if k == "type" else ("sentiment_source", "sent_judge")
        s = d[d[src] == "sonnet_adjudicated"]
        top = s[_coders(k)].apply(lambda r: r.value_counts().index[0] if r.value_counts().iloc[0] >= 2 and r.value_counts().index[0] != "" else None, axis=1)
        has = top.notna()
        out[name] = {"disputed rows": len(s), "with 2-of-3 majority": int(has.sum()), "judge sides with majority (%)": (s.loc[has, judge] == top[has]).mean() * 100,
                     **{f"judge = coder {c} (%)": (s[judge] == s[f"{k}_{c}"]).mean() * 100 for c in "ABC"}}
    return pd.DataFrame(out).round(1)


def reach_by_label(d):
    """Median Brand24 reach by mention type (table) and log10 reach by type and sentiment (boxplots)."""
    r = d[d[eu.REACH] > 0].assign(log_reach=lambda x: np.log10(x[eu.REACH]))
    med = d.pivot_table(index="platform", columns="type", values=eu.REACH, aggfunc="median")[TYPE_ORDER].reindex(eu.PLATFORMS)
    med.loc["TOTAL"] = d.groupby("type")[eu.REACH].median()
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True)
    for a, (col, order) in zip(ax, [("type", TYPE_ORDER), ("sentiment", SENT_ORDER)]):
        sns.boxplot(data=r, x="platform", y="log_reach", hue=col, hue_order=order, order=eu.PLATFORMS, ax=a, fliersize=1)
        a.set_xlabel("platform"); a.set_ylabel("log10(Brand24 reach)"); a.set_title(f"Reach by {col}", fontweight="bold")
        a.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, title=col)
    plt.tight_layout(); plt.show()
    return med.rename(index=eu.NAMES | {"TOTAL": "TOTAL"})
