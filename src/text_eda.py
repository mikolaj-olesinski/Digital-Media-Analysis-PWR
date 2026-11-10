"""E239: preliminary text analysis per platform (5 platforms, 2,500 mentions each).
Per platform: share with text, length, language, hashtags / @mentions / links / emoji / questions, exact duplicate texts,
mention type and sentiment (AI labels, E37 / E176 label files), distinctive words (log-odds, English posts only), and the
median views by mention type where views exist (TikTok, YouTube, X). No author names, handles or links are written; only
counts and aggregate words. Writes data/processed/text_eda_numbers.json and figures/text_eda_pl.png.
    .venv/bin/python -W ignore src/text_eda.py"""
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ["tiktok", "youtube", "twitter", "facebook", "instagram"]; NM = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X", "facebook": "Facebook", "instagram": "Instagram"}
URL = re.compile(r"https?://\S+|www\.\S+"); HASH = re.compile(r"#\w+"); MEN = re.compile(r"@\w+"); EMO = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
WORD = re.compile(r"[a-z][a-z']{2,}")
STOP = set("the and for you with this that are was have has from but not they their your our all can will just out about what when who how his her its more get one has had been were than them then into over now new like also via per".split())


def clean(t):
    t = URL.sub(" ", t); t = MEN.sub(" ", t); return t.lower()


def main():
    raw = eu.load(ROOT).astype({"mention_id": str}); lab = pd.read_csv(ROOT / "data/processed/mention_labels.csv", dtype={"mention_id": str})[["mention_id", "type", "sentiment"]]
    d = raw.merge(lab, on="mention_id", how="left", suffixes=("_b24", ""))
    d["text"] = (d.title.fillna("") + " " + d.content.fillna("")).str.strip(); d["has"] = d.text.str.len() > 0
    d["chars"] = d.text.str.len(); d["words"] = d.text.str.split().str.len()
    out = {}; tok = {}
    for p in P:
        g = d[d.platform == p]; t = g[g.has]; en = t[t.language == "en"]
        out[p] = {"posts": int(len(g)), "share with text": round(float(g.has.mean()), 3), "median chars": int(t.chars.median()), "p90 chars": int(t.chars.quantile(.9)), "median words": int(t.words.median()),
                  "language": {k: round(float(v), 3) for k, v in t.language.value_counts(normalize=True).head(4).items()},
                  "share with hashtag": round(float(t.text.str.contains(HASH).mean()), 3), "hashtags per post (median among those with)": int(t[t.text.str.contains(HASH)].text.str.count(HASH).median()) if t.text.str.contains(HASH).any() else 0,
                  "share with @mention": round(float(t.text.str.contains(MEN).mean()), 3), "share with link": round(float(t.text.str.contains(URL).mean()), 3),
                  "share with emoji": round(float(t.text.str.contains(EMO).mean()), 3), "share with question mark": round(float(t.text.str.contains(r"\?").mean()), 3),
                  "exact duplicate texts": round(float(t.text.duplicated(keep=False).mean()), 3),
                  "type": {k: round(float(v), 3) for k, v in g.type.value_counts(normalize=True).items()}, "sentiment (AI label)": {k: round(float(v), 3) for k, v in g.sentiment.value_counts(normalize=True).items()}}
        if g.view_count.notna().any():
            out[p]["median views by type"] = {k: int(v) for k, v in g[g.view_count > 0].groupby("type").view_count.median().items()}
        tok[p] = Counter(w for s in en.text.map(clean) for w in set(WORD.findall(s)) if w not in STOP)
    # distinctive words per platform (log-odds with a prior), English posts only
    tot = sum(tok.values(), Counter()); N = {p: sum(tok[p].values()) for p in P}; Ns = sum(N.values())
    for p in P:
        sc = {}
        for w, c in tok[p].items():
            if tot[w] < 30: continue
            a = (c + .5) / (N[p] + 1); b = (tot[w] - c + .5) / (Ns - N[p] + 1); sc[w] = np.log(a / b)
        out[p]["distinctive words (English)"] = [w for w, _ in sorted(sc.items(), key=lambda x: -x[1])[:12]]
    (ROOT / "data/processed/text_eda_numbers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    # figure: 4 panels
    INK = "#55544f"; C = ["#3d3c39", "#8a8882", "#c4c2bb", "#c4573a", "#3b6ea8"]
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.2), gridspec_kw={"hspace": .5, "wspace": .28}); x = np.arange(5)
    ax[0, 0].bar(x, [out[p]["median words"] for p in P], color="#3d3c39"); ax[0, 0].set_title("mediana słów w poście", fontsize=12, color=INK, loc="left")
    for xi, p in zip(x, P): ax[0, 0].text(xi, out[p]["median words"] + .5, out[p]["median words"], ha="center", fontsize=11)
    b = np.zeros(5)
    for k, c, lab_ in (("organic", "#c4c2bb", "organiczne"), ("promo", "#3d3c39", "promocyjne"), ("spam", "#c4573a", "spam"), ("unclear", "#e6e5e1", "niejasne")):
        v = np.array([out[p]["type"].get(k, 0) * 100 for p in P]); ax[0, 1].bar(x, v, bottom=b, color=c, label=lab_); b += v
    ax[0, 1].set_title("rodzaj wzmianki (etykieta AI), %", fontsize=12, color=INK, loc="left"); ax[0, 1].legend(frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.18))
    w = .16
    for i, (k, lab_) in enumerate((("share with hashtag", "hasztag"), ("share with link", "link"), ("share with emoji", "emoji"), ("share with question mark", "pytanie"), ("share with @mention", "@wzmianka"))):
        ax[1, 0].bar(x + (i - 2) * w, [out[p][k] * 100 for p in P], w, color=C[i], label=lab_)
    ax[1, 0].set_title("udział postów z elementem, %", fontsize=12, color=INK, loc="left"); ax[1, 0].legend(frameon=False, fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(.5, -.18))
    ax[1, 1].bar(x, [out[p]["share with text"] * 100 for p in P], w * 2.2, color="#8a8882", label="ma tekst"); ax[1, 1].bar(x + w * 2.4, [out[p]["language"].get("en", 0) * 100 for p in P], w * 2.2, color="#3d3c39", label="po angielsku")
    ax[1, 1].set_title("tekst i język, %", fontsize=12, color=INK, loc="left"); ax[1, 1].legend(frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.18))
    for a in ax.ravel():
        a.set_xticks(x if a is not ax[1, 1] else x + w * 1.2); a.set_xticklabels([NM[p] for p in P], fontsize=10.5, color=INK)
        for s_ in ("top", "right"): a.spines[s_].set_visible(False)
        a.tick_params(colors=INK, labelsize=10); a.grid(axis="y", color="#e6e5e1", lw=.8); a.set_axisbelow(True)
    fig.savefig(ROOT / "figures" / "text_eda_pl.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    for p in P: print(p, {k: v for k, v in out[p].items() if k not in ("median views by type",)})


if __name__ == "__main__":
    main()
