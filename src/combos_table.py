"""LaTeX table of all 31 training-platform combinations (E201) for the preprint.
Reads data/processed/train_combos_numbers.json (run src/train_combos.py first).
Writes docs/pl/preprint/combos_table.tex (Polish) and docs/en/preprint/combos_table.tex (English).
    python src/combos_table.py"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = ["tiktok", "youtube", "twitter", "facebook", "instagram"]
AB = {"tiktok": "TT", "youtube": "YT", "twitter": "X", "facebook": "FB", "instagram": "IG"}
EDGES = [2.45, 3.0, 3.35, 3.9, 4.2, 4.7, 5.0]
COLS = ["DDEBFC", "C3DCF8", "A3C8F2", "7FB0EA", "5A95DF", "3779CF", "235FB0", "184A8C"]
TRAIN = {"pl": r"\cellcolor[HTML]{EEEDEA}{\color{black!35}trening}", "en": r"\cellcolor[HTML]{EEEDEA}{\color{black!35}train}"}
LANG = "pl"
SE = False   # deck variant: extra Stack Exchange column (never in training)


def cell(v, bold=False):
    k = int(np.searchsorted(EDGES, v)); t = f"{v:.2f}".replace(".", "," if LANG == "pl" else ".")
    t = rf"\textbf{{{t}}}" if bold else t
    return rf"\cellcolor[HTML]{{{COLS[k]}}}" + (rf"\textcolor{{white}}{{{t}}}" if k >= 5 else t)


def main():
    d = json.loads((ROOT / "data/processed/train_combos_numbers.json").read_text())["E201"]
    c, own = d["combinations"], d["own platform (brand CV)"]
    sx = d["stackexchange (extra test platform, never in training)"]
    EXTRA = ({"youtube", "twitter"}, {"tiktok", "youtube", "twitter"})
    show = lambda a: len(a.split(" + ")) in (1, 4) or set(a.split(" + ")) in EXTRA
    best = {b: min(v[b]["factor"] for a, v in c.items() if b in v and show(a)) for b in P}
    rows = []
    for k in range(1, 5):
        g = [a for a in c if len(a.split(" + ")) == k and show(a)]
        g.sort(key=lambda a: np.mean([c[a][b]["MAE"] for b in c[a]]))
        for a in g:
            tr = a.split(" + ")
            lab = " + ".join(AB[q] for q in sorted(tr, key=lambda q: [c[a].get(b, {"MAE": 0})["MAE"] for b in P] and P.index(q)))
            cells = [TRAIN[LANG] if b in tr else cell(c[a][b]["factor"], round(c[a][b]["factor"], 2) == round(best[b], 2)) for b in P]
            lab = {"YT + X": "X + YT"}.get(lab, lab)
            if SE: cells.append(cell(sx["combinations"][a]["factor"]))
            rows.append(lab + " & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows = [r for i, r in enumerate(rows) if not (r == r"\midrule" and i and rows[i - 1] == r"\midrule")][:-1]
    if LANG == "pl":
        head = r"""\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.05}
\caption{Typowy błąd (\x{}) regresji na lajkach i komentarzach przy różnych platformach treningowych.
\capdet{Ridge na $\log$ lajków i komentarzy, posty z platform treningowych połączone w jeden zbiór. Wiersz: platformy
w treningu (TT TikTok, YT YouTube, X, FB Facebook, IG Instagram): pojedyncze, X + YouTube (zestaw główny), TikTok + YouTube + X
i wszystkie z czterema platformami; pełna tabela 30 zestawów jest w repozytorium. Kolumna: platforma testowa, której nie było
w treningu. Pierwszy wiersz: ten sam ridge uczony na platformie testowej (walidacja po markach), a nie CatBoost
z Rys.~\ref{fig:main}. Pogrubione: najlepszy wynik w kolumnie wśród pokazanych zestawów (bez pierwszego wiersza). Ciemniej: większy błąd.}}
\label{tab:combos}
\begin{tabular}{l|ccccc}
\toprule
Trening na & TikTok & YouTube & X & Facebook & Instagram \\
\midrule
\textit{ridge, ta sama platforma} & """ + " & ".join(cell(own[b]["factor"]) for b in P) + r" \\ \midrule" + "\n"
    if LANG == "en":
        head = r"""\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.05}
\caption{Typical error (\x{}) of the regression on likes and comments with different training platforms.
\capdet{Ridge on $\log$ likes and comments, posts from the training platforms pooled into one set. Row: platforms in
training (TT TikTok, YT YouTube, X, FB Facebook, IG Instagram): single ones, X + YouTube (the main set), TikTok + YouTube + X
and all sets of four platforms; the full table of 30 sets is in the repository. Column: test platform, absent from training. First row:
the same ridge trained on the test platform itself (brand-grouped validation), not the CatBoost of Fig.~\ref{fig:main}. Bold:
best result in the column among the training sets shown (first row excluded). Darker: larger error.}}
\label{tab:combos}
\begin{tabular}{l|ccccc}
\toprule
Trained on & TikTok & YouTube & X & Facebook & Instagram \\
\midrule
\textit{ridge, same platform} & """ + " & ".join(cell(own[b]["factor"]) for b in P) + r" \\ \midrule" + "\n"
    if SE:
        sep = "," if LANG == "pl" else "."
        tab = (r"\begin{tabular}{l|ccccc|c}" + "\n" + r"\toprule" + "\n" + ("Trening na" if LANG == "pl" else "Trained on")
               + r" & TikTok & YouTube & X & Facebook & Instagram & Stack Exchange \\" + "\n" + r"\midrule" + "\n"
               + (r"\textit{ridge, ta sama platforma} & " if LANG == "pl" else r"\textit{ridge, same platform} & ")
               + " & ".join(cell(own[b]["factor"]) for b in P) + " & " + cell(sx["own platform (brand CV)"]["factor"]) + r" \\ \midrule" + "\n"
               + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
        (ROOT / "docs/pl/prezentacja/combos_tabular.tex").write_text(tab); return
    out = head + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}
"""
    (ROOT / f"docs/{LANG}/preprint/combos_table.tex").write_text(out)
    print({b: round(best[b], 2) for b in P})


if __name__ == "__main__":
    for LANG in ("pl", "en"):
        main()
    LANG, SE = "pl", True; main()
