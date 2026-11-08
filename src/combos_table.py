"""LaTeX table of all 31 training-platform combinations (E201) for the preprint, Polish labels.
Reads data/processed/train_combos_numbers.json (run src/train_combos.py first).
Writes docs/pl/preprint/combos_table.tex.
    python src/combos_table.py"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = ["tiktok", "youtube", "twitter", "facebook", "instagram"]
AB = {"tiktok": "TT", "youtube": "YT", "twitter": "X", "facebook": "FB", "instagram": "IG"}
EDGES = [2.45, 3.0, 3.35, 3.9, 4.2, 4.7, 5.0]
COLS = ["DDEBFC", "C3DCF8", "A3C8F2", "7FB0EA", "5A95DF", "3779CF", "235FB0", "184A8C"]
TRAIN = r"\cellcolor[HTML]{EEEDEA}{\color{black!35}trening}"


def cell(v, bold=False):
    k = int(np.searchsorted(EDGES, v)); t = f"{v:.1f}".replace(".", ",")
    t = rf"\textbf{{{t}}}" if bold else t
    return rf"\cellcolor[HTML]{{{COLS[k]}}}" + (rf"\textcolor{{white}}{{{t}}}" if k >= 5 else t)


def main():
    d = json.loads((ROOT / "data/processed/train_combos_numbers.json").read_text())["E201"]
    c, own = d["combinations"], d["own platform (brand CV)"]
    best = {b: min(v[b]["factor"] for v in c.values() if b in v) for b in P}
    rows = []
    for k in range(1, 5):
        g = [a for a in c if len(a.split(" + ")) == k]
        g.sort(key=lambda a: np.mean([c[a][b]["MAE"] for b in c[a]]))
        for a in g:
            tr = a.split(" + ")
            lab = " + ".join(AB[q] for q in sorted(tr, key=lambda q: [c[a].get(b, {"MAE": 0})["MAE"] for b in P] and P.index(q)))
            cells = [TRAIN if b in tr else cell(c[a][b]["factor"], round(c[a][b]["factor"], 2) == round(best[b], 2)) for b in P]
            rows.append(lab + " & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    head = r"""\begin{table}[!htb]
\centering\scriptsize
\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.05}
\caption{Typowy błąd (\x{}) dla wszystkich 31 zestawów platform treningowych.
\capdet{Ridge na $\log$ lajków i komentarzy, posty z platform treningowych połączone w jeden zbiór. Wiersz: platformy
w treningu (TT TikTok, YT YouTube, X, FB Facebook, IG Instagram), w grupach po 1, 2, 3 i 4 platformy, w grupie od
najlepszego średnio. Kolumna: platforma testowa, której nie było w treningu. Pierwszy wiersz: ten sam model uczony na
platformie testowej (walidacja po markach). Pogrubione: najlepszy wynik w kolumnie. Ciemniej: większy błąd.}}
\label{tab:combos}
\begin{tabular}{l|ccccc}
\toprule
Trening na & TikTok & YouTube & X & Facebook & Instagram \\
\midrule
\textit{ta sama platforma (CV po markach)} & """ + " & ".join(cell(own[b]["factor"]) for b in P) + r" \\ \midrule" + "\n"
    out = head + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}
"""
    (ROOT / "docs/pl/preprint/combos_table.tex").write_text(out)
    print({b: round(best[b], 2) for b in P})


if __name__ == "__main__":
    main()
