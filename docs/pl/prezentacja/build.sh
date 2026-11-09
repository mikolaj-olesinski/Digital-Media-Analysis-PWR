#!/bin/bash
# Builds main.pdf (slides, one page = one slide) from the preprint figures and tables.
cd "$(dirname "$0")"
for f in data_overview preprint_main preprint_needs preprint_side preprint_bins final_compare; do cp ../../../figures/${f}_pl.png fig/; done
sed -n '/\\begin{tabular}/,/\\end{tabular}/p' ../preprint/combos_table.tex > combos_tabular.tex
latexmk -xelatex -interaction=nonstopmode -quiet main.tex
