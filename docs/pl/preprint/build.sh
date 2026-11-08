#!/usr/bin/env bash
# Builds main.pdf: latexmk (texlive) if present, else tectonic.
cd "$(dirname "$0")"
for f in data_overview train_combos_simple preprint_main preprint_size preprint_needs preprint_methods preprint_side preprint_bins final_compare preprint_ladder; do cp ../../../figures/${f}_pl.png fig/; done
if command -v latexmk >/dev/null; then latexmk -xelatex -interaction=nonstopmode -quiet main.tex
else tectonic main.tex; fi
