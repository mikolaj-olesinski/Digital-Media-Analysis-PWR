#!/usr/bin/env bash
# Builds main.pdf: latexmk (texlive) if present, else tectonic.
cd "$(dirname "$0")"
for f in data_overview results_a results_b train_combos_simple; do cp ../../../figures/${f}_pl.png fig/; done
if command -v latexmk >/dev/null; then latexmk -xelatex -interaction=nonstopmode -quiet main.tex
else tectonic main.tex; fi
