#!/usr/bin/env bash
# Builds main.pdf: latexmk (texlive) if present, else tectonic.
cd "$(dirname "$0")"
for f in data_overview train_combos_simple preprint_main preprint_size preprint_conditions preprint_missing preprint_side; do cp ../../../figures/${f}_pl.png fig/; done
if command -v latexmk >/dev/null; then latexmk -xelatex -interaction=nonstopmode -quiet main.tex
else tectonic main.tex; fi
