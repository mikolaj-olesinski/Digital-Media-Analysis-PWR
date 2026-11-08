#!/usr/bin/env bash
# Builds main.pdf: latexmk (texlive) if present, else tectonic.
cd "$(dirname "$0")"
cp ../../../figures/data_overview_pl.png fig/
if command -v latexmk >/dev/null; then latexmk -xelatex -interaction=nonstopmode -quiet main.tex
else tectonic main.tex; fi
