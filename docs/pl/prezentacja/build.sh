#!/bin/bash
# Builds main.pdf (slides, one page = one slide) from the preprint figures and tables.
cd "$(dirname "$0")"
for f in data_overview preprint_main preprint_needs preprint_side preprint_bins final_compare preprint_methods author_history calibration_curve; do cp ../../../figures/${f}_pl.png fig/; done
# combos_tabular.tex (with the Stack Exchange column) is written by src/combos_table.py
# bias_views_pl.png = crop of the last panel (error by true views) of bias_pl.png
../../../.venv/bin/python -c "from PIL import Image; im=Image.open('../../../figures/bias_pl.png'); w,h=im.size; im.crop((int(w*0.485),int(h*0.5),w,h)).save('fig/bias_views_pl.png')"
latexmk -xelatex -interaction=nonstopmode -quiet main.tex
