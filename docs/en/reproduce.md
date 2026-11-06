# How to reproduce

From the raw sample to the numbers and figures. On the shared homelab3 machine see `agent/runbook.md` (Kueue queue, image, model weights). Every script writes its own file in `data/processed/`; the order below
is the recommended one, but most scripts are independent.


## Quick start (5 commands)

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/validation_audit.py
.venv/bin/python src/cqr_intervals.py
.venv/bin/python src/final_v3_x.py    # X v3: 0.454 and a 1.99 band (the number in the model card)
.venv/bin/python src/result_figures.py en
.venv/bin/python src/agent_tools.py vault && .venv/bin/python src/agent_tools.py check
.venv/bin/python src/agent_tools_selftest.py
```

What each step does:

1. `pip install` - the environment (Python 3.14, versions in "Versions used for the final numbers").
2. `validation_audit.py` - repeated CV and the recalibration checks; writes `validation_audit_numbers.json`.
3. `cqr_intervals.py` - v2 quantile models and CQR bands; writes `cqr_numbers.json`.
4. `final_v3_x.py` - X with log(age); writes `final_v3_x_numbers.json` (0.454, band 1.99).
5. `result_figures.py` - all result figures from the JSON files, EN and PL.
6. `vault` + `check` - indexes the cards and verifies registers, paths, cards, figures, notebooks.

This reproduces the core numbers and figures; the Facebook steps need the fetched files (section 3). Sections: quick
start, 0 environment, 1 data, 2 core, 3 Facebook, 4 figures, 5 check, 6 map, runtime, data versions, debugging, versions,
missing files, minimal set, notebook hygiene, notebook failures.


Note on the X band: `cqr_intervals.py` trains the v2 model and reports a 2.07 band; `final_v3_x.py` adds log(age) and
gives the current 0.454 MAE and 1.99 band (E141), which is what the model card and the figures use. Run both.

## 0. Environment

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # pandas, scikit-learn, catboost, lightgbm, xgboost, matplotlib, jupyter, yt-dlp
```

Python 3.14 is what the numbers were produced with. Fixed seeds everywhere (`SEED = 42`).

## 1. Data

- `data/raw/reach_samples_students.csv`: the Brand24 sample, read-only, git-ignored.
- `data/processed/mention_labels.csv`: AI labels (type, sentiment) for all 12,500 mentions; produced by
  `src/merge_labels.py` from the three coder files. Already in the repo.
- Facebook views (`data/processed/fb_all_fetched.csv`, `fb_fetched.csv`): fetched from public pages with
  `src/fetch_views.py` on 2026-11-05; git-ignored. Without them, the FB experiments (E93/E95/E111/E123/E150/E154/E165)
  cannot be re-run, but all in-platform numbers can.

## 2. Core model and validation

```bash
.venv/bin/python src/validation_audit.py     # repeated CV, isotonic and tier recalibration (E114)
.venv/bin/python src/cqr_intervals.py        # 90% CQR bands (E118)
.venv/bin/python src/cqr_50.py               # 50% band (E137)
.venv/bin/python src/final_v2.py             # author_te + CQR (E124)
.venv/bin/python src/final_v3_x.py           # X with log(age) (E141)
```

Expected: `data/processed/validation_audit_numbers.json`, `cqr_numbers.json`, `cqr_50_numbers.json`,
`final_v2_numbers.json`, `final_v3_x_numbers.json` — byte-identical to the stored files (verified for two scripts, E29
round).

## 3. Facebook

```bash
.venv/bin/python src/fb_transfer_check.py            # transfer MAE, 130 posts (E93)
.venv/bin/python src/fb_transfer_check.py --followers  # followers penalty (E94)
.venv/bin/python src/fb_extend.py                    # 402 posts (E95)
.venv/bin/python src/fb_cqr.py                       # CQR on FB (E123)
.venv/bin/python src/fb_hybrid.py                    # hybrid >1M (E154)
.venv/bin/python src/fb_rule_vs_model.py             # rule vs model by tier (E165)
```

## 4. Figures

```bash
.venv/bin/python src/result_figures.py en
.venv/bin/python src/result_figures.py pl
```

Writes `figures/cqr_bands_{en,pl}.png`, `fb_coverage_*`, `reach_concentration_*`, `calibration_curve_*`,
`fb_rule_vs_model_*`. Which figures are live and which are historical: `docs/en/scripts.md`.

## 5. Consistency check

```bash
.venv/bin/python src/agent_tools.py vault      # index the cards
.venv/bin/python src/agent_tools.py check      # cards, register, numbers must agree
.venv/bin/python -m py_compile src/*.py
```

`check` must print `0 new failures, 0 known, 0 warnings`.

## 6. Experiment map

- `docs/en/repro_map.md`: experiment -> script -> numbers (E95-E165).
- `docs/en/experiments.md`: the full register (E01-E165) with rows for every experiment, including failed ones.
- `docs/en/scripts.md`: what each helper script does.

## Runtime

- Most scripts: under a minute; the repeated CV and the FB CQR matrices: 2-10 minutes on CPU.
- No GPU needed. LightGBM and CatBoost run single-node.

## Data versions and drifting results

The numbers in this repository correspond to the Brand24 files of 2026-11-03 (history export) and the Facebook checks of 2026-11-05.
A new export or a new fetch will change them: views grow with age, and the November re-fetch was 1.07-18x the stored
snapshot (E95).

If results differ after a new fetch:

1. Compare the **age** of the two checks; do not mix counters read at different ages in one experiment.
2. Re-run the affected scripts and write a **new card** with the new numbers; do not overwrite the old card.
3. Say which data version a number uses (`data_statement.md`, "When the data changes").

## When a script does not reproduce a number

1. **Same data**: same export/fetch dates, same row counts, same filters (`status == "ok"`, `views > 0`).
2. **Same environment**: `pip install -r requirements.txt`; check `pandas`, `scikit-learn`, `catboost`, `lightgbm` versions.
3. **Same seeds**: `SEED = 42` everywhere; avoid `n_jobs` > 1 in stochastic estimators when comparing bits.
4. **Same folds**: brand-grouped CV with the same brand list; a different sort order changes the folds.
5. **Compare with the JSON**, not with the prose: `data/processed/*_numbers.json` and the card's numbers.
6. A drift of ~0.001 is usually library/threading; a larger one is a data filter, fold or feature difference.

## Versions used for the final numbers

Python 3.14.4, pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1, scipy 1.18.1, catboost 1.2.10, lightgbm 4.7.0,
xgboost 3.4.1, torch 2.15.0+cpu, matplotlib 3.11.2, networkx 3.7. `requirements.txt` gives minima; if your numbers
differ by ~0.001, check these versions before anything else.

## If a data file is missing

- **`data/raw/reach_samples_students.csv` missing**: nothing runs; ask the team for the file (it is git-ignored and
  personal).
- **The history file missing**: only notebook 16 and the forecast experiments fail; everything else runs.
- **The fetched Facebook files missing**: the FB experiments (E93/E95/E111/E123/E150/E154/E165/E166) cannot run; the
  in-platform numbers are unaffected.
- **`data/manual/fbig_sample.csv` missing**: the hand-check sample cannot be rebuilt; the fetched files already contain the
  results, but E93/E97 cannot be re-run from scratch.
- Every script that reads the raw CSV (the main loaders, the early scripts and the M1 figure scripts) stops with a
  readable message naming the missing file, not a bare traceback.
- **Never** substitute simulated or re-fetched data for a published number: either reproduce with the same file or write
  a new card with the new version.

## Minimal set to reproduce the numbers

- **Final model and intervals:** pandas, numpy, scipy, scikit-learn, catboost, lightgbm, matplotlib.
- **Fetching (optional):** yt-dlp (the Facebook fetcher calls it as a binary).
- **Experiments only:** tabpfn, tabicl (E47), sentence-transformers, transformers (E50/E53), torch (embeddings and
  GraphSAGE), networkx (graph experiments).
- **Not needed for numbers:** python-pptx (pitchdeck), jupyter (only to run notebooks).

If a package is missing, scripts that do not use it still run; check the import at the top of the script.

## Notebook hygiene

- Run each notebook top to bottom and **store it with outputs**: `jupyter nbconvert --execute --inplace <nb>`.
- `check` fails on a stored error output, and warns when a notebook has code but no outputs.
- Use `eda_utils.preview` for any table with authors or URLs: it drops those columns and links in text. Never store a
  URL or an author column in an output.
- One platform per figure; add the all-platform total when a table describes one platform.

## If a notebook does not run

1. Use the `.venv` kernel: `python -m ipykernel install --user --name amc --display-name amc` (once).
2. Check the data files the notebook needs (raw always; the history file for 16; fetched Facebook files for FB).
3. Run top to bottom; a cell that depends on an earlier one fails when run alone.
4. Heavy notebooks: 19 (TabPFN/TabICL) takes about 17 min on 6 CPU; on homelab3 go through Kueue
   (`agent/runbook.md`).
5. If the outputs differ, follow "When a script does not reproduce a number".
