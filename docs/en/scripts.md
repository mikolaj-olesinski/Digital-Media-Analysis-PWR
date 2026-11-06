# Scripts

Internal index of `src/`. Experiment scripts and their numbers are in `docs/en/repro_map.md`; this page covers the
helpers and the early notebook scripts that are not linked from a card.

## Core

| script | role |
|---|---|
| `eda_utils.py` | loading the raw sample, platform names/colours, reusable EDA blocks |
| `validation.py` | brand-grouped and temporal CV, MAE and bootstrap CI, conformal helpers |
| `modeling.py` | feature groups (`GROUPS`), feature builder, model ladder |
| `extra_models.py` | `ENG` / `ALL` feature sets, final model helpers |
| `reaction_tiers.py` | reaction tiers, the final CatBoost, intervals by tier |
| `agent_tools.py` | vault index and `check` (card and register consistency) |
| `agent_tools_selftest.py` | self-test for `where_paths` and the register row validation (`python src/agent_tools_selftest.py`) |
| `fetch_views.py` | public view-count fetcher (FB watch pages, FxTwitter, yt-dlp); produced the FB data used in E93/E95 |

## Figures and documents

| script | role |
|---|---|
| `result_figures.py` | the result figures (CQR bands, FB coverage, concentration, calibration, rule vs model) |
| `m1_figures.py`, `m1_figures2.py` | M1 card figures and numbers |
| `story_figures.py` | five explanatory figures for the project overview (PL) |
| `guide_figures.py`, `guide_build.py` | figures and the HTML project guide (tables are hand-transcribed from the JSONs; `guide_figures` hardcodes the rule labels 700 / 0.1) |
| `results_numbers.py` | numbers shown on the results slides |

## Early notebook scripts (not linked from a card)

These were used by the notebooks in `notebooks/models/`; their experiments are in the register (E02-E54) but the card
links the notebook, not the script. Kept for reproducibility.

| script | notebook / experiment |
|---|---|
| `transfer.py` | 02 transfer and intervals (E02, E13) |
| `extra_features.py` | 04 extra features (E05) |
| `new_ideas.py` | 04 extra features: paired comparison, engagement target, brand graph |
| `graph_lab.py` | 05 graph lab (E06) |
| `graph_more.py` | 12 graph more (E12) |
| `label_ablation.py` | 09 ablation with labels (E04) |
| `label_utils.py`, `merge_labels.py` | label merging and agreement (E04) |
| `stacked_models.py` | 13 stacked models (E31) |
| `xplat.py` | 06 engagement transfer (E15) |
| `brand_mismatch.py` | 15 brand mismatch (D) |

## Counts

- 121 scripts in `src/`; 106 are referenced by a card or a document; the 15 not referenced are listed above (helpers,
  figure builders and early notebook scripts).

## Command-line arguments

| script | arguments |
|---|---|
| `src/agent_tools.py` | `vault`, `check`, `next` |
| `src/fetch_views.py` | `--input`, `--platforms`, `--limit`, `--seed`, `--out`, `--workers`, `--cookies-from-browser`, `--cookies`, `--refresh`, `--url` (all described in `--help`) |
| `src/result_figures.py` | `en` / `pl` |
| `src/m1_figures.py` | `en` / `pl` |
| `pitchdeck/build.py` | `en` / `pl` |

## Live vs historical figures

| figures | built by | status |
|---|---|---|
| `cqr_bands`, `fb_coverage`, `reach_concentration`, `calibration_curve`, `fb_rule_vs_model` (EN/PL) | `src/result_figures.py` | live, from the JSON files |
| `m1_data_and_error`, `m1_distributions` (EN/PL) | `src/m1_figures.py`, `m1_figures2.py` | live, from the raw data |
| `story_a`-`story_e` (**PL only**) | `src/story_figures.py` | live, from the raw data and labels; PL-only because they feed the Polish guide |
| `baseline_{en,pl}`, `x_rule_{en,pl}` | `src/guide_figures.py` (CLI) | live, from the raw data; the Polish guide uses the PL versions, the EN twins are available but not referenced |
| all other `figures/*.png` (~100) | notebooks and older model scripts | historical; rebuild by running those notebooks |

Refresh the live figures:

```bash
.venv/bin/python src/result_figures.py en && .venv/bin/python src/result_figures.py pl
.venv/bin/python src/m1_figures.py en && .venv/bin/python src/m1_figures.py pl
.venv/bin/python src/story_figures.py
.venv/bin/python src/guide_figures.py            # baseline_{en,pl}, x_rule_{en,pl}
.venv/bin/python src/guide_build.py /tmp/guide.html   # the guide HTML (uses the PNGs above)
```
