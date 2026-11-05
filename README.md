# Digital Media Analysis PWR: estimating the visibility of social media posts

Project for the AMC course (Wrocław University of Science and Technology), client: Brand24.
Question: can the visibility (views) of a post be estimated from the number of followers and reactions better than by the constant × followers rule, and with what uncertainty?

## Contents
- `docs/en/m1_exploration_card.pdf`: M1 data exploration card ([Polish version](docs/pl/m1_exploration_card.pdf))
- `docs/pl/course_rules.pdf`: course rules, grading and schedule (in Polish)
- `notebooks/models/`: `01_first_models` (model ladder, feature importance), `02_transfer_and_intervals` (intervals, transfer to Facebook and Instagram), `03_zoo_strata_graph`, `04_extra_features`, `05_graph_lab`, `06_engagement_transfer`, `07_tuning` (nested search over ten families), `08_transfer_matrix` (out of platform), `09_ablation_labels_bias`, `10_brand24_rule`, `11_promo_flag`, `12_graph_more`, `13_stacked_models`, `14_reaction_tiers`, `15_brand_mismatch`, `16_history_forecast` (growth between checks, forecast from the first check, estimate at posting time), `17_check_age` (age of the view check, newer checks), `18_rolling_window` (train on earlier months), `19_tabular_foundation` (TabPFN v2, TabICL), `20_classification` (viral top 10%, posts without views), `21_context_text` (context from other posts, text embeddings, offset target), `22_posting_time_text` (estimate at posting time with text and context), `23_finetune_text` (fine-tuned transformer), `24_stack_all` (stack of every model family), `25_limits` (leakage, noise floor, quantiles, abstention), `26_history_more` (horizon, early reactions), `27_more_models` (pooled and monotone models, YouTube drift, copied posts, zero-reaction posts), `28_gnn` (GraphSAGE), `29_accounts_clusters` (account types, topic clusters, author history, brand communities), `30_author_more`, `31_author_next`, `32_fbig_more`, `33_context_more`, `34_platform_map` (author history in depth, variance components, brand totals, anomalies, views per like). Map of all analyses: `docs/en/analysis_overview.md`
- `notebooks/eda/`: `01_overview` (all platforms), `02_platforms` (one platform at a time, with totals), `03_labels` (mention labels and coder agreement)
- `src/m1_figures.py`, `src/m1_figures2.py`: figures and numbers for the M1 card (`python src/m1_figures.py en|pl`)
- `figures/`: generated figures, `*_en.png` and `*_pl.png`
- `pitchdeck/`: presentation in English and Polish (`pitchdeck_en.*`, `pitchdeck_pl.*`); build script `build.py en|pl`

## For AI agents
`CLAUDE.md` and `agent/README.md`: work loop, runbook, current findings, backlog, decisions and one card per experiment in `agent/vault/`. Guardrails: `python src/agent_tools.py check`.

## Data
The raw file `data/raw/reach_samples_students.csv` (12,500 mentions from Brand24) is **not** in the repository: it contains personal data of authors and the client's data.
The second file `data/raw/reach_samples_students_with_history.csv` (same mentions plus earlier checks of views and reactions) is also not in the repository; notebook 16 needs it.
To reproduce the results, place them in `data/raw/` and run `python src/m1_figures.py en`. Dependencies: `pip install -r requirements.txt`.

## Team
To be filled in.
