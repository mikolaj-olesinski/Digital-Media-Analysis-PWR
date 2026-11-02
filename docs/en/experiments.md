# Experiment register

One line per experiment. **Add a line whenever an experiment is run, including failures.** Details are in the notebook; exact numbers are in `data/processed/`. Unless noted: target log10(views) on TikTok / YouTube / X, 5-fold CV grouped by brand, MAE on log10 (0.3 is about ×2, 0.5 about ×3, 1.0 ×10). Gain means fall in MAE against the stated base; positive is better. Result types: **gain** (CI above zero and useful), **small** (CI above zero but about 0.01 or less, or isolated), **none**, **worse**, **finding** (says something about the data).

| ID | Date | Question | Setup | Result | Type | Where |
|---|---|---|---|---|---|---|
| L1 | 2026-10 | Can AI label type and sentiment? | 3 Haiku coders, Sonnet on disputes, 12,500 mentions | Kappa 0.77-0.79 type, 0.71-0.73 sentiment. No human gold standard | finding | `eda/03_labels`, `mention_labels.csv` |
| E01 | 2026-11 | Which column groups matter? | Boosting on cumulative groups, against median and constant × followers | Constant 0.92 / 0.86 / 0.88; with reactions 0.29 / 0.46 / 0.50 | gain | `01_first_models`, `results_numbers.json` |
| E02 | 2026-11 | Do intervals hold? Does a model transfer? | Split conformal; ridge and boosting on likes + comments, leave one platform out | 90% coverage 0.89-0.90, width ×16 / ×104 / ×133 in total; transfer 0.31 / 0.61 / 0.65 | finding | `02_transfer_and_intervals` |
| E03 | 2026-11 | Do other models or first graph features help? | 6 families + blend on fixed settings; author and text-neighbour features; strata | Best 0.28 / 0.45 / 0.48. Graph at most about 0.009. Models per follower tier worse | none | `03_zoo_strata_graph`, `zoo_numbers.json` |
| E04 | 2026-11 | Labels vs views | Residual after reactions, by type | Moves views at most about 0.1; only TikTok promo clearly above zero (+0.08) | small | `03_zoo_strata_graph` |
| E05 | 2026-11 | Do extra columns help? | Extra trees; text stats, VADER, TF-IDF, calendar, ratios, brand size | Text +0.011-0.016 on YouTube and X; the rest about zero | small | `04_extra_features`, `extra_numbers.json` |
| E06 | 2026-11 | Brand graph, predicting reactions | Brand similarity graph; targets from other columns | Brand graph small on X only; reactions predictable (R² up to 0.76), followers less | small | `04_extra_features` |
| E07 | 2026-11 | Graph lab | Louvain, PageRank, spectral, author-brand, hashtags, time, duplicates, encodings | At most about 0.007; label propagation worse (X -0.021) | none | `05_graph_lab`, `graphlab_numbers.json` |
| E08 | 2026-11 | What does the text graph find? | Homophily, duplicates, type from text | Duplicates 6-14% of posts; on X 38% spam against 9%; pair view correlation 0.66-0.82 | finding | `05_graph_lab` |
| E09 | 2026-11 | Do reactions transfer to FB/IG? Two-stage reactions then views | Followers + text to reactions across 5 platforms | Spearman 0.3-0.47 inside a platform; two-stage no gain | none | `06_engagement_transfer` |
| E10 | 2026-11-02 | Default model vs baselines | scikit-learn boosting with defaults | With reactions 0.29 / 0.53 / 0.49 | gain | `07_tuning`, `tuning_numbers.json` |
| E11 | 2026-11-02 | Does tuning change the ranking? | 10 families, nested search, 40 settings per fold | Best 0.27 (SVR) / 0.45 / 0.47 (CatBoost); ridge 0.55 on YouTube | small | `07_tuning` |
| E12 | 2026-11-02 | Tuned boosters with text | CatBoost, LightGBM + text stats and TF-IDF | At most +0.013 | small | `07_tuning` |
| E13 | 2026-11-02 | Full transfer matrix | 6 models, 3 column sets, 12 pairings, test brands removed | Ridge on likes + comments best: 0.31 / 0.61 / 0.65, bias +0.03 / +0.21 / -0.40; order kept | finding | `08_transfer_matrix`, `transfer_numbers.json` |
| E14 | 2026-11-02 | Tuning for transfer, label correction | Across-source tuning; reweighting; type as a feature | No consistent gain | none | `08_transfer_matrix` |
| E15 | 2026-11-02 | Calibration with few posts | Shift by median error of n target posts | On X 25-50 posts bring 0.65 to about 0.55; ideal 0.54. Not checked on FB/IG | gain | `08_transfer_matrix` |
| E16 | 2026-11-02 | FB/IG sensitivity to source | Median estimate per source set | Spread ×1.7 and ×2.4. Not a measured error | finding | `08_transfer_matrix` |
| E17 | 2026-11-02 | Three-way ablation with labels | Alone, added, removed; second learner | Removing likes +0.09 / +0.13 / +0.05; labels at most +0.005 | finding | `09_ablation_labels_bias`, `ablation_numbers.json` |
| E18 | 2026-11-02 | Systematic errors | Signed error by subgroup | Lowest fifth overestimated 0.23 / 0.51 / 0.67, top fifth underestimated 0.20 / 0.42 / 0.56 | finding | `09_ablation_labels_bias` |
| E19 | 2026-11-02 | Promo vs views per reaction | Line per type | Promo ×1.5-1.9 at equal likes and comments on TikTok and YouTube | finding | `09_ablation_labels_bias` |
| E20 | 2026-11-02 | Dropping spam from training | Extra trees | At most 0.002 | none | `09_ablation_labels_bias` |
| E21 | 2026-11-02 | Community detection | Louvain on text kNN, no views or labels | NMI with brand 0.49-0.65, type 0.04-0.08; ARI 0.61-0.71 | finding | `09_ablation_labels_bias` |
| E22 | 2026-11-02 | False brand matches | Keyword check of one community | Cricket Wireless 0 of 20 about the company. One example; frequency unknown | finding | `09_ablation_labels_bias` |
| E23 | 2026-11-02 | YouTube transcripts | Flag for mentions starting with a time stamp | 21% of mentions; flag +0.008 | small | `09_ablation_labels_bias` |
| E24 | 2026-11-02 | Brand24 rule | Reconstruct from data | Reproduces 98% / 96% / 98% of X, FB, IG; unconfirmed. Real views per reaction 21 / 20 / 27; rule on X 700 | finding | `10_brand24_rule`, `rule_numbers.json` |
| E25 | 2026-11-02 | Rule vs model on X | Same rows | Rule 1.08, simple rule with learned constants 0.68, extra trees 0.47 | gain | `07_tuning`, `10_brand24_rule` |
| E26 | 2026-11-02 | One is-promo column | 3 learners, 2 column sets | At most +0.006 with followers and reactions; up to +0.012 on likes + comments | small | `11_promo_flag`, `promo_numbers.json` |
| E27 | 2026-11-02 | Is promo behind YouTube errors? | Error by reaction state and promo share | Mostly no. Zero likes and comments: 23% of posts, 35% of error. Possible missing data, unconfirmed | finding | `11_promo_flag` |
| E28 | 2026-11-02 | Removing zero-reaction posts | From test, and from training | Test 0.27 / 0.38 / 0.45 because it is easier; training gain at most 0.002 | none | `11_promo_flag`, `zero_numbers.json` |
| E29 | 2026-11-02 | Second graph round | Fixed aggregation, cross-platform neighbours, community shrinkage | Only cross-platform on YouTube +0.009; a hint given many tests. Slopes differ by community on TikTok and YouTube | small | `12_graph_more`, `graph_more_numbers.json` |
| E30 | 2026-11-02 | Graph bridge across platforms | Neighbours from other platforms, leave one out | Level shift unchanged; ridge on zero-reaction posts only +0.018-0.022 | none | `12_graph_more` |
| E31 | 2026-11-02 | Two-stage models | Tier classifier, routing, residuals, hurdle, promo, community | No clear gain; residual stacking worse | none | `13_stacked_models`, `stacked_numbers.json` |
| E32 | 2026-11-02 | MAE loss | Default CatBoost, RMSE to MAE | +0.011 / +0.007 / +0.022 | gain | `13_stacked_models` |
| E33 | 2026-11-02 | Error by reaction tier | CatBoost MAE; ridge out of platform | YouTube ×4.6 to ×1.8; X ×3.1 to ×2.5; TikTok flat after the first reaction | finding | `14_reaction_tiers`, `tiers_numbers.json` |
| E34 | 2026-11-02 | Interval per reaction tier | Mondrian conformal | One width covers 67-87% at zero reactions; per tier about 90%, except TikTok at zero | gain | `14_reaction_tiers` |

## Planned, not run
From `docs/en/project_plan.md`: own FB/IG sample, abstention, false-match detector, virality classification, quantile models, weighted tails, treating YouTube zeros as missing.

## Not done
Sentence embeddings, GCN or GraphSAGE, multi-task networks, human gold standard.
