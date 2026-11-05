# Analysis overview

Full list of experiments and results: `docs/en/experiments.md`.

Internal map of what was tested, how and where. Not preprint text. Numbers come from `data/processed/*_numbers.json`; every number here can be rebuilt from the notebook named next to it.

## Shared setup
- **Target:** log10(view_count), on TikTok, YouTube and X only. Facebook and Instagram have no views.
- **Validation:** 5 folds grouped by brand. Error is MAE on log10 (0.3 is about a factor of 2, 0.5 a factor of 3, 1.0 a factor of 10). Intervals resample whole brands (300-1000 draws). Comparisons are paired on the same folds.
- **Features built from views** use the training fold only.

## Notebooks
| Notebook | Question | Numbers |
|---|---|---|
| `01_first_models` | First ladder of feature groups | `results_numbers.json` |
| `02_transfer_and_intervals` | Conformal intervals, first transfer test | `results_numbers.json` |
| `03_zoo_strata_graph` | Model zoo with fixed settings, strata, first graph features, labels vs views | `zoo_numbers.json` |
| `04_extra_features` | Text, VADER, TF-IDF, calendar, ratios, brand size, brand graph | `extra_numbers.json` |
| `05_graph_lab` | Communities, centrality, spectral, label propagation, hashtags, time, duplicates | `graphlab_numbers.json` |
| `06_engagement_transfer` | Predicting reactions across platforms | in the notebook |
| `07_tuning` | Default model against baselines, nested search over ten families | `tuning_numbers.json` |
| `08_transfer_matrix` | Every source and target pairing, tuning for transfer, label correction, calibration, FB/IG sensitivity | `transfer_numbers.json` |
| `09_ablation_labels_bias` | Three-way ablation with labels, column importance, bias, communities | `ablation_numbers.json` |
| `10_brand24_rule` | Reconstruction of the reach column and comparison with real views | `rule_numbers.json` |
| `11_promo_flag` | One is-promo column; YouTube errors; removing zero-reaction posts | `promo_numbers.json`, `zero_numbers.json` |
| `12_graph_more` | Second round of graphs: fixed aggregation, cross-platform neighbours, community shrinkage | `graph_more_numbers.json` |
| `13_stacked_models` | Two-stage models: classify then regress, residual stacking, hurdle | `stacked_numbers.json` |
| `14_reaction_tiers` | Error of final models and intervals by reaction tier | `tiers_numbers.json` |
| `15_brand_mismatch` | Brands away from their industry in the text graph (false matches) | `mismatch_numbers.json` |
| `16_history_forecast` | Second Brand24 file: growth between checks, forecast from the first check, estimate at posting time (E38-E43) | `history_numbers.json` |
| `17_check_age` | Age of the view check as a column; newer checks (E44-E45) | `check_age_numbers.json` |
| `18_rolling_window` | Train on earlier months, test on the next (E46) | `rolling_window_numbers.json` |
| `19_tabular_foundation` | TabPFN v2 and TabICL against CatBoost (E47) | `tabular_foundation_numbers.json` |
| `20_classification` | Viral top 10%; posts with missing or zero views (E48) | `classification_numbers.json` |
| `21_context_text` | Context from other posts of the author, hashtags, brand; text embeddings; offset target (E49-E51) | `context_text_numbers.json` |
| `22_posting_time_text` | Estimate at posting time with text and context (E52) | `posting_time_text_numbers.json` |
| `23_finetune_text` | Fine-tuned e5-small: row as text, text as a correction (E53) | `finetune_text_numbers.json` |
| `24_stack_all` | Stack of all model families on the same folds (E54) | `stack_all_numbers.json` |
| `25_limits` | Leakage audit, noise floor, quantile intervals, abstention (E55-E58) | `limits_numbers.json` |
| `26_history_more` | Forecast horizon, all pairs of checks, early reactions without views (E59-E60) | `history_more_numbers.json` |
| `27_more_models` | Pooled and monotone models, YouTube drift, copied posts, zero-reaction posts (E61-E64) | `more_models_numbers.json` |
| `28_gnn` | GraphSAGE on the text graph (E65) | `gnn_numbers.json` |
| `29_accounts_clusters` | Account types, topic clusters, author history, brand communities (E66-E69) | `accounts_clusters_numbers.json` |
| `30_author_more` | Author history in depth, variance components, brand totals, anomalies, views per like (E70-E75) | `author_more_numbers.json` |
| `31_author_next` | Where the error sits, history curve, author level without views, live forecast with history, brand-total intervals (E76-E80) | `author_next_numbers.json` |
| `32_fbig_more` | FB / IG before the hand check (structure, pre-registered estimates, scoring), residual atlas (E81-E83) | `fbig_more_numbers.json`, `fbig_preregistered.csv` |
| `33_context_more` | Country, local time, holidays, industry as columns; where promo over-performs; brand popularity on Wikipedia (E84-E86) | `context_more_numbers.json`, `brand_wikipedia.csv` |
| `34_platform_map` | Mapping reaction scales between platforms before the out-of-platform model (E87) | `platform_map_numbers.json` |
| `35_platform_adapt` | Adapting the out-of-platform model to the target's reaction structure (E88) | `platform_adapt_numbers.json`, `fbig_preregistered_v2.csv` |

## Columns
Base set: followers, likes, comments, shares (logged), and a missing flag for each. Added and tested separately: text statistics, 30 TF-IDF SVD topics, VADER, calendar, ratios, brand size, our type and sentiment labels, Brand24 sentiment, graph features.

## Models and search (`src/tuning.py`)
Ridge, kNN, SVR (RBF), MLP, random forest, extra trees, scikit-learn boosting, LightGBM, XGBoost, CatBoost. Random search, 40 settings per outer fold, inner 3-fold brand-grouped CV, objective MAE. Default settings are the library's, with CatBoost limited to 300 iterations. The family reported per platform is chosen by inner CV. Also a blend of the three best by inner CV.

## Transfer (`src/transfer_matrix.py`)
Train on one or two platforms, test on a platform left out. Test brands are removed from training, also on other platforms. Six models, three feature sets (reactions; + followers; + shares and flags). Settings are chosen in three ways (defaults, inside sources, across sources), never with target views. Label correction: source reweighted to the target's type mix, or type as a feature. Calibration: shift by the median error of n target posts, n from 5 to 500, random posts or posts from a few brands.

## Brand24 rule (`src/brand24_rule.py`)
Reach = max(k × reactions, c(followers) × followers). k is 700 on X (300 above 10 reactions), 100 on Facebook, 20 on Instagram. c falls in steps with account size and is not used on Instagram. Reproduces 98%, 96%, 98% of rows within 2%. Reconstructed by us, not confirmed by Brand24, who describe the column as indicative only.

## Main findings
- A default model with reactions beats constant × followers: 0.29 / 0.53 / 0.49 against 0.92 / 0.86 / 0.88 (TikTok, YouTube, X).
- After tuning the families converge at about 0.27 / 0.45 / 0.47, with ridge on YouTube (0.55) the exception.
- Likes carry most of the signal. Text adds about 0.01-0.02 on YouTube and X. Our labels add at most 0.005. Graph features add at most about 0.007, and label propagation hurts.
- Real views per reaction: 21 TikTok, 20 YouTube, 27 X. Brand24's rule assumes 700 on X.
- Across platforms rank order holds, the level can shift (+0.03, +0.21, -0.40). On X 25-50 posts with true views remove the shift.
- Largest systematic error: regression to the mean. The lowest fifth is overestimated, the top fifth underestimated.

## Later results (notebooks 11-14)
- One is-promo column gains at most 0.006 with followers and reactions. Promo does not explain YouTube's large errors; zero likes and comments do (23% of posts, 35% of error).
- Removing zero-reaction posts lowers the score by making the test easier; removing them from training gains at most 0.002.
- Two-stage models gave no gain. MAE loss did: +0.011 / +0.007 / +0.022 for default CatBoost.
- Second graph round: cross-platform neighbours gain +0.009 on YouTube, the only all-posts CI above zero. The level shift does not shrink.
- Error falls with reactions on YouTube and X. An interval per reaction tier restores about 90% coverage where calibration data suffice.

## Later results (notebooks 15-30)
- **Author history is the largest gain found (E68).** An author's over-performance (views above the final model's prediction) persists: Spearman 0.68 / 0.69 / 0.70 between first and later posts, robust to repeated posts. A correction from the author's earlier posts gains +0.12 / +0.16 / +0.10 MAE on posts that have one (3-16% of the sample).
- **Stacking all families (E54):** 0.260 / 0.425 / 0.441, +0.011 / +0.028 / +0.026 over the final model; most of it comes from a plain mean of diverse members. TabPFN v2 alone +0.005 / +0.010 / +0.002 (E47); CatBoost + MLP mean +0.008 / +0.016 / +0.016 (E65, GraphSAGE itself worse).
- **Text:** frozen embeddings help only without reactions (YouTube +0.051, E50); a fine-tuned transformer as a half correction gains +0.007 / +0.012 / +0.008 (E53). Context from other posts adds ≤ 0.003 next to reactions (E49).
- **History file:** forecast from the first check 0.165 / 0.188 / 0.269 (E40); without the early view count 0.368 / 0.444 / 0.544 (E41); chaining forecast reactions into the now-cast beats that by +0.034 on TikTok (E60). At posting time 0.566 / 0.664 / 0.540 (E43, E52).
- **Time and validation:** training on earlier months is close to brand CV (E46); stricter folds (brand ∪ author ∪ text) cost ≤ 0.011 (E55). The YouTube drift comes from younger check ages (E62); the age of the check helps on X (+0.012, E44).
- **Limits:** noise floor about 0.22 / 0.39 / 0.33 (E56); 10-90% quantile interval covers 73% (E57); scoring only the most certain half cuts YouTube MAE by 27% (E58).
- **Other tasks:** viral top 10% found with precision 83 / 71 / 63% (E48); missing views are predictable (AUC 0.91-0.95), X 2025 posts almost never have views (E48). Copied posts are underestimated (YouTube -0.26, E63).
- **Groups:** official brand accounts are ≤ 5% of posts and underestimated ×2-3 (E66); topic clusters and 9 industry communities of brands are readable but carry little bias beyond reactions (E67, E69).
- **Author level (E70-E72):** two thirds of the remaining error on TikTok and YouTube is shared by posts of the same author (brand ≤ 0.08); account traits do not predict it, the author's past views do: +0.10-0.15 with reactions, +0.14-0.17 at posting time.
- **Client view (E73, E75):** brand totals within ×1.7 / ×2.0 / ×2.1, Brand24 reach ×10 too high on X; views per like 11-64 on TikTok, 17-113 on YouTube, 25-279 on X depending on tier.
- **Remaining error and intervals (E76-E80):** the error is spread over all posts (zero-reaction posts worst), not a few authors; one earlier post of the author gives most of the history gain; the author's level cannot be read from reactions on other posts; an 80% interval for a brand total is ×2.2 / ×3.0 / ×3.3 and calibrated.
- **FB / IG before the hand check (E81-E82):** Instagram's reactions look most like X and YouTube, Facebook like none of the sources; estimates for the 312 hand-check posts are fixed before any view is seen (FB about 200-250, IG 35-95; Brand24 reach ×15-20 above ours on FB). **Atlas (E83):** promo, non-US posts, capitals on X and afternoon UTC on YouTube get more views at equal reactions.
- **Anomalies (E74):** posts ×10 above the prediction are more often promo, copies, brand accounts, repeat authors; the author's history flags them 6-15× better than chance.
- **Did not help:** pooled or monotone models (E61), all pairs of checks (E59), account type as a feature (E66), similar brands for a new brand (E69).

## Corrections made on 2026-11-02
- `extra_features.ratios` inverted log1p columns with base 10. Fixed, notebook 04 re-run. Main results unchanged.
- "Families are equal" now holds only after tuning and not for ridge on YouTube.
- "X reach = 0.1 × followers" holds only for posts without reactions.

## Open
Human gold standard for the labels; whether reach equals views; a Facebook or Instagram sample with real views (312 Reels and videos to check by hand: `docs/en/fbig_manual_check.md`, B14); false brand matches across the whole dataset; author view history from Brand24 (question Q8, E68). Sentence embeddings and a GNN are now tested (E50, E53, E65).

Planned next steps: "Ideas to stand out" in `docs/en/project_plan.md`.
