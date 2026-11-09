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

## Corrections made on 2026-11-02
- `extra_features.ratios` inverted log1p columns with base 10. Fixed, notebook 04 re-run. Main results unchanged.
- "Families are equal" now holds only after tuning and not for ridge on YouTube.
- "X reach = 0.1 × followers" holds only for posts without reactions.

## Open
Human gold standard for the labels; whether reach equals views; a Facebook or Instagram sample with real views (Brand24 has none, so we would collect it ourselves); false brand matches across the whole dataset; sentence embeddings; full GNN.

Planned next steps: "Ideas to stand out" in `docs/en/project_plan.md`.
