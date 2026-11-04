# Findings: what we know (state on 2026-11-04)

Numbers in TikTok / YouTube / X order; MAE of log10(views), brand-grouped 5-fold CV. Every line cites the experiments
that support it (register `docs/en/experiments.md`, cards in `agent/vault/`). Confidence:
**solid** = several experiments agree, CI clear of zero; **likely** = one clean experiment; **tentative** = small
sample, one month, or an unconfirmed assumption. When a new result changes a line, edit the line and add the ID; do not
add a contradicting line below it.

## The task and the ground truth
- Views exist only on TikTok, YouTube and X. Facebook and Instagram have none, so nothing on FB/IG is measured, only transferred. **solid** (E13, E16)
- Brand24's reach column is a rule: max(k × reactions, c(followers) × followers), k = 700 on X, 100 FB, 20 IG; reproduces 96-98% of rows. Reconstructed by us, not confirmed by Brand24. **likely** (E24)
- Real views per reaction are about 21 / 20 / 27; the rule assumes 700 on X. **likely** (E24)
- Views are checked at very different post ages (median 858 h / 600 h / 534 h for the main check), and keep growing between checks. **solid** (E38, E39)
- Knowing the age of the check helps little overall (+0.000 / +0.003 / +0.012, only X clear of zero) but removes the over-estimate of checks younger than a day on TikTok and YouTube. **likely** (E44)
- Trained only on earlier months (expanding window, test months 2026-05 … 08) the final model is close to brand CV: -0.001 / -0.011 / -0.026, the gap mostly in May when little history exists; brand CV is not optimistic beyond that. Recent data alone (previous month) is worse than all past data. No drift in error over these four months. **likely** (E46)
- The newer checks in the history file do not make the main model better; on X worse (-0.046), because views are re-measured while reactions mostly are not. Train on the original columns. **likely** (E45)

## Estimating views from reactions (now-cast)
- Constant × followers is a poor base: 0.92 / 0.86 / 0.88. **solid** (E01)
- A default boosting model with followers + reactions gets 0.29 / 0.53 / 0.49; tuned families converge at about 0.27 / 0.45 / 0.47 (CatBoost on YouTube and X, SVR on TikTok); ridge on YouTube is worse (0.55). **solid** (E01, E10, E11)
- Tabular foundation models are a small gain on the same features: TabPFN v2 +0.005 / +0.010 / +0.002 over the final CatBoost; a plain mean of CatBoost + TabPFN + TabICL +0.004 / +0.010 / +0.003 with the tightest CIs. Only YouTube reaches 0.01. TabPFN 2.5+ weights are non-commercial; v2 needs the attribution "Built with PriorLabs-TabPFN". **likely** (E47)
- Ranking for alerts: the regression estimate finds viral posts (top 10% of views) as well as a dedicated classifier: AUC 0.98 / 0.94 / 0.89, 83% / 71% / 63% of the 10% top-scored are viral, against 36-44% for followers. **likely** (E48)
- Without reactions, text (local embeddings or TF-IDF) is worth +0.05 on YouTube, +0.01-0.02 on TikTok and X over followers alone; next to reactions it adds +0.011 on X only. **likely** (E50)
- Likes carry most of the signal: removing them costs +0.09 / +0.13 / +0.05. **solid** (E17)
- MAE loss instead of RMSE: +0.011 / +0.007 / +0.022. **likely** (E32)
- Error falls with reactions on YouTube (×4.6 at zero to ×1.8 above 1,000) and X (×3.1 to ×2.5); TikTok is flat once a post has any reaction. **solid** (E33)
- Intervals: one width per reaction tier keeps about 90% coverage; one width for all posts under-covers zero-reaction posts. **likely** (E02, E34)
- Systematic error: the lowest fifth of true views is overestimated (0.23 / 0.51 / 0.67), the top fifth underestimated (0.20 / 0.42 / 0.56). Partly an artefact of splitting by the true value. **solid** (E18)
- YouTube posts with zero likes and comments are 23% of posts and 35% of error; possibly missing data rather than true zeros. Unconfirmed. **tentative** (E27)

- Validation: folds that also separate authors and copied texts change MAE by ≤ 0.002 on TikTok and YouTube and -0.011 on X; seeing the brand in training helps ≤ 0.009. **likely** (E55)
- Posts with practically the same followers and reactions differ ×2.8 / ×6.0 / ×4.6 in views; the best possible MAE on them is at least 0.22 / 0.39 / 0.33 (final model 0.33 / 0.58 / 0.47 there). Most remaining error is in the data. **likely** (E56)
- Quantile 10-90% intervals under-cover on held-out brands (73-74%, top fifth 44-54%). Abstaining on the uncertain half cuts MAE by 27% on YouTube, ≤ 10% elsewhere. **likely** (E57, E58)

## Transfer to other platforms
- Ridge on likes + comments transfers best: 0.31 / 0.61 / 0.65 when the platform is left out; the rank order holds but the level shifts (+0.03 / +0.21 / -0.40). **solid** (E02, E13)
- 25-50 posts with true views from the target platform remove most of the shift on X (0.65 → about 0.55). Not checkable on FB/IG. **likely** (E15)
- Tuning for transfer, label reweighting and type as a feature do not help consistently. **likely** (E14)
- Reactions predicted across platforms reach Spearman 0.3-0.47 inside a platform; a two-stage reactions-then-views model gives no gain. **likely** (E09)

## Forecasting and posting time (history file)
- History exists only for about 340 TikTok and 350 X posts, almost all from 2026-08; YouTube has only the main and the latest check. **solid** (E38)
- Views grow like age^b with b ≈ 0.35 / 0.58 / 1.17 between the first and last check; X is observed only in its first day (7 h → 14 h, median ×2.4). **tentative** (E39)
- From one early check, CatBoost forecasts the later check with 0.165 / 0.188 / 0.269 (×1.5 / ×1.5 / ×1.9); better than persistence and than a power law in age (gains +0.03 / +0.05 / +0.08, CI above zero). 90% interval ×2.4 / ×3.8 / ×4.1. **tentative** (E40, E42)
- Without the early view count the forecast is much worse (0.37 / 0.44 / 0.54): early views are worth 0.20-0.28. This is the FB/IG situation. **tentative** (E41)
- At posting time (no reactions) CatBoost on followers, text, labels, calendar and horizon gets 0.575 / 0.684 / 0.550 (×3.8 / ×4.8 / ×3.5), interval ×20 / ×27 / ×14. Reactions remain the main signal. **likely** (E43). Context from other posts of the author, hashtags and brand improves it to 0.566 / 0.664 / 0.540 (YouTube +0.020; upper bound, uses later posts). **likely** (E52)

## Dead ends: tried, do not repeat without a new reason
| Idea | Result | IDs |
|---|---|---|
| Our AI type and sentiment labels as features | at most +0.005 | E04, E17, E26 |
| Graph features (author, text kNN, Louvain, PageRank, spectral, hashtags, brand graph) | at most about +0.007-0.009 | E03, E06, E07, E29 |
| Label propagation on the text graph | worse (X -0.021) | E07 |
| Separate models per follower tier or community | worse than one model | E03, E31 |
| Two-stage models (classify then regress, residual stacking, hurdle) | no gain, stacking worse | E31 |
| Dropping spam or zero-reaction posts from training | at most 0.002 | E20, E28 |
| Author history (smoothed mean of the author's training posts) at posting time | ±0.004, CI includes zero | E43 |
| Tuning across many families beyond CatBoost defaults | small (≤ 0.02) | E11, E12 |
| Context from other posts of the author, hashtags, brand (likes and followers, no views) next to reactions | ≤ +0.003; brand context worse on YouTube (-0.014) | E49 |
| Offset target log(views/likes) against the compression of extremes | +0.001 / +0.007 / +0.000; bias narrows 0.01-0.03 | E51 |
| Tail weighting (×2 on outer fifths) against compression of extremes | ≤ 0.002, bias unchanged | E57 |
| Training on the previous month only (recency) instead of all past months | worse by 0.013-0.032 | E46 |
| Replacing views and reactions by the newer `current_*` check as training data | not better, worse on X | E45 |

## Data quality notes
- On X, posts from 2025 almost never have a view count (12 of 170) and missing counts go with a late check; missing/zero views are predictable (AUC 0.91-0.95), so they look like a collection artefact. On TikTok and YouTube the Brand24 reach column equals the view count. (E48)
- Duplicates or near-duplicates are 6-14% of posts; on X 38% of them are spam against 9% overall. (E08)
- False brand matches exist (Cricket Wireless: 0 of 20 mentions about the company); frequency across the data is unknown. (E22, E35, E37)
- About 21% of YouTube mentions are transcripts starting with a time stamp. (E23)
- Labels: three AI coders agree with kappa 0.77-0.79 on type, 0.71-0.73 on sentiment; there is no human gold standard. (L1)
