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
- A plain MLP on the final-model inputs + 16 embedding components equals CatBoost alone, and their mean gains +0.008 / +0.016 / +0.016 (CIs above zero): the largest simple ensemble gain so far. **likely** (E65)
- A fine-tuned transformer (e5-small) reading the row as text matches CatBoost on YouTube; its mean with CatBoost gains +0.019 there; text as a half-weighted correction to CatBoost gains +0.007 / +0.012 / +0.008. Not more than CatBoost + MLP (E65). **likely** (E53)
- Stacking all model families (CatBoost variants, TabPFN, TabICL, MLP, fine-tuned text) beats the final model by +0.011 / +0.028 / +0.026 and the best single member by 0.008-0.018, CIs above zero; the plain mean of all members gives most of it. Best accuracy reported; final model stays single CatBoost. **likely** (E54)
- Authors have a stable level of views beyond what reactions predict: Spearman 0.68-0.70 between an author's first and later posts (robust to repeats); correcting with the author's earlier posts gains +0.10-0.16 MAE on posts that have one. Largest gain found; needs author view history. **likely** (E68)
- About two thirds of the remaining error on TikTok and YouTube is an author level (intraclass 0.74 / 0.63; brand ≤ 0.08). It cannot be predicted from account traits (Spearman ≈ 0), only from the author's past views; the shrunk correction gains +0.10-0.15 with reactions and +0.14-0.17 at posting time. **likely** (E70-E72)
- Brand totals (≥ 5 posts) are estimated within ×1.7 / ×2.0 / ×2.1 (MAE 0.23 / 0.31 / 0.32) after a one-number calibration; Brand24's reach overestimates X brand totals ×10. **likely** (E73)
- Posts ×10 above the prediction (1-6%) are more often promo (TikTok), copies, brand accounts and repeat authors; ×10 below are zero-reaction posts and spam. Implied views per like: TikTok 50 → 11 as likes grow, X 25-112 against Brand24's 700. **likely** (E74, E75)
- The remaining error is spread over all posts, not over a few authors or replies; zero-reaction posts are the worst group. One earlier post of the author gives most of the history gain (k = 2-3 maybe a bit more on YouTube). Reactions on the author's other posts do not reveal the author's level (E78, with E71): only past views do. Live forecast + history: too few pairs. 80% interval for a brand total: brand-level quantiles, ×2.2 / ×3.0 / ×3.3, calibrated; summing independent post errors is not. (E76-E80)
- FB / IG (no views): Instagram reactions resemble X and YouTube (AUC 0.65), not TikTok (0.85); Facebook resembles no source, worst with followers (0.89-0.98), so transfer to FB should use likes and comments only. Estimates for the 312 hand-check posts are pre-registered (E82, `fbig_preregistered.csv`); Brand24 reach is ×15-20 above ours on FB and below on IG. At equal reactions promo, non-US posts, capitals on X and 12-17 UTC on YouTube get more views (×1.2-1.5); descriptive only. (E81-E83)
- Promo over-performance is real only for promo posts that already have reactions or come from mid-sized accounts (TikTok 10k-100k +0.18, YouTube 11+ reactions +0.13-0.20, X 1k-100k +0.20); promo from tiny accounts or without reactions under-performs, so the average cancels and the label adds only +0.005 (E85). Country, local time, local holidays and industry as columns: at most +0.005 (E84).
- External brand popularity (Wikipedia pageviews, 293 brands) barely relates to post views (Spearman 0.05-0.18) and hurts as columns on YouTube and X (-0.010). (E86)
- Mapping reaction scales between platforms (quantile matching, z-score, ratios) without target views makes out-of-platform estimates much worse (TikTok 0.31 -> 0.76-1.04, X 0.64 -> 1.28-1.43): absolute likes carry the view level across platforms better than within-platform rank. With 25-50 calibration posts a mapped CatBoost gains about +0.03 on X only. (E87)
- Adapting the out-of-platform model to the target's reaction structure without target views (importance weights, per-source mixture, views per like) helps only on X (+0.02 to +0.03) and hurts on TikTok; ridge on raw likes + comments stays the robust default. FB / IG estimates of these variants are pre-registered (`fbig_preregistered_v2.csv`). (E88)
- Literature-based transfer methods (anchor regression, group DRO, stabilised weights, invariant subsets, brand carry-over) do not beat ridge on likes + comments, as the shift benchmarks predict (E90-E92). What helps is a few target posts: with 5-10, a shift shrunk toward zero (τ² from platform pairs) beats the raw median (0.315 / 0.626 / 0.588 vs 0.347 / 0.679 / 0.621); from 25-50, intercept + slope; a few-shot conformal 90% interval is calibrated from about 10 posts. (E89)
- Official brand accounts (≤ 5% of posts) get few views per follower but are underestimated by ×2-3; topic clusters and brand communities (industries) are readable but carry little bias beyond reactions; similar brands do not predict a new brand. **likely** (E66, E67, E69)
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

- Forecast error grows with the gap on TikTok (0.10 at < 12 h to 0.18 at 7-30 d); a first check in the first 12 hours is 2-4× harder. Training on all intermediate pairs does not help. **tentative** (E59)
- Without views (FB/IG case): forecasting later reactions and passing them through the now-cast model beats forecasting views directly from reactions (+0.034 TikTok, +0.033 X, X CI includes zero). **tentative** (E60)
- The main check came ever sooner after posting (Apr-May 78-100 days, Aug < 1 day on YouTube and X); YouTube views per like fell with it. **likely** (E62)
- Copied texts (1-6% of posts) get more views per like and are underestimated (YouTube -0.26); for 6 YouTube and 7 X brands they make up > 50% of estimated reach. **likely** (E63)

## Dead ends: tried, do not repeat without a new reason
| Idea | Result | IDs |
|---|---|---|
| Our AI type and sentiment labels as features | at most +0.005 | E04, E17, E26 |
| GraphSAGE on the text graph | worse than the same MLP without the graph (-0.006 to -0.011) | E65 |
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
| One pooled model for three platforms; monotone constraints | pooled -0.001 to -0.012, monotone up to -0.026; LOPO CatBoost worse than ridge | E61 |
| A separate model for zero-reaction posts | no gain (YouTube -0.010, X +0.011 CI incl. 0) | E64 |
| Training on every intermediate pair of history checks | TikTok -0.020 | E59 |
| Training on the previous month only (recency) instead of all past months | worse by 0.013-0.032 | E46 |
| Replacing views and reactions by the newer `current_*` check as training data | not better, worse on X | E45 |

## Data quality notes
- On X, posts from 2025 almost never have a view count (12 of 170) and missing counts go with a late check; missing/zero views are predictable (AUC 0.91-0.95), so they look like a collection artefact. On TikTok and YouTube the Brand24 reach column equals the view count. (E48)
- Duplicates or near-duplicates are 6-14% of posts; on X 38% of them are spam against 9% overall. (E08)
- False brand matches exist (Cricket Wireless: 0 of 20 mentions about the company); frequency across the data is unknown. (E22, E35, E37)
- About 21% of YouTube mentions are transcripts starting with a time stamp. (E23)
- Labels: three AI coders agree with kappa 0.77-0.79 on type, 0.71-0.73 on sentiment; there is no human gold standard. (L1)
