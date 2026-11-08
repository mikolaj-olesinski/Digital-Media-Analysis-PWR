# Findings: what we know (state on 2026-11-07)

Numbers in TikTok / YouTube / X order; MAE of log10(views), brand-grouped 5-fold CV. Every line cites the experiments
that support it (register `docs/en/experiments.md`, cards in `agent/vault/`). Confidence:
**solid** = several experiments agree, CI clear of zero; **likely** = one clean experiment; **tentative** = small
sample, one month, or an unconfirmed assumption. When a new result changes a line, edit the line and add the ID; do not
add a contradicting line below it.

**Note (2026-11-07):** lines that cite **E01-E37** rely on legacy cards (no JSON, decision not recorded). They are kept
as history; for a current claim use the equivalent in `docs/en/evidence_map.md`, and do not cite a legacy line in the
preprint without re-checking its number.

**What changed since 2026-11-04:** Facebook is now measured (402 posts, model 0.385 vs rule 0.93), the intervals are CQR
(50% and 90%), X moved to v3 with log(age) (0.454), the missing-data switch rules were derived (E163/E164), the tail and
brand totals were quantified (E96/E112/E157/E158), and the graph line was closed with evidence (E127/E128). The
in-platform model itself is unchanged (CatBoost, MAE, ENG features).

**Superseded format:** when a line is overtaken, keep it and append `**superseded by E<new>**`; never delete the old
line. When a line cites a legacy card, a re-run with a JSON is preferred over rewording.

## The task and the ground truth
- Views are public on TikTok, YouTube and X; Facebook Reels/videos expose one in the player (`video_view_count`, not the `play_count` on the Reel card) and we measured 402 of them. Instagram does not without a logged-in session. FB is no longer only transferred: one measured sample exists. **solid** (E13, E16, E93, E95)
- Brand24's reach column is a rule: max(k × reactions, c(followers) × followers), k = 700 on X, 100 FB, 20 IG; reproduces 96-98% of rows. Reconstructed by us, not confirmed by Brand24. **likely** (E99, E111)
- The rule is exact where the platform exports views (TikTok, YouTube) and x9.3 too high on X (median), with 87% of X posts off by >2x; worst for posts with 1-10 reactions (x27) and 1k-100k follower accounts (x16). **solid** (E99)
- Real views per reaction are about 21 / 20 / 27; the rule assumes 700 on X. **likely** (E99, E111)
- Views are checked at very different post ages (median 858 h / 600 h / 534 h for the main check), and keep growing between checks. **solid** (E38, E39)
- Knowing the age of the check helps little overall (+0.000 / +0.003 / +0.012, only X clear of zero) but removes the over-estimate of checks younger than a day on TikTok and YouTube. **likely** (E44)
- Trained only on earlier months (expanding window, test months 2026-05 … 08) the final model is close to brand CV: -0.001 / -0.011 / -0.026, the gap mostly in May when little history exists; brand CV is not optimistic beyond that. Recent data alone (previous month) is worse than all past data. No drift in error over these four months. **likely** (E46)
- The newer checks in the history file do not make the main model better; on X worse (-0.046), because views are re-measured while reactions mostly are not. Train on the original columns. **likely** (E45)

## Estimating views from reactions (now-cast)
- Constant × followers is a poor base: 0.92 / 0.86 / 0.88. **solid** (E99, E111, E165)
- A default boosting model with followers + reactions gets 0.29 / 0.53 / 0.49; tuned families converge at about 0.27 / 0.45 / 0.47 (CatBoost on YouTube and X, SVR on TikTok); ridge on YouTube is worse (0.55). **solid** (E114)
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
- Brand news attention (daily Wikipedia spikes at post time) and company facts (employees, revenue, age from Wikidata) do not help either: gains -0.000 to -0.011, FB transfer unchanged. External brand-level data is closed. (E170, E171)
- Time-varying context computed strictly from the past does not help: brand momentum (leak-free E169 re-test), brand buzz in the Brand24 stream, platform-wide recent level and platform Wikipedia attention all lose 0.000-0.010 in brand CV and more on later months. New text cues (giveaway, call to action, disclosure, mentions, title cues) give +0.009 on X in brand CV only, not confirmed in time. (E172-E175)
- Routing by available fields keeps the bands honest: each pattern (full / no reactions / no followers / nothing) has 90% coverage 0.884-0.900 with its own CQR, while the full model fed missing reactions covers only 0.46 / 0.56 / 0.83. P(within x2) is calibrated per pattern (0.66 TikTok full ... 0.19 YouTube nothing) but does not rank posts within a pattern. **likely** (E176)
- The FB transfer holds only with fresh reactions: on Brand24's own FB likes / comments the same model has MAE 0.831 (x6.8) vs 0.385, still better than the reach rule (1.119). Brand24's FB likes are x2 below current reactions and its FB comments are almost all zero. **likely** (E177)
- Critical review (E178-E188): the noise ceiling is not a check-age artefact (bound unchanged with age matched; a 30-day target changes MAE <= 0.007) (E179). No loss fixes brand totals; the tail stays x9-14 under on YouTube / X (E180). The honest X error over the whole stream is 0.49, not 0.47, and posts without views are ~x1.8 smaller (E181). FB rests on 20 brands; without the 3 largest it is 0.46 (E178), and non-English FB posts are underestimated ~x2.2 (E182). Negative posts are not biased (E187). **likely**
- Business uses (E183-E188): an early view read is the alerting product (top-5% precision 0.82 / 0.86 / 0.71 vs 0.71 / 0.64 / 0.35 from reactions) (E185); brands rank well within industry-months (Kendall 0.73 / 0.70 / 0.60; on X the rule orders slightly better, 0.63) (E186); on X the model puts 47% of brand EMVs within +-50% vs 2% for the rule (E188); creator scores are stable on YouTube (Spearman 0.81, 23 authors) but the sample lacks repeat authors (E183); TikTok anomalies mark promo / spam, YouTube / X do not (E184).
- View ranges (E200): cutting the regression into bins beats a direct classifier on every platform and scheme; accuracy is set by bin width (3 ranges ~0.8, decades ~0.6, half decades ~0.35; X 3 ranges 0.95 = majority). Bins are presentation, not a better model. **likely**
- Mapping reaction scales between platforms (quantile matching, z-score, ratios) without target views makes out-of-platform estimates much worse (TikTok 0.31 -> 0.76-1.04, X 0.64 -> 1.28-1.43): absolute likes carry the view level across platforms better than within-platform rank. With 25-50 calibration posts a mapped CatBoost gains about +0.03 on X only. (E87)
- Adapting the out-of-platform model to the target's reaction structure without target views (importance weights, per-source mixture, views per like) helps only on X (+0.02 to +0.03) and hurts on TikTok; ridge on raw likes + comments stays the robust default. FB / IG estimates of these variants are pre-registered (`fbig_preregistered_v2.csv`). (E88)
- Literature-based transfer methods (anchor regression, group DRO, stabilised weights, invariant subsets, brand carry-over) do not beat ridge on likes + comments, as the shift benchmarks predict (E90-E92). What helps is a few target posts: with 5-10, a shift shrunk toward zero (τ² from platform pairs) beats the raw median (0.315 / 0.626 / 0.588 vs 0.347 / 0.679 / 0.621); from 25-50, intercept + slope; a few-shot conformal 90% interval is calibrated from about 10 posts. (E89)
- Official brand accounts (≤ 5% of posts) get few views per follower but are underestimated by ×2-3; topic clusters and brand communities (industries) are readable but carry little bias beyond reactions; similar brands do not predict a new brand. **likely** (E66, E67, E69)
- Likes carry most of the signal: removing them costs +0.09 / +0.13 / +0.05. **solid** (E17)
- MAE loss instead of RMSE: +0.011 / +0.007 / +0.022. **likely** (E114)
- Error falls with reactions on YouTube (×4.6 at zero to ×1.8 above 1,000) and X (×3.1 to ×2.5); TikTok is flat once a post has any reaction. **solid** (E33)
- Intervals: one width per reaction tier keeps about 90% coverage; one width for all posts under-covers zero-reaction posts. **likely** (E02, E34)
- Systematic error: the lowest fifth of true views is overestimated (0.23 / 0.51 / 0.67), the top fifth underestimated (0.20 / 0.42 / 0.56). Partly an artefact of splitting by the true value. **solid** (E96, E114)
- YouTube posts with zero likes and comments are 23% of posts and 35% of error; possibly missing data rather than true zeros. Unconfirmed. **tentative** (E27)

- Validation: folds that also separate authors and copied texts change MAE by ≤ 0.002 on TikTok and YouTube and -0.011 on X; seeing the brand in training helps ≤ 0.009. **likely** (E55)
- Posts with practically the same followers and reactions differ ×2.8 / ×6.0 / ×4.6 in views; the best possible MAE on them is at least 0.22 / 0.39 / 0.33 (final model 0.33 / 0.58 / 0.47 there). Most remaining error is in the data. **likely** (E56)
- Quantile 10-90% intervals under-cover on held-out brands (73-74%, top fifth 44-54%). Abstaining on the uncertain half cuts MAE by 27% on YouTube, ≤ 10% elsewhere. **likely** (E57, E58)

## Transfer to other platforms
- Ridge on likes + comments transfers best: 0.31 / 0.61 / 0.65 when the platform is left out; the rank order holds but the level shifts (+0.03 / +0.21 / -0.40). **solid** (E131)
- 25-50 posts with true views from the target platform remove most of the shift on X (0.65 → about 0.55). Now checkable on Facebook (E93). **likely** (E115)
- Tuning for transfer, label reweighting and type as a feature do not help consistently. **likely** (E14)
- Reactions predicted across platforms reach Spearman 0.3-0.47 inside a platform; a two-stage reactions-then-views model gives no gain. **likely** (E09)
- First measured FB error (E93): ridge on likes + comments trained on X + YouTube has MAE 0.345 (CI 0.281-0.435, ×2.2) on 130 Facebook Reels/videos, as good as the same features fitted on FB itself (0.355); Brand24 reach 1.145 (×14, Spearman 0.48 against 0.93). X alone is biased high (0.504); a slope fitted on 25-50 FB posts brings it to 0.35-0.37, a median shift does not help X + YouTube. **likely**
- Followers do not carry to FB on the full sample: adding log10(followers) raises MAE 0.345 -> 0.627 (88 posts with followers: 0.360 -> 0.437), followers alone 1.199. **Update (E149/E154):** the penalty is concentrated below 1M followers; above 1M followers help (0.410 -> 0.343) and a hybrid gains 0.410 -> 0.372 there (overall 0.385 -> 0.379, CI includes 0, E155). **likely** (E94, E149)
- Text does not carry to FB: adding 16 e5 components raises MAE 0.345 -> 0.386 (402: 0.385 -> 0.436); text alone 1.24 / 1.31. **likely** (E100)
- Age normalisation (views to 30 days, b = 0.04 / 0.16 / 0.04) changes MAE by <=0.01 but removes the FB level bias (+0.14 -> 0.00): the FB posts are older, so part of the "FB is underestimated" is an age artefact. **likely** (E101)
- Ranking vs level (E102): where Brand24 has views (TikTok/YouTube) its reach is the view count, so brand rankings and top-10% alerts are perfect by construction; on X its brand ranking (Kendall 0.63) and alert precision (0.61) are slightly above the model (0.58 / 0.58) although its level is x9.3 off. The model's value is the calibrated level and intervals, not the ordering. **likely**
- Extreme tail (E104): for the top 1% of views, raw likes beat the MAE model on every platform (0.64 vs 0.56 TikTok, 0.64 vs 0.48 YouTube, 0.47 vs 0.26 X); Brand24 is exact on TikTok/YouTube and 0.42 on X. Alerts at the 1% level need engagement or a dedicated precision@k model, not the reach regression. **likely**
- Same content across platforms is essentially absent (1 text of 12 065 on 2+ platforms), so a platform multiplier cannot be estimated from repeated content. (E103)
- X view counts are missing not at random: 25.4% missing, 66.5% of 2025 posts against 22.4% of 2026, older checks (66 vs 22 days), smaller accounts (239 vs 435 followers); selection AUC 0.84. The observed-only mean reach is 435 against 332 for the full sample (~1.3x), so X averages from posts with views overstate the platform. **likely** (E105)
- Alert budget (E106): to catch half of the top 1% on X the model needs 117 alerts, raw likes 22 and Brand24 25; TikTok 20/16/13, YouTube 26/17/13 (Brand24 is the view count there). The MAE model is the worst alert engine at the 1% level; use engagement or a precision@k model for alerts. **likely**
- Brand totals on X: the rule is x11.2 (median), 96% of 189 brands off by >2x; the model (uncalibrated) x0.66, 49% off, median error 0.30 against 1.05, with the brand order unchanged (Kendall 0.63 vs 0.58). On TikTok/YouTube the rule is exact. **likely** (E107)
- Error is predictable from metadata: AUC 0.936 for |error| > 0.5; flagging the 20% riskiest posts carries 45% of the total error (MAE 0.484 -> 0.333), 50% carries 78% (0.216). A confidence flag is cheap and works; show the reason (platform, age, missing fields). **likely** (E108)
- Calibrated X brand totals: 43% of 189 brands still off by >2x (the rule: 96%), median error 0.26, bias +0.05; an 80% brand interval covers 79.3% of held-out brands. **likely** (E109)
- Posting cadence does not predict reach per post: Spearman 0.03 across 640 brand-platform groups (0.06 after followers and platform), within-brand <1 day vs >7 days differs by 0.004 log10. No saturation in this sample. **likely** (E110)
- Brand24's FB reach, measured on 359 posts: x4.1 too high (median), MAE 0.93, 82% of posts off by >2x; the error is regression to the mean (small posts x34 high, large x2 low). The X+YouTube model has MAE 0.41. **likely** (E111)
- Brand totals need many posts: bootstrapping 5 posts gives a x53 p90/p10 spread and 17% of totals within 2x of the brand total; 20 posts still x6.4 and 66%. Report intervals and the number of posts. **likely** (E112)
- Multi-task across platforms does not help: a shared trunk with per-platform heads is equal on TikTok (0.314 vs 0.315 CatBoost) and worse on YouTube (0.538 vs 0.518) and X (0.504 vs 0.485). Consistent with E61 pooling. **likely** (E113)
- Validation audit (E114): repeated grouped CV gives TikTok 0.269 ± 0.011, YouTube 0.466 ± 0.030, X 0.463 ± 0.018; isotonic recalibration (0.279/0.465/0.476) and a per-reaction-tier shift (0.274/0.457/0.469) do not improve on the point model. Bias by decile stays within ±0.11. **likely**
- Calibration budget (E115): zero-shot transfer MAE 0.313 TikTok, 0.612 YouTube, 0.650 X, 0.385 Facebook; a median shift on target posts does not help TikTok/YouTube/Facebook (it hurts at small n) and cuts X from 0.650 to 0.551-0.563 with 25-50 posts. **likely**
- YouTube zero-reaction posts (27%) are not missing: marking 0 as NaN + flag gains 0.003 MAE (0.456 -> 0.453) and dropping them is much worse (0.499); the error on those posts stays 0.66. The zeros are informative (small posts). **likely** (E116)
- A 0/0 first check (37% of first checks, median age 11.7 h) still gets views: 86% > 0, median 5 (X) / 60 (TikTok), 0.4% > 1k; at the last check 3.5% > 1k. At collection, zero reactions are common (X 64%, YouTube 29%) and 23% of YouTube zero-reaction posts exceed 1k views. **likely** (E117)
- CQR fixes the 90% interval: raw 5-95 quantiles cover 74%, CQR 0.889 / 0.902 / 0.891 at widths 1.14 / 1.71 / 2.07 log10; split conformal 0.883-0.894. Report CQR, not the raw quantile band. **likely** (E118)
- Author target encoding helps only posts whose author is seen in training (X 0.530 -> 0.430, YouTube 0.465 -> 0.446, TikTok unchanged), but that is 2.8% / 8.1% / 1% of posts, so overall <= 0.003. For Brand24's repeated authors this is the largest lever. **likely** (E119)
- Brand history does **not** substitute for a missing author history (E169): with a temporal split, adding a smoothed brand prior changes MAE by -0.005 (TikTok), -0.035 (YouTube) and +0.006 (X, CI includes 0), and only 35-52% of test posts have a known brand. Without author history the model is the base model; the substitutes are an earlier view check (E40) or client calibration (E115). **likely**
- Authors do not repeat across platforms (0-0.1% of posts), so cross-platform author history cannot be used; where it exists it is noise. Consistent with E92 (brand carry-over) and E09. **likely** (E120)
- Recency, sequence, activity and text-surface columns do not help: TikTok -0.002, YouTube +0.004, X -0.001. The feature set stays followers + reactions + flags. **likely** (E121)
- Spectral author-brand embeddings hurt by 0.005-0.023; the graph line stays closed (E03/E06/E07/E12/E29/E65). **likely** (E122)
- CQR on the FB transfer: 90% band covers 0.925 at width 1.99 (split conformal 0.932 at 2.07), calibrated on 25% of the FB brands. **likely** (E123)
- Final estimator v2 (E124): CatBoost + author_te (when the author has history) + CQR: MAE 0.273/0.457/0.466, 90% coverage 0.889/0.902/0.897, widths 1.14/1.72/2.08. Tiny MAE gain, no cost; this is the model card estimator. **likely**
- CQR per reaction tier is not needed: global CQR covers every tier (0.86-0.90); per-tier calibration adds noise. **likely** (E125)
- A 0/0 post at the first check: median 5 views on X (p90 49) and 60 on TikTok (p90 179); at collection the p90 is 124 (X) and 3 383 (YouTube). Report a range. **likely** (E126)
- Graph topology is not informative once reactions are known: metapath2vec on author-brand-hashtag embeddings makes MAE worse (0.278/0.480/0.480 vs 0.273/0.456/0.467), like hand features (<=0.009), GraphSAGE (E65) and spectral embeddings (E122). The only useful history signal is the author's own past views (E68, E119). **solid** (E127)
- Why the graph fails: the mean residual of training posts sharing an author, brand or hashtag does not predict a test post's residual (author rho <= 0.069, X -0.015; brand and hashtag ~0), and adding it always raises MAE. Co-posting edges are not diffusion edges; the author gain (E119) is a level prior, not propagation. **solid** (E128)
- A distributional CatBoost (RMSEWithUncertainty) is slightly worse on MAE and under-covers (0.871/0.884/0.864 at nominal 90%); CQR stays the interval method. **likely** (E129)
- At posting time (no reactions) the MAE is 0.601/0.757/0.611 and the 90% CQR band is 2.4-3.0 log10 wide (x260-900); with reactions it shrinks to 1.1-2.1. That is the quantitative value of waiting for reactions. **likely** (E130)
- Out-of-platform CQR: with 25% target-brand calibration, coverage stays 0.905/0.912/0.908 but the band is 1.51/2.84/3.69 log10 wide (x32/690/4900) vs 1.14/1.71/2.07 in-platform; MAE 0.351/0.658/0.716 vs 0.273/0.456/0.467. Transfer only with the wide band. **likely** (E131)
- The FB band is not uniform: the top quartile of FB views is covered 0.782 and pages above 1M followers 0.857, while small posts are over-covered (global 0.925). Calibrate the FB band by follower tier. **likely** (E132)
- Recency weighting does not help temporal drift: base 0.274/0.440/0.464, best half-life 365 0.272/0.439/0.464. **likely** (E133)
- Seed averaging gives no usable gain (0.272/0.454/0.468 vs 0.273/0.456/0.467); the model is not seed-sensitive. **likely** (E134)
- Follower-tier recalibration makes MAE worse (0.276/0.466/0.481 vs 0.273/0.460/0.470); the point model is already calibrated and the regression to the mean is not a tier shift. **likely** (E135)
- Rolling-origin validation: train before each of the last months, test on the month: TikTok 0.270 ± 0.004, YouTube 0.450 ± 0.018, X 0.473 ± 0.005, close to brand CV; the validation is not over-optimistic in time. **likely** (E136)
- A 50% CQR band is calibrated (0.495/0.483/0.500) and much narrower (0.45/0.74/0.88 log10, x2.8-7.6) than the 90% band (1.14/1.71/2.07). Report both: likely range and worst case. **likely** (E137)
- Missing reactions degrade the estimate: 10% missing costs 0.03-0.08 MAE, 50% costs 0.08-0.37, 100% is much worse than a model trained without reactions (TikTok 1.00 vs 0.60). X is the most robust, TikTok the most fragile. **likely** (E138)
- Target definition: views per day is no better than views (0.276/0.458/0.459); adding log(age) helps X (0.467 -> 0.454) and YouTube slightly (0.456 -> 0.454), TikTok unchanged. Consistent with E44. **likely** (E139)
- Hurdle for zero views is not applicable: zeros are 0.4-2.5% (9/32/63) and the two-stage model equals the single one on the positive sample; close the line. **likely** (E140)
- Final X model v3 with log(age): MAE 0.467 -> 0.454, 90% band 2.08 -> 1.99 (coverage 0.899), 50% band 0.83. Use v3 for X; TikTok/YouTube stay on v2. **likely** (E141)
- Temporal CQR: the bands stay calibrated on future months (0.907/0.901/0.932); temporal validation supports the CQR band. **likely** (E142)
- Adding age to the FB transfer hurts (0.385 -> 0.475, bias +0.35); normalise the target to 30 days instead (E101). **likely** (E143)
- Per-tier CQR on FB does not fix the >1M under-coverage (0.857 unchanged; fewer than 20 calibration posts per tier); widen for large pages. **likely** (E144)
- FB tier calibration with 20 pooled splits: per-tier quantiles do not fix >1M (0.843 -> 0.832) and make 10k-1M slightly worse (0.905 -> 0.891); the gap is not a single-width problem. Widen >1M by a fixed factor and say so. **likely** (E145)
- Widening the FB >1M band by 1.3-2.0 barely moves coverage (0.843 -> 0.848-0.859) at a width cost of 1.46 -> 1.82; the problem is not width. **likely** (E146)
- Even a calibrated level shift (+0.32, the model under-estimates large pages) plus 1.5x width reaches only 0.818 coverage on >1M; those 70 posts cannot be calibrated with this sample. Report the >1M band as uncalibrated/wide. **likely** (E147)
- The FB band is conservative for pages below 1M followers (coverage 0.940, width 1.99); the global 0.925 hides an over-covering majority and an under-covering 70-post tail. **likely** (E148)
- Followers are a large-page feature on FB: on >1M pages they lower MAE from 0.410 to 0.343 and remove the level bias (+0.33 -> -0.06), but on the rest they blow up the error (0.380 -> 0.727). Use followers only for >1M, not globally (E94's "followers hurt" is a whole-sample statement). **likely** (E149)
- A model trained only on source pages above 1M followers beats the global transfer on the FB >1M pages (0.410 -> 0.368, bias +0.33 -> -0.06); the top of the distribution needs its own relation. **likely** (E150)
- Even a dedicated large-page model cannot calibrate the >1M band (coverage 0.796 vs global 0.843); with 70 posts the tier cannot support a 90% claim. Report the point, not a calibrated band. **likely** (E151)
- Error by industry spans 0.22-0.30 MAE (TikTok 0.197-0.421, YouTube 0.388-0.686, X 0.404-0.648); media/entertainment and social media are hardest, consumer goods/food easiest. Brand size is a weak signal (TikTok +0.13, YouTube -0.17). **likely** (E152)
- Hybrid FB (global below 1M, large-page model above): MAE 0.379 vs 0.385; the >1M subset 0.410 -> 0.372. Small overall gain, coherent spec. **likely** (E154)
- The hybrid FB gain is not significant: 0.385 -> 0.379, paired brand bootstrap CI [-0.035, +0.015] includes zero (E155). Treat the hybrid as optional for >1M, not as an overall improvement. **likely** (E155)
- FB error by industry spans 0.263 (beverages) to 0.597 (sportswear), consistent with the viral-prone categories on TT/YT/X (E152). **likely** (E156)
- FB brand totals (19 brands >=5 posts) are low by x2 (median ratio -0.31), 53% off >2x, but the order is good (Kendall 0.801); the hybrid does not improve either. Apply the E109 calibration and intervals. **likely** (E157)
- FB brand totals after calibration: level bias removed (+0.31 -> -0.01), off-2x only 53% -> 47%, and the 80% brand interval covers 0.737 with 19 brands. FB brand totals remain uncertain; report with a warning or avoid small-sample totals. **likely** (E158)
- Scaling the FB brand interval by sqrt(18/n_posts) raises the 80% coverage from 0.737 to 0.789, still below 0.80; use as a heuristic, not a calibrated claim. **likely** (E159)
- Restricting FB brand totals to brands with >=10 posts does not help: coverage 0.714 (14 brands) vs 0.737 (19). **likely** (E160)
- Missing followers hurt YouTube and X (0.456 -> 0.711; 0.467 -> 0.733) but barely TikTok (0.273 -> 0.294). Complementary to E138: TikTok needs reactions, YouTube/X need followers. **likely** (E161)
- The FB transfer runs on reactions (MAE 0.385); followers alone are useless (1.342) and help only above 1M (0.410 -> 0.343) while hurting the rest (0.380 -> 0.727). **likely** (E162)
- When a feature is systematically missing, retrain without it: the specialised model beats the full one in every scenario, e.g. no reactions TikTok 1.139 -> 0.583; no followers YouTube 0.711 -> 0.494, X 0.733 -> 0.515. The E138/E161 numbers are the cost of using the wrong model, not the floor. **likely** (E163)
- Partial-missing crossover: switch to followers-only above ~35% missing reactions on TikTok/YouTube (or ~70% on X); switch to reactions-only above ~20% missing followers on YouTube/X (on TikTok reactions-only is equal even at 0%). **likely** (E164)
- FB rule vs model by follower tier: the rule is 0.958-1.012 MAE in every tier, the reaction model 0.364-0.463; the model wins in every tier, so there is no tier where the rule should be kept. **likely** (E165)
- The 130 hand-checked FB posts are a subset of the 402 fetched and slightly easier (model MAE 0.345 vs 0.385, median views 1 080 vs 1 414); the full-sample 0.385 is the number to report. **likely** (E153)
- FB CQR on the 130 hand-checked posts: 90% coverage 0.898 at width 1.59 (vs 0.925 / 1.99 on 402); the interval procedure holds on the smaller, easier sample. **likely** (E166)
- On all 402 public FB videos the same model has MAE 0.385 (CI 0.312-0.482, x2.4), bias +0.14; Reels 0.383, videos 0.390, no trend with post age. The 68 unavailable posts come from smaller pages (median followers 66k vs 110k) with similar reactions. **likely** (E95)
- Intervals on the FB estimate: source-platform residuals over-cover (80% band -> 0.925 coverage, 90% -> 0.960); split conformal on 50 FB posts gives 0.79 / 0.88. **likely** (E98)
- Hand-check budget: the reaction-stratified design is no better than random; extremes of the estimate or high-leverage points help by at most 0.03 at k = 50, and only 0.008 on the 402 set. **tentative** (E97)

## Reach shape
- Reach is a tail phenomenon: the top 1% of posts carries 60-74% of views (top 5%: 82-90%), Gini 0.92-0.96, 2-4% of posts give 80% of views; for 56% of brands one post is more than half of the brand's reach. Brand totals and averages therefore need intervals and tail monitoring. **solid** (E96)


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

Rows citing E01-E37 are legacy (no JSON); they are kept for the record, not as measured evidence.
| Idea | Result | IDs |
|---|---|---|
| Our AI type and sentiment labels as features | at most +0.005 | E121 |
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
- Brand24's Facebook comments are close to absent (6 of 152 posts non-zero, 22 in total) while the pages show thousands; FB `likes` also differ from the page's total reactions. On Instagram 0 of 160 posts expose a view count without a login, 115 give likes and comments. (B24)

## Platform distribution (reverse-engineered)

- **The visible distribution differs by platform (E167).** Follower elasticity on log10 views is **0.044 TikTok / 0.303
  YouTube / 0.164 X**; reaction elasticity is **0.82 / 0.66 / 1.06**; **43% / 5% / 14%** of posts get more views than the
  author has followers. TikTok is feed-driven (followers barely matter), YouTube is subscriber-driven, X is in between
  with views roughly proportional to reactions. **likely**
- **Brand24's rule is fully recoverable only on Instagram (E168).** A simple max(k x reactions, c(followers) x followers)
  reproduces **97%** of IG rows, **62%** of X, **49%** of FB; on TikTok/YouTube the column is the view count, so the
  near-0 reproduction is correct, not a failure. **likely**

## External sample (2026-11-08)

- **An independent YouTube sample is feasible; TikTok, X and Instagram are not** (E189). `yt-dlp` search fetches 53
  brand-related videos (recent, 2025+), but TikTok reports "No working app info", x.com answers HTTP 520 and Instagram has
  no public views; those need an API, a login or a provider (`docs/en/external_sample.md`). **solid** (for feasibility)
- **The model is calibrated to the Brand24 mention population, not to brand videos in general** (E189): on the external
  sample the relation survives (Spearman 0.93, calibration slope 0.83) but the level does not - median 6,697 views against
  433 in the export, bias +0.71 (x5 low), MAE 0.726; after removing the level shift MAE is 0.540 against 0.456 inside.
  External use needs a small recalibration set (E115). **likely**

## External sample, diverse strata (2026-11-08)

- **Zero-view public videos are not reachable through brand channels or search** (E190: 0 in 131, minimum 71), but
  obscure third-party mentions do include them (E191: 1 zero, several 1-view, 17 below 100). A zero in platform data can
  therefore be hidden/not collected, but real zeros exist among tiny posts. **solid** (reachability)
- **The model is a small-post model** (E190): on the small external stratum MAE 0.289 (bias +0.17), on popular videos
  MAE 1.480 (bias +1.48, ~x30 low); recent 0.620, newest 0.601. The mention population is small posts, so the model works
  there and needs recalibration above it (E115, E132). **likely**

## Third-party mention sample (2026-11-08)

- **The mention-like population is reachable and the model works there** (E191, expanded in E193, E195, E197): 203
  third-party videos from 60 brands (brand channels excluded), median 437 views, 10 zero-view; model MAE **0.536 low /
  0.555 mid / 1.044 top**, overall **0.606** with bias +0.28 (189 validation rows). The earlier x5 shift (E189/E190) came from
  brand-owned and popular videos, not from the model; the remaining boundary is video size. **likely**
- **YouTube export views are much staler than TikTok's** (E194): re-reading 60 posts gives 58 readable, median new/old
  1.516, p90 25.3, only 19% within 5%, Spearman 0.857. Level comparisons need an age correction (E101); the export
  understates current popular videos. **solid**
- **Facebook public view counts are frozen over three days** (E196): 60/60 posts re-read, ratio exactly 1.000 on every
  post, unlike TikTok (+7% median) and YouTube (+52%). The public counter is static or cached, so the FB sample is a valid
  single-time snapshot and drift is not a source of error there. **solid**
- **TikTok export views are stale but rank-stable** (E192): re-reading 60 posts gives 40 readable, median new/old 1.068,
  p90 3.07, only 45% within 5%, but Spearman 0.918. Level comparisons need a common check date or an age correction
  (E101); a third of the posts cannot be read anonymously. **solid**

## Facebook measurability (2026-11-08)

- **Only 470 of the 2,500 Facebook rows are videos with a public view count; 402 were readable.** A full sweep of all
  2,500 URLs gives 402 ok, 68 removed and **2,030 ordinary posts/photos with no video id** - so the 402 is the complete
  measurable subset, not a stopping point (E200). Ordinary Facebook posts are unmeasurable without the owner's insights,
  like Instagram. **solid**

## External X sample (2026-11-08)

- **The X API route works and is cheap; the 7-day stream is dominated by tiny posts** (E198): 98 posts for **$1.47**
  (2,500 would be ~$37.50), median **8 views** against 27 in the export. Model MAE **0.565**, bias **-0.29** (over-predicts
  the small posts by ~x2), median baseline 0.747. For a representative X sample we need several weekly windows, not a
  bigger one-off budget. **solid** (feasibility), **likely** (error)

## Open items (state 2026-11-07)

- **Instagram**: no public views (0/160 anonymous); a manual sample, a login or a provider is needed. Everything else in
  the plan was run (see `docs/en/plan_status.md`).
- **False brand matches**: documented on examples only; a detector needs 100-200 hand labels (backlog B15).
