# What is new since E94

Internal changelog for the team: the experiments added in the November 2026 push (snapshot 2026-11-07), grouped by
theme with the one-line result. Not a current-numbers document. Details in the cards (`agent/vault/`) and the register.

## Facebook

- Measured the transfer on 130 hand-checked and 402 fetched posts: MAE 0.345 / 0.385, against 0.93 for the rule
  (E93, E95, E111).
- The rule is ×4.1 too high, 82% of posts off by >2×, and it is wrong in every follower tier (0.96-1.01); the model wins
  in every tier (0.36-0.46) (E165).
- The 130 hand posts are a subset of the 402 and slightly easier (0.345 vs 0.385) (E153).

## FB intervals

- CQR: 90% coverage 0.925 at width 1.99 (E123); 0.898 / 1.59 on the 130 hand posts (E166).
- Pages below 1M followers cover 0.94; above 1M only 0.84 (E132, E148).
- Widening, a level shift, per-tier quantiles and a large-page model do not calibrate >1M (E144-E147, E151); report the
  point only there. A large-page model improves the point (0.410 -> 0.368) and the hybrid 0.385 -> 0.379 (not
  significant, E150/E154/E155).
- Followers are a large-page feature: they help above 1M (0.410 -> 0.343) and hurt below (0.380 -> 0.727) (E149, E162).

## In-platform uncertainty and model quality

- CQR gives honest bands in-platform: 0.889 / 0.902 / 0.899 (X v3); a 50% band is 0.45 / 0.74 / 0.83 (E118, E137, E141).
- The bands hold out of time (0.907 / 0.901 / 0.932) (E142).
- Repeated CV: 0.269 ± 0.011 / 0.466 ± 0.030 / 0.463 ± 0.018; isotonic and tier recalibration do not help (E114).
- Seed averaging (E134), follower-tier recalibration (E135) and recency weighting (E133) do not help.
- Author target encoding helps posts whose author is seen (X 0.530 -> 0.430; YouTube 0.465 -> 0.446), but only 1-8% of
  posts (E119); cross-platform author history does not exist (E120).
- log(age) helps X (0.467 -> 0.454) and narrows its band (E44, E139, E141); the rate target does not (E139).
- Rolling-origin validates the brand CV (0.270 / 0.450 / 0.473) (E136).

## Transfer and calibration budget

- Out-of-platform CQR: coverage 0.905-0.912 but widths 1.51 / 2.84 / 3.69 vs 1.14-2.07 in-platform (E131).
- Only X needs calibration: 25-50 posts bring 0.650 to 0.551-0.563; TikTok/YouTube/FB are zero-shot (E115).

## Missing data

- 25.4% of X views are missing and not at random; observed averages are ~1.3× biased (E105).
- YouTube zeros are not missing (0.003 gain from treating them so, E116); a 0/0 first check still gets views (median 5
  on X, 60 on TikTok) (E117, E126).
- Missing fractions: 10% reactions costs 0.03-0.08 MAE; 50% costs 0.08-0.37 (E138); platforms differ (E161).
- If a feature is missing systematically, retrain without it; switch at ~35% missing reactions (TikTok/YouTube), ~70%
  (X), ~20% missing followers (YouTube/X) (E163, E164).

## Tail, selection and alerts

- Top 1% of posts hold 60-74% of views; 2-4% of posts give 80% (E96).
- Brand totals are unstable on few posts (5 posts: ×53 spread; 20: ×6.4) (E112) and ×2 low before calibration
  (E109, E157, E158).
- Hand-check selection: reaction strata are no better than random; extreme posts and leverage are best at small budgets
  (E97); conformal covers over-conservatively (E98).
- Alerts: raw likes/Brand24 rank better at the top 1% than the MAE model (E104, E106).

## Nulls and closed lines

- Text (E100), extra columns (E121), graph features (E122, E127, E128), multi-task (E113), seed averaging (E134): no
  gain; the graph has no residual signal along its edges (E128).
- Hurdle for zero views (E140): zeros are 0.4-2.5%, no gain.
