# Estimator specification

Status: internal, 2026-11-07. The exact recipe behind the model card. Not preprint text.
Reproduce it: `docs/en/reproduce.md` (six commands).

## Versions

| platform | point model | interval | features |
|---|---|---|---|
| TikTok | CatBoost MAE (v2) | CQR 50%/90% | ENG |
| YouTube | CatBoost MAE (v2) | CQR 50%/90% | ENG |
| X | CatBoost MAE (v3) | CQR 50%/90% | ENG + log(age) |
| Facebook (video/Reel) | ridge on likes + comments (transfer); optional large-page model >1M | CQR 50%/90% for <1M; >1M uncalibrated | likes, comments (+ followers for >1M) |
| Instagram | **not estimated** | — | — |

`ENG` = log10(followers), log1p(likes), log1p(comments), log1p(shares) and a missing flag for each.

## Pipeline

1. **Target:** `y = log10(view_count)`, the view count at the age of the check.
2. **Features:** `ENG`, plus `log10(age_days)` for X only. Age = `metrics_updated_at - created_at`, clipped to
   0.1-3650 days. For FB, age is not used (E143).
3. **Point model:** CatBoost, `loss_function="MAE"`, 300 iterations, default depth/learning rate, seed 42.
   For FB: ridge (alpha 1) on `log1p(likes)`, `log1p(comments)`, trained on X + YouTube. Optional: for pages above
   1M followers use a large-page model (+ followers); the overall gain is 0.385 -> 0.379, CI includes 0 (E154/E155).
4. **Intervals:** LightGBM quantile models at 0.05 and 0.95 on the same features; CQR conformity score
   `max(q_lo - y, y - q_hi)` on 25% of the training brands (brand-grouped); the 50% and 90% bands use the 50th and 90th
   percentile of the score.
5. **Flags (E108):** missing likes/comments/shares; zero reactions; check age > 30 days; platform X; |point - rule| large.
6. **Output:** point, 50% band, 90% band, flags, and the number of posts behind a brand total.

## Parameters

- CatBoost: `iterations=300`, `loss_function="MAE"`, `random_seed=42`, `allow_writing_files=False`, `thread_count=4`.
- LightGBM quantile: `n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `subsample=0.9`, `colsample_bytree=0.9`.
- CQR calibration share: 25% of brands; fallback to the global score when a tier has fewer than 20 calibration posts.
- Brand-grouped 5-fold CV for every number.

## Evidence

| step | evidence |
|---|---|
| features ENG | E114, E124 |
| MAE loss | E114 |
| CQR 90% coverage | E118 (0.889/0.902), E141 (X 0.899), E142 (temporal) |
| 50% band | E137 |
| age for X | E44, E139, E141 |
| transfer FB | E93, E95, E123 |
| calibration budget | E115 |
| missing data | E138 |
| flags | E108 |

## Known limits

- Instagram has no public views; do not estimate.
- Facebook only for video/Reel; the page shows plays, not the counter modelled here.
- The band is wide: x2.8-7.6 at 50%, x14-117 at 90% (platform-dependent).
- Large FB pages are under-covered (`>1M` followers 0.83-0.86); widen the band there.
- The point is calibrated in the middle; tail posts remain over/under-estimated (E96, E114).

## Code

- `src/modeling.py`, `src/extra_models.py`, `src/reaction_tiers.py` (features and final model).
- `src/validation_audit.py` (repeated CV), `src/cqr_intervals.py` (CQR), `src/cqr_50.py` (50% band).
- `src/fb_transfer_check.py` (FB transfer), `src/fb_cqr.py` (FB CQR), `src/confidence_flags.py` (flags).
- Numbers: `data/processed/*_numbers.json`.

## Where the intervals come from (CQR in five steps)

1. Train two quantile models on the training brands: `q05` (5th percentile) and `q95` (95th), same features as the
   point model.
2. On a held-out 25% of the training brands (brand-grouped), compute the conformity score for each post:
   `E = max(q05 - y, y - q95)` - how far the true value falls outside the raw band.
3. Take `q = quantile(E, 0.90)` for the 90% band (or 0.50 for the 50% band).
4. The reported band is `[q05 - q, q95 + q]`: the raw quantile band widened by the calibrated amount.
5. Check coverage on the held-out brands; nominal is 90% for the 90% band (0.889 / 0.902 / 0.899 in-platform, E118/E141).

Why not the raw quantiles: they cover only 74% instead of 90% (E118). Why CQR and not a distributional model: the
Gaussian band of a distributional CatBoost covers 0.86-0.88 (E129). On Facebook the quantile models are trained on
X + YouTube and calibrated on 25% of the FB brands (E123).

## Calibration size and fallbacks

- The calibration split is **25% of the training brands**, grouped by brand, so no post of a brand sits on both sides.
- If a subgroup (a follower tier) has fewer than **20 calibration posts**, fall back to the global conformity score and
  do not claim a calibrated band for that subgroup. This is why the >1M Facebook tier stays uncalibrated (E144/E145).
- With small brands the band is conservative, not narrow: Facebook pages below 1M are covered 0.94 at nominal 90%
  (E148).
- The parameters in this section match the code: `src/cqr_intervals.py` and `src/final_v2.py` use `test_size=0.25` and
  `np.quantile(E, 0.9)`; `src/fb_cqr_tier_fix.py` uses the 20-post fallback.
