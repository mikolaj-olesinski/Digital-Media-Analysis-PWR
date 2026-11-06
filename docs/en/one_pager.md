# One-pager for Brand24

**Question.** Can the visibility (views) of a post be estimated better than by a constant, and with what uncertainty?

## What we did

- 12,500 Brand24 mentions (2,500 each from TikTok, YouTube, X, Facebook, Instagram); public view counts exist on
  TikTok, YouTube and X and serve as ground truth.
- Final model: CatBoost on followers, likes, comments, shares and missing flags; brand-grouped validation; CQR intervals.
- Facebook measured separately: 130 hand-checked and 402 fetched public Reels/videos.

## Results

| platform | model MAE | typical error | 50% band | 90% band | your rule |
|---|---|---|---|---|---|
| TikTok | 0.273 | x1.9 | x2.8 | x14 | exact (already views) |
| YouTube | 0.456 | x2.9 | x5.5 | x51 | exact (already views) |
| X | 0.454 | x2.9 | x7.6 | x98 | x9.3 too high |
| Facebook | 0.385 | x2.4 | — | x98 (below 1M) | x4.1 too high |

- On Facebook the model beats the rule in **every** follower tier (MAE 0.36-0.46 vs ~1.0).
- The bands are calibrated in-platform (90% covers 0.89-0.90); raw quantiles would cover only 74%.

## What we recommend

1. **Use the model for levels and intervals** on X and Facebook; keep your rule on TikTok/YouTube (it is the view count).
2. **Report ranges, not points**: a 50% "likely" band and a 90% band.
3. **Do not use it for alerts**: raw likes / your rule rank better at the top 1%.
4. **Brand totals**: only with an interval and enough posts (20 posts still give a x6.4 spread).
5. **Missing data**: retrain without the missing feature (TikTok without reactions 1.14 -> 0.58).

## What we cannot do

- Instagram: no public views (0 of 160 anonymous checks); no visibility estimate without a login or a provider.
- Facebook ordinary posts (not video/Reel): no view count.
- A calibrated band for Facebook pages above 1M followers (coverage 0.80-0.86 with 70 posts).
- Unique reach (people): only views are public.

## What we need from you

1. Confirm the constants behind `b24_estimated_reach`.
2. Tell us whether a zero YouTube view count can mean "not collected".
3. Decide on Instagram: manual sample, login-based access, or a provider.

Numbers: `docs/en/evidence_map.md`; details: `docs/en/model_card.md`; questions: `docs/en/client_qa.md`.
