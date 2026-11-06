# Reporting guide: what to say about a post's visibility

Internal one-pager for turning the estimator into a sentence a client can read. Not preprint text.

## Three principles

1. **Always a range, never a bare point.** The point is the middle of a wide distribution; the interval is the claim.
2. **Views are not people.** Platforms publish views, not unique reach. Say "views", not "people reached".
3. **Say when the number was read.** Views grow with age; a post read at 1 hour and at 30 days are different targets.

## By moment

| moment | what to report | 90% band (log10) | typical factor |
|---|---|---|---|
| at posting, before reactions | range only, from the posting-time model | 2.42-2.96 | x260-x900 |
| at collection, with reactions | point + CQR band | 1.14 / 1.71 / 1.99 (TikTok / YouTube / X, X is v3) | x14 / x51 / x98 |
| Facebook transfer (video/Reel) | point + CQR band | 1.99 | x98 |
| early scan with 0 reactions | empirical range (E126): X median 5, p90 49; TikTok median 60, p90 179 | — | — |

- With reactions the MAE is 0.273 / 0.456 / 0.454 (x1.9 / x2.9 / x2.8; X is v3).
- A 50% "likely" band is 0.45 / 0.74 / 0.83 log10 wide (x2.8 / x5.5 / x7.6); the 90% band is 1.14 / 1.71 / 1.99
  (E137/E141). Report both.
- At posting time the MAE is 0.601 / 0.757 / 0.611; do not report a point then.
- If the target platform has no data at all, the transfer band is 1.51 / 2.84 / 3.69 (x32 / x690 / x4 900), with 25%
  target-brand calibration (E131).

## By platform

- **TikTok, YouTube:** views are public; use the in-platform model and CQR. This is the best case.
- **X:** views are public but the sample is hard (25% missing, 64% zero reactions); use the model, report the wide band,
  and flag X as less certain.
- **Facebook:** only video/Reel has `video_view_count`; ordinary posts have no view count. Use the transfer model; the
  page shows plays, which differ from the counter we model by a median x3.6.
- **Instagram:** no public views; do not report a visibility estimate - report reactions only, or say "not measurable
  without a login".

## Suggested wording

Say the point, then the two bands, then the age:

> "Estimated 12k views (likely 3k-34k, 90% range 0.9k-170k), read 2 days after posting."

The multipliers by platform (applied to the point estimate):

| platform | likely band (50%) | 90% band | what to add |
|---|---|---|---|
| TikTok | x2.8 | x14 | "views are public on TikTok" |
| YouTube | x5.5 | x51 | "views are public on YouTube" |
| X | x7.6 | x98 | "X counts are incomplete for older posts" |
| Facebook (video/Reel) | not calibrated | x98 below 1M | "Facebook does not show this counter; only for video/Reel" |
| Facebook above 1M | not calibrated | not calibrated | "point only" |
| at posting time | not calibrated | x260-900 | "too early; this is a wide range" |

- "Too early to estimate: this post has no reactions yet; posts scanned at this age usually stay under 200 views
  (median 5 on X), but 0.4% exceed 1k."
- "Facebook Reel: estimated 8k views (90% range 80-800k); Facebook does not expose the view count for this post type."

Never write the point without at least the 90% band; never call the band a guarantee.

## When coverage is not homogeneous

- Report the **mean coverage for the headline**, then the **worst subgroup in the same breath**: "90% band covers
  0.89-0.93 overall; 0.94 below 1M followers, 0.80-0.86 above, and 0.78 in the top view quartile".
- Never say "the band covers 90%" without the subgroup caveat when one exists.
- If a subgroup is uncalibrated, say "point only" and do not draw a band (FB above 1M).
- Prefer a **range** over a single rounded number when experiments disagree (0.80-0.86, not 0.84).

## When a subgroup is uncalibrated (example: FB above 1M)

Report the point, the age of the check, and **say the band is not calibrated for this subgroup**. Do not attach a 90% to
it, do not draw a band, and do not put it in a table next to calibrated cells without a footnote.

Example: "Estimated 1.2M views for this page, read 3 days after posting. The 90% band is not calibrated above 1M
followers, so treat this as an order-of-magnitude estimate (the model tends to under-estimate large pages)."

If an order of magnitude is needed, give a **heuristic** range and label it "heuristic, not a 90% band".


## Figures to show

- `figures/fb_rule_vs_model_{en,pl}.png` - the model against the client's own rule, tier by tier; the best opener.
- `figures/cqr_bands_{en,pl}.png` - what the 50% and 90% bands mean per platform.
- `figures/fb_coverage_{en,pl}.png` - where the Facebook band holds (below 1M) and where it does not (above 1M).

All are built by `src/result_figures.py`; live vs historical figures are listed in `scripts.md`.

## Never

- A point without an interval.
- An alert ranking based on this model: raw likes/Brand24 rank better at the top 1% (E104/E106).
- A brand total from a handful of posts: 5 posts give a x53 p90/p10 spread, 20 posts still x6.4 (E112).
- A comparison of brands on different numbers of posts without saying so.

## Data-quality flags (E108)

Attach a flag when: likes/comments/shares are missing; the post has zero reactions; the check is old; the platform is X;
the reach is far from the model. The flag is a reason to widen the band or re-check, not a reason to hide the estimate.

## Files

- Model card: `docs/en/model_card.md` / `docs/pl/model_card.md`.
- Numbers: E114, E117, E118, E123, E126, E130, E131.
