# Evidence map: what we know, how strongly, and what is unknown

State: 2026-11-07. Internal planning table for the preprint team. Not preprint text. Strength: **solid** = replicated under several
validations; **likely** = one validation, consistent with the rest; **open** = not measured.

| # | Claim | Strength | Evidence |
|---|---|---|---|
| 1 | On TikTok and YouTube the Brand24 reach equals the view count (100% / 99% of rows) | solid | EDA, E01 |
| 2 | Facebook and Instagram have no view count in the sample; only FB video/Reel has a public counter | solid | E93, E95 |
| 3 | The constant/step rule is far off on X (median x9.3, 87% off >2x) and FB (x4.1, 82% off >2x) | solid | E99, E111 |
| 4 | The model beats the rule on MAE: 0.27 / 0.46 / 0.45 (TT/YT/X) vs about 0.93-1.15; FB transfer 0.385 vs 0.93 | solid | E114, E93, E95, E111, E141 |
| 5 | The model is close to the noise ceiling on TikTok and 0.06-0.14 above it on YouTube/X | solid | E56 |
| 6 | CQR gives honest 90% intervals in-platform (0.889 / 0.902 / 0.899 with X v3); raw quantiles cover only 74%; the FB >1M tier is the exception (0.80-0.86, E145/E151) | solid | E118, E141, E147 |
| 7 | Calibration on target posts is only needed for X (25-50 posts, 0.650 -> 0.551-0.563); TT/YT/FB are fine zero-shot | likely | E115 |
| 8 | Author history is the largest lever when it exists (X 0.530 -> 0.430 on seen authors) but covers 1-8% of posts | likely | E68, E119 |
| 9 | Text, labels, time, country, recency columns and graph methods add <= 0.01; graph embeddings hurt | solid | E100, E49, E84, E121, E122, E127, E128 |
| 10 | Reach is tail-dominated: the top 1% of posts hold 60-74% of views; 2-4% of posts give 80% | solid | E96 |
| 11 | At the top 1% a ranking by raw likes (or the rule on TT/YT) beats the MAE model | likely | E104, E106 |
| 12 | Brand totals are unstable on few posts (5 posts: x53 spread; 20: x6.4) and need intervals + calibration | likely | E109, E112 |
| 13 | 25% of X views are missing and not at random; averages from observed posts are biased about 1.3x | likely | E105 |
| 14 | Zero reactions is not zero visibility; a 0/0 first check has median 5 views (X) / 60 (TikTok) | solid | E117, E126 |
| 15 | At posting time MAE is 0.60-0.76 and the 90% band is x260-900 wide; reactions shrink it to x14-117 | likely | E130, E118 |
| 16 | The FB band under-covers the largest pages (top view quartile 0.782, >1M followers 0.857) | likely | E132 |
| 17 | FB brand totals from 19 brands are x2 low before calibration; after it 47% are still off >2x and the 80% interval covers 0.74-0.79 | likely | E157, E158, E159 |
| 18 | When a feature is systematically missing, retrain without it (switch at ~35% missing reactions on TikTok/YouTube, ~70% on X; ~20% missing followers on YouTube/X) | likely | E163, E164 |
| 19 | Data needs differ by platform: TikTok reactions, YouTube/X followers, Facebook reactions (followers only above 1M) | likely | E161, E162 |
| 20 | The visible distribution differs by platform: follower elasticity 0.044 / 0.303 / 0.164 (TT/YT/X), reaction elasticity 0.82 / 0.66 / 1.06; 43% / 5% / 14% of posts beat the follower count | likely | E167 |
| 21 | A simple reverse-engineered Brand24 rule reproduces 97% of IG, 62% of X and 49% of FB rows; on TikTok/YouTube reach is the view count | likely | E168 |
| 22 | Brand history does not substitute for a missing author history: temporal split, brand prior -0.005 (TikTok), -0.035 (YouTube), +0.006 (X, CI includes 0); 35-52% of test posts have a known brand | likely | E169 |
| 23 | The model is calibrated to the Brand24 mention population: on 53 independently fetched YouTube brand videos the relation holds (Spearman 0.93, slope 0.83) but the level is x5 low (MAE 0.726, bias +0.71; 0.540 after removing the shift) | likely | E189 |
| 23 | The noise ceiling is not a check-age artefact: with age matched the bound is unchanged; a 30-day target changes MAE <= 0.007 | likely | E179 |
| 24 | Zero-view videos are not reachable through brand channels or search (0/131, minimum 71 views), but third-party mentions do include them (1 zero, 17 below 100); the model on brand videos fails above small posts (MAE 0.289 small vs 1.480 popular) | likely | E190 |
| 24 | The FB transfer rests on 20 brands; without the 3 largest it is 0.46; non-English FB posts are ~x2.2 underestimated; on Brand24's own FB counts it is 0.83 | likely | E177, E178, E182 |
| 25 | On third-party mention videos (203 videos, 60 brands) the model is close to the export (MAE 0.536 low, 0.555 mid) and fails only on the popular stratum (1.044); the earlier x5 shift came from brand-owned videos | likely | E191, E193, E195, E197 |
| 25 | On third-party mention videos (203 videos, 60 brands) the model is close to the export (MAE 0.536 low, 0.555 mid) and fails only on the popular stratum (1.044); the earlier x5 shift came from brand-owned videos | likely | E191, E193, E195, E197 |
| 26 | TikTok export views are stale but rank-stable: 40/60 readable, median new/old 1.068, p90 3.07, Spearman 0.918 | solid | E192 |
| 26 | An early view read is the alerting signal: top-5% precision 0.82 / 0.86 / 0.71 vs 0.71 / 0.64 / 0.35 from reactions | likely | E185 |
| 27 | YouTube export views are stale: median new/old 1.516, p90 25.3, 19% within 5%, Spearman 0.857; TikTok 1.068 and 0.918 | solid | E194, E192 |
| 27 | Routing by available fields keeps 90% bands at 0.88-0.90; the full model with missing reactions covers 0.46-0.83 | likely | E176 |
| 28 | Facebook public view counts are static over three days (60/60 identical), while TikTok grew 7% and YouTube 52%; FB is a single-time snapshot | solid | E196 |

## Unknown / open

- Instagram visibility: no public views without a login; not measured (blocked, no consent for cookies).
- Facebook ordinary posts (not video/Reel): no view count; cannot be validated.
- Unique reach (people, not views): not observable from public data.
- Brand24's exact constants: reconstructed, not confirmed by the client.
- False brand matches (a post attributed to the wrong brand): needs 100-200 hand labels; not done.
- Causal effect of any feature on views: out of scope.
- Representativeness of the 402 fetched FB posts: stratified by reactions, not random.

## How to read the strength

- **solid:** two or more independent validations (brand CV + temporal + stricter grouping) or a deterministic fact.
- **likely:** one validation with a bootstrap CI, consistent with the rest of the evidence.
- **open:** no measurement; list it as a limitation, do not guess.

Every number above is in `data/processed/*_numbers.json` and in the experiment register (`docs/en/experiments.md`).

Numbers are on log10 views; how to read MAE, band widths and bias: `model_card.md`, section "How to read the numbers".
