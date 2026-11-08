# Model card: visibility (views) estimator

Status: internal, 2026-11-07. This is a description of the estimator and its evidence, not preprint text.
Reproduce it: `docs/en/reproduce.md` (six commands).

## What it estimates

- **Target:** `log10(view_count)` of a social post, i.e. the number of views the platform reports, at the age at which
  the metrics were read.
- **Unit of prediction:** one mention (one post).
- **Scope:** TikTok, YouTube and X (views are public). Facebook Reels/videos: validated on 130 hand-checked and 402
  fetched posts (E93, E95). Instagram: **not validated** (no public views without a login).
- **Not the same as reach:** views are not unique people. Platforms do not publish unique reach for arbitrary posts;
  views are the closest public proxy.

## Inputs

- `log10(followers)`, `log1p(likes)`, `log1p(comments)`, `log1p(shares)` and a missing flag for each (`ENG`).
- For X only: `log10(age in days)` at the check (`log_age`), which narrows the band and lowers MAE (E44, E141).
- Optional, only when the author's earlier views are known: `author_te`, the shrunk mean log views of the author
  (E119).
- Not used: text, labels, graph, country, time, calendar, followers of the brand. Each was tested and adds <= 0.01
  (E100, E121, E122; earlier E05, E49, E84).

## Model

- CatBoost regressor, `loss_function=MAE`, 300 iterations, library defaults otherwise (E114, E141).
- Training: Brand24 sample, 12,500 mentions, posts with `view_count > 0`; 2,491 TikTok, 2,441 YouTube, 1,803 X.
- Features built from views (author history) use the training fold only.

## Validation

- **Primary:** 5-fold CV grouped by brand, so a brand is never in train and test. MAE on log10, brand-cluster bootstrap
  CI (300-1000 draws).
- **Repeated CV (E114):** TikTok 0.269 ± 0.011, YouTube 0.466 ± 0.030, X 0.463 ± 0.018.
- **Temporal:** train on earlier months, test on later ones; close to brand CV, worse only in the first month (E46).
- **Stricter grouping:** folds that also separate authors and copied texts change MAE by <= 0.002 on TikTok/YouTube and
  -0.011 on X (E55).
- **Noise ceiling:** posts with almost the same followers and reactions differ by x2.8 / x6.0 / x4.6 in views; the best
  possible MAE there is at least 0.22 / 0.39 / 0.33 (E56). The model is close to the ceiling on TikTok, about
  0.06-0.14 above it on YouTube and X.
- **Calibration:** the point estimate is calibrated in the middle; isotonic recalibration and per-tier shifts do not
  improve it (E114).
- **Rolling-origin (E136):** train before each of the last months, test on the month: 0.270 ± 0.004 / 0.450 ± 0.018 /
  0.473 ± 0.005, close to brand CV. The validation is not optimistic in time.

## Performance

| platform | MAE log10 | typical factor | 90% CQR width | source of numbers |
|---|---|---|---|---|
| TikTok | 0.273 | x1.9 | 1.14 | E114, E118 |
| YouTube | 0.456 | x2.9 | 1.71 | E114, E118 |
| X | **0.454** | x2.9 | **1.99** | E114, E118, E141 |
| Facebook (transfer, 130) | 0.345 | x2.2 | — | E93 |
| Facebook (transfer, 402) | 0.385 | x2.4 | 1.99 | E95, E123 |

- **Intervals:** use CQR. A 50% band is also calibrated in-platform (0.45 / 0.74 / 0.83 wide, E137/E141). Raw 5-95% quantiles cover only
  74%; CQR reaches 0.889 / 0.902 / 0.899 at nominal 90% (E118, X is v3 E141).
  On FB, CQR covers 0.925 at width 1.99 (E123). Without target labels, the source-width band is the label-free option
  (E98). At posting time, before reactions, the 90% band is 2.4-3.0 log10 wide (E130); with reactions it is 1.1-2.1.
  A distributional CatBoost under-covers (0.86-0.88) and is not used (E129).
- **Calibration budget:** X needs 25-50 target posts (MAE 0.650 -> 0.551-0.563); TikTok, YouTube and Facebook are
  already calibrated and a shift hurts at small n (E115). The client does not have to collect posts.

## Where it fails

- **Tail:** the model over-estimates small posts and under-estimates virals (E96, E114); recalibration does not fix it
  (E114).
- **Missing reactions at prediction:** 10% missing costs 0.03-0.08 MAE, 50% costs 0.08-0.37; X is the most robust and
  TikTok the most fragile (E138). If a feature is systematically missing, **retrain without it**: the specialised model
  beats the full one (E163: TikTok without reactions 1.139 -> 0.583; YouTube without followers 0.711 -> 0.494).
  Switch at ~35% missing reactions (TikTok/YouTube) or ~20% missing followers (YouTube/X); on X the reactions switch is
  later (~70%) (E164).
- **Which data matters per platform:** TikTok needs reactions (missing them: 1.005) but tolerates missing followers
  (0.294); YouTube and X need followers (0.711 / 0.733 without them) more than reactions (E161, E138). On FB reactions
  are the signal; followers help only above 1M (E162).
- **Zero reactions:** 27% of YouTube posts, 64% of X posts; error 0.66 on those posts; treating zeros as missing gains
  only 0.003 (E116). A 0/0 first check still gets views: 86% > 0, median 5 (X) / 60 (TikTok), p90 49 / 179, 0.4% > 1k
  (E117, E126); at collection the p90 is 124 (X) and 3 383 (YouTube). Report a range, not a point.
- **Author level:** about two thirds of the remaining error is the author (E70); the author's past views are the
  largest lever (E68, E119), but only 1-8% of the sample has them. Without them the model is the base model: a brand prior
  does not substitute for author history (E169: -0.005 / -0.035 / +0.006, only 35-52% of test posts have a known brand).
- **Age of the check:** views keep growing; the FB bias +0.14 is removed by normalising to 30 days (E101).
- **FB/IG:** only video/Reel has a view count; ordinary posts have none. Facebook's page shows plays, not the
  `video_view_count` we use (the two differ by a median x3.6). The FB band is not uniform: pages below 1M followers are
  covered 0.940, while pages above 1M only 0.80-0.86 and cannot be calibrated with 70 posts (E132, E144, E148, E151);
  the top view quartile covers 0.78 at the narrowest width (1.49, E132); do not report a
  calibrated band for >1M, say the model under-estimates the mean by about x2. A large-page model improves the point
  (0.368) but the band stays at 0.80 (E150/E151).
- **Sample:** 12,500 mentions supplied by Brand24, not a random sample of social media; 85% English; time not uniform.

- **Outside the export:** on brand-owned/popular videos the model is x5 low (E189/E190, MAE 1.48 on popular), but on
  third-party mention videos - the population it targets - it is close to the export: MAE 0.690 low, 0.441 mid, and only
  the popular stratum fails (1.569) (E191). Zero-view videos are not reachable through brand channels, but exist among
  obscure third-party mentions (E190, E191).

## Intended use

- Estimate visibility of a post for monitoring, benchmarks and brand totals, always with an interval.
- Brand totals need many posts (20 posts still give a x6.4 p90/p10 spread, E112) and a calibration (E109). FB brand
  totals from 19 brands are x2 low before calibration and their 80% interval covers 0.737 (E157/E158).
- **Not for alerts:** at the top 1% a ranking by raw likes beats this model (E104, E106); use engagement or a dedicated
  precision@k model.
- **Not for billing or guarantees.**

## Data protection and terms

- No author names or URLs in outputs; aggregates only.
- The raw Brand24 file is local and git-ignored. The Facebook views were read from public pages (data statement to be
  updated); Instagram requires a login and was not collected.

## Files

- Code: `src/modeling.py`, `src/extra_models.py`, `src/reaction_tiers.py`, `src/validation.py`, `src/validation_audit.py`,
  `src/cqr_intervals.py`, `src/author_te.py`, `src/fb_transfer_check.py`.
- Numbers: `data/processed/*_numbers.json` (per experiment).
- Evidence: `docs/en/experiments.md`, `agent/findings.md`, `agent/vault/`.

## How to read the numbers

- Everything is on **log10 views**. An MAE of 0.3 is a typical error of x2, 0.5 is x3, 1.0 is x10
  (factor = 10^MAE).
- A band width is also in log10: a width of 1.14 is a factor of x14, 1.99 is x98 (factor = 10^width).
- The 50% band is the "likely" range; the 90% band is the honest worst case. They are not the same claim.
- Bias is mean(true - estimate) in log10: +0.14 means the estimate is low by x1.4.
- "Coverage" is the share of posts whose true value falls inside the band; nominal is 90% for the 90% band.


## Platform signature (reverse-engineered, E167)

| platform | follower elasticity | reaction elasticity | posts with more views than followers |
|---|---|---|---|
| TikTok | 0.044 | 0.82 | 43% |
| YouTube | 0.303 | 0.66 | 5% |
| X | 0.164 | 1.06 | 14% |

Elasticities from a per-platform fit of log10(views) on log10(followers) and log10(1+reactions), brand-grouped OOF R2
0.81 / 0.59 / 0.47. TikTok is feed-driven and YouTube subscriber-driven; a follower-based rule is therefore structurally
wrong on TikTok. E168 shows the same from the other side: Brand24's reach is fully recoverable as a rule only on
Instagram (97% of rows), partially on X (62%) and Facebook (49%), and not at all on TikTok/YouTube, where it is the view
count.

## Figures

The result figures are `figures/cqr_bands_*`, `fb_coverage_*`, `reach_concentration_*`, `calibration_curve_*` and
`fb_rule_vs_model_*`, all built by `src/result_figures.py` from the JSON files; which figures are live and which are
historical is in `scripts.md`.
