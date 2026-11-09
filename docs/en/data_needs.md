# Data needs: value of information, data contract, monitoring

Internal, 2026-11-06. What Brand24 should collect, in which form, and how to keep the estimator honest after launch.
Not preprint text. Every number links to an experiment in `experiments.md`.

## 1. Value of information: what each field is worth

MAE on log10 views; lower is better. "Cost" is the effort for Brand24, not for the team.

| field or action | without | with | gain | who has it | cost | evidence |
|---|---|---|---|---|---|---|
| Fresh FB reactions + comments, read with the views | 0.831 (Brand24 counts) | 0.385 | **0.45** | Brand24, by re-reading the page | low: one extra read | E177 |
| Reactions at all (TikTok) | 0.584 | 0.273 | **0.31** | already collected | none | E176 |
| Reactions at all (YouTube / X) | 0.714 / 0.576 | 0.460 / 0.470 | 0.25 / 0.11 | already collected | none | E176 |
| A second read of the view count (first check -> later) | 0.273 / 0.456 / 0.467 | 0.165 / 0.188 / 0.269 | 0.11-0.27 | Brand24, scheduler | medium: a re-check queue | E40 |
| Author's earlier views (seen authors only, 1-8% of posts) | X 0.530 | X 0.430 | 0.10 on those posts | Brand24 archive | low if kept | E119, E77 |
| 25-50 client posts with views (X) | 0.650 | 0.551-0.563 | ~0.09 | the client | medium | E115 |
| Followers (YouTube / X) | 0.500 / 0.515 | 0.460 / 0.470 | 0.04 | already collected | none | E176 |
| Followers (TikTok) | 0.276 | 0.273 | 0.003 | already collected | none | E176 |
| Text, labels, country, time, brand, Wikipedia, Wikidata, graph | — | — | <= 0.01 | — | — | E49, E84, E121, E170-E175 |

Order of priority for Brand24: (1) read FB reactions and comments together with the views; (2) keep reactions complete on
every platform; (3) schedule a second view read; (4) keep authors' past views; (5) offer client calibration on X.

## 2. Data contract for the estimator

| field | required | form | why |
|---|---|---|---|
| platform, mention id, created_at (UTC) | yes | | routing, age |
| likes, comments, shares | yes, or explicitly missing | all reactions (FB: total reactions, not only "Like") | the main signal; a silent 0 for "not collected" breaks the bands (E176: coverage 0.46-0.83) |
| read time of reactions | yes | UTC, same request as the views when views exist | stale reactions double the FB error (E177) |
| author followers | yes, or explicitly missing | at read time | YouTube / X signal; FB only above 1M |
| view count (when public) | for calibration and monitoring | with its read time and counter name (FB: `video_view_count`, not plays) | FB plays and views differ x2.3-3.6 (E95) |
| missing flag per field | yes | null, never 0 | the router picks the model by the available fields (E176) |

Rules: a missing value is null, not zero; every count carries the time it was read; one counter definition per platform.

## 3. Monitoring after launch

| check | how | trigger | action |
|---|---|---|---|
| Level drift | each month, median (true - predicted) on posts whose views are public (TikTok, YouTube, X, FB video) | |median| > 0.10 log10 two months in a row | refit the shift on the last month (E89/E115) |
| Band coverage | share of public-view posts inside the 90% band, per platform and per field pattern | < 0.85 on >= 200 posts | recalibrate CQR on the last two months (E142) |
| Field completeness | share of posts per field pattern (full / no reactions / no followers / nothing) | a pattern share moves > 10 points | check the collector; the router keeps bands honest meanwhile (E176) |
| Counter definition | ratio of plays to views on FB, views per reaction per platform | a step change > x1.5 between weeks | platform changed the counter; freeze, re-measure, refit (E95) |
| Brand totals | per client brand, interval width vs number of posts | brand with < 20 posts | show the total with its interval and the "few posts" flag (E112) |

Retraining: the model held for 2-3 months without drift (E46, E136); refit quarterly or after any trigger above.
