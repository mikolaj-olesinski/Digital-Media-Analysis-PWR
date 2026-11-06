# What could still be measured

Internal list, ordered by value per effort. Nothing here is done unless it appears in the register. Which plan items are
closed and which are open: `plan_status.md`. The open items are tracked as backlog entries B25-B27.

## High value, blocked on access or a decision

| item | why | effort | blocker |
|---|---|---|---|
| **Instagram views** | the only platform with no measurement at all; 0/160 anonymous | 1-2 days (manual sample of 100-150 posts) or a provider | decision on login/provider and terms |
| **Author history at scale** | the largest known lever (+0.10 MAE on seen authors, E68/E119), but only 1-8% of our posts have it | 0.5 day if Brand24 exports several posts per author | data from the client (Q8) |
| **False brand matches** | wrong brand attribution corrupts brand totals, not single posts | 2-3 days for 100-200 hand labels | human labelling capacity |

## Medium value, data already available

| item | why | effort |
|---|---|---|
| **Monthly brand totals with intervals** | the client reports monthly; we only have all-time totals (E109/E112/E157) | 1 day |
| **Multi-language check** | 85% of the sample is English; the model may not hold for other languages | 0.5 day (subset analysis) |
| **Longer temporal horizon** | the rolling-origin check covers 2-3 months (E136); a 2027 refresh would test drift | 0.5 day after new data |

## Low value or blocked by the data

| item | why not now |
|---|---|
| Facebook ordinary posts (not video/Reel) | no view count at all; cannot be validated |
| Unique reach (people) | not public on any platform |
| Real interaction graph (shares, replies) | Brand24 does not have the edges; without them the graph line stays closed (E127/E128) |
| More model families | the noise ceiling (E56) leaves 0.06-0.14 on YouTube/X and less on TikTok; text/graph/columns already tested (E100/E121/E122) |
| Client calibration data | E115 shows what it would buy; the client has said it will not collect posts |

## What we would do first with one more week

1. Instagram: a manual sample of 100-150 Reels with views (the only unmeasured platform).
2. Monthly brand totals with intervals, since the client reports monthly.
3. A second hand sample for false brand matches only if a person is available.

## After the course

- **Publish the Facebook measurement.** 402 public posts with views vs the rule is the most novel piece; it needs a
  data-release and terms check before any public artifact.
- **Author history with Brand24.** If the client exports several posts per author, the largest known gain (+0.10 MAE on
  seen authors) becomes available at scale; it is the natural next paper.
- **Instagram with a provider.** The only unmeasured platform; a paid or logged-in source would close the biggest hole.
- **Monthly totals and a longer horizon.** Both are cheap once data arrives, and monthly is what the client reports.
- **A real interaction graph.** Shares/replies would test whether the graph line reopens; without them it stays closed
  (E127/E128).
- **Calibration under missingness.** A method paper could formalize the "retrain without the feature" switch (E163/E164)
  and the CQR fallbacks (E144/E145).
