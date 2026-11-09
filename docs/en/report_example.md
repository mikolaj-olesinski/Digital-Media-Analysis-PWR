# Example brand report

Illustrative layout, not real posts: the numbers below are made up to show the format, and no author or URL appears.
Units are views; bands are 90% unless marked.

## Brand: example (6 posts, 2 platforms measured + 2 transferred)

| post | platform | read at | reactions | estimate | likely (50%) | 90% band | flag | note |
|---|---|---|---|---|---|---|---|---|
| P1 | TikTok | 2 days | 420 likes, 18 comments | 12k | 4.3k-34k | 0.9k-170k | — | best case, in-platform |
| P2 | YouTube | 3 days | 95 likes, 4 comments | 5k | 0.9k-28k | 0.1k-255k | missing shares | wider band |
| P3 | X | 2 days | 60 likes, 2 comments | 2k | 0.3k-15k | 20-196k | old check, X | least certain |
| P4 | Facebook (Reel) | 4 days | 300 reactions, 40 comments | 8k | — | 80-800k | transfer | FB counter is `video_view_count` |
| P5 | Facebook (page >1M) | 3 days | 1,800 reactions | 1.2M | — | **point only** | large page | band not calibrated |
| P6 | X | 1 hour | 0 reactions | — | — | — | too early | median 5, p90 49 at this age |

## How to read it

- P1-P3 are in-platform estimates. P3's band is wide because X has 25% missing counts and many zero-reaction posts.
- P4 uses the Facebook transfer model; only video/Reel has a view counter.
- P5 is the case where we **do not** give a band: above 1M followers the 90% coverage is 0.80-0.86, so it is a point
  estimate with a caveat.
- P6 is not estimated at all: with zero reactions at one hour, the honest answer is the empirical range, not a point.

## Roll-up (what to send to the client)

- Total estimate: **~1.23M views** (dominated by P5); without P5 the rest is ~27k.
- Report the total with the same caveats as its parts, and say how many posts it contains. Five posts give a x53
  p90/p10 spread, twenty still x6.4 (E112), so a total from six posts is an order-of-magnitude statement.

## Format rules

- One row per post; never merge platforms in one row.
- Show the age of the check; a post read at 1 hour is not comparable with one read at 4 days.
- Mark transferred estimates and uncalibrated bands explicitly.
- No author names, no URLs; post IDs are internal.

## Before sending (six checks)

1. Every row has a point with a band, or says "point only" / "too early".
2. The age of the check is in the row; no row mixes moments.
3. Transferred estimates and uncalibrated bands are marked.
4. The roll-up is the sum of the rows, and the number of posts is stated.
5. No author names, no URLs, no screenshot of a profile.
6. Data-quality flags (missing reactions, old check, X, zero reactions) are attached where they apply.
