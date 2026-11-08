# External sample for YouTube, X and TikTok

Goal: a sample of posts fetched **independently of the Brand24 export**, so the model can be checked outside the data it
was trained on. Same brands, same period, same kind of content (brand-related posts), but the posts are chosen by us, not
by the client's export.

## Feasibility (tested 2026-11-08, `yt-dlp 2026.10.26`)

| platform | method | status |
|---|---|---|
| **YouTube** | `yt-dlp "ytsearchN:<brand>"`, full metadata (views, likes, comments, channel followers, date) | **works** |
| **TikTok** | `yt-dlp` search / hashtag pages | **no**: yt-dlp marks TikTok as broken ("No working app info"); a login or the Research API is needed |
| **X** | `yt-dlp` on a post URL | **no**: x.com answers HTTP 520/blocked; search needs the paid API or a login |
| **Instagram** | anonymous fetch | **no**: 0/160 posts expose views (B14); needs a login or a provider |

So the only external sample we can collect today without credentials is **YouTube**. TikTok and X need an API key, a
logged-in session or the team's decision to collect manually; the plan for them is below.

## YouTube design (implemented, E189)

- **Frame:** the 12 brands with the most mentions in the export (a mix of consumer and B2B: Adidas, The North Face, eToro,
  Mollie, PayPal, Motorola, Hootsuite, Stripe, Avanza, Coinbase, MailerLite, Monster Energy).
- **Query:** the brand name via YouTube search; take the most-viewed 5 videos per brand that pass the filters.
- **Filters:** upload date >= 2025-01-01 (the Brand24 period), duration <= 15 min, views > 0.
- **Fields:** views, likes, comments, channel followers, upload date, duration, channel id (never the channel name),
  video id. The fetched CSV stays local (`data/processed/external_youtube_fetched.csv`, git-ignored).
- **Summary:** `data/processed/external_youtube_numbers.json`; card E189.

### Known biases (report them with any result)

- Search results are not a random sample of mentions: brand-owned channels and reviews are over-represented, third-party
  posts with few views are under-represented.
- The sample is not stratified by reactions, so it cannot be compared one-to-one with the Brand24 YouTube sample.
- Small n (about 60 videos); use it for a sanity check of the views-reactions relation and of the model's error, not for a
  precise estimate.

## Plan for X and TikTok (needs a decision)

1. **X:** the clean route is the official API (paid, `recent search` with `public_metrics`); the free route is a logged-in
   session, which violates the platform terms for automated collection. Recommendation: ask Brand24 whether they can share
   a small export of public post metrics (they already collect them), or budget the API.
2. **TikTok:** the Research API (free for academics, approval takes weeks) or a provider; yt-dlp cannot search without a
   login. Recommendation: apply for the Research API in parallel with the project; do not scrape.
3. **Instagram:** unchanged (B14): a manual sample, a login for a human, or a provider.

## What this buys us

If the views-reactions relation holds on the external YouTube sample, the transfer claim is not an artefact of the
Brand24 export. If it fails, we learn that the export's YouTube rows are special - both outcomes are results.

## Diverse YouTube strata (E190)

A plain search returns only popular videos. The diverse collector therefore draws four strata per brand: most viewed
from search, most recent from search, least viewed from the brand channel, newest from the brand channel. The last two
include small and zero-view videos, which are missing from search. Results: `data/processed/external_youtube_diverse_numbers.json`,
card E190.

## Cost of the X API (checked 2026-11-08)

X no longer sells monthly tiers: Basic ($200/month) and Pro ($5,000/month) were retired in February 2026, and there is no
free read tier. The public API is pay-per-use: **$0.005 per post read**, $0.010 per user read, $0.015 per post written
($0.200 when the post contains a URL), capped at 3 million post reads per month before an Enterprise contract. For our
external sample of 2,500 X posts with author data that is about **$12.50 for the posts + $25 for the authors = $37.50**
(one-off). Sources: X API pricing page (docs.x.com/x-api/getting-started/pricing) and a 2026 rate summary
(twitterapis.com/answers/x-api-pricing-tiers-explained).


### Result of the diverse strata (E190)

131 videos, 12 brands: 0 zero-view videos, minimum 71 views, one below 100. Median views by stratum: popular 1,544,176,
recent 15,661, newest 3,943, small 876. Model MAE: small 0.289 (bias +0.17), recent 0.620, newest 0.601, popular 1.480
(bias +1.48, ~x30 low). Zero-view public videos cannot be fetched at all.


## Third-party mentions (E191) - the population that matters

Brand search and brand channels give brand-owned videos; the Brand24 mentions are third-party posts. The mention
collector therefore searches review/unboxing/haul/honest-review queries, removes the brand's own channel and samples top,
mid and low views. Result (expanded to 60 brands in E197): **203 videos, 60 brands, median 437 views**, **10 zero-view and 83 below 100** -
so zeros exist, just not through brand channels. Model MAE: **0.536 low, 0.555 mid, 1.044 top**; overall **0.606**, bias
+0.28 (189 validation rows).

## X on a 10 PLN budget

Rates (pay-per-use): $0.005 per post read, $0.010 per user (author) read; resources are deduplicated within a 24 h window.
10 PLN is about $2.5-2.7, so:

| what | cost | how many |
|---|---|---|
| posts + authors | $0.015 per post (less with author dedup) | **~165-250 posts** |
| posts only (no followers) | $0.005 per post | ~500 posts |
| first purchase bonus | +$20 free credits after saving a card and the first purchase (expires in 3 months) | up to **~1,500 posts with authors** |

Limit that matters more than money: the pay-per-use **recent search covers only the last 7 days**; older posts need the
full-archive endpoint, which is Enterprise-only. So a cheap X sample is a **7-day window**, not 2025-2026 history:
either repeat a weekly sample going forward, or accept a snapshot of current mentions.


## TikTok: current views vs the export (E192)

60 random TikTok posts were re-read with headless Chromium: 40 were readable, 20 hit a login wall/captcha. Median
new/old views 1.068, p90 3.07, 45% within 5%, but rank correlation 0.918. The export is stale in level and stable in
order; comparisons need a common check date or an age correction.
