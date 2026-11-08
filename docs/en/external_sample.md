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
