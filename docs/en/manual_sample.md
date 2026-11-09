# Manual sample for Instagram and TikTok

Neither platform exposes view counts to an anonymous fetcher (Instagram 0/160, B14; TikTok's `yt-dlp` extractor is
marked broken). Until an API is available, the cheapest compliant route is a **manual sample**: a person opens public
posts and records the counters. This page is the protocol and the data contract; the scoring tool is
`src/manual_sample.py`.

## Sampling frame

- **Brands:** the same 12 brands as the external YouTube sample (top by mention count), plus 8 more if time allows.
- **Posts per brand:** 5 - mix sizes on purpose: 2 newest, 2 from the middle of the profile, 1 most viewed.
- **Format:** public Reels / videos (they show a view count); for TikTok any public video (views are visible).
- **Period:** posted 2025-01-01 or later; note the date of the check.
- **What NOT to record:** author name, handle, URL, screenshots with handles. The sheet stores only the brand, platform,
  counters and dates.

## Data contract (CSV)

| column | meaning |
|---|---|
| `platform` | `instagram` or `tiktok` |
| `brand` | brand name |
| `view_count` | views/plays shown on the post |
| `likes`, `comments`, `shares` | counters; empty if hidden |
| `author_followers` | followers of the account, if visible |
| `posted_at` | YYYY-MM-DD (approximate is fine) |
| `checked_at` | YYYY-MM-DD of the reading |
| `notes` | free text without personal data |

## Effort and analysis

- About **3 minutes per post**; 100 posts = ~5 hours, split across the team.
- `python src/manual_sample.py data/processed/manual_ig.csv` trains the platform model when it exists (TikTok) and reports
  MAE/bias; for Instagram there is no in-platform model, so it reports descriptives (median views, views per follower,
  rank correlation with reactions) - an honest "what we can say without a model".
- The result gets a card and register row like any experiment; the sheet is git-ignored (it contains public counters but
  also brand-specific detail, keep it local).

## Ethics

Only public counters; no author identities; no automated access; a team member does the clicking. This is the same
standard as the Facebook hand check (E93).
