# TikTok data: Research API or manual sample

TikTok's public pages cannot be fetched by `yt-dlp` (the extractor is marked broken), and the search endpoint needs a
logged-in session. Two compliant routes remain.

## 1. TikTok Research API (recommended)

- Free for **academic researchers**; requires an application with an institutional email, a research description and a
  redirect URI.
- Gives public video data: `view_count`, `like_count`, `comment_count`, `share_count`, author followers, `create_time`,
  hashtags - exactly the fields the model needs.
- Approval takes **weeks**; plan it in parallel with the project and submit early.
- Docs: https://developers.tiktok.com/doc/research-api-specs-query-videos

What to submit: a short description of the visibility-estimation project, the sample size (a few thousand videos across
the 12 brands), the fields needed, and a statement that only aggregate results are published.

## 2. Manual sample (fallback, usable now)

- Protocol and data contract: `docs/en/manual_sample.md`.
- 100 public TikTok videos (views visible), ~5 hours of clicking, no login and no scraping.
- Scoring: `python src/manual_sample.py data/processed/manual_tiktok.csv` (trains the TikTok model and reports MAE).

## What not to do

No scraping, no logged-in automation, no third-party "TikTok API" resellers: they break the platform terms and GDPR
hygiene, and the numbers cannot be defended in the report.
