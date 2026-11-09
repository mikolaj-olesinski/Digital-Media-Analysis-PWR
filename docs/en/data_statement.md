# Data statement

## Data
- `data/raw/reach_samples_students.csv`: 12,500 brand mentions from Brand24 (2,500 each from X, Instagram, YouTube, Facebook, TikTok), 348 brands, created 2025-01 to 2026-10 (about 85% from April to August 2026). The raw file is read-only and not committed.
- Fields: platform, text (title/content), author fields, followers, likes/comments/shares, `view_count`, Brand24's `b24_estimated_reach` and `sentiment`.
- Ground truth for visibility (`view_count`) exists only on TikTok, YouTube and X (75%). Facebook and Instagram have none.

## Collection and sampling
The sample was supplied by Brand24 for the course; how mentions were selected is not documented to us. It is not uniform in time, language (85% English) or brand, so results describe this sample, not all social media.

## Labels we added (AI instrument)
- `data/processed/mention_labels.csv`: mention type and sentiment for all 12,500 mentions, defined in `codebook.md`.
- Produced by three independent Haiku coders; disagreements (37% of mentions) decided by a Sonnet adjudicator. Per-coder labels: `mention_labels_coders.csv`.
- Reliability: pairwise kappa 0.77-0.79 (type) and 0.71-0.73 (sentiment). This is agreement between models, **not** accuracy against a human reference. A team member also reviewed a sample of the labels by hand and found them consistent, but this was informal and not scored, so there is no human-AI agreement number; the labels count as an AI instrument validated only by machine-machine agreement and a manual spot check.
- Brand24's own `sentiment` field is 42% missing; our sentiment labels are separate and complete.

## Known limitations
- Missingness depends on the platform and partly on collection time (see `notebooks/eda/02_platforms.ipynb`). Missing engagement counts go with small posts.
- Brand24's reach is a deterministic rule on X, Facebook and Instagram, so it is a baseline to compare against, not truth.
- Posts are public, but authors are individuals.

## Personal data and GDPR
- `author_name`, `author_url`, `url` and post texts are personal or potentially identifying data. We do not publish them, do not name private authors in the preprint or poster, and report only aggregates.
- Post texts were sent to LLM coders inside this project's tooling (text truncated to 400 characters, no author fields). This use must be disclosed in the preprint.

## Additional data collected by the team (November 2026)

- **Facebook views.** Of the 2,500 Facebook rows, 470 are video posts; 402 of them exposed a public view count (68 removed), and the other 2,030 are ordinary posts/photos with no public view count at all (E200). The 402 public Facebook Reels/videos were read on 2026-11-05 (fetcher:
  `src/fetch_views.py`) from the public
  `facebook.com/watch/?v=<id>` page (logged out, desktop user agent). The page embeds `video_view_count` (the counter we
  use), `reaction_count` and `total_comment_count`. The visible play counter on the page is a different field (`play_count`,
  median x3.6 larger in our sample), so the two are not interchangeable. 68 sampled URLs were gone or restricted.
- **130 hand-checked posts.** A team member opened 130 of these pages and recorded views/plays; the parsed values match
  the page. This is the hand-check sample used in E93.
- **Instagram.** An anonymous attempt returned no view count (0 of 160 checked posts); Instagram requires a login for
  views, so no Instagram view data was collected. Likes/comments for 115 of 160 posts were recorded from public pages.
- **Instagram, logged in (2026-11-08).** 500 random reels from the Brand24 Instagram slice (11 brands) were read by a
  browser agent on a team member's own logged-in account: views, likes and comments at the same moment, read-only, values
  from Instagram's web API as the page loads them. 368 had views (132 gone, 30 with hidden likes). Results are local and
  git-ignored (`data/manual/ig_browser_*.csv`); only statistics are published (E202). The counter's field name was not
  recorded.
- **Instagram `/p/` posts.** 62% of the IG slice (1,550 of 2,500) are `/p/` posts. In a check of the first 26, 19 were
  photos or carousels with no public view counter (`media_type` 1 / 8), 7 were gone, none were videos. Instagram results
  therefore cover reels only; ordinary posts have no ground truth.
- **Terms and personal data.** Only public pages were read, without logging in and without cookies; no author names or
  URLs are published. The fetched files are local and git-ignored (`data/processed/*fetched*.csv`); they carry the mention
  id and the counters, not the post URL. Only aggregates appear in the repository.
- Public brand data from Wikimedia (E86, E170, E171): monthly and daily English Wikipedia pageviews and Wikidata company facts (employees, revenue, founding year) for 293 brand articles. Only brand names / article titles were sent out (user consent 2026-11-05 and 2026-11-06); no author names, URLs or post texts. Files: `data/processed/brand_wikipedia.csv`, `brand_wiki_daily.csv`, `brand_wikidata.csv`.

## Removal requests

If a person asks us to remove a post or an author:

1. Locate it in our local files (`data/processed/fb_fetched.csv`, `fb_all_fetched.csv`, any hand-check sheet) and delete
   or pseudonymise the row; the read-only raw file is the client's and is not edited.
2. Re-run the aggregates that included it, so no published number depends on the removed row.
3. Check that no figure, notebook output or document mentions the author or the URL.
4. Reply with what was removed and which numbers changed.

We cannot delete records inside Brand24; that request goes to the client. A removal never invalidates the method, only
the affected aggregates, and we say so.

## When the data changes

- **A new Brand24 export** gets a dated file name (`..._2027-01.csv`); the old raw file is never overwritten. Re-run the
  labels only if the mention set changed, re-run the numbers that depend on it, update this statement, then `check`.
- **A new Facebook fetch** also gets a dated file; views grow with age, so do not mix counters read at different ages in
  one experiment. Compare the drift before replacing the old file (the November 2026 checks were 1.07-18x the stored
  snapshot, E95).
- **Never overwrite** `data/raw/` or a fetched file in place: keep both versions and say which one a number uses.
- **Same-moment re-read (2026-11-08).** TikTok: **only the first 500 of 2,500** posts were read, with a **logged-in headless browser** (exported cookies) on Xvfb; 426 readable (`playCount`, `diggCount`, `commentCount`, `shareCount`). The remaining 2,000 are not collected yet: a team decision on whether a logged-in session is acceptable under the platform terms is still open.
  YouTube rows also carry `likes_hidden` and `comments_off`: `false` where the counter exists, `unknown` where it is missing; a 10-row check showed the missing likes/comments are a yt-dlp extraction gap, not hidden likes or disabled comments. YouTube: `yt-dlp`, 2,297 of 2,500 (`view_count`, `like_count`,
  `comment_count`); X: FxTwitter, 2,013 of 2,500 (`views`, `likes`, `replies`, `retweets`). Rows carry `status` and
  `source_field`; files are git-ignored, no URLs or author names.
- **Independent check (2026-11-09).** 50 X posts: FxTwitter and the paid API agree on 98% of rows (median ratio 1.000), the export is 1.79x lower (Spearman 0.926). On 20 YouTube posts with `likes = 0` in the export, 10 have positive likes on the page, 6 are true zeros and 4 are unknown, so export zeros are treated as missing (E233).
