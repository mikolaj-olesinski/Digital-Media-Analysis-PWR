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

- **Facebook views.** 402 public Facebook Reels/videos from the sample were read on 2026-11-05 (fetcher:
  `src/fetch_views.py`) from the public
  `facebook.com/watch/?v=<id>` page (logged out, desktop user agent). The page embeds `video_view_count` (the counter we
  use), `reaction_count` and `total_comment_count`. The visible play counter on the page is a different field (`play_count`,
  median x3.6 larger in our sample), so the two are not interchangeable. 68 sampled URLs were gone or restricted.
- **130 hand-checked posts.** A team member opened 130 of these pages and recorded views/plays; the parsed values match
  the page. This is the hand-check sample used in E93.
- **Instagram.** An anonymous attempt returned no view count (0 of 160 checked posts); Instagram requires a login for
  views, so no Instagram view data was collected. Likes/comments for 115 of 160 posts were recorded from public pages.
- **Terms and personal data.** Only public pages were read, without logging in and without cookies; no author names or
  URLs are published. The fetched files are local and git-ignored (`data/processed/*fetched*.csv`); they carry the mention
  id and the counters, not the post URL. Only aggregates appear in the repository.

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
