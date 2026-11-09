# Hand check of Facebook and Instagram views (B14)

Goal: the first measured error of the transfer to FB / IG. Only videos and Reels show a public view count, so the sample is drawn from them (photos remain an assumption).

- File: `data/manual/fbig_sample.csv` (local, git-ignored because it holds links). 160 Instagram Reels, 152 Facebook Reels and videos, stratified by reactions (`src/fbig_check.py`, seed 42).
- For each row open `url` and fill, **at the same moment**: `checked_at`, `views` (full number), `likes_now`, `comments_now`, `shares_now`, `views_visible` (`yes` / `no`), optional `note`.
- No author names, no screenshots, no scripts that fetch pages. Numbers only.
- 50-100 posts per platform are enough (E15); rows are in random order, so stopping early keeps the sample random.
- Then `fbig_check.evaluate(ROOT, f, meta)` scores the transfer (ridge on likes and comments from TikTok / YouTube / X).

The GDPR and terms decision belongs to the team (B14).
