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
