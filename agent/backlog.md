# Backlog

Agents pick the top `open` item they can finish (P0 first), set it to `running` with the date, and close it as `done`
with experiment IDs. Effort and blockers for the open items are in `docs/en/future_work.md`; what happened to the original
plan ideas is in `docs/en/plan_status.md`. People set priorities and unblock. Statuses: `open`, `running`, `blocked` (needs a person or the
client), `done`, `dropped`. Plan letters (A-H) refer to `docs/en/project_plan.md`. `python src/agent_tools.py next` lists the open ones.

| ID | Pri | Status | Item | Done when | Notes |
|---|---|---|---|---|---|
| B01 | P0 | done | Remove personal data from `notebooks/eda/02_platforms.ipynb` outputs (author names, about 66 post URLs; repo is public) | `agent_tools.py check` passes on it; user decided on git history | Done 2026-11-04: `eda_utils.preview` drops author and URL columns and links in text; notebook re-run. User chose not to rewrite git history, so older commits still contain the data |
| B02 | P1 | done | Polish twin for `figures/zoo_summary_en.png` | check passes | Done 2026-11-04: `extra_models.summary_figure_pl` from stored numbers |
| B03 | P1 | done | Quantile models (10/50/90%) and tail weighting for the extremes (plan G) | register row; error per fifth of true views and per reaction tier, plus absolute views | Done 2026-11-04 as E57 |
| B04 | P1 | done | Abstention curve: error against share of posts kept, from the conformal width (plan C) | figure + register row | Done 2026-11-04 as E58 |
| B05 | P1 | done | Forecast horizon curve: error of the E40 model by gap and by age of the first check | register row, figure | Done 2026-11-04 as E59 |
| B06 | P1 | done | Use the history file to normalise the now-cast target to a fixed age (e.g. views at 7 days) and re-test E01-style models | register row | Done 2026-11-04 as E44 (age column: small, X only) and E45 (newer checks: not better) |
| B07 | P1 | done | Early-reaction forecast without views (FB/IG setting): predict later likes/comments from early ones on TikTok/X | register row | Done 2026-11-04 as E60 |
| B08 | P2 | done | Virality classification: will a post reach the top 10% of its platform? (plan F) | precision/recall on held-out brands | Done 2026-11-04 as E48 |
| B09 | P2 | done | Copy-paste campaigns: do near-duplicate posts inflate a brand's estimated reach? (plan F) | register row | Done 2026-11-04 as E63 |
| B10 | P2 | done | Separate model for posts without reactions: text, author, community (plan F) | register row | Done 2026-11-04 as E64 |
| B11 | P2 | done | Sentence embeddings (multilingual, local model) as text features | register row | Done 2026-11-04 as E50 (X +0.011 with reactions; YouTube +0.051 without) |
| B12 | P2 | done | Temporal validation (train on earlier months, test on later) for the final models | register row | Done 2026-11-04 as E46 (expanding window close to brand CV; worse only in the first month) |
| B13 | P3 | done | Full GNN (GraphSAGE) on the text graph | register row | Done 2026-11-04 as E65 (graph hurts; CatBoost + MLP mean +0.016) |
| B14 | P0 | blocked | Own FB/IG sample with public view counts (plan A) | data statement updated, sample collected | Needs GDPR/terms decision and people. FB 2026-11-05: 130/152 auto-filled (views, reactions, comments; shares not public), 22 unavailable logged-out. IG scan 2026-11-06: 160 posts, 0 views public, 115 with likes + comments filled, 45 need a login; views need cookies/paid/Brand24 |
| B15 | P1 | blocked | False brand match detector (plan D) | 100-200 hand labels, detector, share of reach | Needs hand labelling by people |
| B16 | P1 | blocked | Ask Brand24: YouTube zeros, rule confirmation, history timezone | answers recorded in `agent/questions_for_brand24.md` | People send the questions |
| B17 | P1 | done | Tabular foundation models: TabPFN v2 (Apache-2.0 regressor weights, attribution needed) and TabICL (BSD) vs and averaged with the final model | register row, card; image with CPU torch | Done 2026-11-04 as E47 (small: +0.010 on YouTube, ≤ 0.005 elsewhere). Pilot was: YouTube +0.014 (CI 0.007-0.020), TikTok +0.005, X +0.002; 20-40 s per platform. Do not use TabPFN 2.5+ weights (non-commercial). Check whether the YouTube gain sits in zero-reaction posts (E27) |
| B18 | P1 | done | Offset target: predict log10(views) - log10(likes+1), add back | register row; error per fifth of true views | Done 2026-11-04 as E51 (small, YouTube +0.007; extremes barely move) |
| B19 | P1 | done | Leakage audit: group folds by brand + author + duplicate cluster | register row | Done 2026-11-04 as E55 |
| B20 | P2 | done | Noise ceiling from main vs current re-checks of the same post | register row | Done 2026-11-04 as E56 |
| B21 | P2 | done | Pooled 3-platform model with platform feature and monotone constraints | register row; in-platform and leave-platform-out | Done 2026-11-04 as E61 |
| B22 | P2 | done | YouTube views per like fell from 1.57 to 1.13 (log10, May-Aug): explained by younger checks? | register row | Done 2026-11-04 as E62 |
| B23 | P2 | done | Posting-time model (E43) with e5 embeddings and author/hashtag context instead of TF-IDF | register row | Done 2026-11-04 as E52 (YouTube +0.020 from context) |
| B24 | P1 | blocked | Re-fetch public view counts programmatically (X, YouTube, TikTok, Facebook; Instagram needs a login) | team terms/GDPR decision, then full run and data statement update | 2026-11-05: FB hand sample 130/152; 2026-11-06: all 470 video-like FB URLs fetched, 402 with views / 68 gone (E95, `fb_all_fetched.csv`, `fb_all_numbers.json`); IG 0/160 anonymous. Pilot X 7/8, YouTube 7/8, TikTok 6/8. Views today 1.07-18x the stored snapshot. FB comments in Brand24 are near-empty (6/152 posts non-zero), the pages show real ones. Code `src/fetch_views.py`. Not committed |
| B25 | P2 | open | Monthly brand totals with intervals (the client reports monthly; we only have all-time totals) | register row + numbers | Data available; method from E109/E112/E157 |
| B26 | P2 | open | Multi-language check: does the model hold outside English (85% of the sample)? | register row + numbers | E84 tested country, not language subgroups |
| B27 | P1 | blocked | Author history at scale: several posts per author from Brand24 (high value, see future_work) | register row | Needs client export (Q8); E68/E119 show +0.10 MAE on seen authors |
