# Backlog

Agents pick the top `open` item they can finish (P0 first), set it to `running` with the date, and close it as `done`
with experiment IDs. People set priorities and unblock. Statuses: `open`, `running`, `blocked` (needs a person or the
client), `done`, `dropped`. Plan letters (A-H) refer to `docs/en/project_plan.md`. `python src/agent_tools.py next` lists the open ones.

| ID | Pri | Status | Item | Done when | Notes |
|---|---|---|---|---|---|
| B01 | P0 | done | Remove personal data from `notebooks/eda/02_platforms.ipynb` outputs (author names, about 66 post URLs; repo is public) | `agent_tools.py check` passes on it; user decided on git history | Done 2026-11-04: `eda_utils.preview` drops author and URL columns and links in text; notebook re-run. User chose not to rewrite git history, so older commits still contain the data |
| B02 | P1 | open | Polish twin for `figures/zoo_summary_en.png` | check passes | Built by `extra_models.summary_figure`; add a `lang` argument |
| B03 | P1 | open | Quantile models (10/50/90%) and tail weighting for the extremes (plan G) | register row; error per fifth of true views and per reaction tier, plus absolute views | Report the trade-off with overall MAE |
| B04 | P1 | open | Abstention curve: error against share of posts kept, from the conformal width (plan C) | figure + register row | Method: Sokol et al. 2024 (`docs/en/literature_notes.md`) |
| B05 | P1 | open | Forecast horizon curve: error of the E40 model by gap and by age of the first check | register row, figure | Data: `history.pairs`; extend to all (earlier, later) pairs, not only first-last |
| B06 | P1 | done | Use the history file to normalise the now-cast target to a fixed age (e.g. views at 7 days) and re-test E01-style models | register row | Done 2026-11-04 as E44 (age column: small, X only) and E45 (newer checks: not better) |
| B07 | P1 | open | Early-reaction forecast without views (FB/IG setting): predict later likes/comments from early ones on TikTok/X | register row | Likes also have history; this one transfers to FB/IG |
| B08 | P2 | open | Virality classification: will a post reach the top 10% of its platform? (plan F) | precision/recall on held-out brands | |
| B09 | P2 | open | Copy-paste campaigns: do near-duplicate posts inflate a brand's estimated reach? (plan F) | register row | E08 gives the duplicate groups |
| B10 | P2 | open | Separate model for posts without reactions: text, author, community (plan F) | register row | Compare with E43 posting-time model on the zero-reaction subset |
| B11 | P2 | open | Sentence embeddings (multilingual, local model) as text features | register row | Must run locally on homelab3; no external API |
| B12 | P2 | done | Temporal validation (train on earlier months, test on later) for the final models | register row | Done 2026-11-04 as E46 (expanding window close to brand CV; worse only in the first month) |
| B13 | P3 | open | Full GNN (GraphSAGE) on the text graph | register row | Low prior: E07, E29 found ≤ 0.009 |
| B14 | P0 | blocked | Own FB/IG sample with public view counts (plan A) | data statement updated, sample collected | Needs GDPR/terms decision and people |
| B15 | P1 | blocked | False brand match detector (plan D) | 100-200 hand labels, detector, share of reach | Needs hand labelling by people |
| B16 | P1 | blocked | Ask Brand24: YouTube zeros, rule confirmation, history timezone | answers recorded in `agent/questions_for_brand24.md` | People send the questions |
