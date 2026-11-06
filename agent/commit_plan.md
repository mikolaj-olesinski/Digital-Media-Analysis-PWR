# Commit plan for the November push (not committed yet)

The working tree holds 331 changed paths (275 untracked, 56 modified). This is a plan for splitting them into small
commits; nothing here is committed. Order matters: tooling first, then experiments, then figures, then documents.

## 1. Tooling and guardrails

- `src/agent_tools.py`, `src/agent_tools_selftest.py`
- `agent/README.md`, `agent/runbook.md`, `agent/session_index.md`
- `requirements.txt`

Message: `Add register/selftest guardrails and runbook self-test`

## 2. Agent records

- `agent/decisions.md`, `agent/findings.md`, `agent/backlog.md`, `agent/questions_for_brand24.md`

Message: `Record decisions, findings, backlog and client questions from the November push`

## 3a. Facebook code and numbers

- `src/fb_*.py` (22 scripts), `src/fetch_views.py` if modified
- `data/processed/fb_*_numbers.json` (24 files) and related CSVs

Message: `Measure Facebook views and the transfer (402 posts)`

## 3b. Facebook cards

- `agent/vault/E93.md`, `E95.md`, `E111.md`, `E123.md`, and the FB cards `E132`-`E167`

Message: `Add the Facebook experiment cards`

## 4. Intervals, model quality and missing data

- `src/cqr_*.py`, `src/final_v*.py`, `src/validation_audit.py`, `src/transfer_cqr.py`, `src/missing_*.py`,
  `src/partial_missing.py`, `src/rate_target.py`, `src/hurdle_zeros.py`, `src/recency_weights.py`, `src/seed_ensemble.py`,
  `src/tier_recal.py`, `src/rolling_origin.py`, `src/author_te.py`, `src/cross_author.py`, `src/new_columns.py`
- their `data/processed/*_numbers.json` and cards

Message: `Add CQR intervals, validation audit and missing-data switch rules`

## 5. Figures

- `src/result_figures.py`, `figures/*_en.png`, `figures/*_pl.png` (10 files)

Message: `Add result figures (EN/PL) and the figure script`

## 6. Documents

- `docs/en/*.md`, `docs/pl/*.md` (24 each), `README.md`

Message: `Update model card, evidence map and add the November documentation`

## 7. Register

- `docs/en/experiments.md`, `docs/pl/experiments.md` (166 rows, includes the failures)

Message: `Register E93-E166 (166 experiments, failures included)`

## Coverage

A classification of the current `git status` (332 paths) puts every path in one of the groups above: 213 in Facebook
(3a+3b), 48 in intervals/missing data, 47 documents, 11 figures, 7 tooling, 4 agent records, 2 register, 0 unclassified.
Re-run that check before committing; if a path falls outside, add a group here first.

## What stays local (never committed)

- `data/raw/` (the Brand24 export), `data/manual/` (hand-check sheets), `data/processed/*fetched*.csv` (fetched FB/IG
  counters with URLs), `logs/`, `AGENTS.md`, `.venv/`, caches (`*_cache.pkl`, `__pycache__`).
- `.gitignore` covers all of these; `check` fails if a fetched CSV or a raw/log file is tracked.
- This was a real near-miss: the fetched CSVs were untracked but **not ignored** before 2026-11-08; a `git add -A` would
  have committed URLs. The glob `data/processed/*fetched*.csv` now prevents it.

## Rules

- Never `git add data/raw`, `logs/`, `AGENTS.md` (check with `git status --short | grep -E 'data/raw|logs/'`).
- Run `vault`, `check` and the self-test after each commit.
- No `git push`; no AI co-author lines.
- If a commit mixes two groups, split it; a reviewer should be able to read one theme per commit.
