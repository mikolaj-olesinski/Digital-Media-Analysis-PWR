# Documentation status map

Internal: which document holds **current** numbers and which is **historical by design**. Rule: when a number changes,
update the model card, the reporting guide and the evidence map; keep the old value only in the register and the card of
the experiment that measured it.

## At a glance: current vs historical

| current (state 2026-11-08) | historical |
|---|---|
| model card, estimator spec, reporting guide | register rows and experiment cards |
| evidence map, status, one-pager, client Q&A, slides | what is new, early sections of analysis overview |
| reproduce, repro map, scripts, future work, FAQ | project plan (EN only) |
| data statement, codebook, release checklist | the Polish guide HTML, the M1 card, the pitch decks |


Sections: at a glance · legacy cards · what the date in a header means · historical artefacts · changing an old result ·
what `check` validates · how to change a number · typical mistakes · intentional repetition · where old and new numbers
coexist.

| document | numbers | update rule |
|---|---|---|
| `model_card.md` | current (X v3: 0.454, bands 1.99/0.83; FB 0.385) | update when the final model changes |
| `estimator_spec.md` | current | update with the model card |
| `reporting_guide.md` | current | update with the model card |
| `evidence_map.md` | current claims, each with a strength | update when an experiment changes a claim |
| `status.md` | current snapshot | refresh after each push |
| `one_pager.md`, `client_qa.md`, `slides.md` | current | update with the model card |
| `preprint_outline.md` | structure only | stable |
| `whats_new.md` | snapshot of the Nov 2026 push | historical by design; do not update |
| `analysis_overview.md` | early sections from notebooks; the Nov section current | the early sections are historical notes |
| `experiments.md` (register) | **historical**: each row keeps the numbers measured at the time (e.g. E118 has the X v2 band 2.07) | append only; never rewrite old rows |
| `agent/vault/E*.md` | historical per experiment | append an update section if a later experiment changes the reading |
| `repro_map.md` | generated from the cards | regenerate after adding a card |
| `scripts.md` | current | update when scripts are added or removed |
| `reproduce.md` | current | update when commands or dependencies change |
| `future_work.md` | current priorities | update when an item is done or blocked |
| `data_statement.md` | current | update when data collection changes |
| `codebook.md` | stable | change only with a new labelling round |
| `literature_notes.md` | dated entries, checked vs search-only | append only |
| `project_plan.md` | EN-only, historical plan | do not update; `plan_status.md` records what happened to each idea |


## If a number looks wrong

1. Check "Typical mistakes when quoting numbers" above; most wrong numbers are one of those mix-ups.
2. Find the source: the JSON in `data/processed/`, the card `agent/vault/E*.md`, the register row.
3. If the **JSON** is wrong, fix the script and re-run it (never edit the JSON by hand).
4. If the **prose** is wrong, follow "How to change a number".
5. If the fix changes a claim, add a note to the card and a row to the register; run `vault` and `check`.
## Legacy cards

Cards **E01-E37** carry a "Legacy note": 21 of them never wrote a `numbers:` JSON and all 37 had no decision
recorded at the time. They are kept for the record, but use them as supporting evidence only and re-check the number
against the notebook output or the register before citing it.


## Legacy cards and what to cite instead

| legacy claim (E01-E37) | cite instead |
|---|---|
| the sample and views availability (E01) | `data_statement.md`, E105 |
| labels and coder agreement (E04) | `data_statement.md` (kappa) |
| model families and tuning (E11, E12) | E114 |
| CatBoost parameters (E32) | E114, E141 |
| text features (E05) | E100 |
| graph features (E06) | E122, E127, E128 |
| tail/bias (E18) | E96, E114 |



## What the date in a header means

- **Model card, estimator spec, status, findings:** the *state date* - the day the numbers were last checked against
  the files. All of them carry the same date (2026-11-07).
- **Register rows and cards:** the *experiment date* - when that experiment ran, which can be older.
- **Literature notes:** the *scan date* - when the batch was searched, not a project state.
- **What is new:** a snapshot of a period by design; its date range is the period it describes.
- **Data statement:** no state date; it describes the files currently in use (see "When the data changes").
## Historical artefacts

- **M1 exploration card** (`docs/en|pl/m1_exploration_card.html|pdf`): the session-2 milestone; its numbers are the state
  at that time (e.g. X 0.47, before v3).
- **The Polish guide HTML** (`src/guide_build.py`): describes the analyses up to 2026-11-04, with a note at the top.
- **The pitch decks**: text and speaker notes only, no model numbers.

## When a new experiment changes an old result

1. Add the new card and the register row (both languages); the new card is the one with the JSON.
2. Add an **Update** section at the end of the old card: date, new experiment ID, what changes, what still holds. Never
   delete or rewrite the old result.
3. Mark the old claim as superseded in `agent/findings.md` (keep the line, add "superseded by E<new>").
4. Update the current documents with "How to change a number"; the old value stays only in the register and the old card.
5. If a document cites the old card, replace the citation with the new one (the legacy table above shows the mapping).
## What `check` validates

`python src/agent_tools.py check` verifies:

- the EN and PL registers have the same IDs, without duplicates;
- every register row has 7 columns and a result type from the known vocabulary (separate EN and PL sets);
- every path named in the register exists, every `numbers:` and `notebooks:` path named in a card exists, and every register row has a vault card;
- every card has `## Result` and `## Decision` sections, and a card with `numbers: []` carries a "Legacy note";
- no tracked `.md`/`.py` file, nor a notebook source cell, contains a social post URL (**warnings**, not failures);
- every `*_en.png` figure has a `*_pl.png` twin (PL-only series such as the story figures are allowed);
- notebooks are stored with outputs, without error outputs, URLs or personal-data columns (documents may cite arxiv/github; real social post URLs warn);
- the client questions table has 5 columns per row;
- `data/raw/`, `logs/`, `AGENTS.md` and fetched CSVs are not tracked by git, and no tracked CSV has an author/URL column
  (checked from the file header); the backlog exists.

It does **not** verify the content of a card (question, result, decision), nor that the numbers written in prose match the
JSON files; those are the manual procedures below.

## How to change a number

1. If the number is new, add a card in `agent/vault/`, a register row and a JSON in `data/processed/` (script first).
2. Update the **model card** (source of truth for current numbers).
3. Update the estimator spec, the reporting guide and the evidence map; the data statement if the data changed.
4. Update the repeating documents: story, one-pager, client Q&A, slides, status.
5. Regenerate the figures: `python src/result_figures.py en` and `pl`.
6. Run `python src/agent_tools.py vault && python src/agent_tools.py check`.
7. Keep the old value only in the register and the card of the experiment that measured it; never rewrite old rows.

## Typical mistakes when quoting numbers

Numbers, not concepts; the concept-level list is in `faq.md` ("Common misunderstandings").

- **X v2 vs v3**: 0.467 and a 2.07 band (E118) are the old model; the current X numbers are 0.454 and 1.99 (E141).
- **130 vs 402 Facebook posts**: 0.345 belongs to the hand sample (E93); 0.385 is the full sample (E95) and is the
  number to quote.
- **Rule error on Facebook**: x14 (130 hand posts, E93) vs x4.1 (402 posts, E111); use x4.1.
- **FB interval**: 0.94 for pages below 1M, 0.80-0.86 above 1M; do not quote one as the other.
- **Posting time vs collection**: 0.60-0.76 MAE at posting time, not 0.27-0.46; the bands are 2.4-3.0 vs 1.1-2.0.
- **50% vs 90% band**: x2.8-7.6 vs x14-98; a sentence that mixes them is wrong.
- **Brand totals**: E109 is X, E157/E158 are Facebook; do not mix the two samples.
- **"Reach" vs "views"**: the estimator gives views; unique reach is not public.

## Intentional repetition

`story.md`, `one_pager.md`, `client_qa.md`, `slides.md` and `status.md` repeat the headline numbers on purpose, for
different audiences. When a headline number changes, update the model card first and then these five; the register keeps
the historical value.

## Known places where old and new numbers coexist

- **X bands**: 2.07 (E118, v2) in the register vs 1.99 (E141, v3) in the model card; the card is current.
- **X MAE**: 0.467 (E118/E114) vs 0.454 (E141); the card is current.
- **FB followers**: E94 says "followers hurt" on the 130 sample; E149/E154 show they help above 1M; the card and the
  E94 update section carry the nuance.
- **FB rule error**: E93 gave x14 on the 130 hand sample, E111 x4.1 on 402; the card uses 0.93 MAE / x4.1.
- **Brand totals**: E109 (X) and E157/E158 (FB) use different samples; do not mix them in one sentence.
