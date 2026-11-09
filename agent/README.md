# Agent workspace

Operating manual for AI agents that work on this project for long stretches without a person watching.
Read this file, then `AGENTS.md` (course rules, local only), then `agent/findings.md`. Agent files are English only;
documents for people stay in `docs/en` + `docs/pl`.

| File | What it is | Who edits it |
|---|---|---|
| `agent/README.md` | this manual: the work loop, definition of done, when to stop | people |
| `agent/runbook.md` | how to run things on homelab3: env, Kueue queue, data files, reproducibility | agents, when the setup changes |
| `agent/findings.md` | what we know now, with experiment IDs and confidence; dead ends not to repeat | agents, after every experiment that changes a conclusion |
| `agent/backlog.md` | the queue of work, priority and status | agents pick and close items; people set priorities |
| `agent/decisions.md` | methodological choices and why (metric, splits, assumptions) | agents, when a choice is made |
| `agent/questions_for_brand24.md` | questions only the client can answer | agents add, people send |
| `agent/session_index.md` | where each long session is recorded (logs are local) | agents, after a long session |
| `agent/commit_plan.md` | how to split the uncommitted November work into small commits | people, before committing |
| `agent/vault/` | one card per experiment (`E40.md`), index `INDEX.md` + `index.json` | agents; index is generated |
| `agent/templates/` | card and notebook skeletons | people |
| `src/agent_tools.py` | `vault`, `check`, `next` | people |
| `agent/check_known.txt` | check failures already in the backlog, reported as KNOWN | agents, only with a backlog ID |

## Starting a new session (five steps)

1. `git status --short` and `git log --oneline -5`: what is uncommitted and what moved last.
2. Read this file, `agent/findings.md`, `agent/backlog.md` and the newest file in `logs/`.
3. Read `docs/en/doc_status.md` (which document holds current numbers) and `docs/en/status.md` (where the project is).
4. Pick the highest-priority `open` backlog item you can finish; copy the card template and write the question before
   running anything.
5. Work, record, `vault` + `check`, commit locally (never push), append the session log.

## The loop

1. **Orient.** `git status`, `git log --oneline -10`, read `agent/findings.md` and the newest file in `logs/`. Run `python src/agent_tools.py next`.
2. **Pick** the highest-priority `open` backlog item you can finish. Mark it `running` with today's date. Do not pick `blocked` items.
3. **Plan in the card first.** Copy `agent/templates/experiment_card.md` to `agent/vault/E<next>.md`. Write the question, a falsifiable hypothesis, the comparison (base vs new), and what result would change a conclusion, before running anything. The next free ID is one above the last row of `docs/en/experiments.md`.
4. **Build** in `src/<topic>.py` (functions) and `notebooks/models/NN_<topic>.ipynb` (calls and outputs), following §8 of `AGENTS.md`. Reuse `src/validation.py` (brand-grouped CV, `mae_ci`) and the loaders in `modeling.py`, `extra_models.py`, `history.py`.
5. **Run** through the queue (`agent/runbook.md`), never as a long process on the host. Store the notebook with outputs.
6. **Record.** Numbers to `data/processed/<topic>_numbers.json`; figures `figures/<topic>_{en,pl}.png`; one row per experiment in both registers (`docs/en|pl/experiments.md`), failures included; fill the card's Result, Decision and Follow-ups; update `agent/findings.md` if a conclusion moved; add follow-ups to `agent/backlog.md`.
7. **Check.** `python src/agent_tools.py vault && python src/agent_tools.py check` must report 0 new failures. A failure you did not cause and cannot fix now: add a backlog item and a line in `agent/check_known.txt`.
8. **Commit** small and descriptive, in English, no AI co-author lines, author = the person running the agent. Never push. Close the backlog item (`done`, with the experiment IDs).
9. **Log** the session in `logs/YYYY-MM-DD_<person>_<agent>.md`: goal, what ran, decisions, what is left. Append, never delete.

## A good card

```
---
id: E165
date: 2026-11-06
type: small
status: done
notebooks: []
numbers: [data/processed/fb_rule_vs_model_numbers.json]
---
# E165 · FB rule vs model by follower tier

## Question and hypothesis
- Question: does the rule beat the model in any tier?
- Hypothesis: the model wins everywhere.
- Refuted if: the rule wins in a tier.

## Setup
- ... data, folds, base comparison ...

## Result
- ... numbers with CI ...

## Decision
- ... what changes; "nothing" is a valid decision for a null ...

## Links
- `src/...py`, `data/processed/..._numbers.json`, related cards
```

A good card has a falsifiable hypothesis and a "refuted if" line before the run, numbers from a JSON, and a decision that
says what changed. Anti-patterns: "see notebook" instead of numbers, no base comparison, no decision, or a result written
after the fact without the refuted-if line.

**Cards E01-E49 predate the "refuted if" rule** and do not have that line. Do not retrofit them: they state the question
and the comparison, and if one is cited in the preprint, re-check its number against its JSON first. **E01-E37** also
carry a "Legacy note" (no JSON and/or the decision was not recorded); treat them as supporting evidence only.

`vault` never overwrites an existing card (tested: a card with different content is left untouched). A card it
generates from a register row is a **stub**: it has no JSON and will show a `numbers: []` warning until someone fills it.
If a register row names a path that does not exist, `vault` skips that row and prints why, instead of writing a card with
a broken link.

**A verifiable Result** states: the metric and its unit (MAE on log10 views), the number of posts, the platform order
(TikTok / YouTube / X), the CI or the paired gain, and the JSON file that contains it. "Much better" is not a result.

## Adding an experiment to the register

- The next ID is one above the last row of `docs/en/experiments.md` (currently E167).
- One row in **both** languages, columns in this order: ID, Date, Question, Setup, Result, Type, Where.
- Type is one of the values below; a null result gets a row too.

| type | when to use |
|---|---|
| `gain` | CI above zero and useful (more than about 0.01, not isolated) |
| `small` | CI above zero but about 0.01 or less, or isolated |
| `none` | no difference; CI includes zero |
| `worse` | clearly worse than the base |
| `finding` | describes the data or the behaviour, not a model comparison (EDA, rule error, coverage) |
| `instrument` | a new labelling instrument (e.g. the AI labels) |
| `pending` | a pre-registered estimate, not yet checked |
- Result states the numbers and the JSON that holds them; Where names the script or notebook.
- Add the card and the numbers in the same commit; run `vault` and `check`.

## Definition of done for an experiment
- A paired comparison against a named base on the same folds, with a brand bootstrap CI (`validation.mae_ci`, or a paired gain like `history.gain_ci`).
- Numbers per platform (TikTok / YouTube / X order) and the number of posts.
- The register line says what was tested and the result type (gain / small / none / worse / finding).
- The card says what the result changes, including "nothing" for a null result.
- Reproducible from raw data by running the notebook top to bottom.

## Standards for claims
- Never write a number that no file in `data/processed/` or a stored notebook output contains.
- A gain counts only if the CI is above zero; about 0.01 or less is "small". Report nulls and failures just as carefully.
- Say what the sample covers (platforms, months, posts). Facebook and Instagram have no ground truth: any statement about them is a transfer assumption, not a measurement.
- No causal language from correlations. No final preprint or poster text (course rule).

## Stop and ask a person (do not continue alone)
- Anything that sends data outside homelab3 (APIs, LLM labelling of mentions, publishing pages, pushing).
- New data sources, scraping, or collecting data from people (plan item A needs a GDPR and terms decision).
- Rewriting git history, deleting or overwriting someone else's results, changing the raw CSV files.
- Changing the shared validation, the metric, or a decision in `agent/decisions.md`.
- Using more than the `amc` queue share for more than a few hours.
- A result that contradicts a finding marked **solid**: re-run once, then report before building on it.

## When a run fails
Record it anyway (register row with type "worse" or "none", card with the error), fix or drop, and move on. Do not retry the same setting more than twice.

## When `check` fails
- "register IDs differ between en and pl": add or remove the same row in both registers.
- "`path` named in the register does not exist": fix the path or create the file.
- "no vault card": run `python src/agent_tools.py vault` or write the card.
- "figure ... has no Polish twin": generate the `_pl` version with the figure script.
- "stored with an error output" / "stored without outputs": re-run the notebook top to bottom.
- "output contains a URL" / "output shows column `author_...`": re-run the notebook through `eda_utils.preview`, which
  drops those columns.
- "tracked by git but must stay local": `git rm --cached <path>` (never delete the local file).
- "duplicate experiment IDs": fix the duplicate in both registers.
- A failure you did not cause and cannot fix now: add a backlog item and a line in `agent/check_known.txt` with its ID.
  Never add a line just to silence a failure you introduced.
