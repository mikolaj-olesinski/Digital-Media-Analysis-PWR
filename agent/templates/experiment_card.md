---
id: E00
date: YYYY-MM-DD
type: planned            # planned → gain | small | none | worse | finding
status: planned          # planned | running | done | failed
backlog: B00
notebooks: [notebooks/models/NN_topic.ipynb]
numbers: [data/processed/topic_numbers.json]
---
# E00 · Question in one line

## Question and hypothesis
- Question:
- Hypothesis (falsifiable):
- Refuted if:

## Setup
- Data and subset (platforms, months, posts):
- Base model vs new model (same folds):
- Validation: 5-fold brand-grouped CV, MAE log10, brand bootstrap CI
- Queue job: `amc-...`, CPU / memory, run time

## Result
Per platform (TikTok / YouTube / X): base, new, gain with CI, posts.

## Decision
What this changes in `agent/findings.md` (or "nothing").

## Follow-ups
New backlog items (IDs).

## Links
- notebook, numbers file, figures
