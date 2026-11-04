---
name: run-experiment
description: Run one experiment of the AMC Brand24 views project end to end - pick a backlog item, write the vault card first, build src + notebook, run it through the Kueue queue on homelab3, record numbers, figures, register rows (en + pl), findings and backlog, check and commit. Use when asked to run, test or try an idea, or to work autonomously through the backlog.
---

# Run an experiment

Follow `agent/README.md` (the loop) exactly. This skill is the checklist.

## 1. Orient
- `git status --short` (clean, or only your own work), `git log --oneline -5`.
- Read `agent/findings.md` (especially Dead ends) and the newest `logs/` file.
- `.venv/bin/python src/agent_tools.py next`; pick the top `open` item you can finish. Set it to `running` with the date in `agent/backlog.md`.

## 2. Card first
- Next ID = last row of `docs/en/experiments.md` + 1.
- `cp agent/templates/experiment_card.md agent/vault/E<ID>.md`; fill the question, a falsifiable hypothesis, "refuted if", base vs new.

## 3. Build
- Functions in `src/<topic>.py`; reuse `validation.mae_ci`, `history.gain_ci` (paired, brand bootstrap), loaders in `modeling.py` / `extra_models.py` / `history.py`.
- CatBoost: `allow_writing_files=False`, `thread_count` = requested CPU.
- Notebook `notebooks/models/NN_<topic>.ipynb`: imports cell, data cell, small cells, show only non-personal columns, tables with a TOTAL row, figures in `_en` and `_pl`, final cell writes `data/processed/<topic>_numbers.json`.
- Use the `jupyter-notebook` and `scientific-visualization` skills for structure and figure checks.

## 4. Run through the queue
```bash
~/claude-sessions/homelab-kueue/amc/submit.sh <name> <cpu> <mem> -- bash -c 'cd notebooks/models && jupyter nbconvert --to notebook --execute --inplace NN_<topic>.ipynb'
kubectl -n amc wait --for=condition=complete job/<job> --timeout=3600s; kubectl -n amc logs job/<job> | tail
```
Look at every figure once (Read the PNG) before recording.

## 5. Record
- Register rows in `docs/en/experiments.md` and `docs/pl/experiments.md` (same ID, same order, Polish decimal commas).
- Card: Result (table per platform with CI and posts), Decision, Follow-ups.
- `agent/findings.md`: edit the affected line, add the ID, set confidence. New dead end → the table.
- `agent/backlog.md`: close the item (`done`, IDs); add follow-ups as new rows.
- `agent/decisions.md` if you made a methodological choice.
- README notebook list if you added a notebook.

## 6. Check and commit
```bash
.venv/bin/python src/agent_tools.py vault && .venv/bin/python src/agent_tools.py check
git add <your files> && git commit -m "<English, what was added and found>"
```
No AI co-author lines, no push. Append to the session log in `logs/`.

## Stop and ask
Anything in "Stop and ask a person" in `agent/README.md`: external services, new data, history rewrites, changing validation or decisions, a result contradicting a **solid** finding.
