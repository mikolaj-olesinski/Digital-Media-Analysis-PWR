# Runbook

## Machines
- **Mac** (the user's): main working copy. `AGENTS.md` and `logs/` live there and are git-ignored. The user pushes with `git push --force-with-lease origin main`.
- **homelab3** (`192.168.2.19`, Ubuntu, 16 CPU, 20 GB RAM, no GPU): heavy runs. Checkout at `~/claude-sessions/Digital-Media-Analysis-PWR`. Single-node k3s with Kueue shared with two other projects (`mra2`, `zmgsn`).
- Commits made on homelab3 are not on GitHub until someone pushes; tell the user which commits are local.

## Data (git-ignored, copied by hand)
| File | Rows × columns | Notes |
|---|---|---|
| `data/raw/reach_samples_students.csv` | 12,500 × 19 | main file. Never edit |
| `data/raw/reach_samples_students_with_history.csv` | 12,500 × 42 | same mentions + `current_*` and `history_1..3_*` (2026-11-03 from Brand24). History timestamps are Polish local time; `src/history.py` converts them |

Copy from the Mac: `scp <file> mikolaj-olesinski@192.168.2.19:~/claude-sessions/Digital-Media-Analysis-PWR/data/raw/`

## Environment
- `.venv` (Python 3.14.4). Install: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` plus the packages in `~/claude-sessions/homelab-kueue/amc/requirements.lock` (exact versions of this machine).
- The Mac may have other library versions. Notebook 15 gives slightly different numbers here (AUC 0.6149 vs 0.6150 committed). When you compare with a committed number, re-run the base on the same machine.
- CatBoost: pass `allow_writing_files=False` so it does not leave `catboost_info/` in the repo.

## Running through the queue
Every run longer than about a minute goes through Kueue, so the three projects share the machine fairly.

```bash
~/claude-sessions/homelab-kueue/amc/submit.sh <name> <cpu> <memory> -- <command>
# e.g. a notebook in place:
~/claude-sessions/homelab-kueue/amc/submit.sh nb17 4 6Gi -- bash -c 'cd notebooks/models && jupyter nbconvert --to notebook --execute --inplace 17_x.ipynb'
kubectl -n amc get workloads,pods            # queued / admitted / running
kubectl -n amc logs -f job/<job>
kubectl -n amc wait --for=condition=complete job/<job> --timeout=1800s
```

- Image `amc/runner:py314` = the `.venv` packages + `libgomp1`, the repo mounted at the same path, runs as uid 1000, so files it writes belong to the user.
- `amc` share: 6 CPU / 7 GiB nominal, can borrow idle capacity, never preempted. Ask for what you use: CPU requests equal limits.
- New package: install in `.venv`, then rebuild the image (`~/claude-sessions/homelab-kueue/README.md`).
- Set `thread_count` / `n_jobs` to the CPU you requested.
- Model weights (TabPFN v2 regressor, TabICL) live in `~/claude-sessions/homelab-kueue/amc/models`, mounted read-only at `/models` with `HF_HUB_OFFLINE=1`; jobs never download. Use only the TabPFN **v2** weights (2.5+ are non-commercial) and pass `ignore_pretraining_limits=True` on CPU with > 1,000 rows. Outputs that use TabPFN must credit "Built with PriorLabs-TabPFN".
- `kubectl wait --for=condition=complete` does not return on a failed job; poll `.status.succeeded` / `.status.failed` instead.

## Typical costs (homelab3)
| Run | Time |
|---|---|
| notebook 15 | 10 s |
| notebook 16 (history, CatBoost on 6,735 posts × 5 folds × 4 sets) | 35 s on 4 CPU |
| notebook 19 (TabPFN v2 + TabICL, 3 seeds × 5 folds × 3 platforms) | about 17 min on 6 CPU |
| nested tuning over ten families (notebook 07) | not timed here yet; record it when you run it |

## Before a commit
```bash
.venv/bin/python src/agent_tools.py vault
.venv/bin/python src/agent_tools.py check
git status --short
```

Checklist:

- `vault` and `check` report 0 new failures (and 0 warnings you caused).
- `python src/agent_tools_selftest.py` passes (where_paths and register parsing).
- Numbers written by a script, not by hand, and the script is included.
- Register row in **both** languages; card filled (Result, Decision, Follow-ups); findings updated if a conclusion moved.
- No `data/raw/`, `logs/`, `AGENTS.md` or fetched personal data in the diff.
- Commit message in English, small and descriptive; **no AI co-author lines**; no `git push`.
- Figures in both `_en` and `_pl` versions when a document uses them.
