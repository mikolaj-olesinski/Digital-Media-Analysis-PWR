# Instructions for Claude Code

Read in this order at the start of every session:
1. `AGENTS.md`: course rules and project conventions (local only, git-ignored; if it is missing, ask the user for it).
2. `agent/README.md`: the work loop, definition of done, when to stop and ask.
3. `agent/findings.md`: what is already known and what not to repeat.
4. `agent/runbook.md` when you run anything on homelab3.

Hard rules (details in the files above):
- Never edit `data/raw/`. Never send data outside this machine without the user's consent. No author names or URLs in outputs.
- Every experiment, failed ones too: a card in `agent/vault/`, a row in `docs/en/experiments.md` and `docs/pl/experiments.md`, numbers in `data/processed/`.
- Runs longer than a minute on homelab3 go through Kueue: `~/claude-sessions/homelab-kueue/amc/submit.sh`.
- Before a commit: `.venv/bin/python src/agent_tools.py vault && .venv/bin/python src/agent_tools.py check`.
- Commit messages in English, small and descriptive, no AI co-author lines. Never push.
- Log each session in `logs/YYYY-MM-DD_<person>_<agent>.md`.

To run a full experiment, use the `run-experiment` skill.
