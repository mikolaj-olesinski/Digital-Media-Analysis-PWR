# Digital Media Analysis PWR: estimating the visibility of social media posts

Project for the AMC course (Wrocław University of Science and Technology), client: Brand24.
Question: can the visibility (views) of a post be estimated from the number of followers and reactions better than by the constant × followers rule, and with what uncertainty?

**At a glance:** MAE 0.27 / 0.46 / 0.45 (TikTok / YouTube / X) on log10 views against about 1.0 for the constant rule; on 402 measured Facebook posts 0.385 against 0.93. CQR intervals are calibrated in-platform; Instagram is unmeasured and the Facebook band above 1M followers is not calibrated. Start with the [ten-sentence story](docs/en/story.md).

## How to read this project

1. **New to the project?** Read the [ten-sentence story](docs/en/story.md), then [status](docs/en/status.md) (where we
   are) and the [one-pager](docs/en/one_pager.md) (one page for the client).
2. **Need the method?** [model card](docs/en/model_card.md) -> [estimator spec](docs/en/estimator_spec.md) ->
   [evidence map](docs/en/evidence_map.md). Quick objections: [FAQ](docs/en/faq.md).
3. **Writing or reviewing?** [reporting guide](docs/en/reporting_guide.md), [client Q&A](docs/en/client_qa.md),
   [preprint outline](docs/en/preprint_outline.md), [slide plan](docs/en/slides.md).
4. **Reproducing?** [reproduce](docs/en/reproduce.md) -> [repro map](docs/en/repro_map.md) -> [scripts](docs/en/scripts.md).
5. **Which numbers are current?** [documentation status](docs/en/doc_status.md).

Most documents have a Polish version in `docs/pl/`; the internal `project_plan.md` is English-only.

### Who reads what

| audience | start | then |
|---|---|---|
| Client | [one-pager](docs/en/one_pager.md) | [client Q&A](docs/en/client_qa.md), [slide plan](docs/en/slides.md) |
| Team | [status](docs/en/status.md) | [evidence map](docs/en/evidence_map.md), [preprint outline](docs/en/preprint_outline.md), [doc status](docs/en/doc_status.md) |
| Newcomer | [story](docs/en/story.md) | [status](docs/en/status.md), [model card](docs/en/model_card.md), [reproduce](docs/en/reproduce.md) |
| Reviewer | [model card](docs/en/model_card.md) | [estimator spec](docs/en/estimator_spec.md), [evidence map](docs/en/evidence_map.md), [data statement](docs/en/data_statement.md) |
| Agent | `CLAUDE.md`, `agent/README.md` | [decisions](agent/decisions.md), `agent/findings.md`, `agent/backlog.md`, `agent/vault/` |

## Documentation map

| area | documents |
|---|---|
| Start here | [status](docs/en/status.md) · [verdict](docs/en/verdict.md) · [one-pager](docs/en/one_pager.md) · [what is new](docs/en/whats_new.md) · [future work](docs/en/future_work.md) · [plan coverage](docs/en/plan_status.md) |
| The estimator | [model card](docs/en/model_card.md) · [estimator spec](docs/en/estimator_spec.md) · [reporting guide](docs/en/reporting_guide.md) |
| Evidence | [evidence map](docs/en/evidence_map.md) · [experiments register](docs/en/experiments.md) · [documentation status](docs/en/doc_status.md) |
| Client and preprint | [client Q&A](docs/en/client_qa.md) · [example report](docs/en/report_example.md) · [preprint outline](docs/en/preprint_outline.md) · [slide plan](docs/en/slides.md) · [release checklist](docs/en/release_checklist.md) |
| Reproduce | [reproduce](docs/en/reproduce.md) · [repro map](docs/en/repro_map.md) · [scripts](docs/en/scripts.md) |
| Data and labels | [data statement](docs/en/data_statement.md) · [codebook](docs/en/codebook.md) |

## Notebooks and other artifacts

- `notebooks/models/`: 36 notebooks; the map (question, numbers) is in [analysis_overview](docs/en/analysis_overview.md).
- `notebooks/eda/`: `01_overview` (all platforms), `02_platforms` (one platform at a time, with totals), `03_labels` (mention labels and coder agreement)
- `docs/en/m1_exploration_card.pdf`: M1 data exploration card, **historical** milestone artefact from session 2 ([Polish version](docs/pl/m1_exploration_card.pdf)); `docs/pl/course_rules.pdf`: course rules (in Polish)
- `src/m1_figures.py`, `src/m1_figures2.py`: figures and numbers for the M1 card (`python src/m1_figures.py en|pl`)
- `figures/`: generated figures, `*_en.png` and `*_pl.png`
- `pitchdeck/`: presentation in English and Polish (`pitchdeck_en.*`, `pitchdeck_pl.*`); build script `build.py en|pl`
- `docs/en/repo_about.md`: the GitHub "About" text (description, topics, three highlights)

## For AI agents

`CLAUDE.md` and `agent/README.md`: work loop, runbook, current findings, backlog, decisions and one card per experiment in `agent/vault/`. Guardrails: `python src/agent_tools.py check`.

## Data

The raw file `data/raw/reach_samples_students.csv` (12,500 mentions from Brand24) is **not** in the repository: it contains personal data of authors and the client's data.
The second file `data/raw/reach_samples_students_with_history.csv` (same mentions plus earlier checks of views and reactions) is also not in the repository; notebook 16 needs it.
To reproduce the results, place them in `data/raw/` and follow `docs/en/reproduce.md`. Dependencies: `pip install -r requirements.txt`.

## Reporting a problem

If a number looks wrong, include: the document, the experiment ID (e.g. E141), the value you expected and the
`data/processed/*_numbers.json` file that should contain it. The internal procedure is in `docs/en/doc_status.md`
("If a number looks wrong"); a maintainer re-runs the script and follows "How to change a number".

For a data or privacy concern (a post or an author that should not appear), contact the team; we do not publish author
names or URLs, and will remove or aggregate anything that slips through. Please do not paste personal data into an issue.

## License and citation

- There is **no license file yet**: by default all rights are reserved. Reuse of code, text or figures needs the team's
  permission until a license is chosen (a human decision, see `docs/en/release_checklist.md`).
- The **numbers and figures** may be quoted with attribution to this project; the authors line is to be filled in.
- **Brand24 data is not redistributed**: the raw file, the labels derived from it and the fetched Facebook files stay
  local; only aggregates are published.

## Team

To be filled in.
