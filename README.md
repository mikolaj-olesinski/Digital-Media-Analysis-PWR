# Digital Media Analysis PWR: estimating the visibility of social media posts

Project for the AMC course (Wrocław University of Science and Technology), client: Brand24.
Question: can the visibility (views) of a post be estimated from the number of followers and reactions better than by the constant × followers rule, and with what uncertainty?

## Contents
- `docs/en/m1_exploration_card.pdf`: M1 data exploration card ([Polish version](docs/pl/m1_exploration_card.pdf))
- `docs/pl/course_rules.pdf`: course rules, grading and schedule (in Polish)
- `notebooks/models/`: `01_first_models` (model ladder, feature importance), `02_transfer_and_intervals` (intervals, transfer to Facebook and Instagram)
- `notebooks/eda/`: `01_overview` (all platforms), `02_platforms` (one platform at a time, with totals), `03_labels` (mention labels and coder agreement)
- `src/m1_figures.py`, `src/m1_figures2.py`: figures and numbers for the M1 card (`python src/m1_figures.py en|pl`)
- `figures/`: generated figures, `*_en.png` and `*_pl.png`
- `pitchdeck/`: presentation in English and Polish (`pitchdeck_en.*`, `pitchdeck_pl.*`); build script `build.py en|pl`

## Data
The raw file `data/raw/reach_samples_students.csv` (12,500 mentions from Brand24) is **not** in the repository: it contains personal data of authors and the client's data.
To reproduce the results, place it in `data/raw/` and run `python src/m1_figures.py en`. Dependencies: `pip install -r requirements.txt`.

## Team
To be filled in.
