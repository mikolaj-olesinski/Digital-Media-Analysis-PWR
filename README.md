# Digital Media Analysis PWR: estimating the visibility of social media posts

Project for the AMC course (Wrocław University of Science and Technology), client: Brand24.
Question: can the visibility (views) of a post be estimated from the number of followers and reactions better than by the constant × followers rule, and with what uncertainty?

## Contents
- `docs/en/m1_exploration_card.pdf` (+ `.html`): M1 data exploration card ([Polish version](docs/pl/m1_exploration_card.pdf))
- `docs/pl/course_rules.pdf`: course rules, grading and schedule (in Polish)
- `notebooks/01_eda.ipynb` (+ `.html`): preliminary data exploration
- `src/m1_figures.py`, `src/m1_figures2.py`: figures and numbers for the M1 card (`python src/m1_figures.py en|pl`)
- `figures/`: generated figures, `*_en.png` and `*_pl.png`
- `pitchdeck/en/`, `pitchdeck/pl/`: presentation in English and Polish (pptx, pdf); build script `pitchdeck/build.py en|pl`

## Data
The raw file `data/raw/reach_samples_students.csv` (12,500 mentions from Brand24) is **not** in the repository: it contains personal data of authors and the client's data.
To reproduce the results, place it in `data/raw/` and run `python src/m1_figures.py en`. Dependencies: `pip install -r requirements.txt`.

## Team
To be filled in.
