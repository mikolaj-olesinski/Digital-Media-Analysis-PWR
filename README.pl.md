# Digital Media Analysis PWR: estymacja widoczności postów społecznościowych

*English: [README.md](README.md)*

Projekt na kurs AMC (Politechnika Wrocławska), zleceniodawca: Brand24.
Pytanie: czy da się oszacować widoczność (wyświetlenia) postu z liczby obserwujących i reakcji lepiej niż stałą × obserwujący i z jaką niepewnością.

## Zawartość
- `docs/m1_exploration_card_pl.pdf` (+ `.html`): karta eksploracji danych (M1) ([wersja angielska](docs/m1_exploration_card_en.pdf))
- `docs/course_rules_pl.pdf`: zasady kursu, ocenianie i harmonogram
- `notebooks/01_eda.ipynb` (+ `.html`): wstępna eksploracja danych (po angielsku)
- `src/m1_figures.py`, `src/m1_figures2.py`: wykresy i liczby do karty M1 (`python src/m1_figures.py pl|en`)
- `figures/`: wygenerowane wykresy, `*_pl.png` i `*_en.png`
- `pitchdeck/`: prezentacja po polsku i angielsku (pptx, pdf, skrypt budujący `build.py`)
- `AGENTS.pl.md` ([angielska](AGENTS.md)): instrukcje dla agentów AI pracujących w repozytorium

## Dane
Surowy plik `data/raw/reach_samples_students.csv` (12 500 wzmianek od Brand24) **nie jest** w repozytorium: zawiera dane osobowe autorów i dane zleceniodawcy.
Aby odtworzyć wyniki, umieść go w `data/raw/` i uruchom `python src/m1_figures.py pl`. Zależności: `pip install -r requirements.txt`.

## Zespół
Do uzupełnienia.
