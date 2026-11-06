# Skrypty

Wewnętrzny indeks `src/`. Skrypty eksperymentów i ich liczby są w `docs/pl/repro_map.md`; ta strona opisuje helpery i
wczesne skrypty notebookowe, których nie linkują karty.

## Rdzeń

| skrypt | rola |
|---|---|
| `eda_utils.py` | wczytywanie surowej próbki, nazwy/kolory platform, bloki EDA |
| `validation.py` | CV po markach i czasowe, MAE i bootstrap CI, helpery konforemne |
| `modeling.py` | grupy cech (`GROUPS`), builder cech, drabinka modeli |
| `extra_models.py` | zbiory cech `ENG` / `ALL`, helpery finalnego modelu |
| `reaction_tiers.py` | tiery reakcji, finalny CatBoost, przedziały per tier |
| `agent_tools.py` | indeks vaulta i `check` (spójność kart i rejestru) |
| `agent_tools_selftest.py` | self-test `where_paths` i walidacji wierszy rejestru (`python src/agent_tools_selftest.py`) |
| `fetch_views.py` | publiczny fetcher wyświetleń (strony FB, FxTwitter, yt-dlp); źródło danych FB do E93/E95 |

## Figury i dokumenty

| skrypt | rola |
|---|---|
| `result_figures.py` | figury wyników (pasma CQR, pokrycie FB, koncentracja, kalibracja, reguła vs model) |
| `m1_figures.py`, `m1_figures2.py` | figury i liczby karty M1 |
| `story_figures.py` | pięć figur wyjaśniających do przeglądu projektu (PL) |
| `guide_figures.py`, `guide_build.py` | figury i HTML-owy przewodnik projektu (tabele przepisane ręcznie z JSON-ów; `guide_figures` ma na sztywno etykiety reguły 700 / 0,1) |
| `results_numbers.py` | liczby ze slajdów wynikowych |

## Wczesne skrypty notebookowe (bez linku z karty)

Były używane przez notebooki w `notebooks/models/`; ich eksperymenty są w rejestrze (E02-E54), ale karta linkuje
notebook, nie skrypt. Zostają dla reprodukowalności.

| skrypt | notebook / eksperyment |
|---|---|
| `transfer.py` | 02 transfer i przedziały (E02, E13) |
| `extra_features.py` | 04 dodatkowe cechy (E05) |
| `new_ideas.py` | 04: porównanie parami, target zaangażowania, graf marek |
| `graph_lab.py` | 05 laboratorium grafów (E06) |
| `graph_more.py` | 12 więcej grafów (E12) |
| `label_ablation.py` | 09 ablacja z etykietami (E04) |
| `label_utils.py`, `merge_labels.py` | scalanie etykiet i zgodność (E04) |
| `stacked_models.py` | 13 modele dwustopniowe (E31) |
| `xplat.py` | 06 transfer zaangażowania (E15) |
| `brand_mismatch.py` | 15 fałszywe dopasowania marek (D) |

## Liczby

- 121 skryptów w `src/`; 106 są wskazane przez kartę lub dokument; 15 bez wskazania to powyższe helpery, buildery figur
  i wczesne skrypty notebookowe.

## Argumenty wiersza poleceń

| skrypt | argumenty |
|---|---|
| `src/agent_tools.py` | `vault`, `check`, `next` |
| `src/fetch_views.py` | `--input`, `--platforms`, `--limit`, `--seed`, `--out`, `--workers`, `--cookies-from-browser`, `--cookies`, `--refresh`, `--url` (wszystkie w `--help`) |
| `src/result_figures.py` | `en` / `pl` |
| `src/m1_figures.py` | `en` / `pl` |
| `pitchdeck/build.py` | `en` / `pl` |

## Figury żywe i historyczne

| figury | buduje | status |
|---|---|---|
| `cqr_bands`, `fb_coverage`, `reach_concentration`, `calibration_curve`, `fb_rule_vs_model` (EN/PL) | `src/result_figures.py` | żywe, z JSON-ów |
| `m1_data_and_error`, `m1_distributions` (EN/PL) | `src/m1_figures.py`, `m1_figures2.py` | żywe, z surowych danych |
| `story_a`-`story_e` (**tylko PL**) | `src/story_figures.py` | żywe, z danych i etykiet; tylko PL, bo zasilają polski przewodnik |
| `baseline_{en,pl}`, `x_rule_{en,pl}` | `src/guide_figures.py` (CLI) | żywe, z surowych danych; polski przewodnik używa wersji PL, bliźniaki EN są dostępne, ale nieużywane |
| pozostałe `figures/*.png` (~100) | notebooki i starsze skrypty | historyczne; odtworzyć przez uruchomienie notebooków |

Odświeżenie figur żywych:

```bash
.venv/bin/python src/result_figures.py en && .venv/bin/python src/result_figures.py pl
.venv/bin/python src/m1_figures.py en && .venv/bin/python src/m1_figures.py pl
.venv/bin/python src/story_figures.py
.venv/bin/python src/guide_figures.py            # baseline_{en,pl}, x_rule_{en,pl}
.venv/bin/python src/guide_build.py /tmp/guide.html   # HTML przewodnika (używa powyższych PNG)
```
