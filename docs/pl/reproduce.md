# Jak odtworzyć

Od surowej próbki do liczb i figur. Na współdzielonej maszynie homelab3 patrz `agent/runbook.md` (kolejka Kueue, obraz, wagi modeli). Każdy skrypt zapisuje własny plik w `data/processed/`; kolejność poniżej jest
zalecana, ale większość skryptów jest niezależna.


## Szybki start (5 komend)

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/validation_audit.py
.venv/bin/python src/cqr_intervals.py
.venv/bin/python src/final_v3_x.py    # X v3: 0,454 i pasmo 1,99 (liczba z model card)
.venv/bin/python src/result_figures.py en
.venv/bin/python src/agent_tools.py vault && .venv/bin/python src/agent_tools.py check
.venv/bin/python src/agent_tools_selftest.py
```

Co robi każdy krok:

1. `pip install` - środowisko (Python 3.14, wersje w „Wersje użyte do finalnych liczb”).
2. `validation_audit.py` - powtórzone CV i kontrole rekalibracji; zapisuje `validation_audit_numbers.json`.
3. `cqr_intervals.py` - modele kwantylowe v2 i pasma CQR; zapisuje `cqr_numbers.json`.
4. `final_v3_x.py` - X z log(age); zapisuje `final_v3_x_numbers.json` (0,454, pasmo 1,99).
5. `result_figures.py` - wszystkie figury wynikowe z JSON-ów, EN i PL.
6. `vault` + `check` - indeksuje karty i sprawdza rejestry, ścieżki, karty, figury, notebooki.

To odtwarza rdzeń liczb i figur; kroki Facebooka wymagają pobranych plików (sekcja 3). Sekcje: szybki start, 0
środowisko, 1 dane, 2 rdzeń, 3 Facebook, 4 figury, 5 kontrola, 6 mapa, czas, wersje danych, debugowanie, wersje, brakujące
pliki, minimalny zestaw, higiena notebooków, awarie notebooków.


Uwaga o paśmie X: `cqr_intervals.py` trenuje model v2 i raportuje pasmo 2,07; `final_v3_x.py` dodaje log(age) i daje
aktualne MAE 0,454 i pasmo 1,99 (E141), których używają model card i figury. Uruchamiać oba.

## 0. Środowisko

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # pandas, scikit-learn, catboost, lightgbm, xgboost, matplotlib, jupyter, yt-dlp
```

Liczby powstały na Pythonie 3.14. Wszędzie stałe ziarna (`SEED = 42`).

## 1. Dane

- `data/raw/reach_samples_students.csv`: próbka Brand24, tylko do czytania, git-ignored.
- `data/processed/mention_labels.csv`: etykiety AI (typ, sentyment) dla 12 500 wzmianek; tworzone przez
  `src/merge_labels.py` z trzech plików koderów. Już w repo.
- Wyświetlenia Facebooka (`data/processed/fb_all_fetched.csv`, `fb_fetched.csv`): pobrane z publicznych stron przez
  `src/fetch_views.py` 2026-11-05; git-ignored. Bez nich eksperymenty FB (E93/E95/E111/E123/E150/E154/E165) nie
  odtworzą się, ale liczby w platformach — tak.

## 2. Model i walidacja

```bash
.venv/bin/python src/validation_audit.py     # powtórzone CV, isotonic i tier (E114)
.venv/bin/python src/cqr_intervals.py        # pasma CQR 90% (E118)
.venv/bin/python src/cqr_50.py               # pasmo 50% (E137)
.venv/bin/python src/final_v2.py             # author_te + CQR (E124)
.venv/bin/python src/final_v3_x.py           # X z log(age) (E141)
```

Oczekiwane: `data/processed/validation_audit_numbers.json`, `cqr_numbers.json`, `cqr_50_numbers.json`,
`final_v2_numbers.json`, `final_v3_x_numbers.json` — bajtowo identyczne z zapisanymi (sprawdzone dla dwóch skryptów).

## 3. Facebook

```bash
.venv/bin/python src/fb_transfer_check.py              # transfer, 130 postów (E93)
.venv/bin/python src/fb_transfer_check.py --followers  # kara followersów (E94)
.venv/bin/python src/fb_extend.py                      # 402 posty (E95)
.venv/bin/python src/fb_cqr.py                         # CQR na FB (E123)
.venv/bin/python src/fb_hybrid.py                      # hybryd >1M (E154)
.venv/bin/python src/fb_rule_vs_model.py               # reguła vs model per tier (E165)
```

## 4. Figury

```bash
.venv/bin/python src/result_figures.py en
.venv/bin/python src/result_figures.py pl
```

Zapisuje `figures/cqr_bands_{en,pl}.png`, `fb_coverage_*`, `reach_concentration_*`, `calibration_curve_*`,
`fb_rule_vs_model_*`. Które figury są żywe, a które historyczne: `docs/pl/scripts.md`.

## 5. Kontrola spójności

```bash
.venv/bin/python src/agent_tools.py vault      # indeks kart
.venv/bin/python src/agent_tools.py check      # karty, rejestr, liczby muszą się zgadzać
.venv/bin/python -m py_compile src/*.py
```

`check` musi wypisać `0 new failures, 0 known, 0 warnings`.

## 6. Mapa eksperymentów

- `docs/pl/repro_map.md`: eksperyment -> skrypt -> liczby (E95-E165).
- `docs/pl/experiments.md`: pełny rejestr (E01-E165), także eksperymenty nieudane.
- `docs/pl/scripts.md`: co robi każdy helper.

## Czas działania

- Większość skryptów: poniżej minuty; powtórzone CV i macierze CQR na FB: 2-10 minut na CPU.
- Bez GPU. LightGBM i CatBoost działają na jednym węźle.

## Wersje danych i dryfujące wyniki

Liczby w tym repo odpowiadają plikom Brand24 z 2026-11-03 (eksport historii) i sprawdzeniom Facebooka z 2026-11-05. Nowy eksport lub
nowe pobranie je zmienią: wyświetlenia rosną z wiekiem, a listopadowe ponowne pobranie było 1,07-18× zapisanego
snapshotu (E95).

Jeśli wyniki różnią się po nowym pobraniu:

1. Porównać **wiek** obu odczytów; nie mieszać liczników z różnych momentów w jednym eksperymencie.
2. Uruchomić ponownie dotknięte skrypty i zapisać **nową kartę** z nowymi liczbami; nie nadpisywać starej karty.
3. Powiedzieć, której wersji danych używa dana liczba (`data_statement.md`, „Gdy dane się zmieniają”).

## Gdy skrypt nie odtwarza liczby

1. **Te same dane**: te same daty eksportu/pobrania, te same liczby wierszy, te same filtry (`status == "ok"`,
   `views > 0`).
2. **To samo środowisko**: `pip install -r requirements.txt`; sprawdzić wersje `pandas`, `scikit-learn`, `catboost`,
   `lightgbm`.
3. **Te same ziarna**: `SEED = 42` wszędzie; przy porównywaniu bitów unikać `n_jobs` > 1 w estymatorach stochastycznych.
4. **Te same foldy**: CV po markach z tą samą listą marek; inna kolejność sortowania zmienia foldy.
5. **Porównywać z JSON-em**, nie z prozą: `data/processed/*_numbers.json` i liczby z karty.
6. Dryf ~0,001 to zwykle biblioteka/wątki; większy to filtr danych, foldy albo cecha.

## Wersje użyte do finalnych liczb

Python 3.14.4, pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1, scipy 1.18.1, catboost 1.2.10, lightgbm 4.7.0,
xgboost 3.4.1, torch 2.15.0+cpu, matplotlib 3.11.2, networkx 3.7. `requirements.txt` podaje minima; jeśli liczby różnią
się o ~0,001, najpierw sprawdzić te wersje.

## Gdy brakuje pliku danych

- **Brak `data/raw/reach_samples_students.csv`**: nic nie działa; poprosić zespół o plik (jest git-ignored i zawiera
  dane osobowe).
- **Brak pliku historii**: padają tylko notebook 16 i eksperymenty prognozy; reszta działa.
- **Brak pobranych plików Facebooka**: eksperymenty FB (E93/E95/E111/E123/E150/E154/E165/E166) nie ruszą; liczby
  w platformach pozostają bez zmian.
- **Brak `data/manual/fbig_sample.csv`**: próbki ręcznej nie da się odtworzyć; pobrane pliki mają już wyniki, ale
  E93/E97 nie ruszą od zera.
- Każdy skrypt czytający surowy CSV (główne loadery, wczesne skrypty i skrypty figur M1) kończy się czytelnym
  komunikatem z nazwą brakującego pliku, nie gołym tracebackiem.
- **Nigdy** nie podstawiać danych symulowanych ani ponownie pobranych za opublikowaną liczbę: albo odtworzyć z tego
  samego pliku, albo napisać nową kartę z nową wersją.

## Minimalny zestaw do odtworzenia liczb

- **Finalny model i przedziały:** pandas, numpy, scipy, scikit-learn, catboost, lightgbm, matplotlib.
- **Pobieranie (opcjonalne):** yt-dlp (fetcher Facebooka wywołuje go jako binarkę).
- **Tylko eksperymenty:** tabpfn, tabicl (E47), sentence-transformers, transformers (E50/E53), torch (embeddingi i
  GraphSAGE), networkx (eksperymenty grafowe).
- **Niepotrzebne do liczb:** python-pptx (pitchdeck), jupyter (tylko do notebooków).

Jeśli brakuje pakietu, skrypty, które go nie używają, nadal działają; sprawdzić import na górze skryptu.

## Higiena notebooków

- Uruchamiać każdy notebook od góry do dołu i **zapisywać z outputami**: `jupyter nbconvert --execute --inplace <nb>`.
- `check` pada na zapisanym błędzie, a ostrzega, gdy notebook ma kod, ale nie ma outputów.
- Używać `eda_utils.preview` dla każdej tabeli z autorami lub URL-ami: usuwa te kolumny i linki w tekście. Nigdy nie
  zapisywać URL-a ani kolumny autora w outputcie.
- Jedna platforma na figurę; dodać sumę wszystkich platform, gdy tabela opisuje jedną platformę.

## Gdy notebook się nie uruchamia

1. Użyć jądra z `.venv`: `python -m ipykernel install --user --name amc --display-name amc` (raz).
2. Sprawdzić pliki danych, których potrzebuje notebook (surowy zawsze; historia dla 16; pobrane FB dla eksperymentów FB).
3. Uruchamiać od góry do dołu; komórka zależna od wcześniejszej pada uruchomiona samodzielnie.
4. Ciężkie notebooki: 19 (TabPFN/TabICL) trwa około 17 min na 6 CPU; na homelab3 przez Kueue (`agent/runbook.md`).
5. Jeśli outputy się różnią, zastosować „Gdy skrypt nie odtwarza liczby”.
