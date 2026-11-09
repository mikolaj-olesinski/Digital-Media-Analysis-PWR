# Przegląd analiz

Pełna lista eksperymentów i wyników: `docs/pl/experiments.md`.

Wewnętrzna mapa tego, co, jak i gdzie sprawdziliśmy. To nie jest tekst preprintu. Liczby pochodzą z `data/processed/*_numbers.json`; każdą można odtworzyć z notebooka podanego obok.

## Wspólne ustawienia
- **Cel:** log10(view_count), tylko na TikToku, YouTube i X. Facebook i Instagram nie mają wyświetleń.
- **Walidacja:** 5 foldów po markach. Błąd to MAE na log10 (0,3 to ok. ×2, 0,5 to ×3, 1,0 to ×10). Przedziały losują całe marki (300–1000 losowań). Porównania są sparowane na tych samych foldach.
- **Cechy zbudowane z wyświetleń** liczymy tylko z części uczącej.

## Notebooki
| Notebook | Pytanie | Liczby |
|---|---|---|
| `01_first_models` | Pierwsza drabina grup cech | `results_numbers.json` |
| `02_transfer_and_intervals` | Przedziały konforemne, pierwszy test transferu | `results_numbers.json` |
| `03_zoo_strata_graph` | Modele na stałych ustawieniach, podgrupy, pierwsze cechy grafowe, etykiety a wyświetlenia | `zoo_numbers.json` |
| `04_extra_features` | Tekst, VADER, TF-IDF, kalendarz, wskaźniki, wielkość marki, graf marek | `extra_numbers.json` |
| `05_graph_lab` | Społeczności, centralność, spektralne, propagacja etykiet, hashtagi, czas, duplikaty | `graphlab_numbers.json` |
| `06_engagement_transfer` | Przewidywanie reakcji między platformami | w notebooku |
| `07_tuning` | Model domyślny kontra punkty odniesienia, zagnieżdżone strojenie 10 rodzin | `tuning_numbers.json` |
| `08_transfer_matrix` | Wszystkie pary źródło i cel, strojenie pod transfer, korekta etykietami, kalibracja, wrażliwość FB/IG | `transfer_numbers.json` |
| `09_ablation_labels_bias` | Ablacja na trzy sposoby z etykietami, ważność kolumn, błędy systematyczne, społeczności | `ablation_numbers.json` |
| `10_brand24_rule` | Odtworzenie kolumny zasięgu i porównanie z prawdziwymi wyświetleniami | `rule_numbers.json` |
| `11_promo_flag` | Jedna kolumna „reklama”; błędy na YouTube; usuwanie postów bez reakcji | `promo_numbers.json`, `zero_numbers.json` |
| `12_graph_more` | Druga runda grafów: stałe agregacje, sąsiedzi z innych platform, korekta na społeczność | `graph_more_numbers.json` |
| `13_stacked_models` | Modele dwuetapowe: klasyfikacja i regresja, poprawianie błędu, osobna ścieżka dla zera | `stacked_numbers.json` |
| `14_reaction_tiers` | Błąd modeli końcowych i przedziały według liczby reakcji | `tiers_numbers.json` |

## Kolumny
Zestaw podstawowy: followersi, lajki, komentarze, udostępnienia (w logarytmie) i znacznik braku dla każdej z nich. Dodane i sprawdzone osobno: statystyki tekstu, 30 tematów TF-IDF SVD, VADER, kalendarz, wskaźniki, wielkość marki, nasze etykiety typu i sentymentu, sentyment Brand24, cechy grafowe.

## Modele i strojenie (`src/tuning.py`)
Ridge, kNN, SVR (RBF), MLP, random forest, extra trees, boosting ze scikit-learn, LightGBM, XGBoost, CatBoost. Losowe przeszukiwanie, 40 ustawień na zewnętrzny fold, wewnętrzna CV po markach na 3 foldach, cel MAE. Ustawienia domyślne to ustawienia biblioteki, przy czym CatBoost ma limit 300 iteracji. Rodzinę raportowaną dla platformy wybiera wewnętrzna CV. Do tego blend trzech najlepszych według wewnętrznej CV.

## Transfer (`src/transfer_matrix.py`)
Uczenie na jednej lub dwóch platformach, test na pominiętej. Marki testowe usuwamy z uczenia także na innych platformach. Sześć modeli, trzy zestawy cech (reakcje; + followersi; + udostępnienia i znaczniki). Ustawienia wybieramy na trzy sposoby (domyślne, wewnątrz źródeł, między źródłami), nigdy z wyświetleniami celu. Korekta etykietami: przeważenie źródła do składu typów celu albo typ jako cecha. Kalibracja: przesunięcie o medianowy błąd n postów celu, n od 5 do 500, posty losowe albo z kilku marek.

## Reguła Brand24 (`src/brand24_rule.py`)
Zasięg = max(k × reakcje, c(followersi) × followersi). k wynosi 700 na X (300 powyżej 10 reakcji), 100 na Facebooku, 20 na Instagramie. c maleje schodkami z wielkością konta, na Instagramie nie występuje. Odtwarza 98%, 96%, 98% wierszy z dokładnością do 2%. Odtworzone przez nas, niepotwierdzone przez Brand24, który opisuje tę kolumnę jako poglądową.

## Główne wyniki
- Model domyślny z reakcjami bije stałą × followersi: 0,29 / 0,53 / 0,49 wobec 0,92 / 0,86 / 0,88 (TikTok, YouTube, X).
- Po strojeniu rodziny zbiegają się do ok. 0,27 / 0,45 / 0,47; wyjątkiem jest ridge na YouTube (0,55).
- Lajki niosą większość sygnału. Tekst dodaje ok. 0,01–0,02 na YouTube i X. Nasze etykiety najwyżej 0,005. Cechy grafowe najwyżej ok. 0,007, a propagacja etykiet szkodzi.
- Prawdziwe wyświetlenia na reakcję: 21 TikTok, 20 YouTube, 27 X. Reguła Brand24 zakłada 700 na X.
- Między platformami kolejność się trzyma, poziom może się przesunąć (+0,03, +0,21, −0,40). Na X 25–50 postów z prawdą usuwa przesunięcie.
- Największy błąd systematyczny to ściąganie do średniej: najniższa piąta część jest zawyżana, najwyższa zaniżana.

## Późniejsze wyniki (notebooki 11–14)
- Jedna kolumna „reklama” daje najwyżej 0,006 przy obserwujących i reakcjach. Reklamy nie tłumaczą dużych błędów na YouTube; tłumaczą je zera lajków i komentarzy (23% postów, 35% błędu).
- Usunięcie postów bez reakcji obniża wynik, bo test jest łatwiejszy; usunięcie z uczenia daje najwyżej 0,002.
- Modele dwuetapowe nie dały zysku. Dała go strata MAE: +0,011 / +0,007 / +0,022 dla domyślnego CatBoosta.
- Druga runda grafów: sąsiedzi z innych platform dają +0,009 na YouTube, jedyny przedział nad zerem na wszystkich postach. Przesunięcie poziomu nie maleje.
- Błąd spada z liczbą reakcji na YouTube i X. Przedział na poziom reakcji przywraca ok. 90% pokrycia tam, gdzie jest dość danych.

## Poprawki z 2026-11-02
- `extra_features.ratios` odwracało kolumny log1p podstawą 10. Naprawione, notebook 04 przeliczony. Główne wyniki bez zmian.
- „Rodziny są równe” jest prawdą dopiero po strojeniu i nie dla ridge na YouTube.
- „Zasięg na X = 0,1 × followersi” dotyczy tylko postów bez reakcji.

## Otwarte
Ludzki gold standard dla etykiet; czy zasięg to to samo co wyświetlenia; próbka z FB lub IG z prawdziwymi wyświetleniami (Brand24 jej nie ma, więc trzeba by zebrać ją samemu); fałszywe dopasowania marek w całych danych; embeddingi zdaniowe; pełny GNN.

Planowane kolejne kroki: „Ideas to stand out” w `docs/en/project_plan.md`.
