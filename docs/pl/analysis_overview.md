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
| `15_brand_mismatch` | Marki daleko od swojej branży w grafie tekstu (fałszywe dopasowania) | `mismatch_numbers.json` |
| `16_history_forecast` | Drugi plik Brand24: wzrost między sprawdzeniami, prognoza z pierwszego sprawdzenia, ocena w chwili publikacji (E38–E43) | `history_numbers.json` |
| `17_check_age` | Wiek sprawdzenia wyświetleń jako kolumna; nowsze sprawdzenia (E44–E45) | `check_age_numbers.json` |
| `18_rolling_window` | Uczenie na wcześniejszych miesiącach, test na następnym (E46) | `rolling_window_numbers.json` |
| `19_tabular_foundation` | TabPFN v2 i TabICL kontra CatBoost (E47) | `tabular_foundation_numbers.json` |
| `20_classification` | Viral (top 10%); posty bez wyświetleń lub z zerem (E48) | `classification_numbers.json` |
| `21_context_text` | Kontekst z innych postów autora, hashtagów, marki; embeddingi tekstu; cel z przesunięciem (E49–E51) | `context_text_numbers.json` |
| `22_posting_time_text` | Ocena w chwili publikacji z tekstem i kontekstem (E52) | `posting_time_text_numbers.json` |
| `23_finetune_text` | Dostrojony e5-small: wiersz jako tekst, tekst jako poprawka (E53) | `finetune_text_numbers.json` |
| `24_stack_all` | Stack wszystkich rodzin modeli na tych samych foldach (E54) | `stack_all_numbers.json` |
| `25_limits` | Audyt wycieków, granica szumu, przedziały kwantylowe, odrzucanie niepewnych (E55–E58) | `limits_numbers.json` |
| `26_history_more` | Horyzont prognozy, wszystkie pary sprawdzeń, wczesne reakcje bez wyświetleń (E59–E60) | `history_more_numbers.json` |
| `27_more_models` | Model wspólny i monotoniczny, dryf YouTube, skopiowane posty, posty bez reakcji (E61–E64) | `more_models_numbers.json` |
| `28_gnn` | GraphSAGE na grafie tekstu (E65) | `gnn_numbers.json` |
| `29_accounts_clusters` | Typy kont, klastry tematów, historia autora, społeczności marek (E66–E69) | `accounts_clusters_numbers.json` |
| `30_author_more` | Historia autora dokładniej, rozkład wariancji, sumy dla marek, anomalie, wyświetlenia na lajk (E70–E75) | `author_more_numbers.json` |
| `31_author_next` | Gdzie zostaje błąd, krzywa historii, poziom autora bez wyświetleń, szacunek na żywo z historią, przedziały sum marek (E76–E80) | `author_next_numbers.json` |
| `32_fbig_more` | FB / IG przed ręcznym sprawdzeniem (struktura, szacunki zapisane z góry, ocena), atlas nadwyżek (E81–E83) | `fbig_more_numbers.json`, `fbig_preregistered.csv` |

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

## Późniejsze wyniki (notebooki 15–30)
- **Historia autora to największy znaleziony zysk (E68).** Nadwyżka autora (wyświetlenia ponad prognozę modelu końcowego) się utrzymuje: Spearman 0,68 / 0,69 / 0,70 między pierwszym a kolejnymi postami, także bez powtórzonych postów. Poprawka z wcześniejszych postów autora daje +0,12 / +0,16 / +0,10 MAE na postach, które je mają (3–16% próbki).
- **Stack wszystkich rodzin (E54):** 0,260 / 0,425 / 0,441, czyli +0,011 / +0,028 / +0,026 względem modelu końcowego; większość daje zwykła średnia różnych modeli. Sam TabPFN v2 +0,005 / +0,010 / +0,002 (E47); średnia CatBoost + MLP +0,008 / +0,016 / +0,016 (E65, sam GraphSAGE gorszy).
- **Tekst:** zamrożone embeddingi pomagają tylko bez reakcji (YouTube +0,051, E50); dostrojony transformer jako połowa poprawki daje +0,007 / +0,012 / +0,008 (E53). Kontekst z innych postów dodaje ≤ 0,003 przy reakcjach (E49).
- **Plik z historią:** prognoza z pierwszego sprawdzenia 0,165 / 0,188 / 0,269 (E40); bez wczesnych wyświetleń 0,368 / 0,444 / 0,544 (E41); łańcuch prognozowane reakcje → model bieżący bije to o +0,034 na TikToku (E60). W chwili publikacji 0,566 / 0,664 / 0,540 (E43, E52).
- **Czas i walidacja:** uczenie na wcześniejszych miesiącach daje wynik bliski CV po markach (E46); ostrzejsze foldy (marka ∪ autor ∪ tekst) kosztują ≤ 0,011 (E55). Dryf na YouTube bierze się z młodszego wieku sprawdzeń (E62); wiek sprawdzenia pomaga na X (+0,012, E44).
- **Granice:** granica szumu ok. 0,22 / 0,39 / 0,33 (E56); przedział kwantylowy 10–90% pokrywa 73% (E57); ocenianie tylko pewniejszej połowy postów obniża MAE na YouTube o 27% (E58).
- **Inne zadania:** viral (top 10%) z precyzją 83 / 71 / 63% (E48); brak wyświetleń da się przewidzieć (AUC 0,91–0,95), posty z X z 2025 prawie nigdy nie mają wyświetleń (E48). Skopiowane posty są zaniżane (YouTube −0,26, E63).
- **Grupy:** oficjalne konta marek to ≤ 5% postów i są zaniżane ×2–3 (E66); klastry tematów i 9 społeczności marek (branże) są czytelne, ale poza reakcjami niosą mały błąd systematyczny (E67, E69).
- **Poziom autora (E70–E72):** dwie trzecie pozostałego błędu na TikToku i YouTube to wspólny poziom postów jednego autora (marka ≤ 0,08); cechy konta go nie przewidują, przewidują go dawne wyświetlenia autora: +0,10–0,15 z reakcjami, +0,14–0,17 w chwili publikacji.
- **Widok klienta (E73, E75):** sumy marek z dokładnością ×1,7 / ×2,0 / ×2,1, zasięg Brand24 na X ×10 za duży; wyświetlenia na lajk 11–64 na TikToku, 17–113 na YouTube, 25–279 na X zależnie od progu.
- **Pozostały błąd i przedziały (E76–E80):** błąd jest rozłożony na wszystkie posty (najgorsze bez reakcji), a nie na kilku autorów; jeden wcześniejszy post autora daje większość zysku z historii; poziomu autora nie widać z reakcji na innych postach; przedział 80% dla sumy marki to ×2,2 / ×3,0 / ×3,3 i jest skalibrowany.
- **FB / IG przed ręcznym sprawdzeniem (E81–E82):** reakcje na Instagramie najbardziej przypominają X i YouTube, a Facebook nie przypomina żadnej platformy; szacunki dla 312 postów z próbki są zapisane, zanim zobaczymy wyświetlenia (FB ok. 200–250, IG 35–95; zasięg Brand24 ×15–20 wyżej niż nasz na FB). **Atlas (E83):** reklamy, posty spoza USA, wielkie litery na X i popołudnie UTC na YouTube mają więcej wyświetleń przy tych samych reakcjach.
- **Anomalie (E74):** posty ×10 ponad prognozę to częściej reklamy, kopie, konta marek, powracający autorzy; historia autora wskazuje je 6–15× lepiej niż losowo.
- **Nie pomogło:** model wspólny i monotoniczny (E61), wszystkie pary sprawdzeń (E59), typ konta jako cecha (E66), podobne marki dla nowej marki (E69).

## Poprawki z 2026-11-02
- `extra_features.ratios` odwracało kolumny log1p podstawą 10. Naprawione, notebook 04 przeliczony. Główne wyniki bez zmian.
- „Rodziny są równe” jest prawdą dopiero po strojeniu i nie dla ridge na YouTube.
- „Zasięg na X = 0,1 × followersi” dotyczy tylko postów bez reakcji.

## Otwarte
Ludzki gold standard dla etykiet; czy zasięg to to samo co wyświetlenia; próbka z FB lub IG z prawdziwymi wyświetleniami (312 Reelsów i filmów do ręcznego sprawdzenia: `docs/pl/fbig_reczne_sprawdzenie.md`, B14); fałszywe dopasowania marek w całych danych; historia wyświetleń autorów od Brand24 (pytanie Q8, E68). Embeddingi zdaniowe i GNN są już sprawdzone (E50, E53, E65).

Planowane kolejne kroki: „Ideas to stand out” w `docs/en/project_plan.md`.
