# Mapa statusu dokumentacji

Wewnętrzne: który dokument trzyma **aktualne** liczby, a który jest **historyczny z założenia**. Zasada: gdy liczba się
zmienia, aktualizujemy model card, reporting guide i evidence map; starą wartość zostawiamy tylko w rejestrze i karcie
eksperymentu, który ją zmierzył.


Sekcje: w skrócie · karty legacy · co znaczy data w nagłówku · artefakty historyczne · zmiana starego wyniku · co waliduje
`check` · jak zmienić liczbę · typowe błędy · zamierzone powtórzenia · gdzie stare i nowe liczby współistnieją.
| dokument | liczby | zasada aktualizacji |
|---|---|---|
| `model_card.md` | aktualne (X v3: 0,454, pasma 1,99/0,83; FB 0,385) | aktualizować przy zmianie finalnego modelu |
| `estimator_spec.md` | aktualne | razem z model card |
| `reporting_guide.md` | aktualne | razem z model card |
| `evidence_map.md` | aktualne twierdzenia z siłą | gdy eksperyment zmienia twierdzenie |
| `status.md` | aktualny snapshot | odświeżać po każdym pushu |
| `one_pager.md`, `client_qa.md`, `slides.md` | aktualne | razem z model card |
| `preprint_outline.md` | tylko struktura | stabilne |
| `whats_new.md` | snapshot pushu XI 2026 | historyczne z założenia; nie aktualizować |
| `analysis_overview.md` | wczesne sekcje z notebooków; sekcja XI aktualna | wczesne sekcje to notatki historyczne |
| `experiments.md` (rejestr) | **historyczne**: każdy wiersz trzyma liczby z czasu pomiaru (np. E118 ma pasmo X v2 2,07) | tylko dopisywać; nie przepisywać starych wierszy |
| `agent/vault/E*.md` | historyczne per eksperyment | dopisać sekcję update, jeśli późniejszy eksperyment zmienia odczyt |
| `repro_map.md` | generowane z kart | regenerować po dodaniu karty |
| `scripts.md` | aktualne | gdy skrypty się zmieniają |
| `reproduce.md` | aktualne | gdy zmieniają się komendy lub zależności |
| `future_work.md` | aktualne priorytety | gdy pozycja jest zrobiona lub zablokowana |
| `data_statement.md` | aktualne | gdy zmienia się zbieranie danych |
| `codebook.md` | stabilne | tylko z nową rundą etykietowania |
| `literature_notes.md` | datowane wpisy, sprawdzone vs tylko wyszukiwarka | tylko dopisywać |
| `project_plan.md` | tylko EN, plan historyczny | nie aktualizować; `plan_status.md` opisuje, co się stało z każdym punktem |


## Jeśli liczba wygląda źle

1. Sprawdź „Typowe błędy przy cytowaniu liczb” powyżej; większość złych liczb to jedna z tych pomyłek.
2. Znajdź źródło: JSON w `data/processed/`, kartę `agent/vault/E*.md`, wiersz rejestru.
3. Jeśli zły jest **JSON**, popraw skrypt i uruchom go ponownie (nigdy nie edytuj JSON-a ręcznie).
4. Jeśli zła jest **proza**, zastosuj „Jak zmienić liczbę”.
5. Jeśli poprawka zmienia twierdzenie, dopisz notatkę w karcie i wiersz w rejestrze; uruchom `vault` i `check`.
## Karty legacy

Karty **E01-E37** mają „Legacy note”: 21 z nich nigdy nie zapisało JSON-a `numbers:`, a wszystkie 37 nie miały
wtedy zapisanej decyzji. Zostają dla historii, ale traktować je jako dowód wspierający i przed cytowaniem sprawdzić liczbę
w outputcie notebooka lub rejestrze.


## Karty legacy i czym je zastąpić

| twierdzenie legacy (E01-E37) | cytować zamiast |
|---|---|
| próbka i dostępność wyświetleń (E01) | `data_statement.md`, E105 |
| etykiety i zgodność koderów (E04) | `data_statement.md` (kappa) |
| rodziny modeli i tuning (E11, E12) | E114 |
| parametry CatBoost (E32) | E114, E141 |
| cechy tekstowe (E05) | E100 |
| cechy grafowe (E06) | E122, E127, E128 |
| ogon/bias (E18) | E96, E114 |



## Co znaczy data w nagłówku

- **Model card, specyfikacja, status, findings:** *data stanu* — dzień, w którym liczby ostatni raz sprawdzono z
  plikami. Wszystkie mają tę samą datę (2026-11-07).
- **Wiersze rejestru i karty:** *data eksperymentu* — kiedy został wykonany; może być starsza.
- **Literature notes:** *data przeglądu* — kiedy przeszukano partię, nie stan projektu.
- **What is new:** snapshot okresu z założenia; zakres dat to okres, który opisuje.
- **Data statement:** bez daty stanu; opisuje pliki aktualnie używane (patrz „Gdy dane się zmieniają”).
## Artefakty historyczne

- **Karta eksploracji M1** (`docs/en|pl/m1_exploration_card.html|pdf`): kamień milowy z sesji 2; jej liczby to stan z
  tamtego czasu (np. X 0,47, przed v3).
- **Polski przewodnik HTML** (`src/guide_build.py`): opisuje analizy do 2026-11-04, z notą na górze.
- **Decki prezentacyjne**: tylko tekst i notatki mówcy, bez liczb modelu.

## Gdy nowy eksperyment zmienia stary wynik

1. Dodać nową kartę i wiersz w rejestrze (oba języki); nowa karta ma JSON.
2. Dopisać sekcję **Update** na końcu starej karty: data, ID nowego eksperymentu, co się zmienia, co nadal obowiązuje.
   Nigdy nie usuwać ani nie przepisywać starego wyniku.
3. Oznaczyć stare twierdzenie jako zastąpione w `agent/findings.md` (zostawić wiersz, dodać „superseded by E<nowe>”).
4. Zaktualizować bieżące dokumenty zgodnie z „Jak zmienić liczbę”; stara wartość zostaje tylko w rejestrze i starej
   karcie.
5. Jeśli dokument cytuje starą kartę, zamienić cytowanie na nowe (tabela legacy powyżej pokazuje mapowanie).
## Co waliduje `check`

`python src/agent_tools.py check` sprawdza:

- rejestry EN i PL mają te same ID, bez duplikatów;
- każdy wiersz rejestru ma 7 kolumn i typ wyniku ze znanego słownika (osobne zbiory EN i PL);
- każda ścieżka z rejestru istnieje, każda ścieżka `numbers:` i `notebooks:` z karty istnieje i każdy wiersz ma kartę w vaultcie;
- każda karta ma sekcje `## Result` i `## Decision`, a karta z `numbers: []` ma „Legacy note”;
- żaden śledzony plik `.md`/`.py` ani komórka źródłowa notebooka nie zawiera URL-a posta społecznościowego (**ostrzeżenia**, nie błędy);
- każda figura `*_en.png` ma bliźniaka `*_pl.png` (serie tylko PL, np. story, są dozwolone);
- notebooki mają zapisane outputy, bez błędów, URLi i kolumn z danymi osobowymi (dokumenty mogą cytować arxiv/github; prawdziwe URL-e postów ostrzegają);
- tabela pytań do klienta ma 5 kolumn w każdym wierszu;
- `data/raw/`, `logs/`, `AGENTS.md` i pobrane CSV nie są śledzone przez git, a żaden śledzony CSV nie ma kolumny
autora/URL (sprawdzane z nagłówka pliku); backlog istnieje.

**Nie** sprawdza treści karty (pytania, wyniku, decyzji) ani tego, czy liczby w prozie zgadzają się z JSON-ami; to robią
ręczne procedury poniżej.

## Jak zmienić liczbę

1. Jeśli liczba jest nowa, dodaj kartę w `agent/vault/`, wiersz w rejestrze i JSON w `data/processed/` (najpierw skrypt).
2. Zaktualizuj **model card** (źródło prawdy dla aktualnych liczb).
3. Zaktualizuj estimator spec, reporting guide i evidence map; data statement, jeśli zmieniły się dane.
4. Zaktualizuj dokumenty powtarzające liczby: story, one-pager, client Q&A, slides, status.
5. Przegeneruj figury: `python src/result_figures.py en` i `pl`.
6. Uruchom `python src/agent_tools.py vault && python src/agent_tools.py check`.
7. Starą wartość zostaw tylko w rejestrze i karcie eksperymentu, który ją zmierzył; nie przepisuj starych wierszy.

## Typowe błędy przy cytowaniu liczb

Liczby, nie pojęcia; lista pojęć jest w `faq.md` („Częste nieporozumienia”).

- **X v2 vs v3**: 0,467 i pasmo 2,07 (E118) to stary model; aktualne X to 0,454 i 1,99 (E141).
- **130 vs 402 posty FB**: 0,345 to próbka ręczna (E93); 0,385 to pełna próbka (E95) i tę liczbę się cytuje.
- **Błąd reguły na FB**: ×14 (130 ręcznych, E93) vs ×4,1 (402, E111); używać ×4,1.
- **Pasmo FB**: 0,94 dla stron <1M, 0,80-0,86 powyżej 1M; nie cytować jednego jako drugiego.
- **Publikacja vs zbieranie**: 0,60-0,76 MAE przy publikacji, nie 0,27-0,46; pasma 2,4-3,0 vs 1,1-2,0.
- **Pasmo 50% vs 90%**: ×2,8-7,6 vs ×14-98; zdanie mieszające je jest błędne.
- **Sumy marek**: E109 to X, E157/E158 to Facebook; nie mieszać próbek.
- **„Zasięg” vs „wyświetlenia”**: estymator daje wyświetlenia; unikalny zasięg nie jest publiczny.

## Zamierzone powtórzenia

`story.md`, `one_pager.md`, `client_qa.md`, `slides.md` i `status.md` celowo powtarzają nagłówkowe liczby dla różnych
odbiorców. Gdy zmienia się liczba nagłówkowa, najpierw aktualizujemy model card, potem te pięć; rejestr trzyma wartość
historyczną.

## Znane miejsca, gdzie stare i nowe liczby współistnieją

- **Pasmo X**: 2,07 (E118, v2) w rejestrze vs 1,99 (E141, v3) w model card; aktualne jest w karcie.
- **MAE X**: 0,467 (E118/E114) vs 0,454 (E141); aktualne jest w karcie.
- **Followersi FB**: E94 mówi „szkodzą” na próbce 130; E149/E154 pokazują, że pomagają powyżej 1M; niuans jest w karcie
  i w sekcji update E94.
- **Błąd reguły FB**: E93 dał ×14 na 130 ręcznych, E111 ×4,1 na 402; karta używa 0,93 MAE / ×4,1.
- **Sumy marek**: E109 (X) i E157/E158 (FB) to różne próbki; nie mieszać ich w jednym zdaniu.
