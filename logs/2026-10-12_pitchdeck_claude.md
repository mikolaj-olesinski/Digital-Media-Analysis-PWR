# Log sesji: pitch deck (agent: claude, Claude Code)

- Data: 2026-10-12. Osoba: właściciel repozytorium (rola do uzupełnienia w AGENTS.md §4).
- Cel: pitch deck ~5 min, 6–7 slajdów, po polsku, z notatkami prelegenta. Prompt: `prompts/pitchdeck.md`
  (wklejony jako treść do wykonania). Wynik: `pitchdeck/claude/`.

## Decyzje
- Narzędzie: python-pptx (doinstalowane przez pip; nie było na maszynie). Renderowanie: PowerPoint przez COM
  (brak LibreOffice).
- Liczby (12 500, 2 500 na platformę, % `view_count`) liczone w `build.py` z CSV; nic nie wpisane z pamięci.
- Wartość stałej Brand24 pominięta zgodnie z promptem (jedno wspomnienie bez liczb: „obserwujący × stała”).
- 7 slajdów; proporcja: zastosowanie + „co/jak” > dane (jeden slajd).
- Notatki: czas liczony automatycznie z liczby słów (~145 słów/min); pierwsza wersja miała 576 słów (~4 min),
  rozbudowano do ~660 słów (~4 min 35 s).

## Problemy
- `python-pptx` brak → `pip install python-pptx`.
- Heredoc w Bash psuł polskie znaki w skrypcie poprawek → poprawki zapisane jako plik UTF-8 (scratchpad).
- Render 1: oś czasu na slajdzie 7 wychodziła poza slajd; ikony na slajdzie 3 (strzałka L/P, dokument) źle się
  renderowały; osierocone słowo na slajdzie 2. Wszystko poprawione, ponowny render sprawdzony.
- Nie usuwano starych PNG (blokada), eksport nadpisał pliki o tych samych nazwach.

## Pliki
`pitchdeck/claude/{build.py,pitchdeck.pptx,pitchdeck.pdf,NOTATKI.md,render/*.PNG}`

## Aktualizacja: przeprojektowanie i notebook EDA
- Slajdy przeprojektowane (nowa paleta, bardziej ilustracyjne); na prośbę użytkownika więcej o problemie,
  jego przyczynie i korzyściach, mniej o danych (dane zeszły do paska na slajdzie 5; nowy slajd 3 o konsekwencjach).
- Dodano `notebooks/01_eda.ipynb` (EN, wykonany): schemat, braki, duplikaty, czas, ground truth, rozkłady,
  outliery, zachowanie mnożnika Brand24, korelacje, wstępny test wykonalności (CV), podsumowanie.
  Notebook nie wypisuje danych osobowych (author_name/url/url).
- Błąd po drodze: `git add pitchdeck` złapał cudzy folder `pitchdeck/codex` z node_modules (8542 plików);
  cofnięto lokalny commit (`reset --soft`), dodano `node_modules/` do .gitignore i ponowiono commit tylko z `pitchdeck/claude`.

## Aktualizacja: dopracowanie wizualne (Claude Opus 5.5)
- Prośba: „niech jeszcze opus mi ją ładnie doszlifuje, zrobi ładniejszą, efekt wow”. Treść i notatki bez zmian.
- Dodano w `build.py`: przejścia Morph (XML `p159:morph`, fallback fade), animacje wejścia (XML `p:timing`, preset
  „Float In”, auto po poprzednim), cienie (`a:outerShdw`), gradienty, półprzezroczyste koła `!!b1–3` i pasek
  postępu `!!progress` łączone przez Morph między slajdami; elementy kart zgrupowane, by animowały się razem.
- Weryfikacja: PowerPoint (COM) otwiera plik bez naprawy, rozpoznaje przejścia i animacje na każdym slajdzie;
  zrzuty ekranu w trakcie pokazu potwierdziły ruch Morph i kolejne wejścia. Rendery PNG obejrzane;
  usunięto numer slajdu (nachodził na koło) i boczny pasek kart na slajdzie 3.

## Aktualizacja 2026-10-13: krótszy notebook EDA
- Prośba: „zrób krótszy, mniej tekstu, bardziej przejrzysty: jakie dane, z czego”.
- `notebooks/01_eda.ipynb` przepisany: 7 kroków zamiast 11, 5 wykresów zamiast ok. 12, tabela ról kolumn
  (cel / ground truth / predyktory / kontekst / dane osobowe nieużywane), pod każdym krokiem jedno zdanie „➡️ wniosek”.
  Usunięto m.in. sekcję czasu, outlier z-score, rozbudowane kontrole, gradient boosting, pola kontekstowe.
- Liczby w wnioskach porównane z wynikami komórek (MAE 0,93→0,28 TikTok, 0,83→0,57 YouTube, 0,78→0,48 Twitter itd.).
- HTML eksportowany bez kodu (`--no-input`) jako czytelny raport.
- Kolejna prośba: „więcej wykresów, zero emotek”. Usunięto emotki; wykresów z 5 do 11: wzmianki w czasie + języki + marki,
  reach vs views (TikTok/YouTube/Twitter), histogramy reach per platforma, krzywa koncentracji zasięgu,
  stosunek reach/followers wg decyla konta, scatter reach vs followers/likes, przewidywane vs prawdziwe views.

## Aktualizacja 2026-10-13: deck pod szablon prowadzącego
- Prośba: dostosować prezentację do maila prowadzącego (hipoteza, koncepcja, klient-społeczeństwo, impact, moat),
  „bardzo nie zmieniać”. Zachowano wygląd, slajdy 1–2 i koncepcję (dawny slajd 5).
- Nowe: slajd 3 hipoteza (z falsyfikacją, wizual „dziś vs pomysł”), slajd 6 impact (lejek 5,24 mld → 2 400+ → ~12 000),
  slajd 7 wykonalność + stan wiedzy (4 źródła). Slajd 5 przerobiony na grupy społeczne. Usunięto „Co tracimy”,
  „Co z tego wyjdzie”, „Plan na semestr”.
- Liczby do impaktu z wyszukiwania (DataReportal 2025, strona klientów Brand24). Cytowania naukowe z wiedzy agenta —
  do weryfikacji przez zespół.
- Wcześniej w tej sesji: odtworzenie reguły `b24_estimated_reach` z danych (TikTok/YouTube = views; X: 0,1×followers
  albo ~700×interakcje; FB: progi 10/6/5/4/2% wg wielkości konta lub 100×interakcje; IG: 20×(likes+comments)).

## Aktualizacja 2026-10-19: karta eksploracji M1 (mail prowadzącego z 16.10)
- Prośba: przerobić prezentację na kartę M1 (PDF, 2-3 str.), „zrobić to porządniej” niż kolega: reach = views na
  TikToku/YouTube to konstrukcja Brand24, a nie wynik, więc traktowane jako założenie, nie walidacja.
- Dwóch subagentów: (1) istniejące dane (brak publicznych views dla FB/IG; YouNiverse, INFORMS 2024 do weryfikacji),
  (2) literatura (Cha 2010, Szabo 2010, Wies 2023, Cheng 2014, Lunde 2023...). Raporty oparte na snippetach, cytowania
  do sprawdzenia przez zespół przed preprintem.
- Nowe: `src/m1_figures.py` (wykres + `data/processed/m1_numbers.json`), `docs/M1_karta_eksploracji.{html,pdf}`.
- Kluczowy wynik: reguła Brand24 da się zestawić z views tylko na X (log-MAE 1,08 vs model 0,47 vs stała×obserwujący 0,88).
- Do uzupełnienia przez zespół: imiona i role, link do repo. Tekst karty to szkic AI, do przepisania własnymi słowami
  i ujawnienia użycia AI (AGENTS.md §2).
