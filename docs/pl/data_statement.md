# Data statement

## Dane
- `data/raw/reach_samples_students.csv`: 12 500 wzmianek o markach od Brand24 (po 2 500 z X, Instagrama, YouTube, Facebooka i TikToka), 348 marek, utworzone 2025-01 do 2026-10 (około 85% z kwietnia-sierpnia 2026). Surowy plik jest tylko do czytania i nie jest commitowany.
- Pola: platforma, tekst (tytuł/treść), pola autora, followersi, lajki/komentarze/udostępnienia, `view_count`, `b24_estimated_reach` i `sentiment` Brand24.
- Prawda dla widoczności (`view_count`) istnieje tylko na TikToku, YouTube i X (75%). Facebook i Instagram jej nie mają.

## Zbieranie i próbkowanie
Próbkę dostarczył Brand24 na potrzeby kursu; sposób wyboru wzmianek nie jest nam znany. Nie jest jednorodna w czasie, języku (85% angielski) ani markach, więc wyniki opisują tę próbkę, a nie całe social media.

## Etykiety dodane przez nas (instrument AI)
- `data/processed/mention_labels.csv`: typ wzmianki i sentyment dla wszystkich 12 500 wzmianek, zdefiniowane w `codebook.md`.
- Tworzone przez trzy niezależne kodery Haiku; spory (37% wzmianek) rozstrzygał adjudykator Sonnet. Etykiety per koder: `mention_labels_coders.csv`.
- Rzetelność: kappa parami 0,77-0,79 (typ) i 0,71-0,73 (sentyment). To zgodność między modelami, **nie** dokładność względem człowieka. Członek zespołu przejrzał próbkę etykiet ręcznie i uznał je za spójne, ale było to nieformalne i niepunktowane, więc nie ma liczby zgodności człowiek-AI; etykiety są instrumentem AI zwalidowanym tylko zgodnością maszyna-maszyna i ręcznym sprawdzeniem.
- Własne pole `sentiment` Brand24 ma 42% braków; nasze etykiety sentymentu są osobne i kompletne.

## Znane ograniczenia
- Braki zależą od platformy i częściowo od czasu zbierania (patrz `notebooks/eda/02_platforms.ipynb`). Braki zaangażowania idą z małymi postami.
- Zasięg Brand24 to deterministyczna reguła na X, Facebooku i Instagramie, więc jest baseline'em do porównania, nie prawdą.
- Posty są publiczne, ale autorzy to osoby prywatne.

## Dane osobowe i RODO
- `author_name`, `author_url`, `url` i teksty postów to dane osobowe lub potencjalnie identyfikujące. Nie publikujemy ich, nie wymieniamy prywatnych autorów w preprintcie ani na posterze, raportujemy tylko agregaty.
- Teksty postów były wysyłane do koderów LLM w narzędziach projektu (tekst przycięty do 400 znaków, bez pól autora). To użycie musi być ujawnione w preprintcie.

## Dodatkowe dane zebrane przez zespół (listopad 2026)

- **Wyświetlenia Facebooka.** 402 publiczne Reels/videos FB z próbki czytane 2026-11-05 (fetcher: `src/fetch_views.py`) z publicznej strony `facebook.com/watch/?v=<id>` (bez logowania, desktop user agent). Strona zawiera `video_view_count` (licznik, którego używamy), `reaction_count` i `total_comment_count`. Widoczny licznik odtworzeń to inne pole (`play_count`, mediana ×3,6 większa w naszej próbce), więc nie są wymienne. 68 próbkowanych URLi zniknęło lub było ograniczonych.
- **130 ręcznie sprawdzonych postów.** Członek zespołu otworzył 130 z tych stron i zapisał wyświetlenia/odtworzenia; sparsowane wartości zgadzają się ze stroną. To próbka ręczna używana w E93.
- **Instagram.** Próba anonimowa nie zwróciła wyświetleń (0 z 160 sprawdzonych postów); Instagram wymaga logowania do wyświetleń, więc nie zebrano danych o wyświetleniach z Instagrama. Lajki/komentarze dla 115 z 160 postów zapisano z publicznych stron.
- **Instagram, po zalogowaniu (2026-11-08).** 500 losowych rolek z wycinka Instagrama od Brand24 (11 marek) odczytał agent przeglądarki na własnym, zalogowanym koncie członka zespołu: wyświetlenia, lajki i komentarze w tym samym momencie, tylko odczyt, wartości z webowego API Instagrama, które ładuje strona. 368 miało wyświetlenia (132 usunięte, 30 z ukrytymi lajkami). Wyniki są lokalne i ignorowane przez git (`data/manual/ig_browser_*.csv`); publikujemy tylko statystyki (E202). Nazwa pola licznika nie została zapisana.
- **Posty `/p/` na Instagramie.** 62% wycinka IG (1 550 z 2 500) to posty `/p/`. Spośród pierwszych 26 sprawdzonych 19 to zdjęcia lub karuzele bez publicznego licznika wyświetleń (`media_type` 1 / 8), 7 usunięto, żadne nie było wideo. Wyniki dla Instagrama obejmują więc tylko rolki; zwykłe posty nie mają ground truth.
- **Regulaminy i dane osobowe.** Czytano tylko publiczne strony, bez logowania i bez cookies; nie publikujemy nazw ani URLi autorów. Pobrane pliki są lokalne i git-ignored (`data/processed/*fetched*.csv`); zawierają mention_id i liczniki, nie URL posta. W repo są tylko agregaty.
- Publiczne dane o markach z Wikimedia (E86, E170, E171): miesięczne i dzienne wyświetlenia artykułów angielskiej Wikipedii oraz fakty o firmach z Wikidata (zatrudnienie, przychód, rok założenia) dla 293 artykułów marek. Na zewnątrz poszły tylko nazwy marek / tytuły artykułów (zgoda użytkownika 2026-11-05 i 2026-11-06); bez autorów, URL-i i treści postów. Pliki: `data/processed/brand_wikipedia.csv`, `brand_wiki_daily.csv`, `brand_wikidata.csv`.

## Prośby o usunięcie

Jeśli osoba prosi o usunięcie posta lub autora:

1. Znaleźć go w lokalnych plikach (`data/processed/fb_fetched.csv`, `fb_all_fetched.csv`, ewentualny arkusz ręczny) i
   usunąć lub pseudonimizować wiersz; surowy plik tylko do czytania należy do klienta i nie jest edytowany.
2. Ponownie uruchomić agregaty, które go zawierały, żeby żadna publikowana liczba nie zależała od usuniętego wiersza.
3. Sprawdzić, czy żadna figura, output notebooka ani dokument nie wspomina autora lub URL-a.
4. Odpowiedzieć, co usunięto i które liczby się zmieniły.

Nie możemy usuwać rekordów w Brand24; taka prośba idzie do klienta. Usunięcie nigdy nie unieważnia metody, tylko
dotknięte agregaty — i tak mówimy.

## Gdy dane się zmieniają

- **Nowy eksport Brand24** dostaje nazwę z datą (`..._2027-01.csv`); starego surowego pliku nigdy nie nadpisujemy.
  Etykiety uruchamiamy ponownie tylko, jeśli zmienił się zbiór wzmianek; przeliczamy zależne liczby, aktualizujemy ten
  statement i robimy `check`.
- **Nowe pobranie Facebooka** też dostaje plik z datą; wyświetlenia rosną z wiekiem, więc nie mieszamy liczników
  czytanych w różnym wieku w jednym eksperymencie. Przed zastąpieniem starego pliku porównać drift (listopadowe odczyty
  2026 były 1,07-18× zapisanego snapshotu, E95).
- **Nigdy nie nadpisywać** `data/raw/` ani pobranego pliku w miejscu: trzymać obie wersje i mówić, której używa dana
  liczba.
