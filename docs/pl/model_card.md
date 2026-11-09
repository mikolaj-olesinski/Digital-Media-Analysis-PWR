# Model card: estymator widoczności (wyświetleń)

Status: wewnętrzny, 2026-11-07. Opis estymatora i dowodów, nie tekst preprintu.
Odtworzenie: `docs/pl/reproduce.md` (sześć komend).

## Co estymuje

- **Cel:** `log10(view_count)` posta, czyli liczba wyświetleń raportowana przez platformę, w wieku, w którym odczytano
  metryki.
- **Jednostka:** jedna wzmianka (jeden post).
- **Zakres:** TikTok, YouTube i X (wyświetlenia są publiczne). Facebook Reels/videos: zmierzone na 130 ręcznie
  sprawdzonych i 402 pobranych postach (E93, E95). Instagram: **niezwalidowany** (brak publicznych wyświetleń bez
  logowania).
- **To nie jest zasięg:** wyświetlenia to nie unikalne osoby. Platformy nie publikują unikalnego zasięgu dla dowolnych
  postów; wyświetlenia są najbliższym publicznym przybliżeniem.

## Wejścia

- `log10(followers)`, `log1p(likes)`, `log1p(comments)`, `log1p(shares)` i flaga braku dla każdej (`ENG`).
- Tylko dla X: `log10(wiek w dniach)` w momencie sprawdzenia (`log_age`), zwęża pasmo i obniża MAE (E44, E141).
- Opcjonalnie, gdy znane są wcześniejsze wyświetlenia autora: `author_te`, ściągnięta średnia log-wyświetleń autora
  (E119).
- Nieużywane: tekst, etykiety, graf, kraj, czas, kalendarz, wielkość marki. Każde było testowane i daje ≤ 0,01
  (E100, E121, E122; wcześniej E05, E49, E84).

## Model

- CatBoost, `loss_function=MAE`, 300 iteracji, poza tym domyślne (E114, E141).
- Trening: próbka Brand24, 12 500 wzmianek, posty z `view_count > 0`; 2 491 TikTok, 2 441 YouTube, 1 803 X.
- Cechy budowane z wyświetleń (historia autora) tylko z foldów treningowych.

## Walidacja

- **Główna:** 5-fold CV po markach; marka nigdy w treningu i teście. MAE na log10, bootstrap po markach (300-1000
  losowań).
- **Powtórzone CV (E114):** TikTok 0,269 ± 0,011; YouTube 0,466 ± 0,030; X 0,463 ± 0,018.
- **Czasowa:** trening na wcześniejszych miesiącach, test na późniejszych; blisko CV po markach, gorzej tylko w
  pierwszym miesiącu (E46).
- **Mocniejsze grupowanie:** foldy rozdzielające też autorów i kopie tekstu zmieniają MAE o ≤ 0,002 na TikToku/YouTube
  i -0,011 na X (E55).
- **Sufit szumu:** posty o prawie identycznych followersach i reakcjach różnią się ×2,8 / ×6,0 / ×4,6; najlepszy
  możliwy MAE to co najmniej 0,22 / 0,39 / 0,33 (E56). Model jest blisko sufitu na TikToku, 0,06-0,14 nad nim na
  YouTube i X.
- **Kalibracja:** punkt jest wyrównany w środku skali; isotonic i przesunięcia per tier nie poprawiają go (E114).
- **Rolling-origin (E136):** trening przed każdym z ostatnich miesięcy, test na miesiącu: 0,270 ± 0,004 / 0,450 ± 0,018 /
  0,473 ± 0,005, blisko CV po markach. Walidacja nie jest optymistyczna w czasie.

## Wyniki

| platforma | MAE log10 | typowy błąd | szerokość 90% CQR | źródło |
|---|---|---|---|---|
| TikTok | 0,273 | ×1,9 | 1,14 | E114, E118 |
| YouTube | 0,456 | ×2,9 | 1,71 | E114, E118 |
| X | **0,454** | ×2,9 | **1,99** | E114, E118, E141 |
| Facebook (transfer, 130) | 0,345 | ×2,2 | — | E93 |
| Facebook (transfer, 402) | 0,385 | ×2,4 | 1,99 | E95, E123 |

- **Przedziały:** używać CQR. Pasmo 50% też jest skalibrowane w platformie (szerokość 0,45 / 0,74 / 0,83, E137/E141). Surowe kwantyle
  5-95% pokrywają tylko 74%; CQR osiąga 0,889 / 0,902 / 0,899 przy
  nominale 90% (E118). Na FB CQR pokrywa 0,925 przy szerokości 1,99 (E123). Bez etykiet celu opcją jest pasmo z reszt
  źródeł (E98). Przy publikacji, bez reakcji, pasmo 90% ma 2,4-3,0 log10 (E130); z reakcjami 1,1-2,1. Model
  dystrybucyjny CatBoost pod-coveruje (0,86-0,88) i nie jest używany (E129).
- **Budżet kalibracji:** X potrzebuje 25-50 postów celu (MAE 0,650 -> 0,551-0,563); TikTok, YouTube i Facebook są już
  skalibrowane i przesunięcie szkodzi przy małym n (E115). Klient nie musi zbierać postów.

## Gdzie zawodzi

- **Ogon:** model zawyża małe posty i zaniża wiralne (E96, E114); kalibracja tego nie naprawia (E114).
- **Braki reakcji przy predykcji:** 10% braków kosztuje 0,03-0,08 MAE, 50% kosztuje 0,08-0,37; X jest najbardziej
  odporny, TikTok najbardziej kruchy (E138). Jeśli cecha brakuje systematycznie, **przetrenować bez niej**: model
  wyspecjalizowany bije pełny (E163: TikTok bez reakcji 1,139 -> 0,583; YouTube bez followersów 0,711 -> 0,494).
  Przełączać przy ~35% braków reakcji (TikTok/YouTube) lub ~20% braków followersów (YouTube/X); na X przełączenie
  reakcji jest później (~70%) (E164).
- **Które dane są ważne:** TikTok potrzebuje reakcji (brak: 1,005), ale toleruje brak followersów (0,294); YouTube i X
  potrzebują followersów (0,711 / 0,733 bez nich) bardziej niż reakcji (E161, E138). Na FB sygnałem są reakcje;
  followersi pomagają tylko powyżej 1M (E162).
- **Zero reakcji:** 27% YouTube, 64% X; błąd 0,66 na tych postach; traktowanie zer jako braków daje tylko 0,003
  (E116). Post 0/0 przy pierwszym skanie ma jednak wyświetlenia: 86% > 0, mediana 5 (X) / 60 (TikTok), p90 49 / 179,
  0,4% > 1 tys. (E117, E126); przy zbieraniu p90 to 124 (X) i 3 383 (YouTube). Podawać zakres, nie punkt.
- **Poziom autora:** około dwóch trzecich pozostałego błędu to autor (E70); jego wcześniejsze wyświetlenia to
  największa dźwignia (E68, E119), ale tylko 1-8% próbki je ma. Bez nich model to model bazowy: prior marki nie zastępuje
  historii autora (E169: -0,005 / -0,035 / +0,006, tylko 35-52% postów testowych ma znaną markę).
- **Wiek sprawdzenia:** wyświetlenia rosną; bias FB +0,14 znika po normalizacji do 30 dni (E101).
- **FB/IG:** licznik mają tylko wideo/Reels; zwykłe posty nie. Karta FB pokazuje plays, a my używamy
  `video_view_count` (mediana różnicy ×3,6). Pasmo FB nie jest jednorodne: strony poniżej 1M followersów mają pokrycie 0,940,
  a powyżej 1M tylko 0,80-0,86 i nie da się ich skalibrować na 70 postach (E132, E144, E148, E151); górny kwartyl
  wyświetleń pokrywa 0,78 przy najwęższym paśmie (1,49, E132); nie podawać skalibrowanego pasma
  dla >1M, tylko zaznaczyć, że model zaniża średnią o około ×2. Model dużych stron poprawia punkt (0,368), ale pasmo
  zostaje na 0,80 (E150/E151).
- **Próbka:** 12 500 wzmianek od Brand24, nie losowa próbka social mediów; 85% angielskich; czas nierówny.

- **Poza eksportem:** na filmach marek i popularnych model jest ×5 za nisko (E189/E190, MAE 1,48 na popularnych), ale na
  filmach trzecich stron o marce — populacji, którą model ma obsługiwać — jest blisko eksportu: MAE 0,690 małe, 0,441
  średnie, zawodzi tylko warstwa popularna (1,569) (E191). Filmów z zerem nie ma przez kanały marek, ale zdarzają się
  wśród małych wzmianek trzecich stron (E190, E191).

## Zastosowanie

- Szacowanie widoczności posta do monitoringu, benchmarków i sum marek — zawsze z przedziałem.
- Sumy marek wymagają wielu postów (20 postów daje jeszcze rozrzut ×6,4 p90/p10, E112) i kalibracji (E109). Sumy FB
  z 19 marek są ×2 za niskie przed kalibracją, a ich 80% przedział pokrywa 0,737 (E157/E158).
- **Nie do alertów:** na top 1% ranking po surowych lajkach bije model (E104, E106); używać zaangażowania albo modelu
  precision@k.
- **Nie do rozliczeń ani gwarancji.**

## Ochrona danych i regulaminy

- Bez nazw autorów i URL-i w wynikach; tylko agregaty.
- Surowy plik Brand24 jest lokalny i git-ignored. Wyświetlenia FB czytane z publicznych stron (data statement do
  aktualizacji); Instagram wymaga logowania i nie był zbierany.

## Pliki

- Kod: `src/modeling.py`, `src/extra_models.py`, `src/reaction_tiers.py`, `src/validation.py`,
  `src/validation_audit.py`, `src/cqr_intervals.py`, `src/author_te.py`, `src/fb_transfer_check.py`.
- Liczby: `data/processed/*_numbers.json`.
- Dowody: `docs/en/experiments.md`, `agent/findings.md`, `agent/vault/`.

## Jak czytać liczby

- Wszystko jest na **log10 wyświetleń**. MAE 0,3 to typowy błąd ×2, 0,5 to ×3, 1,0 to ×10 (czynnik = 10^MAE).
- Szerokość pasma też jest w log10: 1,14 to czynnik ×14, 1,99 to ×98 (czynnik = 10^szerokość).
- Pasmo 50% to zakres „prawdopodobne”; pasmo 90% to uczciwy najgorszy przypadek. To nie to samo twierdzenie.
- Bias to średnia (prawda - estymata) w log10: +0,14 znaczy, że estymata jest za niska o ×1,4.
- „Pokrycie” to udział postów, których prawda wpada w pasmo; nominał dla pasma 90% to 90%.


## Sygnatura platform (reverse-engineering, E167)

| platforma | elastyczność followersów | elastyczność reakcji | posty z wyświetleniami > followersi |
|---|---|---|---|
| TikTok | 0,044 | 0,82 | 43% |
| YouTube | 0,303 | 0,66 | 5% |
| X | 0,164 | 1,06 | 14% |

Elastyczności z dopasowania log10(wyświetleń) na log10(followersów) i log10(1+reakcji) per platforma; OOF R2 0,81 / 0,59 /
0,47. TikTok jest napędzany feedem, YouTube subskrypcjami; reguła oparta na followersach jest więc strukturalnie błędna na
TikToku. E168 pokazuje to samo z drugiej strony: zasięg Brand24 da się w pełni odtworzyć jako regułę tylko na Instagramie
(97% wierszy), częściowo na X (62%) i Facebooku (49%), a wcale na TikToku/YouTube, gdzie jest liczbą wyświetleń.

## Figury

Figury wynikowe to `figures/cqr_bands_*`, `fb_coverage_*`, `reach_concentration_*`, `calibration_curve_*` i
`fb_rule_vs_model_*`, wszystkie budowane przez `src/result_figures.py` z plików JSON; które są żywe, a które historyczne,
jest w `scripts.md`.
