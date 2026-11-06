# Przykładowy raport dla marki

Układ ilustracyjny, nie prawdziwe posty: liczby poniżej są wymyślone, żeby pokazać format; nie ma nazw autorów ani
URLi. Jednostki to wyświetlenia; pasma 90%, jeśli nie zaznaczono inaczej.

## Marka: przykład (6 postów, 2 platformy zmierzone + 2 transferowane)

| post | platforma | odczyt | reakcje | estymata | prawdopodobne (50%) | pasmo 90% | flaga | uwaga |
|---|---|---|---|---|---|---|---|---|
| P1 | TikTok | 2 dni | 420 lajków, 18 komentarzy | 12 tys. | 4,3-34 tys. | 0,9-170 tys. | — | najlepszy przypadek, w platformie |
| P2 | YouTube | 3 dni | 95 lajków, 4 komentarze | 5 tys. | 0,9-28 tys. | 0,1-255 tys. | brak udostępnień | szersze pasmo |
| P3 | X | 2 dni | 60 lajków, 2 komentarze | 2 tys. | 0,3-15 tys. | 20-196 tys. | stary odczyt, X | najmniej pewny |
| P4 | Facebook (Reel) | 4 dni | 300 reakcji, 40 komentarzy | 8 tys. | — | 80-800 tys. | transfer | licznik FB to `video_view_count` |
| P5 | Facebook (strona >1M) | 3 dni | 1 800 reakcji | 1,2 mln | — | **tylko punkt** | duża strona | pasmo nieskalibrowane |
| P6 | X | 1 godzina | 0 reakcji | — | — | — | za wcześnie | mediana 5, p90 49 w tym wieku |

## Jak to czytać

- P1-P3 to estymaty w platformie. Pasmo P3 jest szerokie, bo X ma 25% braków i wiele postów bez reakcji.
- P4 używa modelu transferu Facebooka; tylko wideo/Reel ma licznik wyświetleń.
- P5 to przypadek, w którym **nie dajemy pasma**: powyżej 1M followersów pokrycie 90% to 0,80-0,86, więc to estymata
  punktowa z zastrzeżeniem.
- P6 nie jest szacowany wcale: przy zerze reakcji po godzinie uczciwą odpowiedzią jest zakres empiryczny, nie punkt.

## Suma (co wysłać klientowi)

- Suma estymat: **~1,23 mln wyświetleń** (zdominowana przez P5); bez P5 reszta to ~27 tys.
- Podawać sumę z tymi samymi zastrzeżeniami co części i powiedzieć, z ilu postów powstała. Pięć postów daje rozrzut
  ×53 p90/p10, dwadzieścia wciąż ×6,4 (E112), więc suma z sześciu postów to stwierdzenie rzędu wielkości.

## Zasady formatu

- Jeden wiersz na post; nigdy nie łączyć platform w jednym wierszu.
- Pokazywać wiek odczytu; post czytany po godzinie nie jest porównywalny z czytanym po 4 dniach.
- Oznaczać estymaty transferowane i pasma nieskalibrowane.
- Bez nazw autorów i URLi; ID postów są wewnętrzne.

## Przed wysłaniem (sześć kontroli)

1. Każdy wiersz ma punkt z pasmem albo mówi „tylko punkt” / „za wcześnie”.
2. Wiek odczytu jest w wierszu; żaden wiersz nie miesza momentów.
3. Estymaty transferowane i pasma nieskalibrowane są oznaczone.
4. Suma to suma wierszy, a liczba postów jest podana.
5. Brak nazw autorów, URLi i zrzutów profilu.
6. Flagi jakości danych (braki reakcji, stary odczyt, X, zero reakcji) są tam, gdzie pasują.
