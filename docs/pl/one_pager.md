# One-pager dla Brand24

**Pytanie.** Czy widoczność (wyświetlenia) posta da się oszacować lepiej niż stałą i z jaką niepewnością?

## Co zrobiliśmy

- 12 500 wzmianek Brand24 (po 2 500 z TikToka, YouTube, X, Facebooka, Instagrama); publiczne liczniki wyświetleń są na
  TikToku, YouTube i X i służą jako prawda.
- Finalny model: CatBoost na followersach, lajkach, komentarzach, udostępnieniach i flagach braków; walidacja po markach;
  przedziały CQR.
- Facebook zmierzony osobno: 130 ręcznie sprawdzonych i 402 pobrane publiczne Reels/videos.

## Wyniki

| platforma | MAE modelu | typowy błąd | pasmo 50% | pasmo 90% | wasza reguła |
|---|---|---|---|---|---|
| TikTok | 0,273 | ×1,9 | ×2,8 | ×14 | dokładna (to wyświetlenia) |
| YouTube | 0,456 | ×2,9 | ×5,5 | ×51 | dokładna (to wyświetlenia) |
| X | 0,454 | ×2,9 | ×7,6 | ×98 | ×9,3 za wysoko |
| Facebook | 0,385 | ×2,4 | — | ×98 (poniżej 1M) | ×4,1 za wysoko |

- Na Facebooku model bije regułę w **każdym** tierze followersów (MAE 0,36-0,46 vs ~1,0).
- Pasma są skalibrowane w platformie (90% pokrywa 0,89-0,90); surowe kwantyle pokrywałyby tylko 74%.

## Co rekomendujemy

1. **Model do poziomów i przedziałów** na X i Facebooku; regułę zostawić na TikToku/YouTube (to licznik wyświetleń).
2. **Raportować zakresy, nie punkty**: pasmo 50% („prawdopodobne”) i 90%.
3. **Nie używać do alertów**: surowe lajki / wasza reguła rankują lepiej na top 1%.
4. **Sumy marek**: tylko z przedziałem i odpowiednią liczbą postów (20 postów daje jeszcze rozrzut ×6,4).
5. **Braki danych**: przetrenować bez brakującej cechy (TikTok bez reakcji 1,14 -> 0,58).

## Czego nie możemy zrobić

- Instagram: brak publicznych wyświetleń (0 z 160 prób anonimowych); bez logowania lub dostawcy nie ma estymaty.
- Zwykłe posty na Facebooku (nie wideo/Reel): brak licznika.
- Skalibrowane pasmo dla stron FB powyżej 1M followersów (pokrycie 0,80-0,86 przy 70 postach).
- Unikalny zasięg (osoby): publiczne są tylko wyświetlenia.

## Czego potrzebujemy od was

1. Potwierdzenia stałych za `b24_estimated_reach`.
2. Informacji, czy zero wyświetleń na YouTube może oznaczać „nie zebrano”.
3. Decyzji o Instagramie: próba ręczna, dostęp z logowaniem albo dostawca.

Liczby: `docs/pl/evidence_map.md`; szczegóły: `docs/pl/model_card.md`; pytania: `docs/pl/client_qa.md`.
