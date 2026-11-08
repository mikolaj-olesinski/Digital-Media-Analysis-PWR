# Zewnętrzna próbka dla YouTube, X i TikToka

Cel: próbka postów pobrana **niezależnie od eksportu Brand24**, żeby sprawdzić model poza danymi, na których się uczył.
Te same marki, ten sam okres, ten sam rodzaj treści (posty związane z marką), ale posty wybieramy my, nie eksport klienta.

## Wykonalność (sprawdzone 2026-11-08, `yt-dlp 2026.10.26`)

| platforma | metoda | status |
|---|---|---|
| **YouTube** | `yt-dlp "ytsearchN:<marka>"`, pełne metadane (wyświetlenia, lajki, komentarze, followersi kanału, data) | **działa** |
| **TikTok** | wyszukiwanie / strony hashtagów w `yt-dlp` | **nie**: yt-dlp oznacza TikToka jako zepsutego („No working app info”); potrzebny login albo Research API |
| **X** | `yt-dlp` na URL posta | **nie**: x.com odpowiada HTTP 520/blokuje; wyszukiwanie wymaga płatnego API albo loginu |
| **Instagram** | anonimowe pobieranie | **nie**: 0/160 postów pokazuje wyświetlenia (B14); potrzebny login albo dostawca |

Czyli jedyna zewnętrzna próbka, jaką dziś zbierzemy bez poświadczeń, to **YouTube**. TikTok i X wymagają klucza API,
zalogowanej sesji albo decyzji zespołu o zbieraniu ręcznym; plan poniżej.

## Projekt YouTube (zaimplementowany, E189)

- **Rama:** 12 marek z największą liczbą wzmianek w eksporcie (mieszanka konsumenckich i B2B: Adidas, The North Face,
  eToro, Mollie, PayPal, Motorola, Hootsuite, Stripe, Avanza, Coinbase, MailerLite, Monster Energy).
- **Zapytanie:** nazwa marki w wyszukiwarce YouTube; bierzemy 5 najczęściej oglądanych filmów na markę, które przejdą filtry.
- **Filtry:** data publikacji >= 2025-01-01 (okres Brand24), czas <= 15 min, wyświetlenia > 0.
- **Pola:** wyświetlenia, lajki, komentarze, followersi kanału, data, czas, id kanału (nigdy nazwa kanału), id filmu.
  Pobrany CSV zostaje lokalnie (`data/processed/external_youtube_fetched.csv`, git-ignored).
- **Podsumowanie:** `data/processed/external_youtube_numbers.json`; karta E189.

### Znane obciążenia (raportować razem z wynikiem)

- Wyniki wyszukiwania nie są losową próbką wzmianek: kanały marek i recenzje są nadreprezentowane, małe posty trzecich
  stron niedoreprezentowane.
- Próbka nie jest warstwowana po reakcjach, więc nie da się jej porównać jeden do jednego z youtube'ową próbką Brand24.
- Małe n (około 60 filmów); służy do kontroli relacji wyświetlenia-reakcje i błędu modelu, nie do precyzyjnej estymaty.

## Plan dla X i TikToka (wymaga decyzji)

1. **X:** czysta droga to oficjalne API (płatne, `recent search` z `public_metrics`); darmowa droga to zalogowana sesja, co
   łamie regulamin przy automatycznym zbieraniu. Rekomendacja: zapytać Brand24 o mały eksport publicznych metryk (oni je
   zbierają) albo zabudżetować API.
2. **TikTok:** Research API (darmowe dla akademii, akceptacja tygodnie) albo dostawca; yt-dlp nie wyszuka bez loginu.
   Rekomendacja: złożyć wniosek o Research API równolegle z projektem; nie scrapować.
3. **Instagram:** bez zmian (B14): próbka ręczna, login dla człowieka albo dostawca.

## Co to daje

Jeśli relacja wyświetlenia-reakcje utrzyma się na zewnętrznej próbce YouTube, twierdzenie o transferze nie jest artefaktem
eksportu Brand24. Jeśli się nie utrzyma, dowiadujemy się, że youtube'owe wiersze eksportu są specyficzne — oba wyniki są
wynikami.

## Zróżnicowane warstwy YouTube (E190)

Samo wyszukiwanie zwraca tylko popularne filmy. Kolektor zróżnicowany losuje cztery warstwy na markę: najczęściej oglądane
z wyszukiwania, najnowsze z wyszukiwania, najsłabiej oglądane z kanału marki i najnowsze z kanału marki. Dwie ostatnie
zawierają filmy małe i z zerową liczbą wyświetleń, których w wyszukiwaniu nie ma. Wyniki:
`data/processed/external_youtube_diverse_numbers.json`, karta E190.

## Koszt API X (sprawdzone 2026-11-08)

X nie sprzedaje już abonamentów miesięcznych: Basic (200 USD/mies.) i Pro (5 000 USD/mies.) wycofano w lutym 2026, nie ma
też darmowego odczytu. Publiczne API działa w modelu pay-per-use: **0,005 USD za odczyt posta**, 0,010 USD za odczyt
użytkownika, 0,015 USD za zapis posta (0,200 USD, gdy post zawiera URL), z limitem 3 mln odczytów postów miesięcznie przed
kontraktem Enterprise. Dla naszej zewnętrznej próbki 2 500 postów X z danymi autora to około **12,50 USD za posty + 25 USD
za autorów = 37,50 USD** (jednorazowo). Źródła: strona cenowa X API (docs.x.com/x-api/getting-started/pricing) i
podsumowanie stawek 2026 (twitterapis.com/answers/x-api-pricing-tiers-explained).


### Wynik zróżnicowanych warstw (E190)

131 filmów, 12 marek: 0 filmów z zerem, minimum 71 wyświetleń, jeden poniżej 100. Mediana wyświetleń per warstwa:
popularne 1 544 176, świeże 15 661, najnowsze 3 943, małe 876. MAE modelu: małe 0,289 (bias +0,17), świeże 0,620,
najnowsze 0,601, popularne 1,480 (bias +1,48, ~×30 za nisko). Filmów z zerem nie da się pobrać.
