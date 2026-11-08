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
