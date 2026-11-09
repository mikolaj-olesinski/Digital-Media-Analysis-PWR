# Próbka ręczna dla Instagrama i TikToka

Ani Instagram, ani TikTok nie pokazują liczby wyświetleń anonimowemu pobieraczowi (Instagram 0/160, B14; ekstraktor
TikToka w `yt-dlp` jest oznaczony jako zepsuty). Dopóki nie ma API, najtańszą zgodną z regulaminem drogą jest **próbka
ręczna**: człowiek otwiera publiczne posty i zapisuje liczniki. Ta strona to protokół i kontrakt danych; narzędzie do
analizy to `src/manual_sample.py`.

## Rama losowania

- **Marki:** te same 12 marek co w zewnętrznej próbce YouTube (najwięcej wzmianek), plus 8 więcej, jeśli starczy czasu.
- **Posty na markę:** 5 — celowo różne rozmiary: 2 najnowsze, 2 ze środka profilu, 1 najczęściej oglądany.
- **Format:** publiczne Reels / filmy (pokazują liczbę wyświetleń); na TikToku dowolny publiczny film (wyświetlenia widać).
- **Okres:** publikacja 2025-01-01 lub później; zapisz datę sprawdzenia.
- **Czego NIE zapisywać:** nazwy autora, handle'a, URL-a, zrzutów z handle'ami. Arkusz trzyma tylko markę, platformę,
  liczniki i daty.

## Kontrakt danych (CSV)

| kolumna | znaczenie |
|---|---|
| `platform` | `instagram` lub `tiktok` |
| `brand` | nazwa marki |
| `view_count` | wyświetlenia/plays widoczne na poście |
| `likes`, `comments`, `shares` | liczniki; puste, jeśli ukryte |
| `author_followers` | followersi konta, jeśli widoczni |
| `posted_at` | YYYY-MM-DD (przybliżona data wystarczy) |
| `checked_at` | YYYY-MM-DD odczytu |
| `notes` | dowolny tekst bez danych osobowych |

## Nakład i analiza

- Około **3 minuty na post**; 100 postów = ~5 godzin, do podziału w zespole.
- `python src/manual_sample.py data/processed/manual_ig.csv` uczy model platformy, gdy istnieje (TikTok) i raportuje
  MAE/bias; dla Instagrama nie ma modelu w platformie, więc raportuje opis (mediana wyświetleń, wyświetlenia na
  followersa, korelację rangową z reakcjami) — uczciwe „co można powiedzieć bez modelu”.
- Wynik dostaje kartę i wiersz rejestru jak każdy eksperyment; arkusz jest git-ignored (zawiera publiczne liczniki, ale
  i szczegóły marek — trzymać lokalnie).

## Etyka

Tylko publiczne liczniki; żadnych tożsamości autorów; żadnego automatycznego dostępu; człowiek klika. Ten sam standard co
ręczna kontrola Facebooka (E93).
