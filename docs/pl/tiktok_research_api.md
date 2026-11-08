# Dane z TikToka: Research API albo próbka ręczna

Publicznych stron TikToka nie da się pobrać przez `yt-dlp` (ekstraktor jest oznaczony jako zepsuty), a wyszukiwanie wymaga
zalogowanej sesji. Zostają dwie zgodne z regulaminem drogi.

## 1. TikTok Research API (zalecane)

- Darmowe dla **badaczy akademickich**; wymaga wniosku z uczelnianym e-mailem, opisem badania i redirect URI.
- Daje publiczne dane filmów: `view_count`, `like_count`, `comment_count`, `share_count`, followersów autora,
  `create_time`, hashtagi — dokładnie pola, których potrzebuje model.
- Akceptacja trwa **tygodnie**; planować równolegle i złożyć wniosek wcześnie.
- Dokumentacja: https://developers.tiktok.com/doc/research-api-specs-query-videos

Co wpisać: krótki opis projektu estymacji widoczności, wielkość próbki (kilka tysięcy filmów z 12 marek), potrzebne pola
oraz oświadczenie, że publikowane są tylko wyniki zagregowane.

## 2. Próbka ręczna (fallback, dostępna od razu)

- Protokół i kontrakt danych: `docs/pl/manual_sample.md`.
- 100 publicznych filmów TikToka (wyświetlenia widoczne), ~5 godzin klikania, bez loginu i bez scrapowania.
- Scoring: `python src/manual_sample.py data/processed/manual_tiktok.csv` (uczy model TikToka i raportuje MAE).

## Czego nie robić

Żadnego scrapowania, żadnej automatyzacji na zalogowanej sesji, żadnych „API TikTok” od resellerów: łamią regulamin i
higienę GDPR, a liczb nie da się obronić w raporcie.
