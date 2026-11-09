# Ręczne sprawdzenie wyświetleń na Facebooku i Instagramie (B14)

Cel: pierwszy pomiar błędu przeniesienia modelu na FB / IG. Publiczny licznik wyświetleń mają tylko filmy i Reelsy, więc próbka jest z nich (wniosek dla zdjęć pozostaje założeniem).

- Plik: `data/manual/fbig_sample.csv` (lokalny, poza gitem, bo zawiera linki). 160 Reelsów z IG, 152 Reelsy i filmy z FB, losowane warstwowo po liczbie reakcji (`src/fbig_check.py`, seed 42).
- Dla każdego wiersza otwórz `url` w przeglądarce (bez logowania, jeśli się da) i wpisz **w tym samym momencie**:
  - `checked_at` – data i godzina, np. `2026-11-05 10:30`,
  - `views` – liczba wyświetleń (bez skrótów: 12 300, nie 12,3 tys.),
  - `likes_now`, `comments_now`, `shares_now` – liczby widoczne teraz,
  - `views_visible` – `yes` / `no` (ukryte, usunięte, konto prywatne),
  - `note` – opcjonalnie.
- Nie zapisuj nazw autorów, nie rób zrzutów ekranu, nie używaj skryptów do pobierania stron. Tylko liczby.
- Wystarczy 50–100 postów na platformę (E15: 25–50 usuwa większość przesunięcia); kolejność wierszy jest losowa, więc można przerwać w dowolnym miejscu.
- Potem: `fbig_check.evaluate(ROOT, f, meta)` liczy MAE przeniesienia (ridge na polubieniach i komentarzach z TikToka / YouTube / X).

Decyzja o zgodności z RODO i regulaminem należy do zespołu (B14).
