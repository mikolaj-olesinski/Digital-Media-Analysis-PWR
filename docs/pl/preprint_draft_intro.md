# Wstęp: wersja robocza wygenerowana przez Claude (2026-11-08)

> Szkic do przepisania własnymi słowami. Wygenerowany przez AI na prośbę zespołu; użycie AI ujawnić w preprincie.
> [cyt.] = miejsce na cytat, do uzupełnienia tylko pozycjami sprawdzonymi w `docs/en/literature_notes.md`.

## 1. Wstęp

Firmy monitorujące media, takie jak Brand24, raportują klientom, ile osób mogło zobaczyć wzmiankę o ich marce. Najprostsza
miara takiego zasięgu, liczba wyświetleń, jest jednak publiczna tylko na części platform. TikTok i YouTube podają ją przy
każdym filmie, a X przy większości postów. Facebook i Instagram, czyli platformy, na których marki są najbardziej obecne,
jej nie pokazują. W praktyce zasięg na tych platformach przybliża się więc stałym mnożnikiem liczby obserwujących lub
reakcji. Taki mnożnik nie mówi, jak bardzo może się mylić.

Przewidywanie popularności treści jest dobrze zbadane, ale prawie zawsze w obrębie jednej platformy, na której model jest
uczony i testowany [cyt.]. Pytanie, czy zależność między reakcjami a wyświetleniami przenosi się na platformę, z której
model nie widział żadnego posta, pozostaje otwarte z prostego powodu: na Facebooku i Instagramie nie ma prawdy, z którą
można by porównać oszacowanie.

W tej pracy pytamy, czy wyświetlenia posta na Facebooku i Instagramie da się oszacować z jego reakcji modelem uczonym
wyłącznie na innych platformach, i z jaką niepewnością. Korzystamy z 12 500 wzmianek o 348 markach z pięciu platform,
udostępnionych przez Brand24. Dla Facebooka i Instagrama sami zebraliśmy wyświetlenia razem z reakcjami odczytanymi w tym
samym momencie: 402 filmy z Facebooka i 338 rolek z Instagrama. Traktujemy je jako punkt odniesienia (ground truth). Wszystkie
modele oceniamy na markach, których nie widziały w treningu.

Nasz wkład jest następujący:
- Model uczony tylko na X i YouTube szacuje wyświetlenia na Facebooku z typowym błędem ×2,4, a na Instagramie ×2,65.
  Po kalibracji na 25 postach błąd na Instagramie spada do ×2,41, czyli prawie do poziomu modelu uczonego na samym
  Instagramie (×2,36).
- Prawie cały sygnał niosą reakcje. Tekst, etykiety treści, grafy autorów i marek oraz dane zewnętrzne o markach poprawiają
  błąd o najwyżej 0,01 w skali log10. Bez reakcji błąd rośnie do ×6–11.
- Dla każdego posta podajemy przedział niepewności (pokrycie przedziału 90%: 0,89–0,90), który pozostaje wiarygodny także
  wtedy, gdy części danych brakuje.
- Pokazujemy, jakie dane musi zbierać dostawca monitoringu, żeby takie oszacowanie działało. Najważniejsze jest odczytywanie
  reakcji w tym samym momencie co wyświetleń.

<!-- Do sprawdzenia przed oddaniem:
- liczby IG po partii 3 (rolki 501-950) mogą się zmienić;
- ×2,4 na FB = E93 (402 posty), ×2,65 / ×2,41 / ×2,36 IG = E202, pokrycie = E118, bez reakcji = E202 / E176, inne źródła <= 0,01 = E49, E100, E122, E170-E175;
- zakres: IG tylko rolki, FB tylko filmy; 20 marek FB, 11 marek IG (do ograniczeń). -->
