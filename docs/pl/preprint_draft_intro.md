# Wstęp: wersja robocza wygenerowana przez Claude (2026-11-08)

> Szkic do przepisania własnymi słowami. Wygenerowany przez AI na prośbę zespołu; użycie AI ujawnić w preprincie.
> Cytaty: Tatar, de Amorim, Fdida, Antoniadis 2014, *A survey on predicting the popularity of web content*, J. Internet Serv. Appl. 5:8 (metadane z Crossref);
> Shulman, Sharma, Cosley 2016, ICWSM, arXiv:1603.09436; Roy, Mei, Zeng, Li 2013, IEEE Trans. Multimedia (oba w `literature_notes.md`, checked).

## 1. Wstęp

Firmy monitorujące media, takie jak Brand24, raportują klientom, ile osób mogło zobaczyć wzmiankę o ich marce. Najprostsza
miara takiego zasięgu, liczba wyświetleń, często nie jest jednak widoczna: platformy ograniczają to, co pokazują
publicznie, a zakres tych danych zmienia się między platformami, formatami postów i w czasie. Bez niej trudno ocenić, ile
osób faktycznie zobaczyło dany post, a jeszcze trudniej przewidzieć, ile go zobaczy. W praktyce zasięg przybliża się więc
stałym mnożnikiem liczby obserwujących lub reakcji. Taki mnożnik nie mówi, jak bardzo może się mylić.

Przewidywanie popularności treści w sieci jest szeroko badane [Tatar i in. 2014]. Część prac sprawdza też przenoszenie
modeli między serwisami: Shulman i in. [2016] pokazują, że model uczony na jednym zbiorze (muzyka, książki, zdjęcia, linki)
działa podobnie dobrze na pozostałych, a Roy i in. [2013] przenoszą wiedzę z Twittera na popularność filmów na YouTube.
Prace te opierają się jednak na innych sygnałach (tempo pierwszych odbiorców, tematy) i oceniają model tam, gdzie wynik
jest znany. Nie znaleźliśmy pracy, która sprawdza oszacowanie wyświetleń z reakcji na platformach, gdzie wyświetlenia
są ukryte, wobec niezależnie zmierzonej prawdy; tam, gdzie wyświetleń nie widać, zwykle nie ma też z czym porównać.

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
