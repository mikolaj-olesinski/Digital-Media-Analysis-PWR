# Potrzeby danych: wartość informacji, umowa na dane, monitoring

Wewnętrzne, 2026-11-06. Co Brand24 powinien zbierać, w jakiej formie i jak utrzymać estymator w ryzach po wdrożeniu.
Nie tekst preprintu. Każda liczba odsyła do eksperymentu w `experiments.md`.

## 1. Wartość informacji: ile jest warte każde pole

MAE na log10 wyświetleń; mniej = lepiej. „Koszt” to wysiłek po stronie Brand24, nie zespołu.

| pole lub działanie | bez | z | zysk | kto ma | koszt | dowód |
|---|---|---|---|---|---|---|
| Świeże reakcje + komentarze FB, odczytane razem z wyświetleniami | 0,831 (liczby Brand24) | 0,385 | **0,45** | Brand24, ponowny odczyt strony | niski: jeden odczyt więcej | E177 |
| Reakcje w ogóle (TikTok) | 0,584 | 0,273 | **0,31** | już zbierane | brak | E176 |
| Reakcje w ogóle (YouTube / X) | 0,714 / 0,576 | 0,460 / 0,470 | 0,25 / 0,11 | już zbierane | brak | E176 |
| Drugi odczyt wyświetleń (pierwszy odczyt -> późniejszy) | 0,273 / 0,456 / 0,467 | 0,165 / 0,188 / 0,269 | 0,11-0,27 | Brand24, harmonogram | średni: kolejka ponownych odczytów | E40 |
| Wcześniejsze wyświetlenia autora (tylko znani autorzy, 1-8% postów) | X 0,530 | X 0,430 | 0,10 na tych postach | archiwum Brand24 | niski, jeśli trzymane | E119, E77 |
| 25-50 postów klienta z wyświetleniami (X) | 0,650 | 0,551-0,563 | ~0,09 | klient | średni | E115 |
| Followersi (YouTube / X) | 0,500 / 0,515 | 0,460 / 0,470 | 0,04 | już zbierane | brak | E176 |
| Followersi (TikTok) | 0,276 | 0,273 | 0,003 | już zbierane | brak | E176 |
| Tekst, etykiety, kraj, czas, marka, Wikipedia, Wikidata, graf | — | — | <= 0,01 | — | — | E49, E84, E121, E170-E175 |

Kolejność dla Brand24: (1) odczytywać reakcje i komentarze FB razem z wyświetleniami; (2) trzymać kompletne reakcje na
każdej platformie; (3) zaplanować drugi odczyt wyświetleń; (4) trzymać wcześniejsze wyświetlenia autorów; (5) dać
klientom kalibrację na X.

## 2. Umowa na dane dla estymatora

| pole | wymagane | forma | dlaczego |
|---|---|---|---|
| platforma, id wzmianki, created_at (UTC) | tak | | routing, wiek |
| lajki, komentarze, udostępnienia | tak albo jawny brak | wszystkie reakcje (FB: suma reakcji, nie tylko „Lubię to”) | główny sygnał; cicha 0 zamiast „nie zebrano” psuje pasma (E176: pokrycie 0,46-0,83) |
| czas odczytu reakcji | tak | UTC, w tym samym zapytaniu co wyświetlenia, gdy są | nieaktualne reakcje podwajają błąd FB (E177) |
| followersi autora | tak albo jawny brak | w chwili odczytu | sygnał na YouTube / X; na FB tylko powyżej 1M |
| liczba wyświetleń (gdy publiczna) | do kalibracji i monitoringu | z czasem odczytu i nazwą licznika (FB: `video_view_count`, nie plays) | plays i views na FB różnią się x2,3-3,6 (E95) |
| flaga braku dla każdego pola | tak | null, nigdy 0 | router wybiera model po dostępnych polach (E176) |

Zasady: brak to null, nie zero; każda liczba ma czas odczytu; jedna definicja licznika na platformę.

## 3. Monitoring po wdrożeniu

| kontrola | jak | wyzwalacz | działanie |
|---|---|---|---|
| Dryf poziomu | co miesiąc mediana (prawda - predykcja) na postach z publicznymi wyświetleniami (TikTok, YouTube, X, wideo FB) | |mediana| > 0,10 log10 dwa miesiące z rzędu | przeliczyć przesunięcie na ostatnim miesiącu (E89/E115) |
| Pokrycie pasma | udział postów z publicznymi wyświetleniami w paśmie 90%, per platforma i wzorzec pól | < 0,85 na >= 200 postach | ponownie skalibrować CQR na dwóch ostatnich miesiącach (E142) |
| Kompletność pól | udział postów w każdym wzorcu (pełne / bez reakcji / bez followersów / nic) | udział zmienia się o > 10 pkt | sprawdzić kolektor; router w tym czasie trzyma pasma uczciwie (E176) |
| Definicja licznika | stosunek plays do views na FB, wyświetlenia na reakcję per platforma | skok > x1,5 tydzień do tygodnia | platforma zmieniła licznik; zamrozić, zmierzyć, przeuczyć (E95) |
| Sumy marek | per marka klienta szerokość przedziału vs liczba postów | marka z < 20 postami | pokazać sumę z przedziałem i flagą „mało postów” (E112) |

Przeuczanie: model trzymał się 2-3 miesiące bez dryfu (E46, E136); przeuczać co kwartał albo po każdym wyzwalaczu.
