# Mapa dowodów: co wiemy, jak mocno i czego nie wiemy

Stan: 2026-11-07. Wewnętrzna tabela planistyczna dla zespołu preprintu. Nie tekst preprintu. Siła: **solidne** = powtórzone w kilku
walidacjach; **prawdopodobne** = jedna walidacja, spójna z resztą; **otwarte** = niezmierzone.

| # | Twierdzenie | Siła | Dowody |
|---|---|---|---|
| 1 | Na TikToku i YouTube zasięg Brand24 równa się liczbie wyświetleń (100% / 99% wierszy) | solidne | EDA, E01 |
| 2 | Facebook i Instagram nie mają wyświetleń w próbce; tylko wideo/Reel FB ma publiczny licznik | solidne | E93, E95 |
| 3 | Reguła stałej/schodków jest daleko na X (mediana ×9,3, 87% poza 2×) i FB (×4,1, 82% poza 2×) | solidne | E99, E111 |
| 4 | Model bije regułę na MAE: 0,27 / 0,46 / 0,45 (TT/YT/X) vs około 0,93-1,15; transfer FB 0,385 vs 0,93 | solidne | E114, E93, E95, E111, E141 |
| 5 | Model jest blisko sufitu szumu na TikToku, 0,06-0,14 nad nim na YouTube/X | solidne | E56 |
| 6 | CQR daje uczciwe przedziały 90% w platformie (0,889 / 0,902 / 0,899 z X v3); surowe kwantyle pokrywają tylko 74%; wyjątkiem jest tier FB >1M (0,80-0,86, E145/E151) | solidne | E118, E141, E147 |
| 7 | Kalibracja na postach celu potrzebna tylko na X (25-50 postów, 0,650 -> 0,551-0,563); TT/YT/FB działają zero-shot | prawdopodobne | E115 |
| 8 | Historia autora to największa dźwignia, gdy istnieje (X 0,530 -> 0,430 na znanych autorach), ale obejmuje 1-8% postów | prawdopodobne | E68, E119 |
| 9 | Tekst, etykiety, czas, kraj, kolumny świeżości i metody grafowe dają <= 0,01; embeddingi grafowe szkodzą | solidne | E100, E49, E84, E121, E122, E127, E128 |
| 10 | Zasięg jest zdominowany przez ogon: top 1% postów to 60-74% wyświetleń; 2-4% postów daje 80% | solidne | E96 |
| 11 | Na top 1% ranking po surowych lajkach (lub reguła na TT/YT) bije model MAE | prawdopodobne | E104, E106 |
| 12 | Sumy marek są niestabilne przy małej liczbie postów (5 postów: rozrzut ×53; 20: ×6,4) i wymagają przedziałów + kalibracji | prawdopodobne | E109, E112 |
| 13 | 25% wyświetleń X jest brakujących i nie losowo; średnie z obserwowanych są obciążone około 1,3× | prawdopodobne | E105 |
| 14 | Zero reakcji to nie zero widoczności; post 0/0 przy pierwszym skanie ma medianę 5 (X) / 60 (TikTok) | solidne | E117, E126 |
| 15 | Przy publikacji MAE to 0,60-0,76, a pasmo 90% ma ×260-900; reakcje zwężają je do ×14-117 | prawdopodobne | E130, E118 |
| 16 | Pasmo FB nie pokrywa największych stron (górny kwartyl wyświetleń 0,782, >1M followersów 0,857) | prawdopodobne | E132 |
| 17 | Sumy marek FB z 19 marek są ×2 za niskie przed kalibracją; po niej 47% wciąż poza 2×, a przedział 80% pokrywa 0,74-0,79 | prawdopodobne | E157, E158, E159 |
| 18 | Gdy cecha brakuje systematycznie, przetrenować bez niej (przełączenie przy ~35% braków reakcji na TikTok/YouTube, ~70% na X; ~20% braków followersów na YouTube/X) | prawdopodobne | E163, E164 |
| 19 | Potrzeby danych różnią się: TikTok reakcje, YouTube/X followersi, Facebook reakcje (followers tylko >1M) | prawdopodobne | E161, E162 |
| 20 | Widoczny rozkład różni się między platformami: elastyczność followersów 0,044 / 0,303 / 0,164 (TT/YT/X), reakcji 0,82 / 0,66 / 1,06; 43% / 5% / 14% postów bije liczbę followersów | prawdopodobne | E167 |
| 21 | Prosta zrekonstruowana reguła Brand24 odtwarza 97% wierszy IG, 62% X i 49% FB; na TikToku/YouTube zasięg to liczba wyświetleń | prawdopodobne | E168 |
| 22 | Historia marki nie zastępuje brakującej historii autora: podział czasowy, prior marki -0,005 (TikTok), -0,035 (YouTube), +0,006 (X, CI zawiera 0); 35-52% postów testowych ma znaną markę | prawdopodobne | E169 |
| 23 | Model jest skalibrowany do populacji wzmianek Brand24: na 53 niezależnie pobranych filmach marek na YouTube relacja się trzyma (Spearman 0,93, nachylenie 0,83), ale poziom jest ×5 za nisko (MAE 0,726, bias +0,71; 0,540 po usunięciu przesunięcia) | prawdopodobne | E189 |
| 23 | Sufit szumu nie jest efektem wieku sprawdzenia: z dopasowanym wiekiem granica bez zmian; cel 30 dni zmienia MAE <= 0,007 | likely | E179 |
| 24 | Filmy z zerem nie są osiągalne przez kanały marek i wyszukiwanie (0/131, minimum 71), ale wśród wzmianek trzecich stron się zdarzają (1 zero, 17 poniżej 100); model na filmach marek zawodzi powyżej małych postów (MAE 0,289 małe vs 1,480 popularne) | prawdopodobne | E190 |
| 24 | Transfer FB stoi na 20 markach; bez 3 największych 0,46; posty nieangielskie FB ~x2,2 za nisko; na własnych liczbach Brand24 0,83 | likely | E177, E178, E182 |
| 25 | Na filmach trzecich stron (144 filmy, 40 marek) model jest blisko eksportu (MAE 0,558 małe, 0,553 średnie) i zawodzi tylko na popularnych (1,047); wcześniejsze ×5 wynikało z filmów marek | prawdopodobne | E191, E193, E195 |
| 25 | Na filmach trzecich stron (144 filmy, 40 marek) model jest blisko eksportu (MAE 0,558 małe, 0,553 średnie) i zawodzi tylko na popularnych (1,047); wcześniejsze ×5 wynikało z filmów marek | prawdopodobne | E191, E193, E195 |
| 26 | Wyświetlenia TikToka z eksportu są nieaktualne, ale stabilne rangowo: 40/60 odczytane, mediana 1,068, p90 3,07, Spearman 0,918 | solidne | E192 |
| 26 | Wczesny odczyt wyświetleń to sygnał do alarmów: precyzja top 5% 0,82 / 0,86 / 0,71 vs 0,71 / 0,64 / 0,35 z reakcji | likely | E185 |
| 27 | Wyświetlenia YouTube z eksportu są nieaktualne: mediana 1,516, p90 25,3, 19% w 5%, Spearman 0,857; TikTok 1,068 i 0,918 | solidne | E194, E192 |
| 27 | Routing po dostępnych polach trzyma pasma 90% na 0,88-0,90; pełny model z brakiem reakcji pokrywa 0,46-0,83 | likely | E176 |

## Nieznane / otwarte

- Widoczność na Instagramie: brak publicznych wyświetleń bez logowania; niezmierzone (brak zgody na cookies).
- Zwykłe posty na Facebooku (nie wideo/Reel): brak licznika; nie da się zwalidować.
- Unikalny zasięg (osoby, nie wyświetlenia): nieobserwowalny z danych publicznych.
- Dokładne stałe Brand24: zrekonstruowane, niepotwierdzone przez klienta.
- Fałszywe dopasowania marek (post przypisany złej marce): potrzeba 100-200 ręcznych etykiet; niezrobione.
- Wpływ przyczynowy cech na wyświetlenia: poza zakresem.
- Reprezentatywność 402 pobranych postów FB: warstwowa po reakcjach, nie losowa.

## Jak czytać siłę

- **solidne:** dwie lub więcej niezależnych walidacji (CV po markach + czasowa + mocniejsze grupowanie) albo fakt
  deterministyczny.
- **prawdopodobne:** jedna walidacja z bootstrapem, spójna z resztą dowodów.
- **otwarte:** brak pomiaru; wpisać jako ograniczenie, nie zgadywać.

Wszystkie liczby są w `data/processed/*_numbers.json` i w rejestrze eksperymentów (`docs/pl/experiments.md`).

Liczby są na log10 wyświetleń; jak czytać MAE, szerokości pasm i bias: `model_card.md`, sekcja „Jak czytać liczby”.
