# Co nowego od E94

Wewnętrzny changelog dla zespołu: eksperymenty dodane w pushu listopad 2026 (snapshot 2026-11-07), pogrupowane tematycznie z wynikiem w jednej
linii. Nie jest dokumentem z aktualnymi liczbami. Szczegóły w kartach (`agent/vault/`) i rejestrze.

## Facebook

- Transfer zmierzony na 130 ręcznych i 402 pobranych postach: MAE 0,345 / 0,385 wobec 0,93 dla reguły (E93, E95, E111).
- Reguła jest ×4,1 za wysoko, 82% postów poza ×2 i myli się w każdym tierze (0,96-1,01); model wygrywa w każdym
  (0,36-0,46) (E165).
- 130 ręcznych postów to podzbiór 402 i jest trochę łatwiejszy (0,345 vs 0,385) (E153).

## Przedziały FB

- CQR: pokrycie 90% 0,925 przy szerokości 1,99 (E123); 0,898 / 1,59 na 130 ręcznych (E166).
- Strony poniżej 1M followersów pokryte 0,94; powyżej 1M tylko 0,84 (E132, E148).
- Poszerzanie, przesunięcie poziomu, kwantyle per tier i model dużych stron nie kalibrują >1M (E144-E147, E151);
  tam podawać tylko punkt. Model dużych stron poprawia punkt (0,410 -> 0,368), hybryd 0,385 -> 0,379 (nieistotne,
  E150/E154/E155).
- Followersi to cecha dużych stron: pomagają powyżej 1M (0,410 -> 0,343), psują poniżej (0,380 -> 0,727) (E149, E162).

## Niepewność i jakość modelu w platformie

- CQR daje uczciwe pasma: 0,889 / 0,902 / 0,899 (X v3); pasmo 50% to 0,45 / 0,74 / 0,83 (E118, E137, E141).
- Pasma trzymają się w czasie (0,907 / 0,901 / 0,932) (E142).
- Powtórzone CV: 0,269 ± 0,011 / 0,466 ± 0,030 / 0,463 ± 0,018; isotonic i tiery nie pomagają (E114).
- Uśrednianie ziaren (E134), kalibracja per tier followersów (E135) i ważenie świeżości (E133) nie pomagają.
- Target encoding autora pomaga na znanych autorach (X 0,530 -> 0,430; YouTube 0,465 -> 0,446), ale tylko 1-8% postów
  (E119); historia autora między platformami nie istnieje (E120).
- log(age) pomaga X (0,467 -> 0,454) i zwęża pasmo (E44, E139, E141); target „rate” nie (E139).
- Rolling-origin potwierdza CV po markach (0,270 / 0,450 / 0,473) (E136).

## Transfer i budżet kalibracji

- CQR między platformami: pokrycie 0,905-0,912, ale szerokości 1,51 / 2,84 / 3,69 vs 1,14-2,07 w platformie (E131).
- Tylko X wymaga kalibracji: 25-50 postów zbija 0,650 do 0,551-0,563; TikTok/YouTube/FB są zero-shot (E115).

## Braki danych

- 25,4% wyświetleń X brakuje i nie losowo; średnie z obserwowanych są obciążone ~1,3× (E105).
- Zera YouTube to nie braki (zysk 0,003, E116); post 0/0 przy pierwszym skanie ma jednak wyświetlenia (mediana 5 na X,
  60 na TikToku) (E117, E126).
- Braki: 10% reakcji kosztuje 0,03-0,08 MAE; 50% kosztuje 0,08-0,37 (E138); platformy się różnią (E161).
- Gdy cecha brakuje systematycznie, przetrenować bez niej; przełączać przy ~35% braków reakcji (TikTok/YouTube), ~70%
  (X), ~20% braków followersów (YouTube/X) (E163, E164).

## Ogon, selekcja i alerty

- Top 1% postów to 60-74% wyświetleń; 2-4% postów daje 80% (E96).
- Sumy marek są niestabilne przy małej liczbie postów (5 postów: rozrzut ×53; 20: ×6,4) (E112) i ×2 za niskie przed
  kalibracją (E109, E157, E158).
- Selekcja do ręcznego sprawdzania: warstwy reakcji nie lepsze od losowej; skrajne posty i leverage najlepsze przy
  małym budżecie (E97); conformal pokrywa zachowawczo (E98).
- Alerty: surowe lajki/Brand24 rankują lepiej na top 1% niż model MAE (E104, E106).

## Nulle i zamknięte linie

- Tekst (E100), dodatkowe kolumny (E121), cechy grafowe (E122, E127, E128), multi-task (E113), uśrednianie ziaren
  (E134): brak zysku; graf nie ma sygnału resztowego na krawędziach (E128).
- Hurdle dla zer (E140): zera to 0,4-2,5%, brak zysku.
