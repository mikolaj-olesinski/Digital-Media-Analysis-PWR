# Stan projektu, 2026-11-08

Wewnętrzne podsumowanie dla zespołu. Nie tekst preprintu. Pełne dowody: `docs/pl/evidence_map.md`,
`agent/findings.md`.

## Co robi projekt

Szacuje **widoczność (wyświetlenia) posta** z followersów i reakcji, lepiej niż stała, i z przedziałem. TikTok, YouTube
i X mają publiczne wyświetlenia i służą jako prawda; Facebook został zmierzony na 130 ręcznych i 402 pobranych postach;
Instagram nie ma publicznych wyświetleń.

## Gdzie jest model

| platforma | MAE log10 | typowy błąd | pasmo 90% | reguła (Brand24) |
|---|---|---|---|---|
| TikTok | 0,273 | ×1,9 | 1,14 (×14) | dokładna (to wyświetlenia) |
| YouTube | 0,456 | ×2,9 | 1,71 (×51) | dokładna (to wyświetlenia) |
| X (v3) | 0,454 | ×2,9 | 1,99 (×98) | ×9,3 za wysoko |
| Facebook (402) | 0,385 | ×2,4 | 1,99 (×98) | ×4,1 za wysoko |

- **Pasmo 50%:** 0,45 / 0,74 / 0,83 (E137/E141).
- **Przy publikacji (bez reakcji):** MAE 0,60-0,76, pasmo 2,4-3,0 (×260-900); na reakcje warto poczekać.
- **Walidacja:** CV po markach, powtórzone CV, rolling-origin, mocniejsze grupowanie autor/kopie; wszystko spójne.
- **Sufit szumu:** ~0,22 / 0,39 / 0,33; model jest blisko na TikToku i 0,06-0,14 nad nim na YouTube/X.




**Kontrole zewnętrzne (2026-11-08).** Trzy niezależne próbki YouTube pobrane przez `yt-dlp`: filmy marek (53),
zróżnicowana próbka 131 filmów (od popularnych do małych) i 47 filmów trzecich stron o marce z wyłączonymi kanałami marek.
Na populacji wzmiankowej — tej, którą model ma obsługiwać — błąd jest blisko eksportu (MAE 0,690 małe, 0,441 średnie),
zawodzi tylko warstwa popularna (1,569, ~×30 za nisko); przesunięcie ×5 na filmach marek to efekt próbki, nie modelu.
Filmy z zerem są osiągalne wśród małych wzmianek trzecich stron, ale nie przez kanały marek. X, TikTok i Instagram nadal
wymagają API, loginu albo próbki ręcznej (E189-E191).

## Główne wnioski

1. Reguła stałej/schodków jest daleko na X i FB (87% / 82% postów poza 2×); na TikToku/YouTube jest dokładna.
2. Sygnałem są reakcje; followersi to cecha dużych stron na FB i rzecz niezbędna na YouTube/X (E149/E162).
3. Tekst, etykiety, czas, kraj, metody grafowe dają ≤0,01; linia grafowa jest zamknięta (E127/E128).
4. Wcześniejsze wyświetlenia autora to największa dźwignia, gdy są dostępne (X 0,530 -> 0,430 na znanych autorach),
   ale tylko 1-8% postów je ma (E68/E119).
5. Zasięg jest zdominowany przez ogon: top 1% = 60-74% wyświetleń; 2-4% postów daje 80% (E96).
6. Alerty powinny używać surowych lajków/Brand24, nie modelu MAE (E104/E106).
7. Sumy marek wymagają przedziałów i wielu postów (E109/E112/E157/E158).
8. Braki danych: przetrenować bez brakującej cechy; konkretne punkty przełączenia (E163/E164).
9. CQR jest metodą przedziałów; surowe kwantyle pod-coverują (74%).
10. Tier >1M na FB nie da się skalibrować na 70 postach; podawać punkt, nie pasmo (E147/E151).


## Co się zmieniło w estymatorze od E94

Rdzeń jest ten sam: CatBoost z lossem MAE na followersach, lajkach, komentarzach, udostępnieniach i flagach braków.
Zmienił się kontrakt wokół niego. **X** ma teraz log(age) (v3): MAE 0,467 -> 0,454, a pasmo 90% 2,07 -> 1,99
(E139/E141). **Przedziały** to CQR 50% i 90% zamiast surowych kwantyli, które pokrywały tylko 74% (E118/E137).
**Ściągnięta średnia wyświetleń autora** to opcjonalna cecha, gdy historia istnieje (E119/E124). **Facebook** przeszedł
z twierdzenia do zmierzonego transferu (0,385 na 402 postach), ze skalibrowanym pasmem poniżej 1M followersów i tylko
punktem powyżej 1M (E95/E123/E147). **Braki danych** mają jawne punkty przełączenia (E163/E164), a **flaga pewności**
z metadanych jest częścią wyjścia (E108).

## Otwarte pytania

- Widoczność na Instagramie: potrzebny login albo płatny dostawca; decyzja otwarta.
- Zwykłe posty na Facebooku (nie wideo/Reel): brak licznika; nie da się zwalidować.
- Unikalny zasięg (osoby, nie wyświetlenia): nieobserwowalny z danych publicznych.
- Fałszywe dopasowania marek: potrzeba 100-200 ręcznych etykiet.
- Stałe Brand24: zrekonstruowane, niepotwierdzone przez klienta.

Zwykłych postów na Facebooku i unikalnego zasięgu nie da się domknąć kolejną analizą: to granice danych publicznych.
Pozostałe trzy wymagają decyzji, ręcznych etykiet albo klienta.

## Dokumenty

- **Estymator:** `model_card.md`, `estimator_spec.md`, `reporting_guide.md`.
- **Dowody:** `evidence_map.md`, `experiments.md` (rejestr), `agent/vault/` (karta na eksperyment),
  `data/processed/*_numbers.json`.
- **Dla klienta:** `one_pager.md`, `client_qa.md`, `slides.md`.
- **Dla zespołu:** `whats_new.md`, `future_work.md`, `doc_status.md`, `preprint_outline.md`.
- **Reprodukcja:** `reproduce.md`, `repro_map.md`, `scripts.md`.

## Reprodukowalność

- 197 kart eksperymentów (198 eksperymentów z eksploracją M1); każdy plik `numbers:` i notebook wskazany przez kartę istnieje.
- Ponowne uruchomienie `src/fb_rule_vs_model.py` i `src/cqr_50.py`: pliki JSON są bajtowo identyczne z zapisanymi
  (stałe ziarna).
- Wszystkie skrypty się kompilują (`python -m py_compile src/*.py`).
- Figury buduje `src/result_figures.py` (`cqr_bands`, `fb_coverage`, `reach_concentration`, `calibration_curve`,
  `fb_rule_vs_model`).

## Kluczowe decyzje

- Target: log10 wyświetleń; na Facebooku `video_view_count`, nie plays.
- Przedziały: CQR; raportować pasmo 50% i 90%, nigdy surowych kwantyli.
- Alerty: surowe lajki / reguła, nie model MAE.
- Instagram: brak estymaty widoczności bez logowania lub dostawcy.
- Braki danych: przetrenować bez brakującej cechy.
- Facebook powyżej 1M followersów: tylko punkt.
- Sumy marek: kalibracja + przedział + liczba postów.

Pełna lista: `agent/decisions.md`.

## Znane ryzyka dla preprintu

1. Próbka nie jest losowa i jest przechylona w czasie (85% kwiecień-sierpień 2026); wyniki opisują próbkę Brand24, a nie
   całe social media.
2. 402 posty FB są warstwowe po reakcjach, a 68 URLi zniknęło; 130 ręcznych postów jest trochę łatwiejsze (E153).
3. Instagram jest niezmierzony; każde twierdzenie o widoczności IG byłoby niepoparte.
4. Pasmo FB >1M jest nieskalibrowane (E147/E151); nie twierdzić o pokryciu 90% w tym tierze.
5. Model nie jest silnikiem alertów: lajki/Brand24 rankują lepiej na top 1% (E104/E106).
6. Sumy marek z kilku postów są niestabilne (E112), a ich przedziały na FB pod-coverują (E158).
7. Etykiety AI są zwalidowane tylko maszyna-maszyna; nie ma kappy człowiek-AI.
8. Stałe Brand24 są zrekonstruowane, niepotwierdzone przez klienta.
9. Wszystko jest korelacyjne; nie formułować przyczynowo.
10. Tekst preprintu musi napisać zespół, z ujawnieniem użycia AI.

## Pokrycie planu

Każdy punkt planu „Ideas to stand out” jest zrobiony, zamknięty lub zablokowany: Facebook zmierzony, Instagram
zablokowany, kalibracja, przedziały, graf (zamknięty), mniejsze testy, ekstrema, zera YouTube i przedziały per tier —
wszystko zrobione. Szczegóły: `plan_status.md`.

## Co zabrać na spotkanie

- `one_pager.md` (jedna strona) i `client_qa.md` (odpowiedzi i „czego nie mówić”).
- Trzy figury: `cqr_bands`, `fb_rule_vs_model`, `fb_coverage`.
- `slides.md` jako kolejność opowieści.

## Następne kroki

1. Zespół pisze preprint/poster (agenci nie piszą finalnego tekstu).
2. Zdecydować o Instagramie (próba ręczna, login albo dostawca).
3. Jeśli czas: miesięczne sumy marek (B25), sprawdzenie innych języków (B26).
4. Zadać Brand24 otwarte pytania (`questions_for_brand24.md`).
