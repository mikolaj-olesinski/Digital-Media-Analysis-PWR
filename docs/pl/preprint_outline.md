# Struktura preprintu (≤ 8 stron)

Wewnętrzna struktura dla zespołu. Tekst piszą ludzie; ta strona mówi tylko, co gdzie idzie i który eksperyment to
popiera. Liczby: `docs/pl/evidence_map.md`.

## 1. Wstęp (~0,75 strony)

- Widoczność jest aktywem komercyjnym; platformy ukrywają jej większość (kontekst Brand24).
- Obecna praktyka: `reach ≈ stała × followersi`; zrekonstruowana reguła.
- Pytanie: czy wyświetlenia da się oszacować lepiej i z jaką niepewnością tam, gdzie są (TikTok, YouTube, X) i gdzie ich
  nie ma (Facebook, Instagram)?
- Wkład: (a) zwalidowany estymator ze skalibrowanymi przedziałami; (b) pierwszy zmierzony test FB (130 + 402 posty);
  (c) uczciwe wyniki negatywne (tekst, graf, multi-task, alerty); (d) przewodnik decyzyjny dla klienta.

## 2. Prace pokrewne (~0,75 strony)

- Przewidywanie zasięgu/popularności; zaangażowanie jako proxy; problem transferu.
- Predykcja konforemna i kalibracja przy przesunięciu.
- Metody grafowe i tekstowe; dlaczego potrzebują danych o interakcjach.
- Źródła: `docs/pl/literature_notes.md` (sprawdzone vs tylko wyszukiwarka oznaczone).

## 3. Dane (~1 strona)

- Próbka Brand24: 12 500 wzmianek, 348 marek, przechył czasowy (85% IV-VIII 2026), 85% angielski.
- Dostępność wyświetleń: TikTok 100%, YouTube 99%, X 75%, Facebook 0%, Instagram 0%.
- Pomiar FB: 130 ręcznych, 402 pobrane; views vs plays (×3,6); 68 URLi zniknęło.
- Etykiety AI: trzej koderzy, kappa 0,77-0,79 / 0,71-0,73, tylko maszyna-maszyna.
- E93, E95, E105, E153; data statement (opis próbki).

## 4. Metody (~1,25 strony)

- Target log10(wyświetlenia); wiek sprawdzenia; views vs plays.
- Cechy: followersi, reakcje, flagi braków; wiek dla X.
- Model: CatBoost MAE; dlaczego nie więcej rodzin (E114/E124).
- Walidacja: CV po markach, powtórzone CV (E114), czasowe (E46/E136), mocniejsze grupowanie (E55), bootstrap.
- Przedziały: CQR (E118), pasmo 50% (E137), test czasowy (E142), FB (E123).
- Transfer: leave-one-platform-out (E131), budżet kalibracji (E115).
- Sufit szumu (E56) i dlaczego nie gonimy setnych.

## 5. Wyniki (~2,5 strony)

1. **Model vs reguła** — 0,273/0,456/0,454 vs ~1,0; FB 0,385 vs 0,93; każdy tier FB (E165). Figury `cqr_bands`,
   `fb_rule_vs_model`.
2. **Niepewność** — pokrycie 90%: 0,89-0,90 w platformie, 0,925 na FB <1M, 0,80-0,86 powyżej 1M (E118/E123/E147/E148/E151).
   Figura `fb_coverage`.
3. **Transfer** — koszt braku danych celu: szerokości 1,51/2,84/3,69 (E131); budżet kalibracji X (E115).
   Figura `calibration_curve`.
4. **Co nie pomaga** — tekst (E100), dodatkowe kolumny (E121), graf (E122/E127/E128), multi-task (E113), uśrednianie
   ziaren (E134). Krótka podsekcja, nie zrzut tabel.
5. **Historia autora** — +0,10 na znanych autorach (E68/E119); dźwignia dla powtarzalnych autorów Brand24.
6. **Koncentracja** — top 1% = 60-74% wyświetleń; dlaczego punkty są niestabilne (E96). Figura `reach_concentration`.
7. **Braki danych** — 25% braków X (E105); zera (E116/E117); reguła retreningu (E163/E164).
8. **Sumy marek** — kalibracja + przedziały; małe próbki (E109/E112/E157/E158).
9. **Alerty** — lajki/reguła biją model na top 1% (E104/E106).

## 6. Dyskusja (~0,75 strony)

- Co Brand24 powinno robić per platforma (one-pager).
- Gdzie modelu nie wolno używać (alerty, rozliczenia, Instagram, pasmo >1M).
- Wielkość wygranej: reguła myli się ×4-×9, model ×2-×3.

## 7. Ograniczenia (~0,5 strony)

- Próbka nielosowa, przechył czasowy, warstwowy FB, niezmierzony IG, etykiety tylko maszynowe, zrekonstruowane stałe,
  brak twierdzeń przyczynowych. Pełna lista: „znane ryzyka” w statusie.

## 8. Reprodukowalność (~0,25 strony)

- `src/`, `data/processed/*_numbers.json`, `docs/pl/repro_map.md`; ziarna; dwa skrypty odtworzone bit w bit.

## Figury i tabele

| element | źródło |
|---|---|
| rys. 1 pasma CQR | `figures/cqr_bands_*` |
| rys. 2 pokrycie FB | `figures/fb_coverage_*` |
| rys. 3 koncentracja | `figures/reach_concentration_*` |
| rys. 4 kalibracja | `figures/calibration_curve_*` |
| rys. 5 reguła vs model FB | `figures/fb_rule_vs_model_*` |
| tab. 1 dane i wyświetlenia | E01, data statement |
| tab. 2 model vs reguła per platforma | E114/E93/E95/E111 |
| tab. 3 co nie pomaga | E100/E113/E121/E122/E127/E128 |
