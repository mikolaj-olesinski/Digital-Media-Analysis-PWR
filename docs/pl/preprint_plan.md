# Plan preprintu (≤8 stron), wersja robocza

Wewnętrzny plan zespołu, 2026-11-08. Mówi, co idzie do którego rozdziału i który eksperyment to dowodzi.
To nie jest tekst preprintu: tekst piszą ludzie. Liczby: `docs/pl/evidence_map.md`, `docs/pl/experiments.md`.
Zastępuje układ z `docs/en/preprint_outline.md`, który jest nieaktualny wobec E170-E188.

Zasada: błędy podajemy jako MAE i mnożnik (×10^MAE); Brand24 tylko raz w tekście i na jednym wykresie.

Legenda: **[do zrobienia]** brakuje danych lub eksperymentu; **[decyzja]** zespół musi wybrać.

---

## 1. Wstęp (0,75 str.)

- Problem: platformy ukrywają wyświetlenia; Brand24 przybliża zasięg regułą stała × followersi.
- Brand24 dotąd używał stałego mnożnika; wspominamy raz, porównanie tylko na jednym wykresie końcowym (sekcja 9).
- Pytanie: czy da się oszacować wyświetlenia lepiej, z uczciwą niepewnością, także tam, gdzie nikt ich nie widzi (FB, IG)?
- Wkład:
  - estymator z przedziałami;
  - pierwszy zmierzony test na FB;
  - mapa, które dane mają wartość;
  - rzetelne wyniki negatywne.

## 2. Dane i pomiar (1 str.)

- Próba Brand24: 12 500 wzmianek, 348 marek, 85% z IV-VIII 2026, 85% po angielsku.
- Gdzie znamy prawdę: TikTok 100%, YouTube 99%, X 75% (eksport Brand24); FB i IG tylko z naszych odczytów, które traktujemy jako ground truth.
- Pomiar FB: 130 postów ręcznie, 402 pobrane; wyświetlenia ≠ odtworzenia (×2,3-3,6, E95); część linków już nie działa.
- Pomiar IG: 500 losowych rolek (11 marek = cały wycinek IG), agent przeglądarki na zalogowanym koncie, wyświetlenia + lajki + komentarze w jednym momencie; 368 z wyświetleniami (E202). Partie 3-4 (reszta IG) w trakcie.
- Etykiety AI (typ, sentyment): 3 kodery, kappa 0,77-0,79 / 0,71-0,73; ręczny przegląd ok. 200 etykiet przez członka zespołu.
- RODO i regulaminy: odsyłacz do data statement.

## 3. Metoda i walidacja (1 str.)

- Cel: log10 wyświetleń; miara MAE (0,3 ≈ błąd ×2).
- Walidacja:
  - CV pogrupowane po markach;
  - test w czasie (trening przed 2026-06-01);
  - bootstrap po markach dla zysków.
- Model: CatBoost na followersach, reakcjach i flagach braków; dlaczego nie więcej rodzin modeli (E114, E124).
- Przedziały: CQR 50% i 90%, kalibracja na odłożonych markach (E118, E137).
- Sufit szumu: 0,243 / 0,386 / 0,331 (E56, E179); poniżej nie ma czego gonić.

## 4. Wyniki A: platformy ze znanymi wyświetleniami (1,25 str.)

- Błąd modelu (TikTok / YouTube / X): 0,27 / 0,46 / 0,45, czyli ×1,9 / ×2,9 / ×2,8; ile postów przeszacowanych, ile niedoszacowanych, ile w granicach ×2.
- Pokrycie przedziału 90%: 0,89-0,90.
- Wartość pól (E176, `data_needs.md`):
  - reakcje dają 0,11-0,31;
  - followersi ≤0,04;
  - wszystko inne ≤0,01.
- Co zmienia wynik:
  - drugi odczyt wyświetleń 0,11-0,27 (E40);
  - historia autora 0,10 na znanych autorach (E119);
  - stabilność 2-3 miesiące w teście w czasie (E46, E136).
- X bez braków danych: 0,49 (E181).
- Wykres: pasma CQR.

## 5. Wyniki B: przeniesienie na inne platformy (1,5 str.)

Najmocniejsza część pracy.

- **Scenariusz 1, mamy kilka postów z wyświetleniami:** 25-50 postów klienta obniża błąd X z 0,650 do 0,551 (E115); TikTok, YouTube i FB działają bez kalibracji.
- **Scenariusz 2, nie da się skalibrować:**
  - koszt braku danych z platformy: szerokości przedziałów 1,51 / 2,84 / 3,69 (E131);
  - test na FB: model z X i YouTube na reakcjach odczytanych razem z wyświetleniami daje 0,385 (×2,4; ręcznie 0,345, wewnątrz FB 0,355).
- Warunek: reakcje odczytane w tym samym momencie co wyświetlenia (E177).
- Pokrycie na FB: 0,925 poniżej 1M followersów, 0,80-0,86 powyżej (E123, E151).
- IG (E202, 338 rolek): bez kalibracji ×2,65, po 25 rolkach ×2,41, model uczony na samym IG ×2,36; zaniża ok. ×1,8; bez lajków ×6-11.
- Macierz kombinacji platform treningowych (E201): FB / IG ×2,4-3,1 z dowolnej mieszanki; X nie przenosi się w żadną stronę; proporcje danych w treningu ustawiają poziom. Wykres główny `figures/train_combos_pl.png`.
- Wykres: krzywa kalibracji; FB i IG prawda kontra predykcja.

## 6. Wyniki C: gdy brakuje danych i jak pewni jesteśmy (0,5 str.)

- Router według dostępnych pól (pełne / bez reakcji / bez followersów / nic). Bez routera pokrycie spada do 0,46-0,83, z routerem 0,88-0,90 (E176).
- P(błąd poniżej ×2) jest skalibrowane per wzorzec pól (od 0,66 do 0,19), ale nie porządkuje postów wewnątrz wzorca.
- Przedziały wyświetleń (E199): 3 przedziały ~0,8, 4 ~0,7, dekady ~0,6 trafności; regresja pocięta > osobny klasyfikator; X 0,95 = klasa większościowa. Forma prezentacji, nie lepszy model.

## 7. Co nie działa (0,5 str., jedna tabela)

- Tekst i NLP, embeddingi i transformery, grafy (autor-marka, treść), modele wielozadaniowe, etykiety AI, kraj, czas.
- Dane zewnętrzne: Wikipedia, Wikidata, momentum marki, popularność platform.
- Wszystko ≤0,01 (E49, E100, E113, E121, E122, E127, E128, E170-E175).
- Jedno zdanie dlaczego: sygnał siedzi w reakcjach, reszta go powtarza.

## 8. Zastosowania poboczne (0,5 str.)

- Alert o wybiciu (top 5%): precyzja 0,82 / 0,86 / 0,71 wobec 0,71 / 0,64 / 0,35 przy samych reakcjach (E185).
- Rankingi marek w branży: Kendall 0,73 / 0,70 / 0,60 (E186).
- EMV na X w ±50%: 47% marek (E188).
- Kto się wybija: promo 51% wobec 34% w całej próbie (E184).
- Fałszywe dopasowania marek (Cricket, AT&T Stadium): **[decyzja]** obserwacja w dyskusji albo zmierzyć odsetek.

## 9. Ograniczenia, dyskusja, rekomendacje (0,75 str.)

- FB opiera się na 20 markach; 3 z nich dają 39% postów. Bez nich 0,461; pomijanie po jednej marce 0,36-0,41 (E178).
- FB posty spoza angielskiego: 0,540, przesunięcie +0,35 (E182).
- Viral posty z top 1% niedoszacowane ×9-14 (E180).
- Sumy marek mylą się o więcej niż ×2 w 29-47% przypadków (E180).
- IG: tylko 11 marek, 3 największe = 44% rolek (bez nich ×2,95); tylko rolki. Reakcje i wyświetlenia w jednym momencie tylko na FB i IG; brak wniosków przyczynowych.
- Rekomendacje dla Brand24 (`data_needs.md`):
  1. reakcje odczytywać razem z wyświetleniami;
  2. braki zapisywać jako null, nie jako 0;
  3. drugi odczyt wyświetleń;
  4. trzymać historię autorów.
- Jeden wykres końcowy: nasz model kontra stały mnożnik Brand24 na wszystkich platformach.
- Ujawnienie użycia AI; odtwarzalność (repo, skrypty, rejestr eksperymentów).

---

## Otwarte sprawy

- **[do zrobienia]** IG partie 3-4: przeliczyć E202 i E201.
- **[do zrobienia]** Wykresy: ablacja (co ma największy wpływ), brak reakcji / części reakcji.
- Przedziały wyświetleń (E199): zrobione; regresja pocięta na przedziały > klasyfikator; 4 przedziały ~0,7 trafności.
- **[decyzja]** Alerty na top 1%: E104/E106 (reguła i lajki lepsze od modelu) kontra E185 (model lepszy na top 5%). Którą wersję pokazać, czy obie z różnym progiem?
- **[decyzja]** Sekcja 8 jest najłatwiejsza do zaatakowania: co zostaje, co idzie do dyskusji.
- Uwagi zespołu: (tu dopisujemy)
