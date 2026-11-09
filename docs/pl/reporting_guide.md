# Przewodnik raportowania: co mówić o widoczności posta

Wewnętrzna jedna strona o tym, jak zamienić estymator w zdanie, które klient zrozumie. Nie tekst preprintu.

## Trzy zasady

1. **Zawsze zakres, nigdy sam punkt.** Punkt to środek szerokiego rozkładu; twierdzeniem jest przedział.
2. **Wyświetlenia to nie osoby.** Platformy publikują wyświetlenia, nie unikalny zasięg. Mówić „wyświetlenia”, nie
   „osoby, do których dotarło”.
3. **Mówić, kiedy odczytano liczbę.** Wyświetlenia rosną z wiekiem; post czytany po godzinie i po 30 dniach to różne
   cele.

## Po momencie

| moment | co raportować | pasmo 90% (log10) | typowy czynnik |
|---|---|---|---|
| przy publikacji, przed reakcjami | tylko zakres, z modelu publikacyjnego | 2,42-2,96 | ×260-×900 |
| przy zbieraniu, z reakcjami | punkt + pasmo CQR | 1,14 / 1,71 / 1,99 (TikTok / YouTube / X, X to v3) | ×14 / ×51 / ×98 |
| transfer Facebooka (wideo/Reel) | punkt + pasmo CQR | 1,99 | ×98 |
| wczesny skan z 0 reakcji | zakres empiryczny (E126): X mediana 5, p90 49; TikTok mediana 60, p90 179 | — | — |

- Z reakcjami MAE to 0,273 / 0,456 / 0,454 (×1,9 / ×2,9 / ×2,8; X to v3).
- Pasmo 50% („prawdopodobne”) ma szerokość 0,45 / 0,74 / 0,83 log10 (×2,8 / ×5,5 / ×7,6); pasmo 90% to
  1,14 / 1,71 / 1,99 (E137/E141). Raportować oba.
- Przy publikacji MAE to 0,601 / 0,757 / 0,611; wtedy nie podawać punktu.
- Jeśli platforma celu nie ma żadnych danych, pasmo transferu to 1,51 / 2,84 / 3,69 (×32 / ×690 / ×4 900), przy
  kalibracji na 25% marek celu (E131).

## Po platformie

- **TikTok, YouTube:** wyświetlenia są publiczne; używać modelu w platformie i CQR. To najlepszy przypadek.
- **X:** wyświetlenia są publiczne, ale próbka jest trudna (25% braków, 64% zer reakcji); używać modelu, raportować
  szerokie pasmo i oznaczyć X jako mniej pewny.
- **Facebook:** tylko wideo/Reel ma `video_view_count`; zwykłe posty nie mają licznika. Używać modelu transferu;
  strona pokazuje plays, które różnią się od modelowanego licznika o medianę ×3,6.
- **Instagram:** brak publicznych wyświetleń; nie raportować estymaty widoczności — tylko reakcje albo „niezmierzalne
  bez logowania”.

## Sugerowane sformułowania

Podać punkt, potem dwa pasma, potem wiek:

> „Szacowane 12 tys. wyświetleń (prawdopodobnie 3-34 tys., zakres 90%: 0,9-170 tys.), odczyt 2 dni po publikacji.”

Mnożniki per platforma (stosowane do punktu):

| platforma | pasmo 50% | pasmo 90% | co dodać |
|---|---|---|---|
| TikTok | ×2,8 | ×14 | „wyświetlenia są publiczne na TikToku” |
| YouTube | ×5,5 | ×51 | „wyświetlenia są publiczne na YouTube” |
| X | ×7,6 | ×98 | „liczniki X są niekompletne dla starszych postów” |
| Facebook (wideo/Reel) | brak kalibracji | ×98 poniżej 1M | „Facebook nie pokazuje tego licznika; tylko wideo/Reel” |
| Facebook powyżej 1M | brak kalibracji | brak kalibracji | „tylko punkt” |
| przy publikacji | brak kalibracji | ×260-900 | „za wcześnie; to szeroki zakres” |

- „Za wcześnie na estymatę: post nie ma jeszcze reakcji; posty skanowane w tym wieku zwykle zostają poniżej 200
  wyświetleń (mediana 5 na X), ale 0,4% przekracza 1 tys.”
- „Facebook Reel: szacowane 8 tys. wyświetleń (zakres 90%: 80-800 tys.); Facebook nie udostępnia licznika dla tego typu
  posta.”

Nigdy nie pisać punktu bez co najmniej pasma 90%; nigdy nie nazywać pasma gwarancją.

## Gdy pokrycie nie jest jednorodne

- Podać **średnie pokrycie jako headline**, a **najgorszą podgrupę w tym samym zdaniu**: „pasmo 90% pokrywa 0,89-0,93
  ogólnie; 0,94 poniżej 1M followersów, 0,80-0,86 powyżej i 0,78 w górnym kwartylu wyświetleń”.
- Nigdy nie mówić „pasmo pokrywa 90%” bez zastrzeżenia o podgrupie, gdy taka istnieje.
- Jeśli podgrupa jest nieskalibrowana, mówić „tylko punkt” i nie rysować pasma (FB powyżej 1M).
- Preferować **zakres** zamiast jednej zaokrąglonej liczby, gdy eksperymenty się różnią (0,80-0,86, nie 0,84).

## Gdy podgrupa jest nieskalibrowana (przykład: FB powyżej 1M)

Podać punkt, wiek sprawdzenia i **powiedzieć, że pasmo nie jest skalibrowane dla tej podgrupy**. Nie doczepiać 90%, nie
rysować pasma i nie wstawiać tego do tabeli obok skalibrowanych komórek bez przypisu.

Przykład: „Szacowane 1,2 mln wyświetleń dla tej strony, odczyt 3 dni po publikacji. Pasmo 90% nie jest skalibrowane
powyżej 1M followersów, więc traktować to jako estymatę rzędu wielkości (model zaniża duże strony).”

Jeśli potrzebny jest rząd wielkości, podać zakres **heurystyczny** i oznaczyć go „heurystyka, nie pasmo 90%”.


## Figury do pokazania

- `figures/fb_rule_vs_model_{en,pl}.png` — model przeciw własnej regule klienta, tier po tierze; najlepsze otwarcie.
- `figures/cqr_bands_{en,pl}.png` — co znaczą pasma 50% i 90% na każdej platformie.
- `figures/fb_coverage_{en,pl}.png` — gdzie pasmo Facebooka trzyma (poniżej 1M), a gdzie nie (powyżej 1M).

Wszystkie buduje `src/result_figures.py`; figury żywe vs historyczne są w `scripts.md`.

## Nigdy

- Punkt bez przedziału.
- Ranking alertów na tym modelu: surowe lajki/Brand24 rankują lepiej na top 1% (E104/E106).
- Suma marki z kilku postów: 5 postów daje rozrzut ×53 p90/p10, 20 postów wciąż ×6,4 (E112).
- Porównanie marek na różnej liczbie postów bez powiedzenia tego.

## Flagi jakości danych (E108)

Doczepić flagę, gdy: brakuje lajków/komentarzy/udostępnień; post ma zero reakcji; sprawdzenie jest stare; platformą jest
X; zasięg jest daleko od modelu. Flaga to powód, by poszerzyć pasmo albo sprawdzić ponownie, nie powód, by ukrywać
estymatę.

## Pliki

- Model card: `docs/en/model_card.md` / `docs/pl/model_card.md`.
- Liczby: E114, E117, E118, E123, E126, E130, E131.
