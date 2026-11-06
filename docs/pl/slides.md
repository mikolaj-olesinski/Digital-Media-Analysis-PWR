# Plan slajdów (M3 / spotkanie z klientem)

Wewnętrzny plan: który wynik na który slajd, z komunikatem i zastrzeżeniem. Nie tekst preprintu; właściwe sformułowania
pisze zespół.

| # | slajd | figura / tabela | komunikat | zastrzeżenie |
|---|---|---|---|---|
| 1 | Tytuł | — | Czy da się oszacować widoczność lepiej niż stałą, z niepewnością? | — |
| 2 | Dane | mała tabela: dostępne wyświetlenia (TT 100%, YT 99%, X 75%, FB 0%, IG 0%) | prawda istnieje na trzech platformach; FB zmierzony osobno | próbka nie jest losowa (85% IV-VIII 2026) |
| 3 | Model vs reguła | `figures/cqr_bands_*` + tabela MAE (0,273 / 0,456 / 0,454; FB 0,385) | model zmniejsza błąd o połowę i daje uczciwe pasma | pasma są szerokie (×14-×98) |
| 4 | Reguła vs model na FB | `figures/fb_rule_vs_model_*` | model wygrywa w każdym tierze followersów (0,36-0,46 vs ~1,0) | FB tylko wideo/Reel |
| 5 | Dlaczego to trudne | `figures/reach_concentration_*` | zasięg jest zdominowany przez ogon (top 1% = 60-74%) | punkt to mediana, nie gwarancja |
| 6 | Przedziały FB | `figures/fb_coverage_*` | strony <1M pokryte 0,94; >1M 0,80-0,86 (0,86 na figurze) | nie twierdzić o 90% powyżej 1M |
| 7 | Budżet kalibracji | `figures/calibration_curve_*` | X potrzebuje 25-50 postów (0,650 -> 0,551); TT/YT/FB zero-shot | kalibracja szkodzi przy małym n |
| 8 | Co raportować / czego nie | tabela z Q&A klienta | zakres, nigdy sam punkt; bez alertów i rozliczeń | — |
| 9 | Ograniczenia | „znane ryzyka” w statusie | uczciwy zakres | — |


## Jeśli masz jedną figurę

Użyć **`figures/fb_rule_vs_model_*`**. Stawia model przeciw obecnej regule Brand24 tier po tierze, na zmierzonych
postach, a przekaz działa bez tabeli: model wygrywa wszędzie (0,36-0,46 vs ~1,0). Odpowiada na rzeczywiste pytanie
klienta („czy to lepsze niż to, co robimy?”), a nie na abstrakcyjne MAE.

Zapas: `figures/cqr_bands_*`, jeśli pytanie dotyczy niepewności.

## Jeśli masz 60 sekund

Zacząć od wersji 60-sekundowej z `client_qa.md`, pokazać `fb_rule_vs_model` i skończyć. Nie przechodzić przez model
card.

## Czego nie wkładać na slajd

- „Szacujemy zasięg” — to wyświetlenia.
- Skalibrowane pasmo dla FB powyżej 1M.
- Model jako silnik alertów.
- Liczby dla Instagrama.
- Jakąkolwiek strzałkę przyczynową.

## Figury do wygenerowania przed spotkaniem

Wszystkie są już budowane przez `python src/result_figures.py en|pl`:
`cqr_bands`, `fb_coverage`, `reach_concentration`, `calibration_curve`, `fb_rule_vs_model`.

## Prezentowanie niepewności na slajdzie

- Nigdy sam punkt: rysować **pasmo 50% jako gruby słupek** i **pasmo 90% jako cienką linię** wokół (albo error bars),
  z jednym opisem w legendzie.
- Dla ogólnego odbiorcy tłumaczyć szerokości log10 na czynniki: „×14”, „×51”, „×98”, nie „1,14 log10”.
- Jedno zdanie na liczbę: „Szacowane 12 tys. wyświetleń (prawdopodobnie 3-34 tys., 90%: 0,9-170 tys.), odczyt 2 dni po
  publikacji.”
- Na slajdzie „model vs reguła” pokazać oba punkty i błąd reguły jako odniesienie; pasmo modelu powinno rzadko
  pokrywać się z regułą.
- Na slajdzie FB rozdzielić <1M i >1M: pierwsze ma pasmo, drugie tylko punkt; nie rysować pasma dla >1M.
- Zastrzeżenie pod figurą, nie w przypisie: „pasma 90%, jeśli nie napisano inaczej; wyświetlenia, nie osoby”.
