# Specyfikacja estymatora

Status: wewnętrzny, 2026-11-07. Dokładny przepis stojący za model card. Nie tekst preprintu.
Odtworzenie: `docs/pl/reproduce.md` (sześć komend).

## Wersje

| platforma | model punktowy | przedział | cechy |
|---|---|---|---|
| TikTok | CatBoost MAE (v2) | CQR 50%/90% | ENG |
| YouTube | CatBoost MAE (v2) | CQR 50%/90% | ENG |
| X | CatBoost MAE (v3) | CQR 50%/90% | ENG + log(age) |
| Facebook (wideo/Reel) | ridge na lajkach + komentarzach (transfer); opcjonalny model dużych stron >1M | CQR 50%/90% dla <1M; >1M nieskalibrowane | lajki, komentarze (+ followersi dla >1M) |
| Instagram | **nie estymujemy** | — | — |

`ENG` = log10(followers), log1p(likes), log1p(comments), log1p(shares) i flaga braku dla każdej.

## Pipeline

1. **Target:** `y = log10(view_count)`, liczba wyświetleń w wieku sprawdzenia.
2. **Cechy:** `ENG`, plus `log10(age_days)` tylko dla X. Wiek = `metrics_updated_at - created_at`, przycięty do
   0,1-3650 dni. Dla FB wieku nie używamy (E143).
3. **Model punktowy:** CatBoost, `loss_function="MAE"`, 300 iteracji, domyślna głębokość/lr, ziarno 42.
   Dla FB: ridge (alpha 1) na `log1p(likes)`, `log1p(comments)`, trenowany na X + YouTube. Opcjonalnie: dla stron
   >1M followersów model dużych stron (+ followersi); zysk całości 0,385 -> 0,379, CI zawiera 0 (E154/E155).
4. **Przedziały:** modele kwantylowe LightGBM 0,05 i 0,95 na tych samych cechach; wynik konforemny CQR
   `max(q_lo - y, y - q_hi)` na 25% marek treningowych (po markach); pasma 50% i 90% to 50. i 90. percentyl wyniku.
5. **Flagi (E108):** braki lajków/komentarzy/udostępnień; zero reakcji; wiek sprawdzenia > 30 dni; platforma X;
   duża różnica punktu i reguły.
6. **Wyjście:** punkt, pasmo 50%, pasmo 90%, flagi i liczba postów stojących za sumą marki.

## Parametry

- CatBoost: `iterations=300`, `loss_function="MAE"`, `random_seed=42`, `allow_writing_files=False`, `thread_count=4`.
- LightGBM quantile: `n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `subsample=0.9`, `colsample_bytree=0.9`.
- Udział kalibracji CQR: 25% marek; fallback do wyniku globalnego, gdy tier ma mniej niż 20 postów kalibracyjnych.
- Walidacja: 5-fold CV po markach dla każdej liczby.

## Dowody

| krok | dowody |
|---|---|
| cechy ENG | E114, E124 |
| loss MAE | E114 |
| pokrycie CQR 90% | E118 (0,889/0,902), E141 (X 0,899), E142 (czasowo) |
| pasmo 50% | E137 |
| wiek dla X | E44, E139, E141 |
| transfer FB | E93, E95, E123 |
| budżet kalibracji | E115 |
| braki danych | E138 |
| flagi | E108 |

## Znane ograniczenia

- Instagram nie ma publicznych wyświetleń; nie estymować.
- Facebook tylko wideo/Reel; karta pokazuje plays, a nie licznik modelowany tutaj.
- Pasmo jest szerokie: ×2,8-7,6 przy 50%, ×14-117 przy 90% (zależnie od platformy).
- Duże strony FB są niedopokryte (`>1M` followersów 0,83-0,86); poszerzać pasmo.
- Punkt jest wyrównany w środku; ogon pozostaje zawyżany/zaniżany (E96, E114).

## Kod

- `src/modeling.py`, `src/extra_models.py`, `src/reaction_tiers.py` (cechy i model finalny).
- `src/validation_audit.py` (powtórzone CV), `src/cqr_intervals.py` (CQR), `src/cqr_50.py` (pasmo 50%).
- `src/fb_transfer_check.py` (transfer FB), `src/fb_cqr.py` (CQR FB), `src/confidence_flags.py` (flagi).
- Liczby: `data/processed/*_numbers.json`.

## Skąd biorą się przedziały (CQR w pięciu krokach)

1. Trenujemy dwa modele kwantylowe na markach treningowych: `q05` (5. percentyl) i `q95` (95.), na tych samych cechach
   co model punktowy.
2. Na odłożonych 25% marek treningowych (po markach) liczymy wynik konforemny dla każdego posta:
   `E = max(q05 - y, y - q95)` — jak daleko prawda wypada poza surowe pasmo.
3. Bierzemy `q = quantile(E, 0.90)` dla pasma 90% (albo 0.50 dla pasma 50%).
4. Raportowane pasmo to `[q05 - q, q95 + q]`: surowe pasmo kwantylowe poszerzone o skalibrowaną wartość.
5. Sprawdzamy pokrycie na odłożonych markach; nominał to 90% dla pasma 90% (0,889 / 0,902 / 0,899 w platformie,
   E118/E141).

Dlaczego nie surowe kwantyle: pokrywają tylko 74% zamiast 90% (E118). Dlaczego CQR, a nie model dystrybucyjny: pasmo
Gaussa z CatBoost pokrywa 0,86-0,88 (E129). Na Facebooku modele kwantylowe są trenowane na X + YouTube i kalibrowane na
25% marek FB (E123).

## Rozmiar kalibracji i fallbacki

- Podział kalibracyjny to **25% marek treningowych**, po markach, więc żaden post marki nie leży po obu stronach.
- Jeśli podgrupa (tier followersów) ma mniej niż **20 postów kalibracyjnych**, wracamy do globalnego wyniku konforemnego
  i nie twierdzimy, że pasmo jest skalibrowane dla tej podgrupy. Dlatego tier >1M na Facebooku zostaje nieskalibrowany
  (E144/E145).
- Przy małych markach pasmo jest zachowawcze, nie wąskie: strony FB poniżej 1M mają pokrycie 0,94 przy nominale 90%
  (E148).
- Parametry w tej sekcji zgadzają się z kodem: `src/cqr_intervals.py` i `src/final_v2.py` używają `test_size=0.25` i
  `np.quantile(E, 0.9)`; `src/fb_cqr_tier_fix.py` używa fallbacku przy 20 postach.
