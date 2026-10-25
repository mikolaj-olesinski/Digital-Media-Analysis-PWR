# AGENTS.pl.md — projekt AMC (analiza mediów)

*English: [AGENTS.md](AGENTS.md)*

Instrukcje dla agentów AI (Claude, Codex, Gemini itd.) pracujących w tym repozytorium.
Czytaj w całości przed każdą sesją. Zasady kursu: `docs/course_rules_pl.pdf`.

## 1. Temat projektu

> **Temat (z listy prowadzącego): „Jak estymować widoczność postów społecznościowych”**

### Kontekst biznesowy (od zleceniodawcy — Brand24)
- Platformy społecznościowe **często nie ujawniają rzeczywistej widoczności (zasięgu/wyświetleń)
  postów** — częściowo dlatego, że same zarabiają na reklamach (np. AdSense i odpowiedniki) i
  zasięg jest dla nich zasobem handlowym.
- **Brand24** monitoruje wzmianki o markach i chce umieć **estymować** widoczność tam, gdzie
  platforma jej nie podaje (Facebook, Instagram, częściowo X).
- Obecnie Brand24 robi to **prostą stałą: zasięg ≈ stała × liczba obserwujących autora**
  (pole `b24_estimated_reach`). Zleceniodawca przekazał zespołowi informację o tej stałej —
  **wartość/wartości stałej: _[UZUPEŁNIĆ z informacji od prowadzącego/Brand24]_**.
- Cel badawczy: czy da się estymować widoczność **lepiej niż stałą** (np. z followers,
  engagementu: likes/comments/shares, platformy, języka, czasu) i **z jaką niepewnością**;
  gdzie mamy twarde dane (TikTok, YouTube: `view_count`) służą jako dane odniesienia
  (ground truth), a na ich podstawie oceniamy, czy podejście da się przenieść na platformy bez
  wyświetleń. Uczciwy wynik negatywny („na FB/IG nie da się tego zwalidować”) jest w pełni
  akceptowalny.

Dane: `data/raw/reach_samples_students.csv` — 12 500 wzmianek (po 2 500 z X/Twittera, Instagrama,
YouTube, Facebooka, TikToka) z kolumnami `view_count`, `author_followers`,
`likes/comments/shares` oraz `b24_estimated_reach`. Zasięg jest liczony **różnie zależnie od
platformy** (patrz §3).

Miara sukcesu wg kursu: **rzetelny, zweryfikowany wniosek o zakresie i pewności adekwatnej do
dowodów**. Wynik negatywny (brak efektu / niemożność pomiaru) jest pełnoprawnym wynikiem.
Nie liczy się sam wynik modelu (np. F1).

## 2. Zasady kursu, które wiążą agenta

- Wynik końcowy: 40% preprint (≤8 stron, zajęcia 5), 20% plakat (zajęcia 7), 20% kamienie
  milowe M1–M3, 20% aktywność. Punkty skalowane wkładem indywidualnym.
- **Oceniane są**: historia commitów, **logi sesji agentów AI**, test pangram (czy tekst
  publikacji jest generowany), obrona indywidualna.
- Z tego wynika:
  1. **Zapisuj log każdej sesji** w `logs/RRRR-MM-DD_<osoba>_<agent>.md` (cel, prompty, co
     zmieniono, decyzje). Nie kasuj logów.
  2. **Nie generuj finalnego tekstu preprintu/plakatu.** Pomagaj w analizie, kodzie, wykresach
     i krytyce; tekst piszą ludzie własnymi słowami. Każde użycie AI ujawniamy w preprincie.
  3. Commity małe, opisowe, od konkretnej osoby. Liczba commitów ≠ wkład, ale ich brak = brak śladu.
  4. Wszystko musi być odtwarzalne: skrypt/notebook od surowych danych do wykresu.
- Dane: zgodność z **RODO** i regulaminami platform. Nie wyciągaj ani nie publikuj danych
  osobowych (`author_name`, `author_url`, `url`) poza to, co konieczne; w raportach agreguj
  lub pseudonimizuj. Na plakacie/preprincie żadnych nazw prywatnych autorów.
- Nie fabrykuj liczb, cytowań ani wyników. Niepewność pokazuj (przedziały, bootstrap).

## 3. Dane — co już wiemy (EDA wstępna, 2026-10-12)

`data/raw/reach_samples_students.csv`: 12 500 wierszy × 19 kolumn, 348 marek, głównie EN (85%), kraje
często puste (53%). Daty: 2025-01-01 → 2026-10-10, ale **~85% wierszy z kwietnia–sierpnia 2026**
(wcześniej ~250–330/mies., od 2026-04 ~1 000–1 900/mies.) → próbka nie jest równomierna w czasie.
Brak duplikatów `mention_id` i `url`.

| platforma | `view_count` | `author_followers` | `shares` | reach == views | reach = 0 |
|---|---|---|---|---|---|
| tiktok | 100% | 100% | 100% | **100%** | 0% |
| youtube | 99% | 100% | 59% | **99%** | 2% |
| twitter | 75% | 100% | 100% | 1% | 11% |
| facebook | **0%** | 73% | 100% | – | 9% |
| instagram | **0%** | 8% | **0%** | – | **43%** |

Wnioski robocze (do zweryfikowania, nie do cytowania jako fakt):
- TikTok i YouTube: zasięg ≈ liczba wyświetleń (miara „twarda”).
- Facebook/Instagram: brak wyświetleń → zasięg jest **modelowany/szacowany** (mediana reach/followers
  ok. 6% FB, 3% IG); Instagram ma 43% zer.
- Twitter: reach ≠ views i ≠ followers — wygląda na szacunek (mediana reach/followers ≈ 10%).
- **Stała Brand24 widoczna w danych** (reach / followers, tylko wiersze z followers>0 i reach>0):
  - Twitter: mediana ≈ **0,10** (najczęstsza wartość 0,1; ~75% wierszy ≤ 0,1; reszta to wyjątki,
    prawdopodobnie posty z realnymi wyświetleniami — do sprawdzenia).
  - Facebook: stała **schodkowa, malejąca z rozmiarem konta**: ok. 0,10 (małe konta) → 0,06 →
    0,05 → 0,04 → 0,02 (największe). Czyli nie jedna stała, tylko progi.
  - Instagram: brak czytelnej stałej (tylko 8% wierszy ma followers, 43% zer).
  - Do zweryfikowania na pełnych danych i porównania z informacją od Brand24.
- Rozkłady są silnie prawoskośne (mediana reach 366, średnia 15 652, max ≈ 16 mln) → używaj
  skali log, median, kwantyli; nie średnich.
- Braki: `sentiment` 42% NaN, `title` 20%, `author_*` 24%, `view_count` 45%.
  Braki **nie są losowe** (zależą od platformy) — raportuj je per platforma.
- Typy: `author_followers`, `view_count`, `likes`… są `float` przez NaN.

## 4. Role w zespole (3–4 osoby; uzupełnij imiona)

| Rola | Osoba | Odpowiedzialność |
|---|---|---|
| Lead badawczy | _TBD_ | pytanie, hipotezy, spójność wniosku |
| Inżynier danych | _TBD_ | pozyskanie danych, RODO/regulaminy, *data statement* |
| Inżynier pipeline'u | _TBD_ | codebook, gold standard, walidacja instrumentu, analiza błędów |
| Inżynier analityk | _TBD_ | analizy odporności, wizualizacje, A5 (w 3 os. łączy lead + metrolog) |

Agent ma wspierać rolę osoby, która go uruchomiła — i logować, której.

## 5. Plan krok po kroku

### Zajęcia 1–2 → **M1: karta eksploracji danych** (to jest „jutro”)
Karta ma zawierać: pytanie badawcze + hipotezę, źródła danych, zakres próby, wyniki wstępnej
eksploracji, plan pomiaru zjawiska, ograniczenia danych, możliwe źródła błędu, uzasadnienie
wykonalności w skali kursu. Oceniane: konkretność pytania, jakość eksploracji, spójność planu.

1. **Potwierdź temat** z listy prowadzącego; ewentualnie dostosuj §1.
2. **Skład zespołu i role** (§4); `git init`, pierwszy commit, `logs/`.
3. **Sprecyzuj pytanie i hipotezy** (falsyfikowalne), np.:
   - H1: na TikToku/YouTube reach = views (miara bezpośrednia).
   - H2: na FB/IG/X reach jest funkcją followers/engagement; sprawdź, jak dobrze.
   - H3: błąd/niepewność szacunku różni się między platformami.
   - Każda hipoteza: jaki wynik ją obali?
4. **EDA** (notebook `notebooks/01_eda.ipynb` lub `src/eda.py`):
   braki per platforma, rozkłady (log), reach vs views / followers / engagement,
   zera, outliery (max 16 mln!), rozkład w czasie (skok od 2026-04), marki/języki/kraje,
   duplikaty i bliskie duplikaty treści, boty/spam (np. oferty kuponowe, „AI screener”).
5. **Plan pomiaru**: czym mierzymy „rzetelność zasięgu” bez prawdy bazowej? (np. zgodność z
   `view_count` tam, gdzie jest; regresja/odtworzenie reguły; stabilność; wpływ metrics_updated_at).
   Wskaż, że prawdziwego zasięgu na FB/IG nie znamy → zakres wniosków ograniczony.
6. **Karta M1** (≤2 strony, `docs/m1_exploration_card_pl.*` (+ wersja `_en`)) + 2–4 kluczowe wykresy.
7. Zaplanuj kolejne etapy i podział pracy do zajęć 3.

### Zajęcia 3 — codebook i gold standard; walidacja instrumentu
Zdefiniuj kodowanie (np. typ wzmianki: organiczna / reklama / spam / bot), ręcznie oznacz
losową, warstwowaną próbę (gold standard), policz zgodność koderów (kappa), zwaliduj
instrument na danych odniesienia.

### Zajęcia 4 — pierwszy wynik na pełnych danych
Uruchom pipeline na całości, pierwsze estymacje z niepewnością, analiza błędów.

### Zajęcia 5 — **M2: red teaming raportu** (+ preprint ≤8 str.)
Wersja raportu z metodologią i wynikami; turniej: zgłaszanie i obrona przed uwagami.
Przygotuj listę własnych słabych punktów **przed** turniejem; nie powtarzaj uwag już użytych.

### Zajęcia 6 — konsultacje · Zajęcia 7 — **M3** (plakat) · Zajęcia 8 — sesja plakatowa
Plakat = ta sama treść co preprint w warstwie prezentacji: hierarchia informacji, czytelne
wykresy z niepewnością, zwięzły język, wierność przekazu. Prezentacja 8 min + obrona indywidualna.

## 6. Konwencje repozytorium

```
AGENTS.md, AGENTS.pl.md         # ten plik (EN / PL)
README.md, README.pl.md         # opis projektu (EN / PL)
docs/course_rules_pl.pdf        # zasady kursu (nie zmieniać)
data/raw/reach_samples_students.csv   # surowe dane (poza gitem): NIGDY nie edytować, tylko czytać
data/processed/                 # dane pochodne (generowane skryptem)
notebooks/  src/                # analizy
figures/                        # wykresy (generowane, z podpisami i osiami), *_en.png / *_pl.png
docs/                           # karta M1, codebook, data statement, preprint: wersje *_en i *_pl
pitchdeck/                      # prezentacja, pitchdeck_en.* / pitchdeck_pl.*
logs/                           # logi sesji agentów (obowiązkowe, poza gitem)
```

- Python + pandas; losowość z ustalonym `seed`. Dokumenty prowadzimy w dwóch wersjach językowych
  (angielska `_en`, polska `_pl`, poza tym te same nazwy plików); kod i komentarze po angielsku.
- Komunikaty commitów piszemy po angielsku, krótko i opisowo. Nie dodajemy agentów AI jako współautorów ani contributorów w commitach.
- Wykresy: osie opisane, jednostki, legenda, niepewność; skala log dla zasięgu.
- Przed zmianą danych/skryptów: `git status`; nie nadpisuj cudzej pracy.

## 7. Czego agent NIE robi

- Nie pisze finalnego tekstu preprintu/plakatu ani nie „ulepsza” go tak, by ukryć użycie AI.
- Nie modyfikuje surowego CSV. Nie wysyła danych do zewnętrznych usług bez zgody zespołu.
- Nie wyciąga wniosków przyczynowych z korelacji; nie przesadza z pewnością wyników.
- Nie raportuje metryk modelu jako celu samego w sobie — liczy się wniosek i jego zakres.
