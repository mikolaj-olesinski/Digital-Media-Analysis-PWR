# Pitch deck — wersja „claude”

## Co zrobiono
- 7 slajdów 16:9, po polsku, wygenerowanych skryptem `build.py` (python-pptx): `pitchdeck.pptx` + `pitchdeck.pdf`.
- Układ wg szablonu prowadzącego (mail z 7.10, „5 slide rule” w wersji badawczej):
  1 tytuł + hak · 2 kontekst: platformy ukrywają zasięg · 3 **hipoteza** (+ kiedy obalona) ·
  4 **koncepcja technologiczna** (4 kroki + pasek danych) · 5 **klient = społeczeństwo** (5 grup) ·
  6 **impact** (szacunek-zabawa: lejek) · 7 **moat = wykonalność + ekspertyza** (dane + stan wiedzy).
  Usunięto dawne slajdy „Co tracimy”, „Co z tego wyjdzie” (wizual przeniesiony na slajd hipotezy) i „Plan na semestr”.
- Paleta: głęboki fiolet, koral, żółty, niebieski, mięta na kremowym tle; Calibri; tytuły 38 pt, tekst 20–28 pt.
  Ikony i diagramy z kształtów, bez clipartów.
- Efekty (wersja dopracowana): przejścia **Morph** (kolorowe koła ze slajdu tytułowego płyną do rogów kolejnych
  slajdów, pasek postępu u dołu się wydłuża), **animacje wejścia** (karty wpływają od dołu po kolei, same, bez klikania),
  miękkie cienie, gradienty na kolorowych elementach, gradientowe tło slajdu tytułowego.
  Morph wymaga PowerPointa 2019/365; starsze wersje pokażą zwykłe przenikanie. PDF pokazuje stan końcowy slajdów.
- Notatki prelegenta na każdym slajdzie z szacowanym czasem.
- Slajdy wyrenderowane do PNG przez PowerPoint (`render/`) i obejrzane po każdej zmianie.

## Czas wystąpienia (szacunek z liczby słów, ~145 słów/min)
| slajd | czas |
|---|---|
| 1 | ~30 s |
| 2 | ~40 s |
| 3 | ~35 s |
| 4 | ~50 s |
| 5 | ~30 s |
| 6 | ~30 s |
| 7 | ~35 s |
| **suma** | **~245 s (≈ 4 min)** |

Notatki mają ok. 590 słów. Wolniejsze tempo lub krótkie pauzy dadzą ok. 5 minut.

## Liczby na slajdach (ze skryptu, z `reach_samples_students.csv`)
- 12 500 wzmianek, po 2 500 z 5 platform (slajd 5). Odsetek wzmianek ze znanymi wyświetleniami (`view_count`):
  TikTok 100%, YouTube 99%, X/Twitter 75%, Facebook 0%, Instagram 0% (liczone w `build.py`).
- Nie podano wartości stałej Brand24 ani żadnych wyników modelu (żadnych jeszcze nie ma).
- Wizual „dziś vs nasz pomysł” na slajdzie 3 jest ilustracją idei, nie danymi.
- Impact (slajd 6): 5,24 mld kont — DataReportal/Kepios, Digital 2025 (tożsamości, nie unikalne osoby);
  2 400+ firm — deklaracja Brand24 (brand24.com/customers; trackery zewnętrzne podają mniej);
  ~12 000 osób — założenie 5 czytelników raportów na firmę (zabawa, prowadzący tego nie ocenia).
- Stan wiedzy (slajd 7): Cha i in. 2010 (ICWSM, „million follower fallacy”); Szabo i Huberman 2010 (Communications
  of the ACM, przewidywanie popularności treści); Bakshy i in. 2011 (WSDM, wpływ na Twitterze, pojedyncze kaskady
  trudne do przewidzenia); Angelopoulos i Bates 2023 (Foundations and Trends in ML, wstęp do conformal prediction).
  Cytowania z wiedzy agenta — przed preprintem sprawdzić pełne opisy bibliograficzne.

## Czego nie zweryfikowano
- „Platformy ukrywają zasięg, bo sprzedają reklamy” i „Brand24 musi szacować wzorem obserwujący × stała” — z kontekstu zadania, nie sprawdzano źródłowo.
- Konsekwencje na slajdzie 3 (ślepa ocena, złe porównania, spóźniona reakcja, raporty) to rozumowanie, nie badanie.
- Plan semestru (slajd 7) pochodzi z harmonogramu w `AMC (1).pdf`.
- Wygląd poza PowerPointem (Google Slides, LibreOffice) nie był sprawdzany.

## Odtworzenie
`pip install python-pptx pandas` → `python pitchdeck/claude/build.py`. PDF i PNG: eksport z PowerPointa (`SaveAs` 32 / `Export`).
