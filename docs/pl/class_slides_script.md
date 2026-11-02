# Tekst do slajdów na zajęcia

Roboczy tekst do `pitchdeck/class_pl.pdf`. To nie jest tekst preprintu. Warto powiedzieć go własnymi słowami. Pełna wersja do nauki: `pitchdeck/study_build.py`.

## Slajd 1: co wychodzi z danych

Platformy często ukrywają, ile osób zobaczyło post, a Brand24 i tak musi to oszacować. Mamy 12 500 postów z pięciu platform. Prawdziwe wyświetlenia znamy tylko na TikToku, YouTube i części X, więc tam sprawdzamy, co działa.

„Pomyłka ×3” znaczy, że szacunek jest typowo trzy razy za duży albo za mały. Stała razy liczba obserwujących myli się 7–8 razy. Zwykły model drzewiasty z reakcjami, nawet bez strojenia, myli się 2–3 razy. Na X zasięg Brand24 myli się około 12 razy, a nasz model około 3 razy.

Gdy dajemy modelowi tylko jedną informację, najlepiej działają lajki. Na TikToku same lajki dają pomyłkę około ×2. Sam typ posta nie mówi prawie nic.

Dodaliśmy też etykiety AI: zwykły post, reklama albo spam, oraz wydźwięk. Szacunku prawie nie poprawiają, ale pokazują, że przy tych samych reakcjach reklamy mają 1,5–1,9 razy więcej wyświetleń.

**Co z tego wynika:**
- Do dobrego szacunku potrzebne są reakcje, a nie liczba obserwujących.
- Nie trzeba skomplikowanego modelu, żeby wyraźnie pobić prostą stałą.
- Etykiety przydają się do opisu danych, a nie do samego szacowania.

## Slajd 2: jakie modele i podejścia

Sprawdziliśmy 10 rodzajów modeli: od prostego ridge, przez lasy losowe, po cztery boostingi i małą sieć neuronową. Każdy dostał 40 zestawów ustawień. Ustawienia dobieramy bez zaglądania w dane testowe.

Najlepiej wypadły SVR na TikToku i CatBoost na YouTube i X: pomyłka około ×1,9, ×2,8 i ×3. Różnice między dobrymi modelami są małe.

Dokładaliśmy tekst, kalendarz, etykiety i cechy z grafów. Tekst daje drobną poprawę na YouTube i X, a reszta prawie nic. Sprawdziliśmy też, co się dzieje, gdy usuwamy każdą grupę kolumn. Najbardziej boli usunięcie lajków.

**Co z tego wynika:**
- Najważniejsze są dane, a nie wybór modelu.
- Strojenie jest potrzebne, żeby porównanie było uczciwe, ale nie zmienia obrazu.
- Brak poprawy to też wynik: po uwzględnieniu reakcji niewiele zostaje do zyskania.

## Slajd 3: inna platforma i kalibracja

To najważniejsze pytanie, bo Facebook i Instagram nie mają wyświetleń. Sprawdzamy więc, czy model działa na platformie, której nie widział. Najlepiej wypada tu prosty ridge na samych lajkach i komentarzach, czyli kolumnach dostępnych wszędzie.

Na TikToku i YouTube wynik jest prawie taki sam jak przy uczeniu na tej samej platformie: ×2,1 i ×4,1. Na X jest ×4,5 zamiast ×3,3. Model dobrze szereguje posty, ale zawyża je wszystkie około 2,5 razy.

Kalibracja to proste przesunięcie wszystkich predykcji o typowy błąd na kilku postach z prawdziwymi wyświetleniami. Na X około 50 postów obniża pomyłkę z ×4,5 do ×3,5, prawie jak kalibracja idealna. Przy 5–10 postach wynik mocno zależy od losowania.

Brand24 nie ma takiej próbki z Facebooka ani Instagrama, więc planujemy zebrać własną.

**Co z tego wynika:**
- Kolejność postów się przenosi, a skala niekoniecznie.
- Na X wystarczy kilkadziesiąt postów, żeby naprawić skalę. To konkretna rekomendacja dla firmy.
- Kalibracja usuwa tylko przesunięcie. Pozostała pomyłka, na X około ×3,5, zostaje.
- Na Facebooku i Instagramie to nadal niesprawdzone oszacowanie.

## Slajd 4: grafy

Pomysł był taki, że post może zyskać na tym, co wiemy o podobnych postach. Każdy post to punkt, a łączymy go z dziesięcioma najbardziej podobnymi tekstami. Algorytm Louvain sam dzieli taki graf na gęste grupy, bez wyświetleń i bez etykiet.

Sprawdziliśmy 18 wariantów cech z grafów: grupy, PageRank, grafy autorów, hashtagów, czasu i duplikatów, a także sąsiadów z innych platform. Najwięcej daje to ostatnie: +0,009 na YouTube, czyli bardzo mało. Propagacja etykiet nawet pogarsza wynik.

Grafy dobrze opisują za to dane. Same wychodzą branże: samochody, hotele, fast food. Prawie identyczne posty to na X w 38% spam, a w reszcie w 9%.

Ciekawostka: wzmianki przypisane do sieci komórkowej Cricket Wireless na TikToku są o krykiecie, 0 z 20 o firmie. Monitoring po słowie kluczowym złapał coś innego. To mała próba, więc chcemy zmierzyć, jak często to się zdarza.

**Co z tego wynika:**
- Grafy z podobieństwa tekstu nie poprawiają szacunku.
- Pomagają znaleźć problemy w danych: kampanie, spam i wzmianki nie o marce.
- To może być naszym wyróżnikiem, ale na razie mamy jeden przykład.

## Slajd zapasowy: niepewność

Załóżmy, że model mówi: 1000 wyświetleń. Gdzie jest prawda? To zależy od liczby reakcji. Na YouTube przy braku reakcji 90% postów mieści się w szerokim zakresie około 19 razy w każdą stronę. Powyżej 1000 reakcji to już tylko około 3 razy.

Dlatego zakres podajemy zależnie od liczby reakcji. Znamy ją w chwili szacowania, więc da się to zastosować w praktyce.

**Co z tego wynika:**
- Przy dużej liczbie reakcji można ufać szacunkowi bardziej.
- Post bez reakcji trzeba oznaczać jako mało pewny.
- Zakres zależny od reakcji uczciwiej opisuje niepewność niż jedna szerokość dla wszystkich.
