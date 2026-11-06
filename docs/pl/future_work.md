# Co jeszcze można zmierzyć

Wewnętrzna lista, uporządkowana po wartości na jednostkę pracy. Niczego tu nie ma, dopóki nie pojawi się w rejestrze.
Które punkty planu są zamknięte, a które otwarte: `plan_status.md`. Otwarte pozycje są w backlogu jako B25-B27.

## Duża wartość, zablokowane dostępem lub decyzją

| pozycja | dlaczego | praca | blokada |
|---|---|---|---|
| **Wyświetlenia Instagrama** | jedyna platforma bez jakiegokolwiek pomiaru; 0/160 anonimowo | 1-2 dni (próba ręczna 100-150 postów) albo dostawca | decyzja o logowaniu/dostawcy i regulaminie |
| **Historia autora na skalę** | największa znana dźwignia (+0,10 MAE na znanych autorach, E68/E119), ale tylko 1-8% naszych postów ją ma | 0,5 dnia, jeśli Brand24 wyeksportuje kilka postów na autora | dane od klienta (Q8) |
| **Fałszywe dopasowania marek** | złe przypisanie marki psuje sumy marek, nie pojedyncze posty | 2-3 dni na 100-200 ręcznych etykiet | ludzka capacity do etykietowania |

## Średnia wartość, dane już są

| pozycja | dlaczego | praca |
|---|---|---|
| **Miesięczne sumy marek z przedziałami** | klient raportuje miesięcznie; mamy tylko sumy całościowe (E109/E112/E157) | 1 dzień |
| **Sprawdzenie innych języków** | 85% próbki to angielski; model może nie działać dla innych języków | 0,5 dnia (analiza podzbioru) |
| **Dłuższy horyzont czasowy** | rolling-origin obejmuje 2-3 miesiące (E136); odświeżenie w 2027 przetestowałoby drift | 0,5 dnia po nowych danych |

## Niska wartość lub zablokowane przez dane

| pozycja | dlaczego nie teraz |
|---|---|
| Zwykłe posty na Facebooku (nie wideo/Reel) | brak licznika wyświetleń; nie da się zwalidować |
| Unikalny zasięg (osoby) | niepubliczny na żadnej platformie |
| Prawdziwy graf interakcji (udostępnienia, odpowiedzi) | Brand24 nie ma krawędzi; bez nich linia grafowa pozostaje zamknięta (E127/E128) |
| Więcej rodzin modeli | sufit szumu (E56) zostawia 0,06-0,14 na YouTube/X i mniej na TikToku; tekst/graf/kolumny już testowane (E100/E121/E122) |
| Dane kalibracyjne od klienta | E115 pokazuje, co by dały; klient zapowiedział, że nie będzie zbierał postów |

## Co byśmy zrobili najpierw, mając jeszcze tydzień

1. Instagram: ręczna próbka 100-150 Reels z wyświetleniami (jedyna niezmierzona platforma).
2. Miesięczne sumy marek z przedziałami, bo klient raportuje miesięcznie.
3. Drugą próbkę ręczną tylko pod fałszywe dopasowania marek, jeśli znajdzie się osoba.

## Po zakończeniu kursu

- **Opublikować pomiar Facebooka.** 402 publiczne posty z wyświetleniami vs reguła to najbardziej nowatorski element;
  wymaga sprawdzenia zgody na publikację danych i regulaminów.
- **Historia autora z Brand24.** Jeśli klient wyeksportuje kilka postów na autora, największy znany zysk (+0,10 MAE na
  znanych autorach) staje się dostępny na skalę; to naturalny następny paper.
- **Instagram z dostawcą.** Jedyna niezmierzona platforma; płatne lub zalogowane źródło domknęłoby największą dziurę.
- **Miesięczne sumy i dłuższy horyzont.** Oba tanie, gdy pojawią się dane; miesięczne to to, co raportuje klient.
- **Prawdziwy graf interakcji.** Udostępnienia/odpowiedzi pozwoliłyby sprawdzić, czy linia grafowa się otwiera; bez nich
  pozostaje zamknięta (E127/E128).
- **Kalibracja przy brakach.** Paper metodologiczny mógłby sformalizować przełączanie „przetrenuj bez cechy” (E163/E164)
  i fallbacki CQR (E144/E145).
