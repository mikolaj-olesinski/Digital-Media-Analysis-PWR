# Pytania klienta: co mówić, czego nie mówić

Krótkie odpowiedzi na spotkanie z Brand24, każda poparta eksperymentem. Nie tekst preprintu.

## Co mówić

**„Czy to lepsze niż nasza stała?”**
Tak na X i Facebooku. Na X MAE modelu to 0,454 (×2,8) wobec około 1,0 dla reguły; na Facebooku 0,385 wobec 0,93,
i wygrywa w każdym tierze followersów (E165). Na TikToku i YouTube wasz zasięg **już jest** liczbą wyświetleń, więc nie
ma tam czego poprawiać.

**„Czy da się oszacować Instagram?”**
Nie, nie z danych publicznych. Próba anonimowa nie zwróciła wyświetleń dla 0 z 160 postów; Instagram wymaga logowania.
Dla Instagrama możemy podać reakcje, nie widoczność.

**„Jak dokładne to jest?”**
MAE 0,27 / 0,46 / 0,45 (TikTok / YouTube / X) — typowy czynnik ×1,9 / ×2,9 / ×2,8. Na Facebooku 0,385 (×2,4). Pasmo 90%
to ×14 / ×51 / ×98; pasmo 50% („prawdopodobne”) to ×2,8 / ×5,5 / ×7,6 (E118/E137/E141).

**„Czy można tego użyć do alertów?”**
Nie do alertów. Na top 1% ranking po surowych lajkach (albo wasza reguła na TikToku/YouTube) bije model MAE; model jest
od poziomów i przedziałów, nie od znajdowania następnego wiralnego posta (E104/E106).

**„A sumy marek?”**
Raportować z przedziałem i liczbą postów. 5 postów daje rozrzut ×53 p90/p10, 20 postów wciąż ×6,4 (E112). Sumy
na Facebooku są ×2 za niskie przed kalibracją, a 80% przedział marki pokrywa 0,74 (E157/E158).

**„Ile postów musimy zebrać, żeby skalibrować?”**
Tylko X potrzebuje danych: 25-50 postów zbija MAE z 0,650 do 0,551-0,563. TikTok, YouTube i Facebook są już
skalibrowane zero-shot (E115). Dla nich klient nie musi zbierać postów.

**„A jeśli nasze dane są niekompletne?”**
Przetrenować bez brakującej cechy, zamiast karmić pełny model brakami: TikTok bez reakcji 1,139 -> 0,583, YouTube bez
followersów 0,711 -> 0,494 (E163). Przełączać przy ~35% braków reakcji (TikTok/YouTube) lub ~20% braków followersów
(YouTube/X) (E164).

**„Dlaczego X jest trudniejszy?”**
25% wyświetleń X jest brakujących i nie losowo (starsze i mniejsze konta), a zero reakcji jest częste; model potrzebuje
tam szerokiego pasma (E105/E117).

**„Duże strony na Facebooku?”**
Podawać tylko punkt. Powyżej 1M followersów pasmo 90% pokrywa 0,80-0,86 i nie da się go skalibrować na naszych 70 postach
(E147/E151); model zaniża średnią o około ×2.


## Odpowiedź na „jak dokładne to jest?” bez tabeli

- „Typowo w granicach **×2-3**: ×1,9 na TikToku, ×2,9 na YouTube i X.”
- „Dziewięć razy na dziesięć w granicach **×14** na TikToku, **×51** na YouTube, **×98** na X.”
- „Na Facebooku: typowo ×2,4, a zakres 90% to ×98 dla stron poniżej 1M followersów; powyżej 1M podajemy tylko punkt.”
- „Zanim pojawią się reakcje: znacznie szerzej, ×260-900 — za wcześnie na punkt.”
- **Nie** mówić „±20%” ani „±X wyświetleń”: błąd jest mnożnikowy, nie addytywny.

## Gdy klient prosi o screenshot profilu lub posta

Nie wysyłać zrzutów z nazwą autora, handle'em, awatarem lub URL-em: data statement zobowiązuje nas do agregatów. Zaproponować
zamiast tego: ID posta i liczby zbiorcze, kadr bez bloku autora albo wiersz tabeli. Jeśli klient musi zweryfikować
konkretny post, ma rekord w Brand24; możemy potwierdzić odczytane liczniki bez odtwarzania danych osobowych. Prośba o
usunięcie (post lub autor, którego nie powinno być) jest realizowana od razu: usunąć lub zagregować, potem powiedzieć, co
się zmieniło.

## Czego nie mówić

- „To szacuje **zasięg**” — to szacuje **wyświetlenia**; unikalny zasięg nie jest publiczny.
- „Pasmo 90% jest skalibrowane na Facebooku powyżej 1M” — nie jest.
- „Bije lajki w alertach” — nie bije.
- „Instagram jest zwalidowany” — nie jest.
- „Sumy marek są wiarygodne z kilku postów” — nie są.
- „Wyświetlenia powodują X” — wszystko jest korelacyjne.
- „Używać do rozliczeń” — nie; to estymata z szerokim pasmem.

## Czego jeszcze potrzebujemy od klienta

- Potwierdzenia stałych za `b24_estimated_reach` (zrekonstruowane, niepotwierdzone).
- Odpowiedzi, czy zero wyświetleń na YouTube może oznaczać „nie zebrano”.
- Decyzji o Instagramie: próba ręczna, dostęp z logowaniem albo dostawca.


## Wersja 60-sekundowa

Pokazać `figures/fb_rule_vs_model_pl.png`: model przeciw regule klienta, tier po tierze.

„Sprawdziliśmy, czy wyświetlenia posta da się oszacować lepiej niż stała razy followersi. Na X i Facebooku model
zmniejsza błąd o połowę — MAE 0,45 i 0,39 wobec około 1,0 — i daje uczciwy zakres 90%; na TikToku i YouTube wasz zasięg
już jest liczbą wyświetleń, więc nie ma tam czego poprawiać. Używać do poziomów i zakresów, nie do alertów, i nie na
Instagramie, gdzie wyświetlenia nie są publiczne.”

## Gdy klient pyta o liczbę, której nie mamy

- **„Ile wyświetleń ma ten post na Instagramie?”** Nie mamy. Zaproponować reakcje albo ręczną próbkę 100-150 postów,
  jeśli klient da dostęp; nie wymyślać liczby.
- **„Ile wyświetleń ma ten zwykły post na Facebooku (nie wideo/Reel)?”** Facebook ich nie udostępnia; nie ma z czego
  estymować. Powiedzieć to wprost.
- **„Ile osób go zobaczyło?”** Publiczne są tylko wyświetlenia; unikalny zasięg nie. Podawać wyświetlenia i mówić, że to
  nie osoby.
- **„Podajcie sumę marki z tych trzech postów.”** Podać skalibrowaną sumę z bardzo szerokim przedziałem i liczbą postów
  albo odmówić porównania; 5 postów daje rozrzut ×53 (E112).
- **„Podajcie pasmo 90% dla tej strony powyżej 1M followersów.”** Podać tylko punkt; pasmo jest tam nieskalibrowane
  (E147/E151). Zaproponować zachowawczy zakres zamiast fałszywego 90%.
- **„Czy ten post osiągnie 100 tys.?”** Można podać zakres/probabilistykę, nie gwarancję; ogon jest zdominowany przez
  kilka postów (E96).
- **„Dlaczego wasza liczba jest niższa niż Brand24?”** Na Facebooku reguła jest ×4,1 za wysoko (E111); na X ×9,3 (E99).
  My mierzymy wyświetlenia; reguła to estymata.
- **„Czy możecie zagwarantować estymatę?”** Nie. To estymata z szerokim pasmem, nie podstawa do rozliczeń.
