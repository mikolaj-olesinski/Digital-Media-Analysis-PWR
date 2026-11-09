# Werdykt: czy widoczność da się przewidzieć i czy jesteśmy lepsi od Brand24?

Wewnętrzne, uczciwe podsumowanie odpowiedzi, o które zespół zapyta. Nie tekst preprintu. Liczby: `evidence_map.md`.

## Czy widoczność da się przewidzieć?

- **Typowy post marki:** tak, jako rząd wielkości z zakresem. MAE 0,273 / 0,456 / 0,454 (TikTok / YouTube / X) = ×1,9 /
  ×2,9 / ×2,8; pasmo 90% to ×14 / ×51 / ×98 (E118/E141).
- **Konkretny post, dokładna liczba:** nie. Na X pasmo 90% obejmuje dwa rzędy wielkości.
- **Świeży post bez reakcji:** tylko zakres; MAE 0,60-0,76 i pasmo ×260-900 (E130).
- **Który post będzie viralem:** nie. Surowe lajki i reguła Brand24 rankują lepiej na top 1% (E104/E106).
- **Instagram:** brak jakiegokolwiek pomiaru (0/160 anonimowo).
- **Facebook powyżej 1M followersów:** tylko punkt, bez skalibrowanego pasma (E147/E151).
- **Sufit jest blisko:** szum 0,22 / 0,39 / 0,33 (E56); tekst, graf, kolumny i multi-task dodają ≤0,01.

## Czy jesteśmy lepsi od Brand24?

- **Wobec ich eksportowanej kolumny (`b24_estimated_reach`), na X i Facebooku: tak.** X MAE 0,454 wobec 1,08 (58% mniej),
  Facebook 0,385 wobec 0,93 (59% mniej), a model wygrywa w każdym tierze followersów (E165). Dajemy też przedziały,
  których kolumna nie ma.
- **Na TikToku i YouTube: nie, i nie ma czego wygrywać.** Ich kolumna równa się liczbie wyświetleń (100% / 99% wierszy),
  czyli jest pomiarem z zerowym błędem; nasz model ma 0,273 / 0,456.
- **W rankingu/alertach: oni są lepsi** (lajki i ich reguła biją model na top 1%).
- **Na Instagramie: oni mają liczbę, my nie mamy żadnej.**
- **Wobec ich modeli wewnętrznych: nie wiadomo.** Widzimy eksportowaną regułę, nie to, co Brand24 liczy w środku.

## A jeśli nie znamy historii autora?

- Model zostaje **bazowy** (followersi, reakcje, flagi, wiek). Cecha autora pomaga tylko na 1-8% postów, których autor jest
  znany (E119: X 0,530 -> 0,430, YouTube 0,465 -> 0,446).
- **Historia marki nie zastępuje historii autora** (E169): podział czasowy z wygładzonym priorem marki zmienia MAE o -0,005
  (TikTok), -0,035 (YouTube) i +0,006 (X, CI zawiera 0); tylko 35-52% postów testowych ma znaną markę.
- Zamienniki, które działają: jeden wcześniejszy **odczyt wyświetleń** tego posta (E40: 0,165 / 0,188 / 0,269) oraz
  kalibracja na 25-50 postach klienta na X (E115).

## Jedno zdanie

Widoczność da się przewidzieć jako poziom i zakres dla typowego posta, lepiej niż reguła Brand24 na X i Facebooku; nie da
się jej przewidzieć jako dokładnej liczby, dla świeżego posta, dla wiralności i wcale na Instagramie.
