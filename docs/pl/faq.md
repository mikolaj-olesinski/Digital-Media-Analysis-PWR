# FAQ dla nowej osoby

Krótkie odpowiedzi na pytania, które pojawiają się najpierw, dla zespołu. Sformułowania dla klienta: `client_qa.md`. Liczby i ID eksperymentów: `docs/pl/evidence_map.md`.

**Dlaczego nie ma wyniku dla Instagrama?**
Instagram nie udostępnia licznika wyświetleń bez logowania: próba anonimowa zwróciła wyświetlenia dla 0 z 160 postów.
Lajki i komentarze dla 115 z 160 zapisano, ale bez wyświetleń nie ma czego zwalidować. Opcje: próbka ręczna, konto z
logowaniem albo dostawca (pytanie Q12).

**Dlaczego pasmo X jest tak szerokie (×98), jeśli model jest dobry?**
X ma 25% brakujących wyświetleń, nielosowo (66% postów z 2025 vs 22% z 2026) i 64% postów bez reakcji (E105). Pasmo 90%,
które pokrywa te przypadki, musi być szerokie; pasmo 50% („prawdopodobne”) to ×7,6 (E137).

**Dlaczego nie używać modelu do alertów?**
Na top 1% ranking po surowych lajkach, albo po regule Brand24 na TikToku/YouTube, bije model MAE: precyzja 0,64 vs 0,56
na TikToku, 0,64 vs 0,48 na YouTube, 0,47 vs 0,26 na X (E104); przy stałym budżecie alertów model potrzebuje ich dużo
więcej (E106). Model jest dobry od poziomów, nie od znajdowania następnego wiralnego posta.

**Dlaczego pasmo Facebooka jest nieskalibrowane powyżej 1M followersów?**
Tylko 70 pobranych postów pochodzi ze stron powyżej 1M. Pokrycie zostaje na 0,80-0,86 nawet po poszerzeniu, przesunięciu
poziomu, kwantylach per tier czy modelu dużych stron (E147/E151). Przy takiej próbce tier nie udźwignie twierdzenia o 90%.

**Dlaczego graf nie pomaga?**
Krawędzie, które mamy, to krawędzie wspólnego publikowania (autor-marka-hashtag), a nie dyfuzji. Średnia reszta sąsiednich
postów treningowych nie przewiduje reszty posta (autor rho ≤0,07, marka/hashtag ~0; E128); cechy ręczne, GraphSAGE,
embeddingi spektralne i metapath2vec zawodzą lub szkodzą (E65/E122/E127/E128).

**Dlaczego tekst nie pomaga?**
Reakcje już niosą sygnał. Tekst dodaje ≤0,01 obok reakcji (E100); bez reakcji pomaga więcej, ale przy publikacji model
jest wciąż znacznie gorszy niż z reakcjami (E43/E130).

**Dlaczego zera reakcji nie są traktowane jak braki?**
Na YouTube oznaczenie zer jako braków zmienia MAE o 0,003 (E116); zera to małe posty, nie dziury w zbieraniu. Post 0/0
przy pierwszym skanie ma jednak wyświetlenia: mediana 5 na X, 60 na TikToku (E117/E126).

**Dlaczego sumy marek są takie niestabilne?**
Zasięg jest zdominowany przez ogon: top 1% postów to 60-74% wyświetleń (E96). Pięć postów daje rozrzut ×53 p90/p10 sumy
marki; nawet 20 postów daje ×6,4 (E112). Kalibracja i przedziały pomagają, ale nie wystarczą, by porównywać marki na
kilku postach.

**Dlaczego nie ma twierdzeń przyczynowych?**
Wszystko jest obserwacyjne, a próbka nie jest losowa; model opisuje związki w tej próbce. Słowa „powoduje” czy „napędza”
nie są poparte projektem.

## Częste nieporozumienia

Same pojęcia; którą liczbę cytować i gdzie jest w `doc_status.md` („Typowe błędy przy cytowaniu liczb”).

- **„Wyświetlenia to zasięg.”** Wyświetlenia to nie unikalne osoby; platformy nie publikują unikalnego zasięgu dla
  dowolnych postów.
- **„MAE 0,3 to 30% błędu.”** To log10: 0,3 to czynnik ×2, 0,5 to ×3, 1,0 to ×10.
- **„Pasmo 90% i 50% to to samo twierdzenie.”** 50% to zakres prawdopodobny; 90% to uczciwy najgorszy przypadek.
- **„Liczby dla Facebooka to 0,345.”** To 130 ręcznych postów; pełna próbka 402 to 0,385 i tę się cytuje.
- **„Model bije regułę wszędzie.”** Bije na poziomach (MAE); w rankingu na top 1% wygrywają surowe lajki/reguła.
- **„Działa na Instagramie.”** Nigdy tam nie był walidowany; nie ma publicznych wyświetleń.
- **„Pasmo >1M to 90%.”** Jest nieskalibrowane (0,80-0,86); podawać tylko punkt.
