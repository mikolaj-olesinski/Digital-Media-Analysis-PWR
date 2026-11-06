# Pokrycie planu: co zaplanowano i co się stało

Mapuje punkty „Ideas to stand out” (A-I) z `project_plan.md` na wynik. Każdy punkt został uruchomiony, zamknięty lub
zablokowany na danych; żaden nie jest po cichu otwarty.

| punkt | status | dowody |
|---|---|---|
| **A. Własna próbka FB/Instagram** | **częściowo**: Facebook zmierzony (130 ręcznych + 402 pobrane), Instagram zablokowany (0/160 anonimowo, potrzebny login lub dostawca) | E93, E95, E111, E123, E153, E166; data statement |
| **B. Rekomendacja kalibracji** | **zrobione**: X potrzebuje 25-50 postów (0,650 -> 0,551); TT/YT/FB są zero-shot | E115 |
| **C. Przedziały i abstencja** | **zrobione**: CQR 50%/90%, krzywa abstencji z szerokości pasma | E58, E118, E137 |
| **D. Fałszywe dopasowania marek** | **zablokowane**: potrzeba 100-200 ręcznych etykiet od osoby (nie klienta); przykłady udokumentowane, brak detektora | E37 (legacy), backlog B15 |
| **E. Community jako most do FB/IG** | **zamknięte**: graf nie niesie sygnału resztowego; krawędzie wspólnego publikowania to nie dyfuzja | E69, E122, E127, E128 |
| **F. Mniejsze testy** | **zrobione**: posty bez reakcji, wiralność top 10%, kampanie copy-paste | E48, E63, E64 |
| **G. Poprawa ekstremów** | **zrobione**: kwantyle, rekalibracja, ważenie ogona, hurdle, przesunięcia tierów; nic nie bije zwykłego modelu na tyle, by zostać | E57, E114, E135, E140 |
| **H. Zera YouTube** | **zrobione**: traktowanie ich jako braków zmienia MAE o 0,003; zera to małe posty | E116, E117 |
| **I. Przedział per tier reakcji** | **zrobione**: jedna szerokość per tier przywraca pokrycie; CQR jest teraz główną metodą | E34, E118 |

## Co to znaczy

- **A** to jedyny punkt wciąż otwarty i tylko dla Instagrama; część facebookowa to najbardziej nowatorski wynik projektu.
- **D** to jedyna zablokowana analiza, która wymagała ludzkiego etykietowania.
- Reszta została przetestowana, a tam gdzie zawiodła, porażka jest w rejestrze i wiersz mówi o tym wprost.

Co jest otwarte i jaki ma priorytet: `future_work.md`.
