# Projekt w dziesięciu zdaniach

Wewnętrzne podsumowanie dla nowej osoby, nie tekst preprintu. Finalny tekst zespół pisze własnymi słowami.

1. Brand24 szacuje zasięg posta jako stałą razy followersi autora; projekt pyta, czy **wyświetlenia** da się oszacować
   lepiej i z jaką niepewnością.
2. TikTok i YouTube publikują wyświetlenia (ich zasięg to licznik wyświetleń), X publikuje je dla około 75% postów,
   a Facebook i Instagram nie publikują ich wcale.
3. Próbka to 12 500 wzmianek Brand24, 348 marek, 85% angielskich i 85% z kwietnia-sierpnia 2026, więc wyniki opisują tę
   próbkę, a nie całe social media.
4. Zrekonstruowana reguła jest dokładna na TikToku/YouTube, około **×9,3 za wysoka na X** i **×4,1 za wysoka na
   Facebooku** (87% / 82% postów poza 2×).
5. Model CatBoost na followersach, lajkach, komentarzach, udostępnieniach i flagach braków osiąga MAE
   **0,273 / 0,456 / 0,454** (TikTok / YouTube / X) na log10 wyświetleń — typowy błąd ×1,9 / ×2,9 / ×2,9.
6. Przedziały CQR przywracają uczciwe pokrycie: pasmo 90% pokrywa **0,889 / 0,902 / 0,899**, a pasmo 50%
   („prawdopodobne”) ma szerokość ×2,8 / ×5,5 / ×7,6.
7. Facebook został zmierzony na **130 ręcznych i 402 pobranych postach**: transfer ma MAE **0,345 / 0,385** wobec 0,93
   dla reguły, a model wygrywa w **każdym tierze followersów** (0,36-0,46 vs ~1,0).
8. Co nie pomaga: tekst, dodatkowe kolumny, cechy grafowe (w tym metapath2vec), multi-task i uśrednianie ziaren;
   sygnał jest w reakcjach, a gdy są dostępne — we wcześniejszych wyświetleniach autora.
9. Zasięg jest **zdominowany przez ogon** (top 1% postów to 60-74% wyświetleń), więc sumy marek wymagają przedziałów i
   wielu postów, a alerty powinny używać surowych lajków, nie modelu.
10. Instagram jest niezmierzony, pasmo Facebooka powyżej 1M followersów jest nieskalibrowane, a estymator służy do
    poziomów i zakresów — nie do alertów, rozliczeń ani twierdzeń przyczynowych.

Skąd liczby: `docs/pl/evidence_map.md`; metoda: `docs/pl/model_card.md`.
