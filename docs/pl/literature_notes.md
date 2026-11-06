# Notatki z literatury

Szybki przegląd z 2026-11-02 zrobiony agentami AI, a nie przegląd systematyczny. Źródło trzeba przeczytać w całości, zanim trafi do preprintu. Status: **sprawdzone** znaczy, że tytuł, autorów, rok i podaną tezę potwierdzono na stronie źródła albo na stronie abstraktu w arXiv; **tylko wyszukiwarka** znaczy, że źródło widziano w wynikach wyszukiwania i nie potwierdzono.

## Sprawdzone
| Źródło | Co mówi | Związek z naszą pracą |
|---|---|---|
| Shulman, Sharma, Cosley, 2016, *Predictability of Popularity: Gaps between Prediction and Understanding*, ICWSM. [arXiv:1603.09436](https://arxiv.org/abs/1603.09436) | Modele z podglądem początku rozchodzenia się posta uogólniają się między zbiorami (muzyka, książki, zdjęcia, linki). Większość mocy daje tempo pierwszych reakcji; przy innych cechach nawet kierunek korelacji nie jest stały | Blisko nas w duchu: sygnał po publikacji dominuje i się przenosi. Inny sygnał (tempo, a nie reakcje) |
| Sahu i in., 2026, *Benchmarking Multi-Modal Graph-based Social Media Popularity Prediction*. [arXiv:2606.27539](https://arxiv.org/abs/2606.27539) | Benchmark na czterech zbiorach z Bluesky i Reddita; uogólnianie między platformami; grafy interakcji społecznych | Możliwy kontrapunkt: oni mają prawdziwe grafy interakcji, my podobieństwo tekstu. Tylko abstrakt; wniosek agenta o łączeniu platform niepotwierdzony |
| Tibshirani, Barber, Candès, Ramdas, 2019, *Conformal Prediction Under Covariate Shift*. [arXiv:1904.06019](https://arxiv.org/abs/1904.06019) | Ważone przedziały konforemne, gdy cechy w uczeniu i teście się różnią | Metoda do punktu C planu. Dotyczy zmiany cech; nasze przesunięcie poziomu to zmiana samej zależności, więc może nie wystarczyć |
| Sokol, Moniz, Chawla, 2024, *Conformalized Selective Regression*. [arXiv:2402.16300](https://arxiv.org/abs/2402.16300) | Model wstrzymuje się od odpowiedzi przy dużej niepewności, pewność z predykcji konforemnej, plus sposób oceny | Wprost metoda do punktu C planu (odmowa odpowiedzi) |
| Mannocci, Mazza, Monreale, Tesconi, Cresci, 2024, *Detection and Characterization of Coordinated Online Behavior: A Survey*, ACM Computing Surveys 2026. [arXiv:2408.01257](https://arxiv.org/abs/2408.01257) | Przegląd zachowań skoordynowanych | Odniesienie do punktu F planu (kampanie). Użycie sieci podobieństwa tekstu niepotwierdzone na podstawie abstraktu |
| Bekavac, Mayer, 2026, *Platforms' Research API Data Access: What Users See vs. What Researchers can Retrieve*. [arXiv:2601.12390](https://arxiv.org/abs/2601.12390) | API badawcze pomijają do ok. 50% tego, co widzą użytkownicy, i usuwają do ok. 83% metadanych, limity nawet ok. 1000 zapytań tygodniowo (wartości maksymalne, bez podziału na platformy) | Ograniczenie przy sprawdzaniu FB/IG |
| Siegel, 2019, *Introducing the Brandwatch Reach Score*, blog firmowy. [Link](https://www.brandwatch.com/blog/introducing-the-brandwatch-reach-score-making-reporting-simpler-and-more-trustworthy/) | Zasięg z typowych lajków, retweetów i udostępnień względem followersów; dla innych źródeł regresja na lajkach plus założenia. Brak opisu sprawdzenia na prawdziwych wyświetleniach | Firma, nie badania. Ta sama rodzina pomysłów co reguła Brand24 |
| SocialCrawl, 2026, *Instagram Reels Views: What Counts*, blog firmowy. [Link](https://www.socialcrawl.dev/blog/how-instagram-counts-views) | Próba wygodna: 36 kont, 425 postów. Publiczne wyświetlenia ma 196 z 197 filmów i 0 z 228 zdjęć i karuzel | Firma, nie badania. Wspiera punkt A planu tylko dla filmów, nie dla zwykłych postów |
| Rubio-Madrigal, Burkholz, 2026, *Fixed Aggregation Features Can Rival GNNs*, ICML 2026. [arXiv:2601.19449](https://arxiv.org/abs/2601.19449) | Stałe agregacje cech sąsiadów ze strojonym MLP dorównują GNN lub je biją na 12 z 14 benchmarków | Podstawa wariantu A w notebooku 12. Benchmarki różnią się od naszego zadania |
| Luan i in., 2023, *When Do Graph Neural Networks Help with Node Classification?*, NeurIPS 2023. [arXiv:2304.14274](https://arxiv.org/abs/2304.14274) | Zysk z GNN zależy od rozróżnialności węzłów, a nie tylko od homofilii | Przypomnienie, że homofilia w grafie tekstu nie gwarantuje zysku. Klasyfikacja węzłów, nie regresja |
| Roy, Mei, Zeng, Li, 2013, *Towards Cross-Domain Learning for Social Video Popularity Prediction*, IEEE Transactions on Multimedia. [Link](https://www.microsoft.com/en-us/research/?p=273903) | Transfer z tematów na Twitterze na filmy YouTube, by wyjaśnić nagłą popularność | Wcześniejszy przykład uczenia między platformami. Inne zadanie i sygnał |

## Tylko wyszukiwarka: otworzyć przed użyciem
- Sadeque, Bethard, 2019, przegląd przewidywania zaangażowania, [arXiv:1907.05442](https://arxiv.org/abs/1907.05442).
- Bandari, Asur, Huberman, 2012, prognoza popularności newsów z treści, ICWSM. Kontrast: przed publikacją.
- Cao i in., 2019, sieci grafowe do popularności na grafach dyfuzji, [arXiv:1906.09032](https://arxiv.org/abs/1906.09032).
- Spina, Gonzalo, Amigó, ok. 2013, słowa filtrujące do rozróżniania nazw firm na Twitterze. Metoda do punktu D planu. [Link](https://research-repository.rmit.edu.au/articles/journal_contribution/Discovering_filter_keywords_for_company_name_disambiguation_in_twitter/27516885)
- Grinsztajn i in., 2022, drzewa kontra deep learning na danych tabelarycznych, NeurIPS. Bez zmiany rozkładu.
- Jonkers, Van Wallendael, 2024, konforemne systemy predykcyjne przy zmianie rozkładu, [arXiv:2404.15018](https://arxiv.org/abs/2404.15018).
- Minimum-risk recalibration of classifiers, NeurIPS 2023. Mało etykiet celu; tylko klasyfikatory.
- Wedel, Ohme, 2025, oraz Chouaki i in., 2024: donacja danych. Mówi, co widział użytkownik, a nie ile wyświetleń miał post.
- Pierri i in., 2025, dostęp badaczy według DSA, [arXiv:2512.14223](https://arxiv.org/abs/2512.14223); strona pomocy LoudCrowd; benchmarki zaangażowania SocialCrawl.

## Co z tego wynika
- Żadne znalezione źródło nie przeczy naszym głównym wynikom. To szybki przegląd, więc nie dowodzi, że takich źródeł nie ma.
- Nie znaleźliśmy recenzowanej pracy z liczbą wyświetleń na reakcję ani sprawdzenia szacunków zasięgu na prawdziwych wyświetleniach. To może być luka warta podkreślenia, ale po dokładniejszym wyszukiwaniu.
- Nie znaleźliśmy porównania ridge i boostingu przy zmianie platformy. Nasz wynik podajemy jako empiryczny.
- Nie znaleźliśmy publicznego zbioru z prawdziwymi wyświetleniami z FB ani IG, co wspiera zebranie własnej próbki (punkt A planu).

## Przegląd 2026-11-06 (nowa partia, tylko wyszukiwarka)

- *A Data-Driven Study on Instagram Reach Prediction*, IEEE. [Link](https://ieeexplore.ieee.org/abstract/document/11469564) — badanie predykcji zasięgu na Instagramie; przeczytać przed twierdzeniami o IG. Tylko wyszukiwarka.
- *Predictability of Extreme Events in Social Media*, PLOS ONE. [Link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0111506) — wczesna uwaga powyżej progu zapowiada zdarzenia skrajne; blisko naszego wyniku o ogonie (E96). Tylko wyszukiwarka.
- *Adapting prediction sets to distribution shifts without labels*, UAI 2025. [Link](https://dl.acm.org/doi/10.5555/3762387.3762475) — zbiory konforemne przy przesunięciu bez etykiet; kandydat do przedziałów FB/IG (E98). Tylko wyszukiwarka.
- *Wasserstein-Regularized Conformal Prediction under General Distribution Shift*, ICLR 2025. [Link](https://proceedings.iclr.cc/paper_files/paper/2025/hash/283066055b0256ca8e3e0c8c96019357-Abstract-Conference.html) — konforemność przy ogólnym przesunięciu. Tylko wyszukiwarka.
- CreatorIQ, *Estimate performance with greater accuracy*. [Link](https://www.creatoriq.com/whats-new-in-creatoriq/estimate-performance-with-accuracy) — produkt konkurencji estymujący wyniki; materiał do ramy produktowej Brand24, nie dowód. Tylko wyszukiwarka.

## Przegląd metod grafowych 2026-11-06 (tylko wyszukiwarka)

- Dong, Chawla, Swami, 2017, *metapath2vec*, KDD. [Link](https://www.kdd.org/kdd2017/papers/view/metapath2vec-scalable-representation-learning-for-heterogeneous-networks) — spacery po metapathach (autor-marka-autor) + skip-gram; standard dla grafów heterogenicznych. Podstawa E127.
- *A Deep Prediction Framework for Multi-Source Information via Heterogeneous GNN*, KDD 2024. [Link](https://dl.acm.org/doi/abs/10.1145/3637528.3671966) — heterogeniczny GNN łączący wiele źródeł; znaczenie typów węzłów i krawędzi. Tylko wyszukiwarka.
- SW-Metapath2vec, PLOS ONE 2024. [Link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0315507) — ważone spacery po metapathach. Tylko wyszukiwarka.
- Przegląd grafów czasowych, arXiv 2208.08606. [Link](http://export.arxiv.org/pdf/2208.08606) — krawędzie z czasem; kandydat na graf osi czasu autora. Tylko wyszukiwarka.

Nasze próby: cechy grafowe (E03/E06/E12/E29) <= +0,009; GraphSAGE na grafie tekstu gorzej (E65); spektralne embeddingi autor-marka gorzej (E122). E127 sprawdza metapath2vec na autor-marka-hashtag.

Wynik testu grafowego (E127): embeddingi metapath2vec na grafie autor-marka-hashtag pogorszyły MAE na wszystkich
platformach (0,278/0,480/0,480 vs 0,273/0,456/0,467). Powyższe metody zostają odniesieniem dla danych z prawdziwymi
krawędziami interakcji; nasz graf ma tylko krawędzie wspólnego publikowania, które nie wnoszą sygnału.
