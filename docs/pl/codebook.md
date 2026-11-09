# Codebook: typ wzmianki i sentyment

Używany do etykietowania każdej wzmianki przez LLM (patrz `src/make_label_chunks.py`). Wzmianka jest oceniana tylko na
podstawie widocznego tekstu (marka, platforma, język, tytuł/treść przycięte do 400 znaków).

## typ (dokładnie jeden)
| etykieta | definicja | typowe sygnały |
|---|---|---|
| `organic` | osoba pisze lub reaguje własnym głosem, o własnym doświadczeniu, opinii, pytaniu lub rozmowie, niezależnie od tego, czy marka jest głównym tematem | osobisty ton, odpowiedzi innym, opinie, historie |
| `promo` | komunikacja komercyjna lub instytucjonalna: własne ogłoszenia marki, reklamy, posty sponsorowane, promocja influencera, news lub marketing firmy albo medium | wezwania do działania, „link w bio”, premiery produktów, styl komunikatu prasowego, rabat samej marki |
| `spam` | treść niezwiązana z prawdziwym tematem marki lub masowo produkowana: farmienie kuponów/ofert, oferty w stylu „AI screener”, oszustwa, treści dla dorosłych, upychanie słów kluczowych, powtarzane szablony, konkursy proszące o oznaczenia | listy niepowiązanych marek lub hashtagów, „kod promocyjny”, „kliknij tutaj”, tekst szablonowy |
| `unclear` | zbyt krótka, pusta lub niejednoznaczna, by zdecydować (np. same emoji, sam link, jedno słowo) | używać oszczędnie |

Rozstrzyganie: jeśli post to osoba promująca coś za pieniądze — `promo`; jeśli jest masowo produkowany i niezwiązany z
marką — `spam`; w pozostałych przypadkach `organic`.

## sentyment (dokładnie jeden, o tekście jako całości)
`positive`, `neutral`, `negative`. `neutral` dla faktów, pytań oraz tonu mieszanego lub niejasnego.
