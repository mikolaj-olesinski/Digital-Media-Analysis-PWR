# Checklista wydania

Przed publikacją repozytorium, preprintu lub posteru.

## Dane i prywatność

- [ ] `data/raw/` nie jest śledzone; `logs/` nie jest śledzone; `AGENTS.md` nie jest śledzone.
- [ ] Brak nazw autorów, URLi autorów i URLi postów w śledzonych plikach, outputach notebooków i figurach.
- [ ] Brak handle’i społecznościowych (`@nazwa`) w dokumentach; `check` ostrzega o prawdziwych URL-ach, handle to ręczny grep.
- [ ] Pobrane pliki Facebooka są lokalne i git-ignored; w repo są tylko agregaty.
- [ ] Data statement jest aktualny (publiczne pobieranie FB, views vs plays, anonimowa próba IG).
- [ ] Użycie LLM do etykiet jest ujawnione (trzy kodery, kappa maszyna-maszyna, brak ludzkiego gold standardu).
- [ ] Nie zebrano danych z Instagrama z zalogowanej sesji; jeśli później pojawi się dostawca, regulamin jest sprawdzony.

## Liczby

- [ ] Każda liczba pochodzi z `data/processed/*_numbers.json`; żadnych ręcznie edytowanych wartości.
- [ ] `python src/agent_tools.py vault && python src/agent_tools.py check` raportuje 0 nowych błędów.
- [ ] `doc_status.md` jest aktualny; lista „typowych błędów” została przejrzana.
- [ ] Figury istnieją w `_en` i `_pl` i zgadzają się z liczbami (`src/result_figures.py`).
- [ ] Etykiety figur nie zawierają URL-i ani handle’i: grep po skryptach (`set_title|xlabel|ylabel|text`).
- [ ] Wygenerowany HTML przewodnika i oba PPTX-y są przeskanowane pod URL-e i handle (sprawdzone 2026-11-08: brak).
- [ ] `reproduce.md` działa w czystym środowisku.

## Tekst

- [ ] Tekst preprintu/posteru napisał zespół, nie agent.
- [ ] Użycie AI jest ujawnione w preprintcie, zgodnie z wymogami kursu.
- [ ] Twierdzenia zgadzają się z `evidence_map.md` (siła) i nie ma języka przyczynowego.
- [ ] Ograniczenia: pasmo FB >1M, Instagram, próbka — są opisane.

## Atrybucje

- [ ] Wagi TabPFN v2: „Built with PriorLabs-TabPFN”; bez wag 2.5+.
- [ ] Wpisy literatury oznaczone „search only” nie są cytowane jako zweryfikowane.
- [ ] Biblioteki i źródła danych są wymienione.

## Repozytorium

- [ ] Commity małe, po angielsku, bez AI co-author; nic nie wypchnięte bez użytkownika.
- [ ] `requirements.txt` odpowiada środowisku, w którym powstały finalne liczby.
- [ ] `README.md` linkuje aktualne dokumenty (model card, status, reproduce, doc status).

## Licencja i wykorzystanie danych

- [ ] Repozytorium **nie ma jeszcze pliku licencji**; domyślnie oznacza to wszystkie prawa zastrzeżone. Wybór
      MIT/Apache-2.0 to decyzja zespołu (ludzi, nie agenta) i musi być zapisany przed publikacją.
- [ ] Dane Brand24 (surowy CSV, etykiety z niego, pobrane pliki FB) **nie są redystrybuowane**; publikujemy tylko
      agregaty, liczby i kod. Dane należą do klienta.
- [ ] Publiczne strony Facebooka: pobrane liczniki zostają lokalnie i nie są republikowane; paper raportuje agregaty i
      metodę.
- [ ] Figury i tekst w repo są zespołu; figury osób trzecich nie są kopiowane.
- [ ] Wagi TabPFN v2 są używane zgodnie z ich licencją, z atrybucją „Built with PriorLabs-TabPFN”.
