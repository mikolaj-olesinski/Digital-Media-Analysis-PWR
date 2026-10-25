"""English texts for build.py: EN maps the Polish on-slide strings to English, NOTES_EN holds speaker notes by slide number."""

EN = {
    "Ile osób naprawdę\nzobaczyło ten post?": "How many people really\nsaw this post?",
    "Jak estymować widoczność\npostów społecznościowych": "How to estimate the visibility\nof social media posts",
    "Projekt AMC  ·  dla Brand24": "AMC project  ·  for Brand24",
    "Platformy ukrywają widoczność postów": "Platforms hide post visibility",
    "Ile osób to zobaczyło?": "How many people saw this?",
    "Zasięg często jest ukryty": "Reach is often hidden",
    "Zasięg to ich towar:\nsprzedają reklamy": "Reach is their product:\nthey sell ads",
    "Brand24 musi więc szacować": "So Brand24 has to estimate",
    "Nasza hipoteza": "Our hypothesis",
    "HIPOTEZA": "HYPOTHESIS",
    "Widoczność posta oszacujemy lepiej niż reguła Brand24,\n"
    "łącząc obserwujących z reakcjami: polubieniami, komentarzami, udostępnieniami":
        "We will estimate post visibility better than the Brand24 rule\n"
        "by combining followers with reactions: likes, comments, shares",
    "DZIŚ": "TODAY",
    "Jedna liczba": "A single number",
    "obserwujący × stała": "followers × constant",
    "NASZ POMYSŁ": "OUR IDEA",
    "Estymacja + przedział": "Estimate + interval",
    "Wiemy też, kiedy jej nie ufać": "We also know when not to trust it",
    "Obalona, jeśli model nie pobije reguły Brand24 na postach ze znanymi wyświetleniami":
        "Refuted if the model does not beat the Brand24 rule on posts with known views",
    "Jak to ugryziemy: koncepcja": "How we will tackle it: the concept",
    "Eksploracja": "Exploration",
    "Braki, rozkłady, dziwne przypadki": "Gaps, distributions, odd cases",
    "Estymacja": "Estimation",
    "Model z obserwujących i zaangażowania": "Model from followers and engagement",
    "Walidacja": "Validation",
    "Test tam, gdzie znamy wyświetlenia": "Tested where views are known",
    "Niepewność": "Uncertainty",
    "Przedziały i odporność wyników": "Intervals and robustness of results",
    "NA CZYM PRACUJEMY": "WHAT WE WORK WITH",
    "WZMIANKI Z ZNANYMI WYŚWIETLENIAMI": "MENTIONS WITH KNOWN VIEWS",
    "Kto na tym skorzysta?": "Who benefits?",
    "Firmy i marki": "Companies and brands",
    "Wiedzą, ile osób naprawdę zobaczyło wzmiankę": "They know how many people really saw a mention",
    "Małe firmy i NGO": "Small businesses and NGOs",
    "Rzetelny obraz bez drogich danych platform": "A reliable picture without costly platform data",
    "Dziennikarze": "Journalists",
    "Widzą, jak daleko zaszła dezinformacja": "They see how far disinformation has spread",
    "Badacze mediów": "Media researchers",
    "Miara z niepewnością zamiast zgadywania": "A measure with uncertainty instead of guessing",
    "Każdy z nas": "All of us",
    "Mniej napompowanych statystyk w obiegu": "Fewer inflated statistics in circulation",
    "Impact: komu to coś da?": "Impact: who gains from this?",
    "kont w mediach społecznościowych:\nich posty mogą być mierzone uczciwiej":
        "social media accounts:\ntheir posts could be measured more honestly",
    "5,24 mld": "5.24 bn",
    "2 400+": "2,400+",
    "~12 000": "~12,000",
    "DataReportal / Kepios, 2025": "DataReportal / Kepios, 2025",
    "firm korzysta z Brand24": "companies use Brand24",
    "dane podawane przez Brand24": "figure reported by Brand24",
    "osób czyta ich raporty": "people read their reports",
    "założenie: ok. 5 osób na firmę": "assumption: about 5 people per company",
    "Szacunek-zabawa: rzędy wielkości, nie pomiar": "A playful estimate: orders of magnitude, not a measurement",
    "Dlaczego nam się uda?": "Why will we succeed?",
    "WYKONALNOŚĆ": "FEASIBILITY",
    "Legalne dane od zleceniodawcy: Brand24": "Legal data from the client: Brand24",
    "Prawda bazowa: TikTok i YouTube": "Ground truth: TikTok and YouTube",
    "Bez danych osobowych autorów (RODO)": "No personal data of authors (GDPR)",
    "Wstępny test w eksploracji: obiecujący": "Preliminary exploration test: promising",
    "EKSPERTYZA: STAN WIEDZY": "EXPERTISE: STATE OF KNOWLEDGE",
    "Obserwujący to nie realny wpływ": "Followers are not real influence",
    "Popularność treści da się przewidywać": "Content popularity can be predicted",
    "Pojedynczy post trudno przewidzieć": "A single post is hard to predict",
    "Przedziały niepewności z gwarancją": "Uncertainty intervals with a guarantee",
    "Cha i in., 2010": "Cha et al., 2010",
    "Szabo i Huberman, 2010": "Szabo and Huberman, 2010",
    "Bakshy i in., 2011": "Bakshy et al., 2011",
    "Angelopoulos i Bates, 2023": "Angelopoulos and Bates, 2023",
}

NOTES_EN = {
    1: "Hello. Let me start with a simple question: how many people actually saw a given social media post? "
       "It seems like basic information, and very often we simply do not have it. Our project is about how to "
       "estimate such visibility when the platform does not show it. We are doing it as part of the media "
       "analysis course, for the company Brand24. In the next few minutes I will show our hypothesis, how we "
       "want to test it, who it will help and why we will succeed.",
    2: "Let us start with the problem. Social media platforms very often do not reveal how many people "
       "actually saw a post. One reason is that they earn money from ads themselves, so reach is a commercial "
       "asset for them, not information they readily give away. Brand24 is a tool that monitors mentions of "
       "brands on the internet. So it sees that someone wrote about a brand, but does not always know how many "
       "people read it. Imagine a post about a brand: we do not know whether ten people saw it or ten "
       "thousand, and for a brand that is a huge difference. That is why it has to estimate visibility. And "
       "this is where our hypothesis comes in.",
    3: "Instead of a problem, we state a hypothesis. It goes like this: the visibility of a post can be "
       "estimated better than by today's Brand24 rule if, besides the number of followers, we take reactions "
       "into account, that is likes, comments and shares. Today the result is a single number: followers "
       "times a constant. We want to give an estimate together with an uncertainty interval, so that it is "
       "clear when not to trust it. The hypothesis is falsifiable: if, on posts for which we know the true "
       "views, our model is not better than the Brand24 rule, we consider it refuted. Such a result is also "
       "valuable.",
    4: "How do we want to tackle it technologically? Four steps. First, data exploration: gaps, "
       "distributions and odd cases, for example posts with huge reach. Then estimation: a model in Python "
       "on a log scale that uses followers, likes, comments, shares and the platform. The third step is "
       "cross-validation where we know the true views. The fourth is uncertainty: instead of a single number "
       "we give an interval. We work on twelve and a half thousand mentions from five platforms. At the "
       "bottom you can see for what share of mentions we know the true views: TikTok {tiktok} percent, "
       "YouTube {youtube}, Twitter {twitter}, and Facebook and Instagram zero. This is our reference point.",
    5: "Our client is society, so who gains? First, companies and brands: they know how many people really "
       "saw a mention. Second, small businesses and non-governmental organisations that cannot afford paid "
       "data from the platforms. Third, journalists and fact-checkers: they can judge how far a false piece "
       "of information has spread. Fourth, media researchers, who get a measure together with its "
       "uncertainty. And finally all of us: fewer inflated numbers in reports and in the media.",
    6: "Instead of market size we estimate impact, and we treat it as a game of orders of magnitude. At the "
       "top: there are about five billion social media accounts in the world, according to the DataReportal "
       "report. Every one of their posts could be measured more honestly. Closer to us: Brand24 says it is "
       "used by over two thousand four hundred companies. If about five people read the reports in each, "
       "then directly some twelve thousand people get better numbers. This is not a measurement, just an "
       "estimate, but it shows the scale.",
    7: "Why will we succeed? Two reasons. The first is feasibility: we have legal data from the client, the "
       "company Brand24, that is twelve and a half thousand public mentions from five platforms. On TikTok "
       "and YouTube we know the true views, and we do not use authors' personal data. The preliminary test "
       "in the exploration is promising. The second is the state of knowledge we build on: Cha and "
       "co-authors showed that the number of followers is not real influence. Szabo and Huberman, that "
       "content popularity can be predicted. Bakshy and co-authors, that a single post is hard to predict, "
       "which is why we need uncertainty. And conformal methods, described by Angelopoulos and Bates, give "
       "intervals with a guarantee. Thank you.",
}
