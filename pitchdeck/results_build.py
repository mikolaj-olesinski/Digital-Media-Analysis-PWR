"""Builds results_<lang>.pptx: three slides on what the data say (numbers from data/processed/results_numbers.json, made by src/results_numbers.py).
Run: python pitchdeck/results_build.py [pl|en]"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).parent; ROOT = HERE.parent; LANG = sys.argv[1] if len(sys.argv) > 1 else "pl"
R = json.load(open(ROOT / "data" / "processed" / "results_numbers.json"))
BG, PLUM, CORAL, INK, GRAY, SOFT, LINE = "FFF8EF", "2D1E5F", "E8475A", "1E1740", "5E5878", "F0E9F7", "E6DDEE"
PCOL = {"tiktok": "#3f2a6b", "youtube": "#e4572e", "twitter": "#2a9d8f", "facebook": "#2f6db5", "instagram": "#c9a227"}
NAMES = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X", "facebook": "Facebook", "instagram": "Instagram"}
PL = LANG == "pl"
n = lambda x, d=2: (f"{x:.{d}f}".replace(".", ",") if PL else f"{x:.{d}f}")
T = lambda pl, en: pl if PL else en
L = R["ladder"]; X_RULE = L["twitter"]["Brand24 reach as-is"][0]
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 14})


def fig1(path):
    fig, ax = plt.subplots(figsize=(6.6, 4.4)); ps = ["tiktok", "youtube", "twitter", "facebook", "instagram"]; v = [R["views_known_pct"][p] for p in ps]
    ax.barh([NAMES[p] for p in ps][::-1], v[::-1], color=[PCOL[p] for p in ps][::-1])
    for i, x in enumerate(v[::-1]): ax.text(x + 2, i, f"{x}%", va="center", fontsize=14)
    ax.set_xlim(0, 118); ax.set_xlabel(T("wzmianki ze znanymi wyświetleniami (%)", "mentions with known views (%)")); ax.set_ylabel(T("platforma", "platform"))
    ax.set_title(T("Gdzie znamy prawdę o wyświetleniach", "Where the true views are known"), loc="left"); plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig2(path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4)); ps = ["tiktok", "youtube", "twitter"]; w = .2
    rows = [("median views", T("mediana", "median"), "#c9c4d6"), ("constant x followers", T("stała × followers (idea Brand24)", "constant × followers (Brand24 idea)"), "#9a9aa8"),
            ("gradient model: + followers", T("model: followers", "model: followers"), "#7a6bb0"), ("gradient model: + time", T("model: followers + lajki, komentarze, tekst, etykiety, czas", "model: followers + engagement, text, labels, time"), "#2D1E5F")]
    for k, (key, lab, c) in enumerate(rows):
        m = [L[p][key][0] for p in ps]; lo = [L[p][key][0] - L[p][key][1] for p in ps]; hi = [L[p][key][2] - L[p][key][0] for p in ps]
        ax.bar(np.arange(3) + (k - 1.5) * w, m, w, color=c, label=lab, yerr=[lo, hi], ecolor="black", capsize=2)
    ax.set_xticks(range(3)); ax.set_xticklabels([NAMES[p] for p in ps]); ax.set_xlabel(T("platforma (znane wyświetlenia)", "platform (views known)")); ax.set_ylabel(T("MAE, log10 wyświetleń", "MAE, log10 views"))
    ax.set_title(T("Błąd oszacowania wyświetleń (CV po markach)", "Error of the views estimate (brand-grouped CV)"), loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.3), frameon=False, fontsize=11, ncol=1); plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig3(path):
    fig, ax = plt.subplots(figsize=(6.6, 4.4)); T3 = R["transfer"]; names = ["TikTok", "YouTube", "X (Twitter)"]; x = np.arange(3); w = .38
    ax.bar(x - w / 2, [T3[k]["in-platform MAE"] for k in names], w, color="#9a9aa8", label=T("uczony na tej samej platformie", "trained on the same platform"))
    ax.bar(x + w / 2, [T3[k]["transfer MAE"] for k in names], w, color=[PCOL[p] for p in ["tiktok", "youtube", "twitter"]], label=T("uczony na dwóch pozostałych", "trained on the other two"),
           yerr=[[T3[k]["transfer MAE"] - T3[k]["CI low"] for k in names], [T3[k]["CI high"] - T3[k]["transfer MAE"] for k in names]], ecolor="black", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(["TikTok", "YouTube", "X"]); ax.set_xlabel(T("platforma testowa (pominięta w uczeniu)", "held-out platform")); ax.set_ylabel(T("błąd MAE, log10 wyświetleń", "MAE of log10 views"))
    ax.set_title(T("Czy model przenosi się na inną platformę?", "Does a model transfer to another platform?"), loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.3), frameon=False, fontsize=11); plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


rgb = RGBColor.from_string
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)


def tb(s, x, y, w, h, t, size=18, bold=False, color=INK, anchor=MSO_ANCHOR.TOP):
    b = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); f = b.text_frame; f.word_wrap = True; f.vertical_anchor = anchor
    f.margin_left = f.margin_right = Inches(.05); p = f.paragraphs[0]; r = p.add_run(); r.text = t
    r.font.size, r.font.bold, r.font.name = Pt(size), bold, "Calibri"; r.font.color.rgb = rgb(color); return b


def bullets(s, x, y, w, items, size=18):
    cy = y
    for head, body in items:
        dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(cy + .09), Inches(.16), Inches(.16)); dot.fill.solid(); dot.fill.fore_color.rgb = rgb(CORAL); dot.line.fill.background()
        b = tb(s, x + .3, cy, w - .3, 1.3, "", size)
        p = b.text_frame.paragraphs[0]; r1 = p.runs[0]; r1.text = head + " "; r1.font.bold = True; r1.font.color.rgb = rgb(PLUM)
        r2 = p.add_run(); r2.text = body; r2.font.size = Pt(size); r2.font.name = "Calibri"; r2.font.color.rgb = rgb(INK)
        cy += 1.28


def slide(title, kicker, pic, items, foot, notes):
    s = prs.slides.add_slide(prs.slide_layouts[6]); s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(BG)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(.22), prs.slide_height); bar.fill.solid(); bar.fill.fore_color.rgb = rgb(PLUM); bar.line.fill.background()
    tb(s, .6, .3, 12, .4, kicker, 14, True, CORAL); tb(s, .6, .65, 12.1, 1.2, title, 30, True, PLUM)
    s.shapes.add_picture(str(pic), Inches(.6), Inches(2.0), width=Inches(6.6 if "1" in pic.name or "3" in pic.name else 7.0))
    bullets(s, 7.8, 2.1, 5.1, items, 17)
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(.6), Inches(6.72), Inches(12.1), Inches(.5)); box.fill.solid(); box.fill.fore_color.rgb = rgb(SOFT); box.line.fill.background()
    tb(s, .75, 6.75, 11.8, .45, foot, 13, False, GRAY, MSO_ANCHOR.MIDDLE); s.notes_slide.notes_text_frame.text = notes


fd = ROOT / "figures"; p1, p2, p3 = (fd / f"results_{i}_{LANG}.png" for i in (1, 2, 3)); fig1(p1); fig2(p2); fig3(p3)
imp = R["importance"]; U = R["unknown"]; ratio = 10 ** (U["Facebook"]["brand24_median"] - U["Facebook"]["our_median"]); band = 10 ** U["half_width_log10"]
slide(T(f"Wyświetlenia znamy tylko na trzech z pięciu platform, a reguła Brand24 na X myli się około {10 ** X_RULE:.0f} razy", f"We know views on only three of five platforms, and Brand24's rule on X is off by a factor of about {10 ** X_RULE:.0f}"),
      T("1 · DANE I PUNKT ODNIESIENIA", "1 · DATA AND BASELINE"), p1,
      [(T("TikTok i YouTube:", "TikTok and YouTube:"), T("zasięg Brand24 to dokładnie liczba wyświetleń, więc to nasza prawda.", "Brand24 reach is exactly the view count, so this is our ground truth.")),
       (T("X:", "X:"), T(f"wyświetlenia są dla {R['views_known_pct']['twitter']}% postów, a zasięg Brand24 to osobne oszacowanie z błędem średnio {n(X_RULE)} w log10.", f"views exist for {R['views_known_pct']['twitter']}% of posts, and Brand24's reach is a separate estimate that is off by {n(X_RULE)} log10 on average.")),
       (T("Facebook i Instagram:", "Facebook and Instagram:"), T("brak wyświetleń, więc nie da się sprawdzić żadnego oszacowania wprost.", "no views at all, so no estimate can be checked directly."))],
      T("12 500 wzmianek, po 2 500 z każdej platformy. Próba nie jest równomierna w czasie (ok. 85% z kwietnia–sierpnia 2026).", "12,500 mentions, 2,500 per platform. The sample is not uniform in time (about 85% from April to August 2026)."),
      T("Punkt wyjścia: prawdę znamy tylko tam, gdzie platforma podaje wyświetlenia.", "Starting point: we know the truth only where the platform reports views."))
slide(T("Wyświetlenia wyjaśniają głównie lajki i komentarze, a nie followersi, tekst czy typ wzmianki", "Views are explained mainly by likes and comments, not by followers, text or mention type"),
      T("2 · OD CZEGO ZALEŻĄ WYŚWIETLENIA", "2 · WHAT DRIVES VIEWS"), p2,
      [(T("Engagement:", "Engagement:"), T(f"dodanie lajków, komentarzy i udostępnień obniża błąd z {n(L['tiktok']['gradient model: + followers'][0])} do {n(L['tiktok']['gradient model: + time'][0])} na TikToku i z {n(L['youtube']['gradient model: + followers'][0])} do {n(L['youtube']['gradient model: + time'][0])} na YouTube.", f"adding likes, comments and shares cuts the error from {n(L['tiktok']['gradient model: + followers'][0])} to {n(L['tiktok']['gradient model: + time'][0])} on TikTok and from {n(L['youtube']['gradient model: + followers'][0])} to {n(L['youtube']['gradient model: + time'][0])} on YouTube.")),
       (T("Stała × followers:", "Constant × followers:"), T("najgorszy model, gorszy niż zgadywanie mediany na TikToku i X.", "the worst model, worse than guessing the median on TikTok and X.")),
       (T("Tekst, nasze etykiety, czas:", "Text, our labels, time:"), T("nie poprawiają wyniku w granicach szumu.", "do not improve the result beyond noise."))],
      T("MAE na log10: 0,3 oznacza typowy błąd około dwukrotny. Walidacja: 5 grup marek, słupki = 95% przedział (bootstrap po markach). Zależność, nie przyczyna.", "MAE on log10: 0.3 means a typical error of about a factor of 2. Validation: 5 brand groups, bars = 95% CI (brand-cluster bootstrap). Association, not cause."),
      T("Najprostszy model liniowy w skali log jest prawie tak dobry jak drzewa gradientowe na TikToku i X.", "A plain log-log linear model is nearly as good as gradient trees on TikTok and X."))
slide(T(f"Na Facebooku i Instagramie da się tylko oszacować wyświetlenia, z niepewnością rzędu ×{band:.0f} w każdą stronę", f"On Facebook and Instagram views can only be estimated, with an uncertainty of about ×{band:.0f} each way"),
      T("3 · PRZENIESIENIE NA PLATFORMY BEZ PRAWDY", "3 · TRANSFER TO PLATFORMS WITHOUT GROUND TRUTH"), p3,
      [(T("Przenosi się dobrze:", "Transfers well:"), T(f"na TikToku i YouTube błąd po uczeniu na innych platformach to {n(R['transfer']['TikTok']['transfer MAE'])} i {n(R['transfer']['YouTube']['transfer MAE'])}, jak przy uczeniu na własnej.", f"on TikTok and YouTube the error after training on other platforms is {n(R['transfer']['TikTok']['transfer MAE'])} and {n(R['transfer']['YouTube']['transfer MAE'])}, as good as training on the platform itself.")),
       (T("Facebook:", "Facebook:"), T(f"Brand24 podaje ok. {ratio:.0f} razy więcej niż wynika z engagementu, ale prawdy brak.", f"Brand24 reports about {ratio:.0f} times more than engagement implies, but there is no ground truth.")),
       (T("Uwaga:", "Caution:"), T(f"przedziały z innych platform pokrywają tylko {R['cross_platform_coverage']['YouTube'] * 100:.0f}% przypadków przy celu 90%, więc są zbyt optymistyczne.", f"intervals built from other platforms cover only {R['cross_platform_coverage']['YouTube'] * 100:.0f}% of cases against a 90% target, so they are optimistic."))],
      T("Etykiety (typ, sentyment) tworzy instrument AI: kappa 0,77–0,79 i 0,71–0,73 między trzema modelami. To spójność, nie trafność.", "Labels (type, sentiment) come from an AI instrument: kappa 0.77-0.79 and 0.71-0.73 between three models. Consistency, not accuracy."),
      T("Uczciwy wniosek: oszacowanie z szerokim przedziałem, nie wynik do zacytowania jako prawda o Facebooku.", "Honest conclusion: an estimate with a wide interval, not a number to quote as the truth about Facebook."))

Z = json.load(open(ROOT / "data" / "processed" / "zoo_numbers.json")); ZS = Z["summary"]; PS = ["tiktok", "youtube", "twitter"]
TYPECOL = ["#4c9f70", "#e0a030", "#b5403a"]; TYPES = ["organic", "promo", "spam"]; TYPE_LAB = T(["organiczne", "promocyjne", "spam"], ["organic", "promo", "spam"])


def fig4(path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4)); w = .26; steps = [("median", T("mediana", "median"), "#c9c4d6"), ("zoo_best", T("najlepszy z 7 modeli", "best of 7 models"), "#7a6bb0"), ("graph_best", T("+ cechy grafowe", "+ graph features"), "#2D1E5F")]
    for k, (key, lab, c) in enumerate(steps):
        v = [ZS[p][key] for p in PS]; ax.bar(np.arange(3) + (k - 1) * w, v, w, color=c, label=lab)
        for i, x in enumerate(v): ax.text(i + (k - 1) * w, x + .01, n(x), ha="center", fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels([NAMES[p] for p in PS]); ax.set_xlabel(T("platforma (znane wyświetlenia)", "platform (views known)")); ax.set_ylabel(T("MAE, log10 wyświetleń", "MAE, log10 views"))
    ax.set_title(T("Więcej modeli i grafy nie pomagają", "More models and graphs do not help"), loc="left"); ax.legend(loc="upper center", bbox_to_anchor=(.5, -.25), frameon=False, fontsize=11, ncol=3)
    plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig5(path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4)); w = .26; lab = Z["labels"]
    for k, t in enumerate(TYPES):
        r = [next(x for x in lab if x["type"] == t and x["platform"] == {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X (Twitter)"}[p]) for p in PS]
        v = np.array([x["resid"] for x in r]); lo = v - np.array([x["resid lo"] for x in r]); hi = np.array([x["resid hi"] for x in r]) - v
        ax.bar(np.arange(3) + (k - 1) * w, v, w, color=TYPECOL[k], label=TYPE_LAB[k], yerr=[lo, hi], ecolor="black", capsize=2)
    ax.axhline(0, color="grey", lw=1); ax.set_xticks(range(3)); ax.set_xticklabels([NAMES[p] for p in PS]); ax.set_xlabel(T("platforma", "platform"))
    ax.set_ylabel(T("log10(wyświetlenia) minus predykcja\n(+ = więcej niż wynika z reakcji)", "log10(views) minus prediction\n(+ = more than engagement implies)"))
    ax.set_title(T("Typ wzmianki (nasza etykieta) a wyświetlenia", "Mention type (our label) vs views"), loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.25), frameon=False, fontsize=11, ncol=3, title=T("nasza etykieta typu", "our type label")); plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def stratum(var, name, plat):
    return next(r["MAE"] for r in Z["strata"] if r["stratifier"] == var and r["stratum"] == name and r["platform"] == plat)


p4, p5 = fd / f"results_4_{LANG}.png", fd / f"results_5_{LANG}.png"; fig4(p4); fig5(p5)
sy = "YouTube"; sx = "X (Twitter)"
slide(T("Sześć rodzin modeli, grafy autorów i tekstów oraz podział na grupy nie pobiły samych lajków i komentarzy", "Six model families, author and text graphs and splitting into groups did not beat likes and comments alone"),
      T("4 · CO JESZCZE SPRAWDZILIŚMY", "4 · WHAT ELSE WE TRIED"), p4,
      [(T("Modele:", "Models:"), T(f"ridge, kNN, lasy losowe, boosting i blend mieszczą się w swoich przedziałach (TikTok {n(ZS['tiktok']['zoo_best'])}, YouTube {n(ZS['youtube']['zoo_best'])}, X {n(ZS['twitter']['zoo_best'])}).", f"ridge, kNN, forests, boosting and a blend sit within each other's CI (TikTok {n(ZS['tiktok']['zoo_best'])}, YouTube {n(ZS['youtube']['zoo_best'])}, X {n(ZS['twitter']['zoo_best'])}).")),
       (T("Grafy:", "Graphs:"), T(f"średnia autora, sąsiedzi z grafu podobieństwa tekstu i wygładzanie po sąsiadach dają co najwyżej {n(ZS['youtube']['engagement_only'] - ZS['youtube']['graph_best'], 3)} na YouTube, w granicach błędu.", f"author mean, text-similarity neighbours and neighbour smoothing give at most {n(ZS['youtube']['engagement_only'] - ZS['youtube']['graph_best'], 3)} on YouTube, within noise.")),
       (T("Gdzie błąd jest największy:", "Where the error is largest:"), T(f"posty z najmniejszą liczbą reakcji (YouTube {n(stratum('engagement tier', 'Q1 lowest', sy))} wobec {n(stratum('engagement tier', 'Q4 highest', sy))} w górnym kwartylu) i spam na X ({n(stratum('our type label', 'spam', sx))} wobec {n(stratum('our type label', 'organic', sx))}).", f"posts with the fewest reactions (YouTube {n(stratum('engagement tier', 'Q1 lowest', sy))} vs {n(stratum('engagement tier', 'Q4 highest', sy))} in the top quartile) and spam on X ({n(stratum('our type label', 'spam', sx))} vs {n(stratum('our type label', 'organic', sx))})."))],
      T("Ta sama walidacja co wcześniej (5 grup marek, MAE na log10). Cechy zależne od wyświetleń liczone tylko z części treningowej. Osobny model na przedział followersów wypadł gorzej niż jeden wspólny.", "Same validation as before (5 brand groups, MAE on log10). Features that use views are computed from the training fold only. One model per follower tier was worse than one pooled model."),
      T("Wynik negatywny, ale mocny: sygnał jest w reakcjach, nie w sposobie uczenia.", "A negative but strong result: the signal is in engagement, not in the learner."))
slide(T("Nasze etykiety AI niemal nie zmieniają wyświetleń przy tych samych reakcjach: promocyjne na TikToku mają o około 20% więcej", "Our AI labels barely change views at equal engagement: promo on TikTok gets about 20% more"),
      T("5 · NASZE ETYKIETY A WYŚWIETLENIA", "5 · OUR LABELS VS VIEWS"), p5,
      [(T("Co mierzymy:", "What we measure:"), T("różnicę między faktycznymi wyświetleniami a tym, co przewidują lajki i komentarze, osobno dla organicznych, promocyjnych i spamu.", "the gap between actual views and what likes and comments predict, for organic, promo and spam separately.")),
       (T("TikTok:", "TikTok:"), T("promocyjne +0,08 log10 (około +20%), przedział ufności nad zerem; reszta w granicach błędu.", "promo +0.08 log10 (about +20%), CI above zero; the rest within noise.")),
       (T("Wniosek:", "Takeaway:"), T("etykiety typu nie poprawiają oszacowania wyświetleń; przy tych samych reakcjach spam i reklama nie wyróżniają się wyraźnie od zwykłych postów.", "the type labels do not improve the views estimate; at equal engagement, promo and spam posts do not stand out clearly from ordinary ones."))],
      T("Etykiety tworzą trzy modele Haiku, a 4 668 niezgodnych wzmianek rozstrzyga Sonnet. Zgodność trzech modeli 79%, kappa typu 0,77–0,79. To zgodność maszyn, nie sprawdzenie względem ludzkiego wzorca.", "Labels come from three Haiku coders, with Sonnet adjudicating 4,668 disputed mentions. All three agree on 79%, type kappa 0.77-0.79. This is machine-machine agreement, not a check against a human gold standard."),
      T("Przedziały to bootstrap po markach. Etykiety są kodowane maszynowo, więc efekt traktujemy ostrożnie.", "Intervals are brand-cluster bootstrap. Labels are machine-coded, so the effect is treated with caution."))


def architecture():
    s = prs.slides.add_slide(prs.slide_layouts[6]); s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(BG)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(.22), prs.slide_height); bar.fill.solid(); bar.fill.fore_color.rgb = rgb(PLUM); bar.line.fill.background()
    tb(s, .6, .3, 12, .4, T("9 · ARCHITEKTURA I PRACA Z NARZĘDZIAMI AI", "9 · ARCHITECTURE AND HOW WE USE AI TOOLS"), 14, True, CORAL)
    tb(s, .6, .65, 12.1, 1.0, T("Od surowego CSV do wyniku: każdy krok to skrypt lub notebook, a AI koduje wzmianki i pomaga w kodzie, ale nie pisze tekstu pracy", "From raw CSV to result: every step is a script or notebook; AI codes mentions and helps with code but does not write the paper"), 28, True, PLUM)
    steps = [(T("Dane surowe", "Raw data"), T("CSV, 12 500 wzmianek, tylko do odczytu", "CSV, 12,500 mentions, read-only")), (T("Kodowanie AI", "AI coding"), T("3× Haiku + Sonnet dla niezgodnych; codebook", "3x Haiku + Sonnet on disputes; codebook")),
             (T("Etykiety", "Labels"), T("typ i sentyment, kappa, źródło decyzji", "type and sentiment, kappa, decision source")), (T("Cechy", "Features"), T("followersi, reakcje, braki, tekst, etykiety, graf", "followers, engagement, missingness, text, labels, graph")),
             (T("Walidacja i modele", "Validation and models"), T("CV po markach, bootstrap, przedziały, transfer", "brand-grouped CV, bootstrap, intervals, transfer")), (T("Wyniki", "Results"), T("notebooki, rysunki, slajdy z liczb w JSON", "notebooks, figures, slides built from JSON numbers"))]
    bw, gap, x0, y0 = 1.85, .2, .6, 2.1
    for i, (h, b) in enumerate(steps):
        x = x0 + i * (bw + gap); box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y0), Inches(bw), Inches(1.9)); box.fill.solid(); box.fill.fore_color.rgb = rgb(SOFT if i % 2 == 0 else "FFFFFF"); box.line.color.rgb = rgb(PLUM)
        tb(s, x + .05, y0 + .1, bw - .1, .5, h, 15, True, PLUM); tb(s, x + .05, y0 + .65, bw - .1, 1.2, b, 12, False, INK)
        if i < len(steps) - 1:
            a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + bw + .01), Inches(y0 + .8), Inches(.18), Inches(.3)); a.fill.solid(); a.fill.fore_color.rgb = rgb(CORAL); a.line.fill.background()
    bullets(s, .6, 4.3, 6.0, [(T("Gdzie używamy AI:", "Where we use AI:"), T("kodowanie wzmianek (instrument), pomoc przy kodzie i notebookach, sprawdzanie błędów.", "coding mentions (the instrument), help with code and notebooks, bug checks.")),
                              (T("Czego AI nie robi:", "What AI does not do:"), T("nie pisze tekstu preprintu ani plakatu, nie rusza surowych danych.", "does not write the preprint or poster text, never touches raw data."))], 16)
    bullets(s, 6.9, 4.3, 5.9, [(T("Kontrola:", "Control:"), T("log każdej sesji w repozytorium, małe commity, stałe ziarna, kod od surowych danych do rysunku.", "a log of every session in the repo, small commits, fixed seeds, code from raw data to figure.")),
                               (T("Granica:", "Limit:"), T("etykiety AI nie były oceniane względem punktowanego ludzkiego wzorca; to nasze główne ograniczenie.", "AI labels were not scored against a human gold standard; this is our main limitation."))], 16)
    s.notes_slide.notes_text_frame.text = T("5–10 minut o architekturze i narzędziach AI na zajęcia.", "5-10 minutes on architecture and AI tools for class.")


E = json.load(open(ROOT / "data" / "processed" / "extra_numbers.json"))
def gain(plat, grp, key="gain vs base"): return next(r[key] for r in E["gains"] if r["platform"] == plat and r["model"] == grp)
def loss(plat, grp): return next(r["loss"] for r in E["ablation"] if r["platform"] == plat and r["removed group"] == grp)
def tgt(plat, name, key="R2"): return next(r[key] for r in E["targets"] if r["platform"] == plat and r["target"] == name)
bgx = next(r for r in E["brand_graph"] if r["platform"] == "X (Twitter)")
slide(T("Tekst daje drobny, ale realny zysk na YouTube i X; reakcje są niezbędne, a followersi i graf marek pomagają tylko miejscami", "Text adds a small but real gain on YouTube and X; engagement is essential, while followers and a brand graph help only in places"),
      T("6 · DODATKOWE KOLUMNY I ABLACJA", "6 · EXTRA COLUMNS AND ABLATION"), fd / f"ablation_{LANG}.png",
      [(T("Tekst (statystyki i tematy TF-IDF):", "Text (statistics and TF-IDF topics):"), T(f"poprawia błąd o {n(gain('YouTube', '+ TF-IDF SVD (30)'), 3)} na YouTube i {n(gain('X (Twitter)', '+ TF-IDF SVD (30)'), 3)} na X; na TikToku nic. Zysk jest mały, ale przedział ufności nie obejmuje zera.", f"improves the error by {n(gain('YouTube', '+ TF-IDF SVD (30)'), 3)} on YouTube and {n(gain('X (Twitter)', '+ TF-IDF SVD (30)'), 3)} on X; nothing on TikTok. Small, but the CI excludes zero.")),
       (T("Ablacja:", "Ablation:"), T(f"po usunięciu reakcji błąd rośnie o {n(loss('TikTok', 'engagement'))} (TikTok) i {n(loss('YouTube', 'engagement'))} (YouTube). Sentyment VADER, kalendarz i wielkość marki nic nie wnoszą.", f"removing engagement raises the error by {n(loss('TikTok', 'engagement'))} (TikTok) and {n(loss('YouTube', 'engagement'))} (YouTube). VADER sentiment, calendar and brand size add nothing.")),
       (T("Odwrotnie:", "The other way round:"), T(f"lajki, komentarze i udostępnienia da się przewidzieć (R² {n(tgt('TikTok', 'likes'))} lajki na TikToku), a followersów prawie nie (R² {n(tgt('TikTok', 'followers'))}). Graf marek poprawia tylko X o {n(bgx['gain'], 3)}.", f"likes, comments and shares are predictable (R² {n(tgt('TikTok', 'likes'))} for TikTok likes), followers hardly (R² {n(tgt('TikTok', 'followers'))}). The brand graph helps only X, by {n(bgx['gain'], 3)}."))],
      T("Słupki = wzrost błędu po usunięciu grupy z pełnego modelu (oś symlog), 95% przedział z bootstrapu po markach. Reakcje wyjaśniają większość, reszta to poprawki rzędu 0,01.", "Bars = error increase when a group is removed from the full model (symlog axis), 95% brand-cluster bootstrap CI. Engagement explains most; the rest are corrections around 0.01."),
      T("Przedziały sparowane na tych samych foldach. Cechy tekstu i kalendarza nie używają wyświetleń.", "Intervals are paired on the same folds. Text and calendar features use no views."))

GL = json.load(open(ROOT / "data" / "processed" / "graphlab_numbers.json"))
def glg(plat, name): return next(r["gain vs base"] for r in GL["variants"] if r["platform"] == plat and r["model"] == name)
def gld(plat, key): return next(r[key] for r in GL["duplicates"] if r["platform"] == plat)
def glh(plat, key): return next(r[key] for r in GL["homophily"] if r["platform"] == plat)
def glt(plat, model): return next(r["macro F1"] for r in GL["type_prediction"] if r["platform"] == plat and r["model"] == model)
pc = lambda x: f"{100 * x:.0f}%"
slide(T("Dziesięć pomysłów grafowych i kodowań cech nie poprawia oszacowania wyświetleń, a propagacja etykiet je pogarsza", "Ten graph and encoding ideas do not improve the views estimate, and label propagation makes it worse"),
      T("7 · LABORATORIUM GRAFOWE", "7 · GRAPH LAB"), fd / f"graphlab_gains_{LANG}.png",
      [(T("Co sprawdziliśmy:", "What we tried:"), T("społeczności (Louvain), PageRank, embedding spektralny, graf autor–marka, hashtagi, czas, near-duplikaty i kodowania kategorii.", "communities (Louvain), PageRank, spectral embedding, author-brand graph, hashtags, time, near-duplicates and category encodings.")),
       (T("Wynik:", "Result:"), T(f"największe zyski to ok. {n(glg('TikTok', '+ category encodings (language, country, hour, weekday, month)'), 3)} (kodowania kategorii, TikTok) i {n(glg('YouTube', '+ spectral'), 3)} (spektralny, YouTube), czyli w praktyce zero.", f"the biggest gains are about {n(glg('TikTok', '+ category encodings (language, country, hour, weekday, month)'), 3)} (category encodings, TikTok) and {n(glg('YouTube', '+ spectral'), 3)} (spectral, YouTube), practically zero.")),
       (T("Propagacja etykiet:", "Label propagation:"), T(f"pogarsza wynik na wszystkich platformach (X {n(glg('X (Twitter)', '+ label propagation (2 hops)'), 3)}), bo na treningu widzi własne wyniki odbite przez pętle grafu.", f"hurts on all platforms (X {n(glg('X (Twitter)', '+ label propagation (2 hops)'), 3)}) because on the training fold it sees its own results reflected through graph loops."))],
      T("Zysk sparowany względem followersi + reakcji, CV po markach, 95% bootstrap. Cechy zależne od wyświetleń liczone tylko z części treningowej.", "Paired gain over followers + engagement, brand-grouped CV, 95% bootstrap. Features using views are computed from the training fold only."),
      T("Wynik negatywny, ale po szerokim przeszukaniu: w danych nie ma ukrytej struktury, którą graf by wyciągnął.", "A negative result after a wide search: there is no hidden structure in the data that a graph would extract."))


def wide_slide(title, kicker, pic, cols, foot, notes):
    s = prs.slides.add_slide(prs.slide_layouts[6]); s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(BG)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(.22), prs.slide_height); bar.fill.solid(); bar.fill.fore_color.rgb = rgb(PLUM); bar.line.fill.background()
    tb(s, .6, .3, 12, .4, kicker, 14, True, CORAL); tb(s, .6, .65, 12.1, 1.0, title, 28, True, PLUM)
    s.shapes.add_picture(str(pic), Inches(1.9), Inches(1.75), height=Inches(3.45))
    for i, (h, b) in enumerate(cols):
        x = .6 + i * 4.1; tb(s, x, 5.3, 3.9, .35, h, 14, True, PLUM); tb(s, x, 5.65, 3.9, 1.0, b, 13, False, INK)
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(.6), Inches(6.72), Inches(12.1), Inches(.5)); box.fill.solid(); box.fill.fore_color.rgb = rgb(SOFT); box.line.fill.background()
    tb(s, .75, 6.75, 11.8, .45, foot, 12, False, GRAY, MSO_ANCHOR.MIDDLE); s.notes_slide.notes_text_frame.text = notes


wide_slide(T("Graf nie poprawia oszacowania, ale pokazuje kampanie copy-paste: promocje i spam tworzą ciasne skupiska prawie identycznych postów", "The graph does not improve the estimate, but it exposes copy-paste campaigns: promo and spam form tight clusters of near-identical posts"),
      T("8 · CO GRAF POKAZUJE O WZMIANKACH", "8 · WHAT THE GRAPH SHOWS ABOUT MENTIONS"), fd / f"graphlab_map_{LANG}.png",
      [(T("Kopie:", "Copies:"), T(f"od {pc(gld('TikTok', 'share with near-duplicate'))} do {pc(gld('YouTube', 'share with near-duplicate'))} wzmianek ma niemal identyczną kopię; spamu jest wśród nich {pc(gld('X (Twitter)', 'spam share, duplicates'))} vs {pc(gld('X (Twitter)', 'spam share, others'))} na X.", f"{pc(gld('TikTok', 'share with near-duplicate'))} to {pc(gld('YouTube', 'share with near-duplicate'))} of mentions have a near-identical copy; spam is {pc(gld('X (Twitter)', 'spam share, duplicates'))} among them vs {pc(gld('X (Twitter)', 'spam share, others'))} on X.")),
       (T("Wyświetlenia kopii:", "Views of copies:"), T(f"pary kopii mają skorelowane wyświetlenia ({n(gld('TikTok', 'corr of log views, duplicate pairs'))}–{n(gld('YouTube', 'corr of log views, duplicate pairs'))}) wobec ok. 0 dla losowych par.", f"copy pairs have correlated views ({n(gld('TikTok', 'corr of log views, duplicate pairs'))} to {n(gld('YouTube', 'corr of log views, duplicate pairs'))}) against about 0 for random pairs.")),
       (T("Etykiety i typ:", "Labels and type:"), T(f"krawędzie łączą ten sam typ częściej niż z przypadku ({pc(glh('X (Twitter)', 'same-type edges'))} vs {pc(glh('X (Twitter)', 'expected by chance'))} na X), ale głos sąsiadów nie poprawia przewidywania typu (F1 {n(glt('X (Twitter)', 'text only'))} z samego tekstu).", f"edges link the same type more often than by chance ({pc(glh('X (Twitter)', 'same-type edges'))} vs {pc(glh('X (Twitter)', 'expected by chance'))} on X), but a neighbour vote does not improve type prediction (F1 {n(glt('X (Twitter)', 'text only'))} from text alone)."))],
      T("Mapa t-SNE z 30 tematów TF-IDF, kolor = nasza etykieta typu. Duplikaty: cosinus > 0,8. Społeczności to głównie branże (auta, banki, fast food, linie lotnicze).", "t-SNE map of 30 TF-IDF topics, colour = our type label. Duplicates: cosine > 0.8. Communities are mostly industries (cars, banks, fast food, airlines)."),
      T("Zbyt mało wzmianek ma kopię w innej marce, by to poprawiło estymację w CV po markach.", "Too few mentions have a copy in another brand for this to improve the estimate in brand-grouped CV."))

architecture()
out = HERE / f"results_{LANG}.pptx"; prs.save(out); print("saved", out)
