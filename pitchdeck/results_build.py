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
out = HERE / f"results_{LANG}.pptx"; prs.save(out); print("saved", out)
