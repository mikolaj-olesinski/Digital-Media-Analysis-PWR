"""Four slides for the class on what comes out of the data: pitchdeck/class_<lang>.pptx. Numbers from data/processed/*.json.
Run after notebooks 07-10: python pitchdeck/class_build.py [pl|en]"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from pptx.util import Inches, Pt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src")); import slide_graph as sg

HERE = Path(__file__).parent; ROOT = HERE.parent; LANG = sys.argv[1] if len(sys.argv) > 1 else "pl"; PL = LANG == "pl"; FIG = ROOT / "figures"; D = ROOT / "data" / "processed"
T = lambda a, b: a if PL else b; n = lambda x, d=2: f"{x:.{d}f}".replace(".", ",") if PL else f"{x:.{d}f}"
TU, TR, AB, RU = (json.load(open(D / f"{k}_numbers.json")) for k in ("tuning", "transfer", "ablation", "rule"))
BG, INK, ACC, GRAY, SOFT, RED = "FAFAF8", "1F2A33", "2F6F93", "5F6368", "ECEFF1", "A61E1E"; rgb = RGBColor.from_string
NM = {"TikTok": "TikTok", "YouTube": "YouTube", "X (Twitter)": "X"}
def base(p, m): return next(r["MAE"] for r in TU["baseline"] if r["platform"] == p and r["model"] == m)
def lopo(t):
    r = next(r for r in TR["matrix"] if r["model"] == TR["best_model"] and r["features"] == TR["features"] and r["target"] == t and r["kind"] == "2 source(s)"); o = next(r for r in TR["matrix"] if r["model"] == TR["best_model"] and r["features"] == TR["features"] and r["target"] == t and r["kind"] == "in-platform"); return r, o
def fit(p): return next(r["matched by the rule"] for r in RU["fit"] if r["platform"] == p)
def real(p, k="views per reaction"): return next(r[k] for r in RU["real_vs_rule"] if r["platform"] == p)
def views(k, p, g="group"): return next(r for r in AB["three_views"] if r["platform"] == p and r[g] == k)
def line(p, t): return next(r["views at 100 reactions"] for r in AB["type_lines"] if r["platform"] == p and r["type"] == t)
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)


def tb(s, x, y, w, h, t, size=18, bold=False, color=INK, anchor=MSO_ANCHOR.TOP):
    b = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); f = b.text_frame; f.word_wrap = True; f.vertical_anchor = anchor; f.margin_left = f.margin_right = Inches(.04); r = f.paragraphs[0].add_run(); r.text = t; r.font.size, r.font.bold, r.font.name = Pt(size), bold, "Calibri"; r.font.color.rgb = rgb(color); return b


def frame(kicker, title, foot, notes):
    s = prs.slides.add_slide(prs.slide_layouts[6]); s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(BG); tb(s, .5, .22, 12.3, .35, kicker, 14, True, ACC); tb(s, .5, .55, 12.3, 1.1, title, 27, True, INK)
    bx = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(.5), Inches(6.82), Inches(12.33), Inches(.5)); bx.fill.solid(); bx.fill.fore_color.rgb = rgb(SOFT); bx.line.fill.background(); tb(s, .6, 6.84, 12.1, .46, foot, 12, False, GRAY, MSO_ANCHOR.MIDDLE); s.notes_slide.notes_text_frame.text = notes; return s


def pic(s, name, x, y, w=None, h=None):
    p = FIG / name; iw, ih = Image.open(p).size; r = iw / ih
    if w and h: w, h = (w, w / r) if w / r <= h else (h * r, h)
    s.shapes.add_picture(str(p), Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def points(s, x, y, w, items, size=16, gap=1.45):
    for i, (a, b) in enumerate(items):
        bb = tb(s, x, y + i * gap, w, gap, "", size); p = bb.text_frame.paragraphs[0]; r = p.runs[0]; r.text = a + " "; r.font.bold = True; r.font.color.rgb = rgb(ACC); q = p.add_run(); q.text = b; q.font.size = Pt(size); q.font.name = "Calibri"; q.font.color.rgb = rgb(INK)


P = ["TikTok", "YouTube", "X (Twitter)"]; xr = TU["x_rule"]; SH = {"TikTok": "TikTok", "YouTube": "YouTube", "X (Twitter)": "X"}; X = lambda v: f"×{n(10 ** v, 1)}"; BLUE, GR = "#2f6f93", "#a9a9a4"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 12})


def save(fig, name):
    path = FIG / f"{name}_{LANG}.png"; fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig); return path.name


def chart_error():
    """Typical error as a factor, for the constant and for the default model."""
    fig, a = plt.subplots(figsize=(4.2, 2.5)); yy = range(3)
    for y, p in zip(yy, P):
        u, v = base(p, "constant x followers"), base(p, "default model: followers + reactions"); a.plot([10 ** v, 10 ** u], [y, y], color="#d0d0cc", lw=3, zorder=1); a.scatter(10 ** u, y, s=90, color=GR, zorder=2, label=T("stała × obserwujący", "constant × followers") if y == 0 else None); a.scatter(10 ** v, y, s=90, color=BLUE, zorder=2, label=T("model z reakcjami", "model with reactions") if y == 0 else None)
        a.text(10 ** u, y - .3, X(u), ha="center", fontsize=11); a.text(10 ** v, y - .3, X(v), ha="center", fontsize=11, color=BLUE, fontweight="bold")
    a.set_xscale("log"); a.set_xlim(1.2, 13); a.set_ylim(2.5, -.75); a.set_yticks(list(yy)); a.set_yticklabels([SH[p] for p in P]); a.set_xticks([2, 5, 10]); a.set_xticklabels(["×2", "×5", "×10"]); a.minorticks_off(); a.set_xlabel(T("typowa pomyłka (ile razy)", "typical error (how many times)")); a.set_ylabel(T("platforma", "platform")); a.legend(loc="upper center", bbox_to_anchor=(.4, 1.28), ncol=2, frameon=False, fontsize=10, handletextpad=.1, columnspacing=.8)
    return save(fig, "class_error")


def chart_alone():
    """Error of a model that knows only one thing about the post."""
    rows = [("likes", T("lajki", "likes")), ("comments", T("komentarze", "comments")), ("shares", T("udostępnienia", "shares")), ("followers", T("obserwujący", "followers")), ("our type label", T("typ posta", "type of post"))]; col = {"TikTok": "#3f2a6b", "YouTube": "#e4572e", "X (Twitter)": "#2a9d8f"}; fig, a = plt.subplots(figsize=(4.2, 2.5))
    for y, (k, _) in enumerate(rows):
        for p in P: a.scatter(10 ** views(k, p)["MAE alone"], y, s=70, color=col[p], alpha=.9, zorder=2, label=SH[p] if y == 0 else None)
    a.set_xscale("log"); a.set_xlim(1.7, 10.5); a.set_yticks(range(5)); a.set_yticklabels([r[1] for r in rows]); a.set_xticks([2, 3, 5, 10]); a.set_xticklabels(["×2", "×3", "×5", "×10"]); a.minorticks_off(); a.grid(axis="x", alpha=.25); a.set_ylim(4.5, -.5); a.set_xlabel(T("typowa pomyłka (ile razy)", "typical error (how many times)")); a.set_ylabel(T("co model wie", "what the model knows")); a.legend(loc="upper center", bbox_to_anchor=(.42, 1.2), ncol=3, frameon=False, fontsize=10, handletextpad=.1, columnspacing=.8)
    return save(fig, "class_alone")


def chart_labels():
    """Views at the same number of reactions: ordinary posts against promo."""
    fig, a = plt.subplots(figsize=(4.2, 2.5)); w = .36; ps = ["TikTok", "YouTube"]; o = [10 ** line(p, "organic") for p in ps]; q = [10 ** line(p, "promo") for p in ps]
    a.bar([i - w / 2 for i in range(2)], o, w, color=GR, label=T("zwykły post", "ordinary post")); a.bar([i + w / 2 for i in range(2)], q, w, color=BLUE, label=T("reklama", "promo"))
    fm = lambda v: f"{round(v, -2):,.0f}".replace(",", " " if PL else ",")
    for i in range(2): a.text(i - w / 2, o[i] + 60, fm(o[i]), ha="center", fontsize=11); a.text(i + w / 2, q[i] + 60, fm(q[i]), ha="center", fontsize=11, color=BLUE, fontweight="bold")
    a.set_xticks(range(2)); a.set_xticklabels(ps); a.set_ylim(0, 3900); a.set_yticks([0, 1000, 2000, 3000]); a.set_xlabel(T("platforma", "platform")); a.set_ylabel(T("wyświetlenia przy\n100 lajkach i komentarzach", "views at 100\nlikes and comments"), fontsize=10); a.legend(loc="upper center", bbox_to_anchor=(.5, 1.22), ncol=2, frameon=False, fontsize=10)
    return save(fig, "class_labels")


def lo(p, t): return 10 ** (line(p, "promo") - line(p, "organic"))
s = frame(T("1 · CO WYCHODZI Z DANYCH", "1 · WHAT THE DATA SHOW"), T("Lajki mówią o wyświetleniach dużo więcej niż liczba obserwujących", "Likes say far more about views than the follower count"),
          T("„Pomyłka ×3” = typowo 3 razy za dużo lub za mało. Reakcje = lajki, komentarze, udostępnienia. Test na markach, których model nie widział.", "\"Off by ×3\" = typically 3 times too high or too low. Reactions = likes, comments, shares. Tested on brands the model has not seen."),
          T("Platformy często ukrywają wyświetlenia, więc Brand24 je szacuje. 12 500 postów z 5 platform; prawdziwe wyświetlenia tylko na TikToku, YouTube i części X. Przedział na 90% postów: ×4 (TikTok), ×10 (YouTube), ×11 (X) w każdą stronę. Zasięg Brand24 może nie być tym samym co wyświetlenia. Pełny tekst: docs/pl/class_slides_script.md. Model z lewego wykresu to HistGradientBoosting bez strojenia, a porównanie na X używa extra trees. Środkowy wykres też używa extra trees. Na X różnica między reklamą a zwykłym postem ma ten sam kierunek, ale szerokie przedziały.", "Platforms often hide views, so Brand24 estimates them. 12,500 posts from 5 platforms; true views only on TikTok, YouTube and part of X. Range for 90% of posts: ×4 (TikTok), ×10 (YouTube), ×11 (X) each way. Brand24's reach may not be the same as views. Full text: docs/en/class_slides_script.md. The left chart's model is HistGradientBoosting without tuning; the comparison on X and the middle chart use extra trees. On X the promo difference points the same way but has wide intervals."))
cols = [(T("Ile się mylimy?", "How far off are we?"), chart_error(), T(f"Stała × obserwujący: ×7–8. Model z reakcjami: ×2–3. Na X Brand24 ok. ×{10 ** xr['Brand24 MAE']:.0f}, my ok. ×{10 ** xr['model MAE']:.0f}.", f"Constant × followers: ×7-8. Model with reactions: ×2-3. On X Brand24 about ×{10 ** xr['Brand24 MAE']:.0f}, ours about ×{10 ** xr['model MAE']:.0f}.")),
        (T("Co się liczy najbardziej?", "What matters most?"), chart_alone(), T("Same lajki wystarczą najbardziej. Typ posta prawie nic nie mówi.", "Likes alone go furthest. The type of post says almost nothing.")),
        (T("Czy pomaga rodzaj posta?", "Does the kind of post help?"), chart_labels(), T(f"Etykiety AI prawie nie poprawiają szacunku, ale reklamy mają {n(lo('TikTok', 0), 1)}–{n(lo('YouTube', 0), 1)}× więcej wyświetleń.", f"AI labels barely help the estimate, but promo gets {n(lo('TikTok', 0), 1)}-{n(lo('YouTube', 0), 1)}× the views."))]
for i, (h, f_, b) in enumerate(cols):
    x = .5 + i * 4.17; tb(s, x, 1.75, 4.0, .4, h, 18, True, ACC); pic(s, f_, x, 2.25, 4.0, 3.2); tb(s, x, 4.85, 4.0, 1.4, b, 16)
s = frame(T("2 · JAKIE MODELE I PODEJŚCIA", "2 · WHICH MODELS AND APPROACHES"), T("Sprawdziliśmy 10 rodzajów modeli: najlepsze mylą się 2–3 razy, a dodatkowe dane i grafy prawie tego nie zmieniają", "We tested 10 kinds of model: the best are off 2-3 times, and extra data and graphs barely change that"),
          T("Najlepsze: SVR (TikTok), CatBoost (YouTube, X). 40 ustawień na model.", "Best: SVR (TikTok), CatBoost (YouTube, X). 40 settings per model."),
          T("Wejście: obserwujący, reakcje, informacja o brakach. Etykiet AI nie sprawdził jeszcze człowiek. Strojenie daje dużo tylko modelom, które na domyślnych ustawieniach się przeuczają (extra trees na X 0,59 do 0,47). Po strojeniu rodziny się zbiegają, wyjątek to ridge na YouTube (0,55).", "Input: followers, reactions, missing-value flags. No human has checked the AI labels yet. Tuning matters only for models that overfit by default (extra trees on X 0.59 to 0.47). After tuning the families converge, except ridge on YouTube (0.55)."))
pic(s, f"approach_map_{LANG}.png", 1.15, 1.72, 11.0, 5.0)
C = {r["platform"]: r for r in AB["communities"]}; FP = {r["brand"]: r for r in AB["false_positive"]}; lp = [lopo(t) for t in ("tiktok", "youtube", "twitter")]; CAL = TR["calibration"]["twitter"]; cal = CAL["random"]; c50 = next(r for r in cal if r["n"] == 50); usp = {r["platform"]: r for r in TR["unknown_spread"]}


def chart_calibration():
    """How many posts with true views fix the level on X, as a factor."""
    fig, a = plt.subplots(figsize=(4.2, 2.9)); x = [r["n"] for r in cal]; g = "#2a9d8f"
    a.fill_between(x, [10 ** r["low"] for r in cal], [10 ** r["high"] for r in cal], color=g, alpha=.2, label=T("90% losowań", "90% of draws")); a.plot(x, [10 ** r["median"] for r in cal], "o-", color=g, ms=4, label=T("z kalibracją", "calibrated"))
    a.axhline(10 ** cal[0]["uncalibrated"], color="#a61e1e", ls="--", lw=1.4, label=T("bez kalibracji", "no calibration")); a.axhline(10 ** cal[0]["oracle"], color="black", ls=":", lw=1.4, label=T("idealna kalibracja", "perfect calibration"))
    a.set_xscale("log"); a.set_xticks([5, 10, 25, 50, 100, 500]); a.set_xticklabels(["5", "10", "25", "50", "100", "500"]); a.minorticks_off(); a.set_ylim(3.2, 6); a.set_yticks([3.5, 4, 4.5, 5, 5.5]); a.set_yticklabels([f"×{n(v, 1)}" for v in [3.5, 4, 4.5, 5, 5.5]]); a.set_xlabel(T("posty z prawdziwymi wyświetleniami", "posts with true views")); a.set_ylabel(T("pomyłka na X", "error on X")); a.legend(frameon=False, fontsize=9, loc="upper right")
    return save(fig, "class_calibration")


def chart_community():
    """TikTok communities only: readable on a slide."""
    r = [q for q in AB["community_profiles"] if q["platform"] == "TikTok"]; fig, a = plt.subplots(figsize=(6.1, 4.3)); big = lambda q: 10 ** q["median log10 views"]
    a.scatter([q["spam"] * 100 for q in r], [big(q) for q in r], s=[q["size"] * 2.4 for q in r], color="#7fa3bb", edgecolor="#2f6f93", alpha=.75)
    L = {"cricket": T("krykiet i vlogi (Cricket, Aon)", "cricket and vlogs (Cricket, Aon)"), "adidas": T("buty sportowe", "sneakers"), "lexus": T("samochody", "cars"), "seo": T("SEO i marketing", "SEO and marketing")}
    for q in r:
        k = q["top terms"].split(", ")[0]
        if k in L: a.annotate(L[k], (q["spam"] * 100, big(q)), xytext=(-8, 6) if k == "cricket" else (8, 5), textcoords="offset points", fontsize=10.5, ha="right" if k == "cricket" else "left", fontweight="bold" if k == "cricket" else "normal", bbox=dict(fc="white", ec="none", alpha=.8, pad=1))
    a.set_yscale("log"); a.set_yticks([300, 1000, 3000, 10000]); a.set_yticklabels(["300", "1000", "3000", "10 000" if PL else "10,000"]); a.minorticks_off(); a.grid(alpha=.2); a.set_xlim(0, 62); a.set_ylim(250, 12000); a.set_xlabel(T("spam w grupie (%)", "spam in the group (%)")); a.set_ylabel(T("mediana wyświetleń (skala log)", "median views (log scale)")); a.set_title(T("TikTok: każde koło to grupa podobnych tekstów", "TikTok: each circle is a group of similar texts"))
    return save(fig, "class_community")


s = frame(T("3 · INNA PLATFORMA I KALIBRACJA", "3 · ANOTHER PLATFORM AND CALIBRATION"), T("Model z innej platformy dobrze szereguje posty, ale może się mylić co do skali; około 50 postów to naprawia", "A model from another platform ranks posts well but can miss the scale; about 50 posts fix that"),
          T("Ridge uczony na dwóch platformach, test na trzeciej. Kalibrację sprawdziliśmy tylko tam, gdzie znamy prawdę.", "Ridge trained on two platforms, tested on the third. Calibration tested only where the truth is known."),
          T(f"Marki testowe usunięte z uczenia. Na FB i IG szacunek zależy od źródła o ×{n(usp['Facebook']['factor'], 1)} i ×{n(usp['Instagram']['factor'], 1)}. Kolumny po lewej to posty bez reakcji: dostają tę samą predykcję. Kalibracja: przesuwamy wszystkie predykcje o medianowy błąd n postów i liczymy błąd na reszcie. Na TikToku i YouTube nie ma czego poprawiać.", f"Test brands removed from training. On FB and IG the estimate depends on the source by ×{n(usp['Facebook']['factor'], 1)} and ×{n(usp['Instagram']['factor'], 1)}. Columns on the left are posts without reactions: they get the same prediction. Calibration: shift all predictions by the median error of n posts and score the rest. TikTok and YouTube have nothing to fix."))
pic(s, f"lopo_{LANG}.png", .5, 1.72, 8.2, 3.4); pic(s, chart_calibration(), 8.85, 1.72, 4.0, 3.4)
tb(s, .5, 4.82, 6.0, .35, T("Inna platforma", "Another platform"), 17, True, ACC); tb(s, .5, 5.2, 6.0, 1.2, T(f"TikTok {X(lp[0][0]['MAE'])}, YouTube {X(lp[1][0]['MAE'])}: prawie jak na własnych danych. X: {X(lp[2][0]['MAE'])} zamiast {X(lp[2][1]['MAE'])}, bo wszystko zawyżone ok. {n(10 ** abs(lp[2][0]['bias']), 1)}×.", f"TikTok {X(lp[0][0]['MAE'])}, YouTube {X(lp[1][0]['MAE'])}: nearly as on own data. X: {X(lp[2][0]['MAE'])} instead of {X(lp[2][1]['MAE'])}, as everything is about {n(10 ** abs(lp[2][0]['bias']), 1)}× too high."), 16)
tb(s, 6.85, 4.82, 6.0, .35, T("Mała próbka wystarcza", "A small sample is enough"), 17, True, ACC); tb(s, 6.85, 5.2, 6.0, 1.2, T(f"50 postów z X: pomyłka {X(c50['uncalibrated'])} → {X(c50['median'])}. Dla FB i IG trzeba zebrać taką próbkę.", f"50 posts from X: error {X(c50['uncalibrated'])} → {X(c50['median'])}. FB and IG need such a sample."), 16)
TR_ = AB["transcripts"]; dup = json.load(open(D / "graphlab_numbers.json"))["duplicates"]; dx = next(r for r in dup if r["platform"] == "X (Twitter)")
s = frame(T("4 · GRAFY", "4 · GRAPHS"), T("Grafy prawie nie poprawiają szacunku, ale pokazują, co naprawdę jest w danych", "Graphs barely improve the estimate, but they show what is really in the data"),
          T("Branże marek przypisał model AI, bez sprawdzenia. Część kandydatów to tylko inny język (Decathlon, Avanza), a Cricket Wireless tą metodą nie wypada nisko.", "Brand industries were assigned by an AI model, unchecked. Some candidates are just another language (Decathlon, Avanza), and Cricket Wireless does not score low with this method."),
          T(f"Grupy umiarkowanie stabilne (ARI {n(min(v['stability (ARI across seeds)'] for v in C.values()))}–{n(max(v['stability (ARI across seeds)'] for v in C.values()))}), pokrywają się z markami (NMI {n(min(v['NMI with brand'] for v in C.values()))}–{n(max(v['NMI with brand'] for v in C.values()))}), nie z typem posta (NMI {n(min(v['NMI with our type'] for v in C.values()))}–{n(max(v['NMI with our type'] for v in C.values()))}). Fałszywe dopasowania sprawdzone na jednym przykładzie. Graf tekstu: każdy post łączymy z 10 najbardziej podobnymi (TF-IDF, cosinus). Louvain szuka gęstych grup. Sprawdziliśmy też grafy autor–marka, hashtagów, czasu, duplikatów, PageRank, spektralne i propagację etykiet.", f"Groups moderately stable (ARI {n(min(v['stability (ARI across seeds)'] for v in C.values()))}-{n(max(v['stability (ARI across seeds)'] for v in C.values()))}), match brands (NMI {n(min(v['NMI with brand'] for v in C.values()))}-{n(max(v['NMI with brand'] for v in C.values()))}), not post type (NMI {n(min(v['NMI with our type'] for v in C.values()))}-{n(max(v['NMI with our type'] for v in C.values()))}). False matches checked on one example. Text graph: each post links to its 10 most similar (TF-IDF, cosine). Louvain finds dense groups. We also tried author-brand, hashtag, time and duplicate graphs, PageRank, spectral features and label propagation."))
CG = sg.cricket_numbers(ROOT); json.dump({k: CG[k] for k in ("cw_in_group", "cw_phone_edges", "group_brands")} | {"cricket wireless mentions": len(CG["cw"]), "mobile network posts": len(CG["phone"]), "group size": len(CG["group"])}, open(D / "cricket_graph_numbers.json", "w"), indent=1); sg.plot_cricket(CG, FIG / f"class_cricket_{LANG}.png", LANG)
pic(s, f"class_cricket_{LANG}.png", .5, 1.7, 6.5, 5.0)
MM = json.load(open(D / "mismatch_numbers.json")); AU = MM["flag_auc"]
points(s, 7.25, 1.75, 5.6, [(T("Jak:", "How:"), T("każdy post łączymy z 10 najbardziej podobnymi tekstami. Louvain dzieli graf na grupy, a słowa w ramkach mówią, o czym jest grupa.", "each post links to its 10 most similar texts. Louvain splits the graph into groups; the boxed words say what a group is about.")),
                          (T("Złe dopasowanie:", "False match:"), T(f"wszystkie {CG['cw_in_group']} wzmianek „Cricket Wireless” są w grupie o krykiecie, nie przy sieciach komórkowych. Ręcznie: 0 z {FP['Cricket Wireless']['mentions']} o firmie.", f"all {CG['cw_in_group']} \"Cricket Wireless\" mentions are in the cricket group, not with mobile networks. By hand: 0 of {FP['Cricket Wireless']['mentions']} about the company.")),
                          (T("Na wszystkich markach:", "Across all brands:"), T("pytamy, czy sąsiedzi wzmianki są z tej samej branży. Najsłabiej: Frontier na X (AI), Gap na YouTube (AI), Guardian Life (Minecraft). Kandydaci, jeszcze nie sprawdzeni ręcznie.", "we ask whether a mention's neighbours share its industry. Weakest: Frontier on X (AI), Gap on YouTube (AI), Guardian Life (Minecraft). Candidates, not yet checked by hand.")),
                          (T("Do szacowania:", "For the estimate:"), T("grafy dają najwyżej +0,009.", "graphs give at most +0.009."))], 15, 1.25)
s = frame(T("ZAPAS · NIEPEWNOŚĆ", "BACKUP · UNCERTAINTY"), T("Bez reakcji szacunek jest mało pewny; z reakcjami zakres mocno się zwęża", "Without reactions the estimate is uncertain; with reactions the range narrows sharply"),
          T("CatBoost uczony na tej samej platformie. Ciemny pasek: typowa pomyłka; jasny: 90% postów.", "CatBoost trained on the same platform. Dark bar: typical error; light: 90% of posts."),
          T("Test na markach spoza uczenia. Przedział liczony osobno dla poziomów reakcji pokrywa ok. 90% postów; wyjątek to TikTok przy zerze reakcji (mało danych). Czytaj tak: czarna linia to szacunek modelu, tu 1000. Ciemny pasek to typowa pomyłka. Jasny to zakres dla 90% postów, a liczba obok mówi, ile razy w każdą stronę.", "Tested on unseen brands. An interval per reaction tier covers about 90% of posts, except TikTok at zero reactions (little data). Read it this way: the black line is the estimate, here 1,000. The dark bar is the typical error. The light bar is the range for 90% of posts, and the number beside it says how many times each way."))
pic(s, f"reaction_ranges_{LANG}.png", .6, 1.65, 12.1, 5.05)
out = HERE / f"class_{LANG}.pptx"; prs.save(out); print("saved", out)
