"""Builds pitchdeck_<lang>.pptx (7 slides, speaker notes, Morph transitions, entrance animations).
Run: python pitchdeck/build.py [pl|en]"""
import re
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).parent
CSV = HERE.parent / "data" / "raw" / "reach_samples_students.csv"
LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
if LANG == "en":
    sys.path.insert(0, str(HERE))
    from strings_en import EN, NOTES_EN

FONT = "Calibri"
BG, PLUM, CORAL, SUN, BLUE, MINT = "FFF8EF", "2D1E5F", "E8475A", "FFC247", "3B5BDB", "1FB59A"
INK, GRAY, LINE, WHITE, SOFT = "1E1740", "5E5878", "E6DDEE", "FFFFFF", "F0E9F7"
N_SLIDES = 7


def rgb(h):
    return RGBColor.from_string(h)


def lighten(h, f):
    c = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return "".join(f"{round(v + (255 - v) * f):02X}" for v in c)


# ---- numbers from the CSV (nothing typed from memory) ----
df = pd.read_csv(CSV, usecols=["platform", "view_count"])
N_ROWS = len(df)
PER_PLATFORM = int(df.platform.value_counts().min())
VIEWS_PCT = {p: round(g.view_count.notna().mean() * 100) for p, g in df.groupby("platform")}
fmt = (lambda n: f"{n:,}") if LANG == "en" else (lambda n: f"{n:,}".replace(",", " "))


def tr(t):
    """On-slide text in the chosen language (Polish is the source language)."""
    return EN.get(t, t) if LANG == "en" else t

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


# ---- primitives ----
def shape(slide, kind, x, y, w, h, fill=None, line=None, line_w=1.0, adj=None, rot=0, alpha=None):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
        if alpha is not None:
            clr = s._element.spPr.find(qn("a:solidFill")).find(qn("a:srgbClr"))
            clr.append(parse_xml(f'<a:alpha xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" val="{alpha * 1000}"/>'))
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(line_w)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if adj is not None:
        s.adjustments[0] = adj
    s.rotation = rot
    return s


def text(slide, x, y, w, h, t, size=20, bold=False, color=INK, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, line in enumerate(tr(t).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.name, r.font.size, r.font.bold = FONT, Pt(size), bold
        r.font.color.rgb = rgb(color)
    return tb


def label(s, t, size=20, bold=False, color=WHITE):
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.1)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = tr(t)
    r.font.name, r.font.size, r.font.bold = FONT, Pt(size), bold
    r.font.color.rgb = rgb(color)


def shadow(s, blur=16, dist=5, alpha=16, color=PLUM):
    """Soft shadow under a shape (values in pt / %)."""
    eff = s._element.spPr.find(qn("a:effectLst"))
    eff.append(parse_xml(
        '<a:outerShdw xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        f'blurRad="{blur * 12700}" dist="{dist * 12700}" dir="5400000" algn="t" rotWithShape="0">'
        f'<a:srgbClr val="{color}"><a:alpha val="{alpha * 1000}"/></a:srgbClr></a:outerShdw>'))
    return s


def grad(fill_owner, c1, c2, angle=0):
    f = fill_owner.fill if hasattr(fill_owner, "fill") else fill_owner
    f.gradient()
    f.gradient_angle = angle
    st = f.gradient_stops
    st[0].position, st[0].color.rgb = 0.0, rgb(c1)
    st[1].position, st[1].color.rgb = 1.0, rgb(c2)


class Group:
    """A group of shapes animated as one; the helpers accept it in place of a slide."""
    def __init__(self, slide):
        self.g = slide.shapes.add_group_shape()
        self.shapes = self.g.shapes


def card(s, x, y, w, h, fill=WHITE, line=LINE, line_w=1.5):
    c = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill=fill, line=line, line_w=line_w, adj=0.07)
    return shadow(c)


def eye(s, cx, cy, w, ring=PLUM, bg=WHITE, pupil=CORAL):
    h = w * 0.55
    shape(s, MSO_SHAPE.OVAL, cx - w / 2, cy - h / 2, w, h, fill=bg, line=ring, line_w=3)
    d = h * 0.7
    shape(s, MSO_SHAPE.OVAL, cx - d / 2, cy - d / 2, d, d, fill=pupil)
    d2 = d * 0.35
    shape(s, MSO_SHAPE.OVAL, cx - d2 / 2, cy - d2 / 2, d2, d2, fill=ring)
    shape(s, MSO_SHAPE.OVAL, cx - d2 * 0.2, cy - d2 * 1.1, d2 * 0.55, d2 * 0.55, fill=WHITE)


# ---- "floating" background between slides (Morph links shapes named !!...) ----
# (x, y, diameter, opacity in %) for the coral, yellow and blue circles
BLOBS = {
    1: [(9.7, 0.4, 3.2, 100), (11.5, 4.4, 1.5, 100), (8.4, 4.8, 1.0, 100)],
    2: [(11.4, -1.5, 3.2, 14), (12.5, 6.6, 1.5, 40), (-0.5, 6.9, 1.0, 22)],
    3: [(11.9, -1.2, 2.6, 14), (-0.7, 6.5, 1.6, 40), (12.7, 3.4, 0.9, 22)],
    4: [(12.3, -0.9, 2.2, 14), (12.4, 6.5, 1.8, 40), (-0.5, 6.8, 1.1, 22)],
    5: [(11.2, -1.6, 3.0, 14), (-0.8, 6.6, 1.6, 40), (12.8, 6.4, 1.0, 22)],
    6: [(12.0, -1.1, 2.6, 14), (12.6, 4.9, 1.4, 40), (-0.5, 6.9, 1.0, 22)],
    7: [(10.6, -1.4, 3.4, 14), (12.3, 6.3, 1.8, 40), (-0.4, 6.9, 1.0, 22)],
}


def new_slide(title=None, dark=False, accent=CORAL):
    s = prs.slides.add_slide(BLANK)
    n = len(prs.slides)
    if dark:
        grad(s.background, "3A2878", "170E3A", angle=315)
    else:
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = rgb(BG)
    for name, col, (x, y, d, a) in zip(("!!b1", "!!b2", "!!b3"), (CORAL, SUN, BLUE), BLOBS[n]):
        b = shape(s, MSO_SHAPE.OVAL, x, y, d, d, fill=col, alpha=None if a == 100 else a)
        b.name = name
    # progress bar (Morph lengthens it)
    p = shape(s, MSO_SHAPE.RECTANGLE, 0, 7.38, 13.333 * n / N_SLIDES, 0.12, fill=SUN if dark else CORAL)
    p.name = "!!progress"
    if title:
        d = shape(s, MSO_SHAPE.OVAL, 0.7, 0.62, 0.34, 0.34, fill=accent)
        d.name = "!!dot"
        text(s, 1.2, 0.4, 11.4, 0.8, title, size=38, bold=True, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    return s


# ---- transitions and animations (raw XML, python-pptx has no API) ----
P_NS = 'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'


def transition(slide, morph=True):
    if morph:
        xml = ('<mc:AlternateContent xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006">'
               '<mc:Choice xmlns:p159="http://schemas.microsoft.com/office/powerpoint/2015/09/main" Requires="p159">'
               f'<p:transition {P_NS} xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" '
               'spd="slow" p14:dur="1200"><p159:morph option="byObject"/></p:transition></mc:Choice>'
               f'<mc:Fallback><p:transition {P_NS} spd="slow"><p:fade/></p:transition></mc:Fallback>'
               '</mc:AlternateContent>')
    else:
        xml = f'<p:transition {P_NS} spd="slow"><p:fade/></p:transition>'
    slide._element.append(parse_xml(xml))


def animate(slide, steps, dur=500, gap=120):
    """Each step (a list of shapes/groups) flies in from below with a fade, on its own, after the previous one."""
    nid = iter(range(3, 100000))

    def effect(spid, node):
        e, s1, a1, a2, a3 = (next(nid) for _ in range(5))
        tgt = f'<p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl>'

        def mv(i, attr, start):
            return (f'<p:anim calcmode="lin" valueType="num"><p:cBhvr additive="base"><p:cTn id="{i}" dur="{dur}" fill="hold"/>'
                    f'{tgt}<p:attrNameLst><p:attrName>{attr}</p:attrName></p:attrNameLst></p:cBhvr><p:tavLst>'
                    f'<p:tav tm="0"><p:val><p:strVal val="{start}"/></p:val></p:tav>'
                    f'<p:tav tm="100000"><p:val><p:strVal val="#{attr}"/></p:val></p:tav></p:tavLst></p:anim>')
        return (f'<p:par><p:cTn id="{e}" presetID="42" presetClass="entr" presetSubtype="0" fill="hold" nodeType="{node}">'
                '<p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
                f'<p:set><p:cBhvr><p:cTn id="{s1}" dur="1" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn>'
                f'{tgt}<p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr>'
                '<p:to><p:strVal val="visible"/></p:to></p:set>'
                f'<p:animEffect transition="in" filter="fade"><p:cBhvr><p:cTn id="{a1}" dur="{dur}"/>{tgt}</p:cBhvr></p:animEffect>'
                + mv(a2, "ppt_x", "#ppt_x") + mv(a3, "ppt_y", "#ppt_y+.08") +
                '</p:childTnLst></p:cTn></p:par>')

    groups, t = [], 0
    for step in steps:
        gid = next(nid)
        effs = "".join(effect(sh.shape_id, "afterEffect" if k == 0 else "withEffect") for k, sh in enumerate(step))
        groups.append(f'<p:par><p:cTn id="{gid}" fill="hold"><p:stCondLst><p:cond delay="{t}"/></p:stCondLst>'
                      f'<p:childTnLst>{effs}</p:childTnLst></p:cTn></p:par>')
        t += dur + gap
    xml = (f'<p:timing {P_NS}><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>'
           '<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>'
           '<p:par><p:cTn id="99999" fill="hold"><p:stCondLst><p:cond delay="indefinite"/>'
           '<p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond></p:stCondLst>'
           f'<p:childTnLst>{"".join(groups)}</p:childTnLst></p:cTn></p:par>'
           '</p:childTnLst></p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
           '<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq>'
           '</p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>')
    slide._element.append(parse_xml(xml))


def finish(slide, steps, morph=True):
    transition(slide, morph)
    if steps:
        animate(slide, [[x.g if isinstance(x, Group) else x for x in st] for st in steps])


def notes(s, body):
    if LANG == "en":
        body = NOTES_EN[len(prs.slides)].format(**VIEWS_PCT)
    words = len(re.findall(r"\w+", body))
    sec = max(5, round(words / 2.4 / 5) * 5)
    s.notes_slide.notes_text_frame.text = f"{body} (~{sec} s)"
    NOTES.append((words, sec))


NOTES = []

# ============ 1. Title ============
s = new_slide(dark=True)
for sx, sy, sd, sc in [(8.1, 1.2, 0.35, SUN), (12.6, 3.3, 0.28, WHITE), (7.6, 3.9, 0.22, WHITE),
                       (10.2, 6.3, 0.3, SUN)]:
    shape(s, MSO_SHAPE.STAR_4_POINT, sx, sy, sd, sd, fill=sc)
post = Group(s)
pc = shape(post, MSO_SHAPE.ROUNDED_RECTANGLE, 8.9, 2.3, 3.5, 2.5, fill=WHITE, adj=0.09, rot=-6)
shadow(pc, blur=30, dist=12, alpha=40, color="0B0620")
shape(post, MSO_SHAPE.OVAL, 9.2, 2.65, 0.55, 0.55, fill=SOFT, rot=-6)
shape(post, MSO_SHAPE.ROUNDED_RECTANGLE, 9.9, 2.75, 1.7, 0.15, fill=SOFT, adj=0.5, rot=-6)
eye(post, 10.65, 3.85, 1.3)
q = shape(post, MSO_SHAPE.ROUNDED_RECTANGLE, 11.3, 4.15, 0.9, 0.5, fill=SUN, adj=0.5, rot=-6)
shadow(q, blur=10, dist=3, alpha=30, color="0B0620")
label(q, "???", size=22, bold=True, color=INK)
head = Group(s)
shape(head, MSO_SHAPE.RECTANGLE, 0.9, 1.95, 0.14, 2.9, fill=SUN)
text(head, 1.3, 1.8, 7.3, 1.9, "Ile osób naprawdę\nzobaczyło ten post?", size=52, bold=True, color=WHITE)
sub = text(s, 1.3, 3.95, 7.0, 1.0, "Jak estymować widoczność\npostów społecznościowych", size=28, color="FFD98A")
foot = text(s, 1.3, 6.3, 7, 0.5, "Projekt AMC  ·  dla Brand24", size=22, color="C9C2E0")
finish(s, [[head], [sub, foot], [post]], morph=False)
notes(s, "Dzień dobry. Zacznę od prostego pytania: ile osób naprawdę zobaczyło dany post w mediach "
         "społecznościowych? Wydaje się, że to podstawowa informacja, a bardzo często po prostu jej "
         "nie mamy. Nasz projekt dotyczy tego, jak taką widoczność oszacować, kiedy platforma jej nie "
         "pokazuje. Robimy go w ramach kursu analizy mediów, na potrzeby firmy Brand24. Przez najbliższe "
         "minuty pokażę naszą hipotezę, jak chcemy ją sprawdzić, komu się to przyda i dlaczego nam się uda.")

# ============ 2. Problem ============
s = new_slide("Platformy ukrywają widoczność postów")
mock = Group(s)
card(mock, 0.7, 1.7, 4.9, 5.0)
shape(mock, MSO_SHAPE.OVAL, 1.05, 2.05, 0.75, 0.75, fill=BLUE)
shape(mock, MSO_SHAPE.ROUNDED_RECTANGLE, 2.0, 2.15, 2.6, 0.2, fill=SOFT, adj=0.5)
shape(mock, MSO_SHAPE.ROUNDED_RECTANGLE, 2.0, 2.5, 1.7, 0.2, fill=SOFT, adj=0.5)
shape(mock, MSO_SHAPE.ROUNDED_RECTANGLE, 1.05, 3.1, 4.2, 0.2, fill=SOFT, adj=0.5)
shape(mock, MSO_SHAPE.ROUNDED_RECTANGLE, 1.05, 3.45, 3.4, 0.2, fill=SOFT, adj=0.5)
eye(mock, 2.3, 5.05, 1.5)
pill = shape(mock, MSO_SHAPE.ROUNDED_RECTANGLE, 3.45, 4.7, 1.7, 0.75, fill=SUN, adj=0.5)
shadow(pill, blur=8, dist=3, alpha=22)
label(pill, "???", size=34, bold=True, color=INK)
text(mock, 1.05, 5.95, 4.3, 0.5, "Ile osób to zobaczyło?", size=24, bold=True, color=PLUM, align=PP_ALIGN.CENTER)
rows = [("1", "Zasięg często jest ukryty", PLUM), ("2", "Zasięg to ich towar:\nsprzedają reklamy", BLUE),
        ("3", "Brand24 musi więc szacować", CORAL)]
steps = [[mock]]
for i, (n, t, col) in enumerate(rows):
    y = 1.7 + i * 1.7
    g = Group(s)
    r = shape(g, MSO_SHAPE.ROUNDED_RECTANGLE, 6.1, y, 6.55, 1.55, fill=col, adj=0.12)
    grad(r, col, lighten(col, 0.25), angle=0)
    shadow(r, blur=14, dist=5, alpha=22, color=col)
    c = shape(g, MSO_SHAPE.OVAL, 6.4, y + 0.35, 0.85, 0.85, fill=WHITE)
    label(c, n, size=30, bold=True, color=col)
    text(g, 7.5, y, 5.0, 1.55, t, size=26, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    steps.append([g])
finish(s, steps)
notes(s, "Zacznijmy od problemu. Platformy społecznościowe bardzo często nie ujawniają, ile osób "
         "faktycznie zobaczyło dany post. Jednym z powodów jest to, że same zarabiają na reklamach, "
         "więc zasięg jest dla nich zasobem handlowym, a nie informacją, którą chętnie oddają. Brand24 "
         "to narzędzie, które monitoruje wzmianki o markach w internecie. Widzi więc, że ktoś napisał o "
         "marce, ale nie zawsze wie, ilu ludzi to przeczytało. Wyobraźcie sobie post o marce: nie wiemy, "
         "czy zobaczyło go dziesięć osób, czy dziesięć tysięcy, a dla marki to ogromna różnica. "
         "Dlatego musi tę widoczność szacować. I tu pojawia się nasza hipoteza.")

# ============ 3. Hypothesis (instead of the problem) ============
s = new_slide("Nasza hipoteza", accent=CORAL)
hyp = Group(s)
hc = shape(hyp, MSO_SHAPE.ROUNDED_RECTANGLE, 0.7, 1.6, 11.95, 1.55, fill=PLUM, adj=0.12)
grad(hc, PLUM, "4B3591", angle=0)
shadow(hc, blur=18, dist=6, alpha=30)
text(hyp, 1.05, 1.72, 4, 0.4, "HIPOTEZA", size=18, bold=True, color=SUN)
text(hyp, 1.05, 2.1, 11.3, 0.95, "Widoczność posta oszacujemy lepiej niż reguła Brand24,\n"
     "łącząc obserwujących z reakcjami: polubieniami, komentarzami, udostępnieniami",
     size=24, bold=True, color=WHITE)
left = Group(s)
card(left, 0.7, 3.4, 5.4, 2.3)
text(left, 1.0, 3.52, 4.8, 0.45, "DZIŚ", size=20, bold=True, color=GRAY)
text(left, 1.0, 3.92, 4.8, 0.5, "Jedna liczba", size=28, bold=True)
shape(left, MSO_SHAPE.RECTANGLE, 1.0, 4.78, 4.8, 0.06, fill=LINE)
shape(left, MSO_SHAPE.OVAL, 3.15, 4.58, 0.46, 0.46, fill=CORAL)
text(left, 1.0, 5.15, 4.8, 0.45, "obserwujący × stała", size=20, color=GRAY)
right = Group(s)
ar = shape(right, MSO_SHAPE.RIGHT_ARROW, 6.2, 4.25, 0.65, 0.6, fill=SUN)
shadow(ar, blur=8, dist=3, alpha=25)
card(right, 6.95, 3.4, 5.7, 2.3, line=BLUE, line_w=3)
text(right, 7.25, 3.52, 5.1, 0.45, "NASZ POMYSŁ", size=20, bold=True, color=BLUE)
text(right, 7.25, 3.92, 5.1, 0.5, "Estymacja + przedział", size=28, bold=True)
shape(right, MSO_SHAPE.RECTANGLE, 7.25, 4.78, 5.1, 0.06, fill=LINE)
band = shape(right, MSO_SHAPE.ROUNDED_RECTANGLE, 8.4, 4.53, 2.6, 0.56, fill="C9D6FF", adj=0.5)
grad(band, "EEF2FF", "C9D6FF", angle=0)
shape(right, MSO_SHAPE.OVAL, 9.46, 4.58, 0.46, 0.46, fill=BLUE)
text(right, 7.25, 5.15, 5.1, 0.45, "Wiemy też, kiedy jej nie ufać", size=20, color=GRAY)
fal = Group(s)
fp = card(fal, 0.7, 5.95, 11.95, 0.95, fill=WHITE, line=CORAL, line_w=2.5)
text(fal, 1.05, 5.95, 11.3, 0.95, "Obalona, jeśli model nie pobije reguły Brand24 na postach ze znanymi wyświetleniami",
     size=22, bold=True, color=CORAL, anchor=MSO_ANCHOR.MIDDLE)
finish(s, [[hyp], [left], [right], [fal]])
notes(s, "Zamiast problemu stawiamy hipotezę. Brzmi tak: widoczność posta da się oszacować lepiej niż "
         "dzisiejsza reguła Brand24, jeśli oprócz liczby obserwujących weźmiemy pod uwagę reakcje, czyli "
         "polubienia, komentarze i udostępnienia. Dziś wynikiem jest jedna liczba: obserwujący razy stała. "
         "My chcemy dać estymację razem z przedziałem niepewności, żeby było wiadomo, kiedy jej nie ufać. "
         "Hipoteza jest falsyfikowalna: jeśli na postach, dla których znamy prawdziwe wyświetlenia, nasz "
         "model nie będzie lepszy od reguły Brand24, uznajemy ją za obaloną. Taki wynik też jest wartościowy.")

# ============ 4. Concept (technology) ============
s = new_slide("Jak to ugryziemy: koncepcja", accent=MINT)
plan5 = [("1", "Eksploracja", "Braki, rozkłady, dziwne przypadki", PLUM, WHITE),
         ("2", "Estymacja", "Model z obserwujących i zaangażowania", BLUE, WHITE),
         ("3", "Walidacja", "Test tam, gdzie znamy wyświetlenia", CORAL, WHITE),
         ("4", "Niepewność", "Przedziały i odporność wyników", SUN, INK)]
steps = []
for i, (n, t, sub, col, fg) in enumerate(plan5):
    x = 0.7 + i * 3.0
    g = Group(s)
    chev = shape(g, MSO_SHAPE.PENTAGON, x, 1.75, 3.1, 1.4, fill=col, adj=0.3)
    grad(chev, col, lighten(col, 0.22), angle=0)
    shadow(chev, blur=12, dist=4, alpha=22, color=col)
    text(g, x + 0.3, 1.75, 0.7, 1.4, n, size=44, bold=True, color=fg, anchor=MSO_ANCHOR.MIDDLE)
    text(g, x + 1.0, 1.75, 1.7, 1.4, t, size=24, bold=True, color=fg, anchor=MSO_ANCHOR.MIDDLE)
    card(g, x + 0.05, 3.3, 2.75, 1.3, line=col, line_w=2.5)
    text(g, x + 0.3, 3.3, 2.3, 1.3, sub, size=20, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    steps.append([g])
strip = Group(s)
st = shape(strip, MSO_SHAPE.ROUNDED_RECTANGLE, 0.7, 4.95, 11.95, 1.85, fill=PLUM, adj=0.12)
grad(st, PLUM, "4B3591", angle=0)
shadow(st, blur=18, dist=6, alpha=30)
text(strip, 1.05, 5.1, 3.6, 0.4, "NA CZYM PRACUJEMY", size=18, bold=True, color=SUN)
text(strip, 1.05, 5.5, 3.6, 0.7, fmt(N_ROWS), size=40, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
text(strip, 1.05, 6.2, 3.6, 0.45, f"mentions, {fmt(PER_PLATFORM)} from each of 5 platforms" if LANG == "en" else f"wzmianek, po {fmt(PER_PLATFORM)} z 5 platform", size=18, color="C9C2E0")
shape(strip, MSO_SHAPE.RECTANGLE, 4.55, 5.2, 0.02, 1.4, fill="6A58A8")
text(strip, 4.85, 5.1, 7.6, 0.4, "WZMIANKI Z ZNANYMI WYŚWIETLENIAMI", size=18, bold=True, color=SUN)
NAMES = {"tiktok": "TikTok", "youtube": "YouTube", "twitter": "X (Twitter)", "facebook": "Facebook",
         "instagram": "Instagram"}
for i, p in enumerate(NAMES):
    vx, pct = 4.85 + i * 1.53, VIEWS_PCT[p]
    col = MINT if pct >= 90 else SUN if pct > 0 else CORAL
    text(strip, vx, 5.45, 1.4, 0.55, f"{pct}%", size=28, bold=True, color=col, anchor=MSO_ANCHOR.MIDDLE)
    shape(strip, MSO_SHAPE.ROUNDED_RECTANGLE, vx, 6.05, 1.3, 0.16, fill="4E3C8C", adj=0.5)
    if pct:
        shape(strip, MSO_SHAPE.ROUNDED_RECTANGLE, vx, 6.05, 1.3 * pct / 100, 0.16, fill=col, adj=0.5)
    text(strip, vx, 6.3, 1.5, 0.4, NAMES[p], size=16, color="C9C2E0")
steps.append([strip])
finish(s, steps)
notes(s, "Jak chcemy to ugryźć od strony technologii? Cztery kroki. Najpierw eksploracja danych: braki, "
         "rozkłady i dziwne przypadki, na przykład posty z ogromnym zasięgiem. Potem estymacja: model w "
         "Pythonie na skali logarytmicznej, który korzysta z obserwujących, polubień, komentarzy, "
         "udostępnień i platformy. Trzeci krok to walidacja krzyżowa tam, gdzie znamy prawdziwe "
         "wyświetlenia. Czwarty to niepewność: zamiast jednej liczby podajemy przedział. Pracujemy na "
         "dwunastu i pół tysiąca wzmianek z pięciu platform. Na dole widać, dla jakiej części wzmianek "
         f"znamy prawdziwe wyświetlenia: TikTok {VIEWS_PCT['tiktok']} procent, YouTube "
         f"{VIEWS_PCT['youtube']}, Twitter {VIEWS_PCT['twitter']}, a Facebook i Instagram zero. "
         "To nasz punkt odniesienia.")

# ============ 5. Client: society ============
s = new_slide("Kto na tym skorzysta?", accent=BLUE)
apps = [
    ("Firmy i marki", "Wiedzą, ile osób naprawdę zobaczyło wzmiankę", MSO_SHAPE.DONUT, CORAL, WHITE),
    ("Małe firmy i NGO", "Rzetelny obraz bez drogich danych platform", MSO_SHAPE.STAR_5_POINT, BLUE, WHITE),
    ("Dziennikarze", "Widzą, jak daleko zaszła dezinformacja", MSO_SHAPE.LIGHTNING_BOLT, SUN, INK),
    ("Badacze mediów", "Miara z niepewnością zamiast zgadywania", MSO_SHAPE.UP_ARROW, MINT, INK),
    ("Każdy z nas", "Mniej napompowanych statystyk w obiegu", None, PLUM, WHITE),
]
pos = [(0.7, 1.65), (4.77, 1.65), (8.84, 1.65), (2.735, 4.4), (6.805, 4.4)]
steps = []
for (t, sub, ic, col, fg), (x, y) in zip(apps, pos):
    g = Group(s)
    card(g, x, y, 3.8, 2.55, line=col, line_w=2.5)
    c = shape(g, MSO_SHAPE.OVAL, x + 0.3, y + 0.28, 0.95, 0.95, fill=col)
    grad(c, lighten(col, 0.2), col, angle=315)
    shadow(c, blur=10, dist=4, alpha=28, color=col)
    if ic is None:
        for j, bh in enumerate([0.22, 0.38, 0.54]):
            shape(g, MSO_SHAPE.RECTANGLE, x + 0.55 + j * 0.16, y + 1.0 - bh, 0.11, bh, fill=fg)
    else:
        shape(g, ic, x + 0.53, y + 0.51, 0.5, 0.5, fill=fg)
    text(g, x + 0.3, y + 1.35, 3.3, 0.5, t, size=24, bold=True)
    text(g, x + 0.3, y + 1.85, 3.3, 0.65, sub, size=20, color=GRAY)
    steps.append([g])
finish(s, steps)
notes(s, "Naszym klientem jest społeczeństwo, więc kto na tym zyska? Po pierwsze firmy i marki: wiedzą, "
         "ile osób naprawdę zobaczyło wzmiankę. Po drugie małe firmy i organizacje pozarządowe, których nie "
         "stać na płatne dane od platform. Po trzecie dziennikarze i fact-checkerzy: mogą ocenić, jak daleko "
         "zaszła fałszywa informacja. Po czwarte badacze mediów, którzy dostają miarę razem z niepewnością. "
         "I wreszcie każdy z nas: mniej napompowanych liczb w raportach i w mediach.")

# ============ 6. Impact (instead of TAM) ============
s = new_slide("Impact: komu to coś da?", accent=SUN)
funnel = [
    ("5,24 mld", "kont w mediach społecznościowych:\nich posty mogą być mierzone uczciwiej", "DataReportal / Kepios, 2025",
     11.95, PLUM, "4B3591"),
    ("2 400+", "firm korzysta z Brand24", "dane podawane przez Brand24", 9.4, BLUE, lighten(BLUE, 0.2)),
    ("~12 000", "osób czyta ich raporty", "założenie: ok. 5 osób na firmę", 6.9, CORAL, lighten(CORAL, 0.2)),
]
steps = []
for k, (big, desc, src, w, c1, c2) in enumerate(funnel):
    x, y = (13.333 - w) / 2, 1.6 + k * 1.6
    g = Group(s)
    bar = shape(g, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, 1.4, fill=c1, adj=0.15)
    grad(bar, c1, c2, angle=0)
    shadow(bar, blur=14, dist=5, alpha=25, color=c1)
    text(g, x + 0.35, y, 2.6, 1.4, big, size=38, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    text(g, x + 3.0, y + 0.12, w - 3.3, 0.85, desc, size=20, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    text(g, x + 3.0, y + 1.0, w - 3.3, 0.35, src, size=14, color="E8E2F5")
    steps.append([g])
cap = text(s, 0.7, 6.55, 11.95, 0.4, "Szacunek-zabawa: rzędy wielkości, nie pomiar", size=18, bold=True,
           color=GRAY, align=PP_ALIGN.CENTER)
steps.append([cap])
finish(s, steps)
notes(s, "Zamiast wielkości rynku szacujemy impact, i traktujemy to jako zabawę w rzędy wielkości. Na górze: "
         "na świecie jest około pięciu miliardów kont w mediach społecznościowych, tak podaje raport "
         "DataReportal. Każdy ich post może być mierzony uczciwiej. Bliżej nas: Brand24 podaje, że korzysta z "
         "niego ponad dwa tysiące czterysta firm. Jeśli w każdej raporty czyta około pięciu osób, to "
         "bezpośrednio jakieś dwanaście tysięcy ludzi dostaje lepsze liczby. To nie jest pomiar, tylko "
         "oszacowanie, ale pokazuje skalę.")

# ============ 7. Moat: feasibility + expertise ============
s = new_slide("Dlaczego nam się uda?", accent=MINT)
feas = Group(s)
card(feas, 0.7, 1.6, 5.85, 5.25, line=MINT, line_w=2.5)
text(feas, 1.05, 1.8, 5.2, 0.45, "WYKONALNOŚĆ", size=22, bold=True, color=MINT)
for k, t in enumerate(["Legalne dane od zleceniodawcy: Brand24", f"{fmt(N_ROWS)} public mentions, 5 platforms" if LANG == "en" else f"{fmt(N_ROWS)} publicznych wzmianek, 5 platform",
                       "Prawda bazowa: TikTok i YouTube", "Bez danych osobowych autorów (RODO)",
                       "Wstępny test w eksploracji: obiecujący"]):
    y = 2.45 + k * 0.85
    shape(feas, MSO_SHAPE.OVAL, 1.05, y + 0.12, 0.3, 0.3, fill=MINT)
    text(feas, 1.55, y, 4.8, 0.6, t, size=20, color=INK, anchor=MSO_ANCHOR.MIDDLE)
exp = Group(s)
card(exp, 6.8, 1.6, 5.85, 5.25, line=BLUE, line_w=2.5)
text(exp, 7.15, 1.8, 5.2, 0.45, "EKSPERTYZA: STAN WIEDZY", size=22, bold=True, color=BLUE)
refs = [("Cha i in., 2010", "Obserwujący to nie realny wpływ"),
        ("Szabo i Huberman, 2010", "Popularność treści da się przewidywać"),
        ("Bakshy i in., 2011", "Pojedynczy post trudno przewidzieć"),
        ("Angelopoulos i Bates, 2023", "Przedziały niepewności z gwarancją")]
for k, (r, d) in enumerate(refs):
    y = 2.45 + k * 1.07
    shape(exp, MSO_SHAPE.RECTANGLE, 7.15, y + 0.05, 0.08, 0.8, fill=BLUE)
    text(exp, 7.45, y, 5.0, 0.45, r, size=20, bold=True, color=INK)
    text(exp, 7.45, y + 0.42, 5.0, 0.45, d, size=18, color=GRAY)
finish(s, [[feas], [exp]])
notes(s, "Dlaczego nam się uda? Dwa powody. Pierwszy to wykonalność: mamy legalne dane od zleceniodawcy, "
         "firmy Brand24, czyli dwanaście i pół tysiąca publicznych wzmianek z pięciu platform. Na TikToku i "
         "YouTubie znamy prawdziwe wyświetlenia, a danych osobowych autorów nie używamy. Wstępny test w "
         "eksploracji jest obiecujący. Drugi to stan wiedzy, na którym się opieramy: Cha i współautorzy "
         "pokazali, że liczba obserwujących to nie realny wpływ. Szabo i Huberman, że popularność treści da "
         "się przewidywać. Bakshy i współautorzy, że pojedynczy post trudno przewidzieć, dlatego potrzebujemy "
         "niepewności. A metody konformalne, opisane przez Angelopoulosa i Batesa, dają przedziały z "
         "gwarancją. Dziękuję.")

out = HERE / f"pitchdeck_{LANG}.pptx"
prs.save(out)
tot_w = sum(w for w, _ in NOTES)
tot_s = sum(t for _, t in NOTES)
print(f"saved {out}; slides={len(prs.slides)}; words in notes={tot_w}; time={tot_s}s")
