"""Excalidraw scene of what we tested: pitchdeck/approach_map_<lang>.excalidraw, rendered to figures/approach_map_<lang>.png with the Excalidraw CLI.
Run: python pitchdeck/approach_map.py [pl|en], then: npx @excalidraw/cli render pitchdeck/approach_map_pl.excalidraw --out figures/approach_map_pl.png --scale 2"""
import json, sys
from pathlib import Path

LANG = sys.argv[1] if len(sys.argv) > 1 else "pl"; PL = LANG == "pl"; T = lambda a, b: a if PL else b
INK, BLUE, SAND, GREY, RES = "#1e1e1e", "#d7e6f0", "#f3ead7", "#ececec", "#a61e1e"
E = []; n = [0]


def base(kind, x, y, w, h, **kw):
    n[0] += 1
    return {"id": f"e{n[0]}", "type": kind, "x": x, "y": y, "width": w, "height": h, "angle": 0, "strokeColor": INK, "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None, "roundness": None, "seed": 1000 + n[0], "version": 1, "versionNonce": 2000 + n[0], "isDeleted": False, "boundElements": None, "updated": 1, "link": None, "locked": False, **kw}


def text(x, y, w, s, size=20, align="left", color=INK):
    lines = s.count("\n") + 1
    E.append(base("text", x, y, w, lines * size * 1.25, text=s, originalText=s, fontSize=size, fontFamily=5, textAlign=align, verticalAlign="top", containerId=None, autoResize=True, lineHeight=1.25, strokeColor=color))


def box(x, y, w, h, title, body, fill):
    E.append(base("rectangle", x, y, w, h, backgroundColor=fill, roundness={"type": 3}))
    text(x + 16, y + 12, w - 32, title, 25)
    text(x + 16, y + 54, w - 32, body, 19)


def arrow(x, y, dx, dy=0):
    E.append(base("arrow", x, y, abs(dx), abs(dy), points=[[0, 0], [dx, dy]], lastCommittedPoint=None, startBinding=None, endBinding=None, startArrowhead=None, endArrowhead="arrow"))


def card(x, y, w, h, title, body, result, fill):
    """A box with what we did and, in colour, what came out of it."""
    box(x, y, w, h, title, body, fill); text(x + 16, y + h - 74, w - 32, result, 19, color=RES)


W = 345; G = 45; H = 275; X = [0, W + G, 2 * (W + G), 3 * (W + G)]
text(0, -66, 1500, T("Co sprawdziliśmy i co z tego wyszło", "What we tested and what came out"), 38)
card(X[0], 0, W, H, T("Dane", "Data"), T("12 500 wzmianek, 5 platform\nwyświetlenia: TikTok,\nYouTube, część X\nFB, IG: brak", "12,500 mentions, 5 platforms\nviews: TikTok, YouTube,\npart of X\nFB, IG: none"), T("prawda tylko\nna 3 z 5 platform", "truth only\non 3 of 5 platforms"), BLUE)
card(X[1], 0, W, H, T("Kolumny", "Columns"), T("obserwujący; reakcje = lajki,\nkomentarze, udostępnienia;\ntekst, kalendarz, marka,\netykiety AI, cechy z grafów", "followers; reactions = likes,\ncomments, shares;\ntext, calendar, brand,\nAI labels, graph features"), T("lajki niosą prawie\ncały sygnał", "likes carry nearly\nall the signal"), SAND)
card(X[2], 0, W, H, T("Modele", "Models"), T("ridge, kNN, SVR, MLP, 2 lasy,\nboosting, LightGBM, XGBoost,\nCatBoost; 40 ustawień na fold,\nblend najlepszych 3", "ridge, kNN, SVR, MLP, 2 forests,\nboosting, LightGBM, XGBoost,\nCatBoost; 40 settings per fold,\nblend of the top 3"), T("najlepsze: SVR (TikTok ×1,9),\nCatBoost (YouTube ×2,8, X ×3)", "best: SVR (TikTok ×1.9),\nCatBoost (YouTube ×2.8, X ×3)"), SAND)
card(X[3], 0, W, H, T("Sprawdzanie", "Validation"), T("test na markach spoza uczenia\nustawienia dobierane bez testu\nniepewność: losowanie marek\ntest na innej platformie", "tested on unseen brands\nsettings chosen without the test\nuncertainty: resampling brands\ntest on another platform"), T("na X ok. ×3,\nzasięg Brand24 ok. ×12", "on X about ×3,\nBrand24's reach about ×12"), SAND)
for i in range(3): arrow(X[i] + W + 6, H / 2, G - 12)
Y2 = H + 40
card(X[0], Y2, W, H, T("Grafy", "Graphs"), T("tekst kNN, autor-marka,\nmarka-marka, hashtagi, czas,\nduplikaty, Louvain, PageRank,\nspektralne, propagacja", "text kNN, author-brand,\nbrand-brand, hashtags, time,\nduplicates, Louvain, PageRank,\nspectral, propagation"), T("prawie bez poprawy; ale wykrywa\nkampanie i złe dopasowania", "barely any gain; but finds\ncampaigns and false matches"), GREY)
card(X[1], Y2, W, H, T("Ablacja i etykiety", "Ablation and labels"), T("każda grupa 3 razy: sama,\ndodana, usunięta\nważność kolumn\nbłędy w podgrupach", "each group 3 ways: alone,\nadded, removed\ncolumn importance\nerrors in subgroups"), T("etykiety do estymacji: ok. 0;\ndo opisu danych: tak", "labels for the estimate: ~0;\nfor describing data: yes"), GREY)
card(X[2], Y2, W, H, T("Inne podejścia", "Other approaches"), T("najpierw reakcje, potem\nwyświetlenia; modele dla grup;\nprzedziały konforemne;\netykiety jako korekta", "reactions, then views;\nmodels per group;\nconformal intervals;\nlabels as a correction"), T("bez zysku; przedziały\nszerokie, ale się trzymają", "no gain; intervals\nwide but they hold"), GREY)
card(X[3], Y2, W, H, T("Poza platformą", "Out of platform"), T("6 modeli uczonych na 1-2\nplatformach, test na pozostałej\nkalibracja kilkoma postami\nszacunek dla FB i IG", "6 models trained on 1-2\nplatforms, tested on the other\ncalibration with a few posts\nestimate for FB and IG"), T("najlepszy: ridge; kolejność się\nprzenosi, poziom do ×2,5", "best: ridge; order transfers,\nlevel off by up to ×2.5"), GREY)
scene = {"type": "excalidraw", "version": 2, "source": "https://excalidraw.com", "elements": E, "appState": {"viewBackgroundColor": "#ffffff", "exportBackground": True}, "files": {}}
out = Path(__file__).parent / f"approach_map_{LANG}.excalidraw"; out.write_text(json.dumps(scene, ensure_ascii=False, indent=1)); print("saved", out)
