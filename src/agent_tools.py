"""Tools for agents working in this repo. Run from the repo root.

python src/agent_tools.py vault   # create a card in agent/vault/ for every experiment in the register that has none; rebuild the index
python src/agent_tools.py check   # guardrails before a commit; exit code 1 if anything fails
python src/agent_tools.py next    # print the open backlog items, highest priority first

The register `docs/en/experiments.md` (with its Polish twin) is the source of truth for IDs. Cards hold the detail
the one-line register cannot: hypothesis, setup, numbers, decision, what to try next. Existing cards are never overwritten."""
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG_EN, REG_PL = ROOT / "docs/en/experiments.md", ROOT / "docs/pl/experiments.md"
VAULT, BACKLOG, KNOWN = ROOT / "agent/vault", ROOT / "agent/backlog.md", ROOT / "agent/check_known.txt"
ROW = re.compile(r"^\|\s*((?:E|L)\d+)\s*\|(.*)\|\s*$")
PII_COLS = ("author_name", "author_url", "attributes_url")
TYPES_EN = {"gain", "small", "none", "worse", "finding", "instrument", "pending"}
TYPES_PL = {"zysk", "mały", "brak", "gorzej", "wniosek", "instrument", "w toku"}
MALFORMED = []
URL = re.compile(r"https?://(?!(?:arxiv\.org|github\.com|www\.brandwatch|www\.socialcrawl|www\.microsoft|docs\.|ubuntu\.com))\S+")
SOCIAL = re.compile(r"(facebook\.com/(?:watch|reel|videos)[^\s)\"']*\d|twitter\.com/[A-Za-z0-9_]+/status/\d|instagram\.com/(?:p|reel)/[A-Za-z0-9_-]+|tiktok\.com/@[A-Za-z0-9_.]+)")


def register(path=REG_EN):
    """Rows of the register as dicts: id, date, question, setup, result, type, where."""
    rows = []
    for line in path.read_text().splitlines():
        m = ROW.match(line)
        if m:
            cells = [c.strip() for c in m.group(2).split("|")]
            if len(cells) != 6:
                MALFORMED.append((m.group(1), len(cells) + 1))
                continue
            rows.append(dict(zip(["id", "date", "question", "setup", "result", "type", "where"], [m.group(1)] + cells)))
    return rows


def where_paths(where):
    """Notebook names and data files mentioned in the `where` cell."""
    out = []
    for tok in re.findall(r"`([^`]+)`", where):
        tok = tok.strip()
        if tok.endswith(".ipynb"):
            out.append(tok)
        elif tok.endswith((".json", ".csv")):
            out.append(tok if "/" in tok else f"data/processed/{tok}")
        elif "/" in tok and tok.endswith(".py"):
            out.append(tok)
        elif re.match(r"^(eda/)?\d\d_\w+$", tok):
            out.append(f"notebooks/{tok}.ipynb" if tok.startswith("eda/") else f"notebooks/models/{tok}.ipynb")
    return out


CARD = """---
id: {id}
date: {date}
type: {type}
status: done
notebooks: [{nbs}]
numbers: [{nums}]
---
# {id} · {question}

**Register line.** {result}

## Question and hypothesis
{question} (Hypothesis not recorded when this card was generated from the register; add it if you rerun.)

## Setup
{setup}

## Result
{result}

## Decision
_Not recorded. Fill in what this result changed (kept / dropped / follow-up)._

## Follow-ups
_None recorded._

## Links
{links}
"""


def vault():
    VAULT.mkdir(parents=True, exist_ok=True); made = 0; rows = register()
    for r in rows:
        card = VAULT / f"{r['id']}.md"
        if card.exists():
            continue
        paths = where_paths(r["where"])
        missing = [p for p in paths if not (ROOT / p).exists()]
        if missing:
            print(f"vault: {r['id']} skipped, missing {missing}")
            continue
        nbs = [p for p in paths if p.endswith(".ipynb")]; nums = [p for p in paths if not p.endswith(".ipynb")]
        card.write_text(CARD.format(**r, nbs=", ".join(nbs), nums=", ".join(nums), links="\n".join(f"- `{p}`" for p in paths) or "- none"))
        made += 1
    idx = ["# Experiment vault", "", "One card per experiment. Generated index; do not edit by hand (run `python src/agent_tools.py vault`).", "",
           "| ID | Type | Question | Card |", "|---|---|---|---|"]
    for r in rows:
        idx.append(f"| {r['id']} | {r['type']} | {r['question']} | [{r['id']}]({r['id']}.md) |")
    (VAULT / "INDEX.md").write_text("\n".join(idx) + "\n")
    json.dump([{**r, "paths": where_paths(r["where"])} for r in rows], open(VAULT / "index.json", "w"), indent=1, ensure_ascii=False)
    print(f"vault: {made} new cards, {len(rows)} experiments indexed")


def check():
    fails, warns = [], []
    en, pl = register(REG_EN), register(REG_PL)
    if [r["id"] for r in en] != [r["id"] for r in pl]:
        fails.append(f"register IDs differ between en and pl: only en {sorted(set(r['id'] for r in en) - set(r['id'] for r in pl))}, only pl {sorted(set(r['id'] for r in pl) - set(r['id'] for r in en))}")
    if MALFORMED:
        fails.append(f"register rows with the wrong number of columns (7 expected): {MALFORMED}")
    for r in en:
        if r["type"] not in TYPES_EN:
            fails.append(f"{r['id']}: unknown result type `{r['type']}` (EN register)")
    for r in pl:
        if r["type"] not in TYPES_PL:
            fails.append(f"{r['id']}: unknown result type `{r['type']}` (PL register)")
    ids = [r["id"] for r in en]
    if len(ids) != len(set(ids)):
        fails.append("duplicate experiment IDs in the register")
    for r in en:
        for p in where_paths(r["where"]):
            if not (ROOT / p).exists():
                fails.append(f"{r['id']}: `{p}` named in the register does not exist")
        if not (VAULT / f"{r['id']}.md").exists():
            fails.append(f"{r['id']}: no vault card (run `python src/agent_tools.py vault`)")
    for card in sorted(VAULT.glob("E*.md")):
        text = card.read_text()
        if "## Result" not in text:
            warns.append(f"{card.name}: no ## Result section")
        if "## Decision" not in text:
            warns.append(f"{card.name}: no ## Decision section")
        if re.search(r"^numbers:\s*\[\s*\]", text, re.M) and "Legacy note." not in text:
            warns.append(f"{card.name}: numbers: [] and no Legacy note")
        m = re.search(r"^numbers:\s*\[(.*?)\]", text, re.M)
        if m:
            for tok in re.findall(r"([^,\[\]`]+?\.(?:json|csv))", m.group(1)):
                if not (ROOT / tok.strip()).exists():
                    fails.append(f"{card.name}: numbers path `{tok.strip()}` does not exist")
        m = re.search(r"^notebooks:\s*\[(.*?)\]", text, re.M)
        if m:
            for tok in re.findall(r"([^,\[\]`]+?\.ipynb)", m.group(1)):
                if not (ROOT / tok.strip()).exists():
                    fails.append(f"{card.name}: notebook `{tok.strip()}` does not exist")
    for f in sorted((ROOT / "figures").glob("*_en.png")):
        if not f.with_name(f.name.replace("_en.png", "_pl.png")).exists():
            fails.append(f"figure {f.name} has no Polish twin")
    for nb in sorted((ROOT / "notebooks").rglob("*.ipynb")):
        if ".ipynb_checkpoints" in nb.parts:
            continue
        cells = json.loads(nb.read_text())["cells"]; code = [c for c in cells if c["cell_type"] == "code"]
        src_all = "\n".join("".join(c.get("source", [])) for c in cells)
        if SOCIAL.search(src_all):
            warns.append(f"{nb.relative_to(ROOT)}: source contains a social URL")
        if code and not any(c.get("outputs") for c in code):
            warns.append(f"{nb.relative_to(ROOT)}: stored without outputs")
        for c in code:
            for o in c.get("outputs", []):
                if o.get("output_type") == "error":
                    fails.append(f"{nb.relative_to(ROOT)}: stored with an error output ({o.get('ename')})")
                text = json.dumps(o.get("data", o.get("text", "")))
                if URL.search(text):
                    fails.append(f"{nb.relative_to(ROOT)}: an output contains a URL (personal data rule)")
                for col in PII_COLS:
                    if col in text:
                        fails.append(f"{nb.relative_to(ROOT)}: an output shows column `{col}` (personal data rule)")
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    for t in tracked:
        if t.endswith((".md", ".py")) and (ROOT / t).exists():
            m = SOCIAL.search((ROOT / t).read_text(errors="ignore"))
            if m:
                warns.append(f"tracked file `{t}` contains a social URL: {m.group(0)[:60]}")
        if t.endswith(".csv") and (ROOT / t).exists():
            header = [c.strip().strip('"').lower() for c in (ROOT / t).read_text(errors="ignore").splitlines()[0].split(",")]
            bad = [c for c in header if c in PII_COLS or c == "url"]
            if bad:
                fails.append(f"tracked CSV `{t}` has personal-data column(s) {bad}")
    for t in tracked:
        if t.startswith("data/raw/") or t.startswith("logs/") or t == "AGENTS.md" or ("fetched" in t and t.endswith(".csv")):
            fails.append(f"`{t}` is tracked by git but must stay local")
    q = ROOT / "agent/questions_for_brand24.md"
    if q.exists():
        for line in q.read_text().splitlines():
            if line.startswith("|") and "---" not in line:
                if len(line.strip().strip("|").split("|")) != 5:
                    fails.append(f"questions_for_brand24.md: row with the wrong number of columns: {line[:60]}")
    if not BACKLOG.exists():
        fails.append("agent/backlog.md is missing")
    known = [l.split("#")[0].strip() for l in KNOWN.read_text().splitlines()] if KNOWN.exists() else []
    known = [k for k in known if k]; new, old = [], []
    for f in dict.fromkeys(fails):
        (old if any(k in f for k in known) else new).append(f)
    for w in warns:
        print("WARN ", w)
    for f in old:
        print("KNOWN", f)
    for f in new:
        print("FAIL ", f)
    print(f"check: {len(new)} new failures, {len(old)} known (in agent/check_known.txt), {len(warns)} warnings")
    return 1 if new else 0


def next_items():
    items = re.findall(r"^\|\s*(B\d+)\s*\|\s*(P\d)\s*\|\s*(open|running|blocked)\s*\|([^|]*)\|", BACKLOG.read_text(), re.M)
    for i, p, s, t in sorted(items, key=lambda x: (x[1], x[0])):
        print(f"{i} {p} {s:8s} {t.strip()}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    sys.exit({"vault": lambda: vault() or 0, "check": check, "next": lambda: next_items() or 0}[cmd]())
