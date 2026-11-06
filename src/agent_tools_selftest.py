"""Self-test for the agent tools (no pytest needed): where_paths parsing and register row validation.

    .venv/bin/python src/agent_tools_selftest.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_tools as at  # noqa: E402


def main():
    cases = {
        "`99_missing`": ["notebooks/models/99_missing.ipynb"],
        "`notebooks/models/99_missing.ipynb`": ["notebooks/models/99_missing.ipynb"],
        "`eda/03_labels`": ["notebooks/eda/03_labels.ipynb"],
        "`fb_cqr_numbers.json`": ["data/processed/fb_cqr_numbers.json"],
        "`src/final_v3_x.py`": ["src/final_v3_x.py"],
    }
    bad = [(k, at.where_paths(k), v) for k, v in cases.items() if at.where_paths(k) != v]
    tmp = Path(tempfile.mkstemp(suffix=".md")[1])
    tmp.write_text(
        "| ID | Date | Question | Setup | Result | Type | Where |\n|---|---|---|---|---|---|---|\n"
        "| E999 | 2026-01-01 | q | s | r | finding | x |\n"
        "| E1000 | 2026-01-01 | q | s | r | finding | x | extra |\n"
    )
    at.MALFORMED.clear()
    rows = at.register(tmp)
    tmp.unlink()
    malformed_ok = len(rows) == 1 and at.MALFORMED == [("E1000", 8)]
    print(f"where_paths: {len(cases) - len(bad)}/{len(cases)} OK")
    print(f"register malformed detection: {'OK' if malformed_ok else 'FAIL ' + str(at.MALFORMED)}")
    if bad or not malformed_ok:
        for b in bad:
            print("  mismatch:", b)
        sys.exit(1)


if __name__ == "__main__":
    main()
