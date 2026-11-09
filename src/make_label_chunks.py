"""Splits mentions into chunk files for LLM labelling. Only id, platform, language, brand, content
(title only if content is empty; truncated) are exported: no author fields, no URLs. Run: python src/make_label_chunks.py <out_dir> [chunk_size]"""
import re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]); size = int(sys.argv[2]) if len(sys.argv) > 2 else 200
out.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(ROOT / "data" / "raw" / "reach_samples_students.csv", usecols=["mention_id", "platform", "language", "brand", "title", "content"])
d = d.sample(frac=1, random_state=0).reset_index(drop=True)  # shuffled so every chunk mixes platforms
clean = lambda s: re.sub(r"\s+", " ", str(s) if pd.notna(s) else "").strip()
for k in range(0, len(d), size):
    rows = []
    for r in d.iloc[k:k + size].itertuples():
        text = clean(r.content) or clean(r.title)  # titles mostly repeat the content
        rows.append("\t".join([str(r.mention_id), r.platform, str(r.language), clean(r.brand), text[:400]]))
    (out / f"chunk_{k // size:03d}.tsv").write_text("id\tplatform\tlanguage\tbrand\ttext\n" + "\n".join(rows) + "\n", encoding="utf-8")
print(len(list(out.glob("chunk_*.tsv"))), "chunks")
