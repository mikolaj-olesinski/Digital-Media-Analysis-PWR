"""Merges the three Haiku coders and prepares/finishes adjudication of disagreements.
  python src/merge_labels.py merge <lab_dir>   -> writes <lab_dir>/merged.tsv, <lab_dir>/disputed/dispute_NNN.tsv, prints agreement stats
  python src/merge_labels.py final <lab_dir>   -> combines unanimous Haiku labels with adjudicated ones into data/processed/mention_labels.csv
<lab_dir> contains chunks/chunk_NNN.tsv (id, platform, language, brand, text) and out/coder{A,B,C}_chunk_NNN.tsv (id, type, sentiment);
adjudicated rows are expected in out/judge_dispute_NNN.tsv (id, type, sentiment)."""
import itertools, sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
TYPES = {"organic", "promo", "spam", "unclear"}
SENT = {"positive", "neutral", "negative"}
lab = Path(sys.argv[2]); cmd = sys.argv[1]
read = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)


def merge():
    chunks = pd.concat([read(p) for p in sorted((lab / "chunks").glob("chunk_*.tsv"))], ignore_index=True)
    coders = {}
    for c in "ABC":
        d = pd.concat([read(p) for p in sorted((lab / "out").glob(f"coder{c}_chunk_*.tsv"))], ignore_index=True)
        assert d.type.isin(TYPES).all() and d.sentiment.isin(SENT).all(), f"invalid labels for coder {c}"
        d = d.drop_duplicates("id", keep="first")  # a coder may repeat a row; its missing row stays empty and goes to adjudication
        coders[c] = d.set_index("id").reindex(chunks.id).fillna("")
    m = chunks.set_index("id")[["platform", "language", "brand", "text"]]
    print("rows:", len(m), "rows without a label per coder:", {c: int((coders[c].type == "").sum()) for c in coders})
    for c in "ABC":
        m[f"type_{c}"] = coders[c].loc[m.index, "type"]; m[f"sent_{c}"] = coders[c].loc[m.index, "sentiment"]
    for k, col in (("type", "type"), ("sent", "sentiment")):
        for a, b in itertools.combinations("ABC", 2):
            ok = (m[f"{k}_{a}"] != "") & (m[f"{k}_{b}"] != "")
            print(col, a, b, "agree", round((m[f"{k}_{a}"] == m[f"{k}_{b}"])[ok].mean(), 3),
                  "kappa", round(cohen_kappa_score(m[f"{k}_{a}"][ok], m[f"{k}_{b}"][ok]), 3))
    m["type_unan"] = m[["type_A", "type_B", "type_C"]].nunique(axis=1) == 1
    m["sent_unan"] = m[["sent_A", "sent_B", "sent_C"]].nunique(axis=1) == 1
    print("unanimous type:", round(m.type_unan.mean(), 3), "sentiment:", round(m.sent_unan.mean(), 3))
    m.to_csv(lab / "merged.tsv", sep="\t")
    disp = m[~(m.type_unan & m.sent_unan)].reset_index()
    out = lab / "disputed"; out.mkdir(exist_ok=True)
    for i in range(0, len(disp), 200):
        disp.iloc[i:i + 200][["id", "platform", "language", "brand", "text"]].to_csv(out / f"dispute_{i // 200:03d}.tsv", sep="\t", index=False)
    print("disputed rows:", len(disp), "files:", (len(disp) + 199) // 200)


def final():
    m = read(lab / "merged.tsv").set_index("id")
    j = pd.concat([read(p) for p in sorted((lab / "out").glob("judge_dispute_*.tsv"))]).set_index("id") if list((lab / "out").glob("judge_dispute_*.tsv")) else pd.DataFrame(columns=["type", "sentiment"])
    m["type_unan"] = m.type_unan == "True"; m["sent_unan"] = m.sent_unan == "True"
    need = m.index[~(m.type_unan & m.sent_unan)]
    assert need.isin(j.index).all(), f"{(~need.isin(j.index)).sum()} disputed rows have no adjudication"
    m["type"] = m.type_A.where(m.type_unan, j.reindex(m.index).type)
    m["sentiment"] = m.sent_A.where(m.sent_unan, j.reindex(m.index).sentiment)
    m["type_source"] = m.type_unan.map({True: "haiku_unanimous", False: "sonnet_adjudicated"})
    m["sentiment_source"] = m.sent_unan.map({True: "haiku_unanimous", False: "sonnet_adjudicated"})
    res = m.reset_index().rename(columns={"id": "mention_id"})[["mention_id", "type", "sentiment", "type_source", "sentiment_source"]]
    res.to_csv(ROOT / "data" / "processed" / "mention_labels.csv", index=False)
    # per-coder labels, kept so agreement (kappa etc.) can be recomputed in notebooks/eda/03_labels.ipynb
    ju = j.reindex(m.index).rename(columns={"type": "type_judge", "sentiment": "sent_judge"})
    m.join(ju)[["type_A", "type_B", "type_C", "type_judge", "sent_A", "sent_B", "sent_C", "sent_judge"]].reset_index().rename(
        columns={"id": "mention_id"}).to_csv(ROOT / "data" / "processed" / "mention_labels_coders.csv", index=False)
    print(len(res), "labels written;", res.type.value_counts().to_dict(), res.sentiment.value_counts().to_dict())


{"merge": merge, "final": final}[cmd]()
