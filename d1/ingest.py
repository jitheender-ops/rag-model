"""Stage 0: pull the real corpus -- ai4bharat/MSMARCO-XI -- into data/raw/.

    make corpus            # ~50k passages: hi + ta + bn + the English originals
    make corpus ROWS=200   # smaller slice while iterating

Why streaming: the train split is ~48 GB (13 x 3.7 GB parquet) and validation is ~460 MB
per language. We read row groups over HTTP and stop after ROWS rows per language, so a
usable corpus costs minutes and a few hundred MB of traffic instead of a day and 48 GB.

What each MS MARCO row gives us, and why it matters here:
  query / Eng_Query            a real spoken-style question, not a template
  passages.Translated_passages 10 candidate passages in the target language
  passages.English_passages    the same 10 in English -- a genuine parallel corpus
  passages.is_selected         REAL QRELS. This is the part that makes D1's table honest:
                               relevance is human-labelled, not derived from the generator
                               that wrote the corpus.
  Answer / Eng_Answer          the gold answer, used by D4 for control rows and grading

Rows with no selected passage are dropped -- a query with no qrel cannot be scored.
"""
from __future__ import annotations

import argparse
import json
import os

LANGS = {                     # code -> (parquet stem, script)
    "hi": ("hinval", "Devanagari"),
    "ta": ("tamval", "Tamil"),
    "bn": ("benval", "Bengali"),
    "mr": ("marval", "Devanagari"),
    "te": ("telval", "Telugu"),
    "gu": ("gujval", "Gujarati"),
    "kn": ("kanval", "Kannada"),
    "ml": ("malval", "Malayalam"),
    "pa": ("panval", "Gurmukhi"),
    "ur": ("urdval", "Arabic"),
    "as": ("asmval", "Bengali"),
    "or": ("orival", "Odia"),
    "ne": ("nepval", "Devanagari"),
    "sa": ("sanval", "Devanagari"),
}
REPO = "hf://datasets/ai4bharat/MSMARCO-XI/validation"
RAW = "data/raw"
QUERIES = "data/msmarco_queries.jsonl"


COLUMNS = ["query_id", "query", "Eng_Query", "Answer", "Eng_Answer", "passages"]


def _rows_remote(lang: str):
    from datasets import load_dataset
    stem, _ = LANGS[lang]
    return load_dataset("parquet", data_files={"v": f"{REPO}/{stem}.parquet"},
                        split="v", streaming=True)


def _rows_local(path: str, batch: int = 256):
    """Batch-iterate a parquet already on disk.

    Measured, because the obvious claim about this is wrong: the hub's train shards are one
    single row group, and `iter_batches` does NOT make that cheap. Reading the first 300 rows
    of the 3.79 GB Assamese shard peaks at 3.86 GB resident -- the row group's column chunks
    are decoded whole whatever batch_size says. It is fast (first row in 1.6 s) and it is fine
    on a laptop with the RAM to spare; it is not the constant-memory stream it looks like.
    Budget for the shard's full size in memory, or re-shard the file before ingesting it."""
    import pyarrow.parquet as pq
    for b in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=COLUMNS):
        yield from b.to_pylist()


def stream(lang: str, rows: int, local: str | None = None):
    src = _rows_local(local) if local else _rows_remote(lang)
    n = 0
    for r in src:
        if n >= rows:
            break
        p = r["passages"]
        if not any(p["is_selected"]):        # no qrel -> unscoreable, drop it
            continue
        n += 1
        yield r


def ingest(langs: list[str], rows: int, english_from: str | None = "hi",
           local: dict[str, str] | None = None) -> dict:
    os.makedirs(RAW, exist_ok=True)
    qout = open(QUERIES, "w", encoding="utf-8")
    stats, seen_en = {}, False
    for lang in langs:
        _, script = LANGS[lang]
        n_doc = n_pas = n_q = 0
        en_docs, en_queries = [], []
        with open(f"{RAW}/msmarco_{lang}.jsonl", "w", encoding="utf-8") as fh:
            for r in stream(lang, rows, (local or {}).get(lang)):
                qid, p = r["query_id"], r["passages"]
                doc_id = f"{lang}:{qid}"
                passages = [{"pid": f"{doc_id}:{i}", "text": t}
                            for i, t in enumerate(p["Translated_passages"]) if t.strip()]
                if not passages:
                    continue
                qrels = [f"{doc_id}:{i}" for i, s in enumerate(p["is_selected"])
                         if s and i < len(passages)]
                if not qrels:
                    continue
                fh.write(json.dumps({"doc_id": doc_id, "lang": lang, "script": script,
                                     "passages": passages}, ensure_ascii=False) + "\n")
                qout.write(json.dumps({"qid": f"q:{doc_id}", "lang": lang,
                                       "query": r["query"], "qrels": qrels,
                                       "answer": r.get("Answer", "")},
                                      ensure_ascii=False) + "\n")
                n_doc += 1
                n_pas += len(passages)
                n_q += 1
                if lang == english_from and not seen_en:
                    en_doc_id = f"en:{qid}"
                    en_pass = [{"pid": f"{en_doc_id}:{i}", "text": t}
                               for i, t in enumerate(p["English_passages"]) if t.strip()]
                    en_qrels = [f"{en_doc_id}:{i}" for i, s in enumerate(p["is_selected"])
                                if s and i < len(en_pass)]
                    if en_pass and en_qrels:
                        en_docs.append({"doc_id": en_doc_id, "lang": "en", "script": "Latin",
                                        "passages": en_pass})
                        en_queries.append({"qid": f"q:{en_doc_id}", "lang": "en",
                                           "query": r["Eng_Query"].strip(" ."),
                                           "qrels": en_qrels,
                                           "answer": r.get("Eng_Answer", "")})
        stats[lang] = {"docs": n_doc, "passages": n_pas, "queries": n_q}
        if en_docs:
            seen_en = True
            with open(f"{RAW}/msmarco_en.jsonl", "w", encoding="utf-8") as fh:
                for d in en_docs:
                    fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            for q in en_queries:
                qout.write(json.dumps(q, ensure_ascii=False) + "\n")
            stats["en"] = {"docs": len(en_docs),
                           "passages": sum(len(d["passages"]) for d in en_docs),
                           "queries": len(en_queries)}
        print(f"  {lang}: {stats[lang]}", flush=True)
    qout.close()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="hi,ta,bn")
    ap.add_argument("--rows", type=int, default=1250, help="rows per language")
    ap.add_argument("--local", default=None,
                    help="lang=path/to.parquet for a shard already downloaded, "
                         "repeatable with commas (as=data/shards/asmtrain.parquet)")
    args = ap.parse_args()
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    local = dict(kv.split("=", 1) for kv in args.local.split(",")) if args.local else {}
    for lang in local:
        if lang not in LANGS:
            raise SystemExit(f"--local names {lang!r}, which is not in LANGS: "
                             f"{', '.join(sorted(LANGS))}")
    print(f"streaming {args.rows} rows x {langs} (+ English originals) from MSMARCO-XI")
    stats = ingest(langs, args.rows, local=local)
    tot = sum(s["passages"] for s in stats.values())
    print(f"total: {tot} passages, {sum(s['queries'] for s in stats.values())} queries "
          f"-> {RAW}/ + {QUERIES}")


def demo():
    """Offline check of the row -> doc mapping, no network."""
    row = {"query_id": 7, "query": "क्या है?", "Eng_Query": ". what is it?",
           "Answer": "उत्तर", "Eng_Answer": "answer",
           "passages": {"Translated_passages": ["अ", "ब"], "English_passages": ["a", "b"],
                        "is_selected": [0, 1]}}
    p = row["passages"]
    qrels = [f"hi:7:{i}" for i, s in enumerate(p["is_selected"]) if s]
    assert qrels == ["hi:7:1"], qrels
    assert any(p["is_selected"]), "rows with no qrel must be dropped"
    assert not any({"is_selected": [0, 0]}["is_selected"])
    print("ingest mapping ok", qrels)


if __name__ == "__main__":
    import sys

    demo() if "--selfcheck" in sys.argv else main()
