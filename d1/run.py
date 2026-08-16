"""D1 -- chunking breadth: 8 strategies x 6 columns, one marked winner.

Four stages, all offline:
  1 CORPUS  freeze the subset, hash it
  2 CHUNK   8 chunkers, same interface, own artifact dir, no shared state
  3 INDEX   same embedder, same params, for all eight
  4 EVAL    one loop, 8 indexes, held-out queries, 3 repeats, median

The one thing that makes this table valid: score at PASSAGE granularity. Every retrieved
chunk is mapped back to its source passage id and deduped preserving best rank BEFORE
scoring against the passage-level qrels. Without that step a sentence chunker and a
512-token chunker are not comparable and the table is noise.

    make chunking     ->  reports/chunking.md + reports/chunking.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time

from d1 import corpus
from d1.chunkers import ALL, Doc
from d1.index import BACKEND, DIM, EF_CONSTRUCTION, M, MODEL_NAME, Index, embed_many
from harness.env import header, pin_randomness
from harness.spans import NS_PER_MS, now_ns

ART = "artifacts/d1"
EMBED_CHUNK = 1024        # chunks per batched embed call
QUERIES = "data/queries_chunking.jsonl"


# ---------- metrics, all computed on deduped PASSAGE ids ----------

def dedupe_to_passages(hits: list[tuple[str, float]], chunk2passage: dict[str, str]) -> list[str]:
    """Map chunk -> source passage, dedupe preserving best rank."""
    seen, out = set(), []
    for cid, _ in hits:
        pid = chunk2passage[cid]
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def recall_at(passages: list[str], rel: set[str], k: int) -> float:
    return len(rel & set(passages[:k])) / len(rel) if rel else 0.0


def ndcg_at(passages: list[str], rel: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, p in enumerate(passages[:k]) if p in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal else 0.0


def mrr_at(passages: list[str], rel: set[str], k: int) -> float:
    for i, p in enumerate(passages[:k]):
        if p in rel:
            return 1.0 / (i + 1)
    return 0.0


# ---------- stages ----------

def build(strategy, docs: list[Doc]) -> dict:
    """Stages 2+3 for one strategy: chunk, index, freeze, write manifest."""
    t0 = now_ns()
    out_dir = os.path.join(ART, strategy.key.split()[0])
    os.makedirs(out_dir, exist_ok=True)
    ix, chunk2passage, n, batch = Index(), {}, 0, []
    with open(os.path.join(out_dir, "chunks.jsonl"), "w", encoding="utf-8") as fh:
        for doc in docs:
            for c in strategy.chunk(doc):
                batch.append((c.id, c.text, {"pid": c.doc_id, "parent": c.parent_id,
                                             "span": list(c.span), "lang": c.lang}))
                chunk2passage[c.id] = c.doc_id
                fh.write(json.dumps({"id": c.id, "doc_id": c.doc_id, "parent_id": c.parent_id,
                                     "span": list(c.span), "n_tokens": c.n_tokens,
                                     "text": c.text}, ensure_ascii=False) + "\n")
                n += 1
                if len(batch) >= EMBED_CHUNK:     # embed in batches, not one call per chunk
                    ix.add_many(batch)
                    batch = []
    if batch:
        ix.add_many(batch)
    ix.freeze()
    size = ix.save(os.path.join(out_dir, "index.jsonl"))
    build_s = (now_ns() - t0) / 1e9
    manifest = {"key": strategy.key, "params": strategy.params, "n_chunks": n,
                "build_s": round(build_s, 2), "bytes": size,
                "corpus_sha": corpus.sha(), "embedder": BACKEND,
                "model": MODEL_NAME if BACKEND == "st" else "hashed-bow",
                "embed_dim": DIM(),
                "m": M, "ef_construction": EF_CONSTRUCTION,
                "sampled": getattr(strategy, "sampled", False)}
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return {"index": ix, "chunk2passage": chunk2passage, "manifest": manifest}


def evaluate(built: dict, queries: list[dict], repeats: int, warm: int) -> dict:
    """Stage 4: one loop, same for all 8 indexes. Dense only -- hybrid is off here so the
    number is a chunking signal, not a BM25 signal."""
    ix, c2p = built["index"], built["chunk2passage"]
    if built["manifest"]["sampled"]:
        # a sampled strategy is scored only on the queries whose gold passage is inside
        # its slice; scoring it on the full query set would measure the sampling.
        indexed = set(c2p.values())
        queries = [q for q in queries if set(q["qrels"]) <= indexed]
    # embedding is excluded from the search timer: it is constant across strategies
    qvecs = list(zip(queries, embed_many([q["query"] for q in queries], "query")))
    r50, n10, m10 = [], [], []
    for q, v in qvecs:
        passages = dedupe_to_passages(ix.search(v, k=50), c2p)
        rel = set(q["qrels"])
        r50.append(recall_at(passages, rel, 50))
        n10.append(ndcg_at(passages, rel, 10))
        m10.append(mrr_at(passages, rel, 10))
    # search P50: ANN search only, warm queries, embedding excluded, median of `repeats`
    timing_set = qvecs[:warm] or qvecs
    medians = []
    for _ in range(repeats):
        samples = []
        for _, v in timing_set:
            t = now_ns()
            ix.search(v, k=50)
            samples.append((now_ns() - t) / NS_PER_MS)
        medians.append(statistics.median(samples))
    return {"recall@50": statistics.mean(r50), "nDCG@10": statistics.mean(n10),
            "MRR@10": statistics.mean(m10), "search_p50_ms": statistics.median(medians),
            "n_queries": len(queries)}


def parallel_twins(raw: list[dict]) -> int:
    """Documents that are a translation of another document in the same frozen corpus.

    MSMARCO-XI is parallel, and MS MARCO's qrels are per-passage: a query answered correctly
    from its translated twin still scores as a miss. Counted, not hand-waved."""
    seen: dict[str, int] = {}
    for d in raw:
        rid = d["doc_id"].split(":", 1)[-1]
        seen[rid] = seen.get(rid, 0) + 1
    return sum(v for v in seen.values() if v > 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=400)   # ~2.2k passages, so 2000 held-out queries exist
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warm", type=int, default=500)
    ap.add_argument("--report-only", action="store_true",
                    help="re-render reports/chunking.md from reports/chunking.json, no embedding")
    args = ap.parse_args()

    if args.report_only:
        # the prose is re-rendered from the saved rows; everything else is re-derived from
        # the frozen files, so this works on a chunking.json written by any older run.
        with open("reports/chunking.json") as fh:
            saved = json.load(fh)
        winner = next(r for r in saved["rows"] if r["key"] == saved["winner"])
        raw = corpus.load()
        with open(QUERIES, encoding="utf-8") as fh:
            n_q = sum(1 for _ in fh)
        write_report(saved["rows"], winner, saved["corpus_sha"], len(raw), n_q,
                     saved.get("wall_s", 0.0), parallel_twins(raw))
        print("re-rendered reports/chunking.md")
        return

    pin_randomness()
    t_start = time.time()
    sha, n_docs = corpus.freeze(args.docs)
    raw = corpus.load()
    docs = [Doc(d["doc_id"], d["lang"], d["script"], d["passages"], d.get("text", "")) for d in raw]
    n_q = corpus.make_queries(raw, args.queries, QUERIES)
    with open(QUERIES, encoding="utf-8") as fh:
        queries = [json.loads(l) for l in fh]
    print(f"corpus {sha} docs={n_docs} passages={sum(len(d.passages) for d in docs)} queries={n_q}")

    rows = []
    for cls in ALL:
        s = cls()
        built = build(s, docs)
        m = evaluate(built, queries, args.repeats, args.warm)
        row = {**built["manifest"], **m,
               "size_mb": built["manifest"]["bytes"] / 1e6,
               "build_min": built["manifest"]["build_s"] / 60}
        row.pop("params", None)
        rows.append(row)
        print(f"  {s.key:24s} r@50={m['recall@50']:.3f} nDCG={m['nDCG@10']:.3f} "
              f"MRR={m['MRR@10']:.3f} {row['size_mb']:.1f}MB {row['build_min']:.2f}m "
              f"p50={m['search_p50_ms']:.2f}ms")

    # recall@50 first, nDCG@10 breaks ties, then the smaller index wins.
    # Sampled rows are ineligible: they were scored on their own slice's queries, so their
    # numbers are reportable but not comparable.
    winner = max([r for r in rows if not r["sampled"]],
                 key=lambda r: (r["recall@50"], r["nDCG@10"], -r["size_mb"]))
    twins = parallel_twins(raw)
    wall_s = time.time() - t_start
    write_report(rows, winner, sha, n_docs, n_q, wall_s, twins)
    with open("reports/chunking.json", "w") as fh:
        json.dump({"rows": rows, "winner": winner["key"], "corpus_sha": sha,
                   "n_docs": n_docs, "n_queries": n_q, "wall_s": wall_s,
                   "parallel_twins": twins}, fh, indent=2)
    print(f"winner: {winner['key']}  -> reports/chunking.md")


def table(rows: list[dict]) -> str:
    head = ("| strategy                   | recall@50 | nDCG@10 | MRR@10 | size  | build | p50   |\n"
            "|----------------------------|-----------|---------|--------|-------|-------|-------|")
    out = [head]
    for r in rows:
        key = r["key"] + (" (sampled)" if r["sampled"] else "")
        out.append(f"| {key:26.26s} | {r['recall@50']:9.3f} | {r['nDCG@10']:7.3f} | "
                   f"{r['MRR@10']:6.3f} | {r['size_mb']:5.1f} | {r['build_min']:5.2f} | "
                   f"{r['search_p50_ms']:5.2f} |")
    return "\n".join(out)


def write_report(rows, winner, sha, n_docs, n_q, wall_s, twins=0):
    best_ndcg = max(rows, key=lambda r: r["nDCG@10"])
    cheapest = min(rows, key=lambda r: r["size_mb"])
    loser = min(rows, key=lambda r: r["recall@50"])
    os.makedirs("reports", exist_ok=True)
    md = [
        # the sum of the measured per-strategy builds, not the process's wall clock: a
        # backgrounded run that the OS suspends reports hours of "wall clock" that nobody
        # spent computing, and this column exists to say what the table costs to reproduce.
        header("D1 - Chunking breadth", {"corpus_sha": sha, "docs": n_docs,
                                         "held-out queries": n_q,
                                         "build compute": f"{sum(r['build_min'] for r in rows):.1f} min"}),
        "\nUnits: size = MB on disk after freeze, build = minutes (chunk+embed+index, "
        "corpus download excluded), p50 = ms of ANN search only (query embedding excluded, "
        "it is constant across strategies). Scoring is at passage granularity: every chunk "
        "is mapped back to its source passage and deduped preserving best rank before "
        "scoring against passage-level qrels.\n",
        table(rows),
        f"\n**{winner['key']}** wins recall@50 at {winner['recall@50']:.3f}"
        + (f" and leads nDCG@10 at {winner['nDCG@10']:.3f}. "
           if best_ndcg["key"] == winner["key"] else
           f"; {best_ndcg['key']} leads nDCG@10 at {best_ndcg['nDCG@10']:.3f}. ") +
        f"It costs {winner['size_mb']:.1f} MB and {winner['build_min']:.2f} min to build, "
        f"against {cheapest['size_mb']:.1f} MB for the smallest index ({cheapest['key']}). "
        f"Search stays under {max(r['search_p50_ms'] for r in rows):.2f} ms for every row, "
        f"so the chunking choice does not spend the latency budget.",
        f"\n{loser['key']} lost at {loser['recall@50']:.3f} recall@50, "
        f"{winner['recall@50'] - loser['recall@50']:.3f} behind the winner and at "
        f"{loser['size_mb'] / winner['size_mb']:.1f}x the index size - the honest line: "
        "splitting finer puts several chunks of one passage into the same ranking, and after "
        "the dedupe to passage granularity they collapse back to one hit. The extra chunks "
        "compete with each other for top-50 slots instead of adding coverage, and you pay "
        "for them twice: once on disk and once on every search.",
        "\n_Caveats:_ rows marked (sampled) ran on a deterministic 10% slice - proposition "
        "decomposition is millions of LLM calls at full corpus. Index size is this "
        "repository's own serialisation (vectors + payload), not an HNSW graph.\n",
        (f"\n{twins} of the {n_docs} frozen documents are translations of a document that is "
         "also in the index (MSMARCO-XI is parallel). MS MARCO's qrels are per-passage, so a "
         "query answered correctly from its translated twin scores as a miss: every recall "
         "and nDCG number in the table above is a **floor** for a multilingual retriever, "
         "and the gap is widest for exactly the strategies that retrieve across scripts.\n"
         if twins else ""),
    ]
    with open("reports/chunking.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))


def demo():
    c2p = {"a:1": "p1", "a:2": "p1", "a:3": "p2"}
    hits = [("a:1", 0.9), ("a:2", 0.8), ("a:3", 0.7)]
    assert dedupe_to_passages(hits, c2p) == ["p1", "p2"], "dedupe must preserve best rank"
    assert recall_at(["p1", "p2"], {"p1", "p3"}, 50) == 0.5
    assert mrr_at(["p9", "p1"], {"p1"}, 10) == 0.5
    assert abs(ndcg_at(["p1"], {"p1"}, 10) - 1.0) < 1e-9
    assert abs(ndcg_at(["p9", "p1"], {"p1"}, 10) - (1 / math.log2(3))) < 1e-9
    assert mrr_at(["p9"] * 20 + ["p1"], {"p1"}, 10) == 0.0
    twins = parallel_twins([{"doc_id": "hi:7"}, {"doc_id": "en:7"}, {"doc_id": "ta:9"}])
    assert twins == 2, twins        # both sides of a pair count; the lone doc does not
    print("d1 metrics ok")


if __name__ == "__main__":
    import sys

    if "--selfcheck" in sys.argv:
        demo()
    else:
        main()
