# D1 - Chunking breadth

| field | value |
|---|---|
| machine | arm64 / 10 vCPU |
| region | local |
| provider | Darwin |
| date | 2026-08-17 18:22 IST |
| commit | 2b3c461 |
| embedder | intfloat/multilingual-e5-small / 384d |
| python | 3.13.7 |
| seed | 42 |
| corpus_sha | e3e9652b8936ae16 |
| docs | 1200 |
| held-out queries | 1200 |
| build compute | 17.6 min |


Units: size = MB on disk after freeze, build = minutes (chunk+embed+index, corpus download excluded), p50 = ms of ANN search only (query embedding excluded, it is constant across strategies). Scoring is at passage granularity: every chunk is mapped back to its source passage and deduped preserving best rank before scoring against passage-level qrels.

| strategy                   | recall@50 | nDCG@10 | MRR@10 | size  | build | p50   |
|----------------------------|-----------|---------|--------|-------|-------|-------|
| s1 fixed 256/64            |     0.888 |   0.577 |  0.501 |  20.0 |  1.09 |  0.34 |
| s2 recursive 320/80        |     0.889 |   0.577 |  0.501 |  19.9 |  1.11 |  0.29 |
| s3 sentence-window         |     0.746 |   0.453 |  0.392 |  85.0 |  2.11 |  1.27 |
| s4 semantic drift          |     0.787 |   0.488 |  0.425 |  55.1 |  5.17 |  0.97 |
| s5 proposition (sampled)   |     0.832 |   0.492 |  0.430 |  12.1 |  0.18 |  0.09 |
| s6 parent-document         |     0.878 |   0.569 |  0.494 |  20.8 |  1.39 |  0.32 |
| s7 metadata-filtered       |     0.892 |   0.580 |  0.504 |  20.0 |  1.57 |  0.29 |
| s8 multi-granularity       |     0.812 |   0.517 |  0.451 | 125.2 |  4.92 |  2.25 |

**s7 metadata-filtered** wins recall@50 at 0.892 and leads nDCG@10 at 0.580. It costs 20.0 MB and 1.57 min to build, against 12.1 MB for the smallest index (s5 proposition). Search stays under 2.25 ms for every row, so the chunking choice does not spend the latency budget.

s3 sentence-window lost at 0.746 recall@50, 0.146 behind the winner and at 4.3x the index size - the honest line: splitting finer puts several chunks of one passage into the same ranking, and after the dedupe to passage granularity they collapse back to one hit. The extra chunks compete with each other for top-50 slots instead of adding coverage, and you pay for them twice: once on disk and once on every search.

_Caveats:_ rows marked (sampled) ran on a deterministic 10% slice - proposition decomposition is millions of LLM calls at full corpus. Index size is this repository's own serialisation (vectors + payload), not an HNSW graph.


690 of the 1200 frozen documents are translations of a document that is also in the index (MSMARCO-XI is parallel). MS MARCO's qrels are per-passage, so a query answered correctly from its translated twin scores as a miss: every recall and nDCG number in the table above is a **floor** for a multilingual retriever, and the gap is widest for exactly the strategies that retrieve across scripts.
