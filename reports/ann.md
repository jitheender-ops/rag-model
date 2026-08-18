# ANN vs exact - where the vector index earns its place

| field | value |
|---|---|
| machine | arm64 / 10 vCPU |
| region | local |
| provider | Darwin |
| date | 2026-08-18 16:55 IST |
| commit | 4433cea |
| embedder | hashed-bow / 4096d |
| python | 3.13.7 |
| seed | 42 |
| strategy | artifacts/d1/s7 |
| queries | 50 |
| k | 50 |
| M | 32 |
| efConstruction | 200 |
| efSearch | 128 |
| real corpus | 12024 chunks |

Search latency is the dense ANN lookup only: no query embedding, no BM25, no fusion. Vectors above the real corpus size are resampled real vectors plus noise -- honest for latency, meaningless for retrieval quality, so recall is reported only at the real corpus size.

| chunks | vectors | exact P50 | exact P100 | hnsw P50 | hnsw P100 | build | recall@50 vs exact |
|---|---|---|---|---|---|---|---|
| 12,024 | 18 MB | 0.28 ms | 1.76 ms | 0.13 ms | 0.84 ms | 0.3 s | 0.999 |
| 50,000 | 77 MB | 1.45 ms | 1.67 ms | 0.21 ms | 0.30 ms | 9.3 s | — |
| 200,000 | 307 MB | 5.27 ms | 6.14 ms | 0.39 ms | 0.47 ms | 98.4 s | — |

**The crossover is at or below the smallest size measured (12,024 chunks)** — the graph is already the faster index on the corpus this repo actually serves. Exact's cost grows with the corpus and the graph's barely moves: 12,024 -> 2x, 50,000 -> 7x, 200,000 -> 13x (exact / hnsw, P50).

At the real corpus (12,024 chunks) the graph returns **99.9%** of exact's top-50, at **0.13 ms** against exact's **0.28 ms**.

**And yet `INDEX=exact` stays the default here, on the same evidence.** Dense search is not the pipeline's bottleneck at this size: the whole `dense + bm25 + rrf` stage is 1.3 ms P50 of a 48 ms request, so the 0.14 ms the graph saves is under one percent of a request and inside the run-to-run noise of the D3 table. What it costs is 0.1% of the true neighbours, a build step, and a dependency. Paying that for noise is not a trade, it is a habit.

The number that matters is the last column of the top row and the shape of the one before it: the graph is *ready*, its recall cost is measured rather than assumed, and the corpus size where it stops being optional is now a row in a table instead of a sentence in a README. `INDEX=hnsw` is one environment variable, and every gate, floor and report above it stays as it is — the scores are cosines on the same scale, which is why `build_hnsw` uses inner product rather than L2.
