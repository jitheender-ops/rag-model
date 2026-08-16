# D3 - Latency analytics

| field | value |
|---|---|
| machine | arm64 / 10 vCPU |
| region | local |
| provider | Darwin |
| date | 2026-08-16 10:12 IST |
| commit | uncommitted |
| embedder | intfloat/multilingual-e5-small / 384d |
| python | 3.13.7 |
| seed | 42 |
| n | 500 |
| budget | 200 ms |


> **200 ms budget: PASS.** 1500/1500 requests inside the window across every mode (warm 0/500, cold 0/500, conc 0/500 over budget). Slowest single request 58.3 ms. Excluded legs are listed below and are not part of this verdict.


## The window

```
  [ mic + VAD ]   [ STT round trip ]   |=== t0 ---> t1 MEASURED ===|   [ TTS + net ]
     excluded          excluded          guards -> retrieve -> rerank      excluded
                                         -> generate -> verify
  t0 = server receives the STT is_final event, stamped server-side
  t1 = last answer token flushed to the socket, after the grounding verdict
```

excluded legs: not measured on this run -- `make stt` with SARVAM_API_KEY set and clips in data/audio/ writes `data/excluded_legs.json`, and this line becomes the measurement. They are excluded from the budget, not hidden from the report.


## Table shape - warm, n = 500

| stage             |  P50 |  P70 |  P95 | P100 |
|-------------------|------|------|------|------|
| input guards      | 0.01 | 0.01 | 0.01 | 0.02 |
| embed query       | 7.10 | 7.52 | 8.10 | 14.89 |
| dense + bm25 + rrf | 1.07 | 1.55 | 2.29 | 3.28 |
| rerank            | 0.00 | 0.00 | 0.00 | 0.00 |
| generate          | 0.10 | 0.11 | 0.27 | 1.12 |
| verify            | 0.03 | 0.04 | 0.05 | 0.51 |
| END-TO-END (warm) | 8.38 | 8.93 | 10.13 | 15.65 |
| END-TO-END (cold) | 8.61 | 9.20 | 10.72 | 17.11 |

degradation rate: 0.0%   cache hit rate: 10.2%   n=500, seed 42   over-budget: 0/500 (0.0%)


> 10.2% of warm queries hit the semantic cache. The cache-off cold row is published beside the warm one; read it as the cost of a first-time question.


## P100, said out loud

P100 over 500 samples is one observation: it is the max and it is unstable by construction. P95 = 10.1 ms, P99 = 12.6 ms, P100 = 15.7 ms. P100 = 15.7 ms on `l143` (in_domain, hi); the dominant stage was **embed_query** at 12.1 ms, fallbacks fired: none.


## Concurrency

A separate `--concurrency 4` pass over the same 500 queries: P50 36.0 ms, P95 47.3 ms, over-budget 0/500.


![per-stage latency](latency.svg)


_All stages are milliseconds. The `embed query` row is a real transformer forward pass (multilingual-e5-small on CPU). Generation is still extractive and the reranker is lexical, so those two rows are floors, not an LLM's cost: budget for ~15 ms of cross-encoder and the generator's own time on top._


_Cold = fresh process, semantic cache off, encoder already resident: a server loads its model before it accepts traffic, so that ~10 s belongs to startup and not to the first caller's 200 ms. Warm = 50 discarded warmups first, cache on; the cache-off run is the `cold` row and the cache hit rate is printed above, so a repeat-heavy query file cannot flatter the P50 unnoticed._
