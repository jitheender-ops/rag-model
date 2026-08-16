# D3 - Latency analytics

| field | value |
|---|---|
| machine | arm64 / 10 vCPU |
| region | local |
| provider | Darwin |
| date | 2026-08-16 11:27 IST |
| commit | 4035141 |
| embedder | intfloat/multilingual-e5-small / 384d |
| python | 3.13.7 |
| seed | 42 |
| n | 500 |
| budget | 200 ms |


> **200 ms budget: PASS.** 1500/1500 requests inside the window across every mode (warm 0/500, cold 0/500, conc 0/500 over budget). Slowest single request 63.8 ms. Excluded legs are listed below and are not part of this verdict.


## The window

```
  [ mic + VAD ]   [ STT round trip ]   |=== t0 ---> t1 MEASURED ===|   [ TTS + net ]
     excluded          excluded          guards -> retrieve -> rerank      excluded
                                         -> generate -> verify
  t0 = server receives the STT is_final event, stamped server-side
  t1 = last answer token flushed to the socket, after the grounding verdict
```

excluded legs (P50 / P95 / P100, ms): STT _ / _ / _    TTS 679.7    client RTT _    (`_` = no such stage in this repo, or not measured)


## Table shape - warm, n = 500

| stage             |  P50 |  P70 |  P95 | P100 |
|-------------------|------|------|------|------|
| input guards      | 0.01 | 0.01 | 0.01 | 0.02 |
| embed query       | 6.91 | 7.39 | 7.99 | 15.68 |
| dense + bm25 + rrf | 1.05 | 1.50 | 2.33 | 3.15 |
| rerank            | 0.00 | 0.00 | 0.00 | 0.00 |
| generate          | 0.10 | 0.11 | 0.26 | 1.14 |
| verify            | 0.03 | 0.03 | 0.05 | 0.50 |
| END-TO-END (warm) | 8.22 | 8.77 | 10.04 | 16.33 |
| END-TO-END (cold) | 8.49 | 9.00 | 11.87 | 22.26 |

degradation rate: 0.0%   cache hit rate: 10.2%   n=500, seed 42   over-budget: 0/500 (0.0%)


> 10.2% of warm queries hit the semantic cache. The cache-off cold row is published beside the warm one; read it as the cost of a first-time question.


## P100, said out loud

P100 over 500 samples is one observation: it is the max and it is unstable by construction. P95 = 10.0 ms, P99 = 13.4 ms, P100 = 16.3 ms. P100 = 16.3 ms on `o043` (ood, en); the dominant stage was **embed_query** at 13.8 ms, fallbacks fired: none.


## Concurrency

A separate `--concurrency 4` pass over the same 500 queries: P50 40.8 ms, P95 53.9 ms, over-budget 0/500.


![per-stage latency](latency.svg)


_All stages are milliseconds. The `embed query` row is a real transformer forward pass (multilingual-e5-small on CPU). Generation is still extractive and the reranker is lexical, so those two rows are floors, not an LLM's cost: budget for ~15 ms of cross-encoder and the generator's own time on top._


_Cold = fresh process, semantic cache off, encoder already resident: a server loads its model before it accepts traffic, so that ~10 s belongs to startup and not to the first caller's 200 ms. Warm = 50 discarded warmups first, cache on; the cache-off run is the `cold` row and the cache hit rate is printed above, so a repeat-heavy query file cannot flatter the P50 unnoticed._
