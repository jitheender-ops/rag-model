# The build spec vs what is built

Reviewed against the repository at the commit that produced the current reports.

## Verdict

**The spec is not a different architecture. It is this architecture, plus eight specific
upgrades** — one of which, the cross-encoder reranker, is now built. Its defining idea — §0,
"the answer is produced by extraction, not generation; the LLM is an optional downstream
upgrade outside the latency budget and can be discarded at any time" — is what ships today,
and it was not copied from the spec: it was arrived at by measuring four providers and finding
the fastest one 2.8x over budget in isolation and worse on real passages.

So **do not rewrite.** A rewrite would re-derive the same shape while discarding a
reproducible 200 ms PASS, 23 self-checks, and four regenerable reports. What the spec is
genuinely worth is its **gap list**, which is real and ranked at the bottom of this file.

## Point by point

### Already built, and matching

| spec | state |
|---|---|
| §0 extraction is the answer; LLM optional, outside the budget, discardable | `GENERATOR=extractive` default; LLM unreachable unless explicitly enabled |
| §1 `t0` = STT `is_final`, `t1` = answer flushed; STT reported separately | identical, and both excluded legs are measured (STT 520 ms, TTS 793 ms) |
| §1 target P50 ≤ 80 ms | **48.0 ms** with the reranker in, 20.2 ms before it — inside the target either way |
| §2 stage budgets, RUN/DEGRADE/SKIP before entry | `@stage(budget_ms=..., degradable=...)`, checked before entry |
| §2.5 RRF on rank, not score; dedupe to parent passage | yes, and the BM25 weight (0.1) was swept, not guessed |
| §2.8 grounding gate on a **calibrated** score, value and derivation recorded | `service/calibrate.py`, fitted on a set that does not grade it, stamped with its corpus sha |
| §4 normalize, freeze, hash, chunk by many strategies, evaluate, freeze winner | eight strategies, exceeding the spec's minimum of five |
| §4 **score at passage granularity** | the rule D1 is built around |
| §5 deadline propagation; never retry inside the deadline | both, and the encoder deadline is *enforced*, not just logged |
| §5 trace JSONL, one line per request, feeds both report and replay | `harness/spans.py` |
| §6 150 should-abstain + 130 should-answer; four numbers; per-gate attribution with latency | exactly this |
| §7 500 queries 350/100/50, cold + warm + concurrency-4, nearest-rank, no column summing, P100 attributed | exactly this |
| §9 build order: span recorder first, then index, then path, then guardrails | the order it was built in |
| §10 invariants 1, 2, 4, 5, 6, 7, 8 | all hold |

### Gaps — real, ranked by value

| # | gap | spec | now | cost | worth it |
|---|---|---|---|---|---|
| ~~1~~ | ~~**cross-encoder reranker**~~ | §2.6 MiniLM int8, top-20 → top-4 | **CLOSED** — mMiniLMv2-L12-H384 over the top 4, `RERANK=cross` by default. +0.023 answer F1 [+0.011, +0.035] over the shipped row, top-1 34.6% → 41.5%, budget still PASS. See the note below on the two things the spec's version of this would have got wrong | done | it was |
| 2 | **two-phase delivery** | §3 flush fast answer, then polish on the same connection, discard on novel fact | LLM replaces inline or not at all | ~1 day | **yes** — this is the spec's one genuinely new idea, and it makes the LLM free |
| 3 | **NLI hallucination check** | §6 automatic NLI + 50-answer human review | lexical support; human sample generated, unfilled | ~half day | **yes** — near_miss catches 17/30 (was 13/30 before the reranker, which improved gate 4's inputs rather than gate 4); lexical rejects 22.7% of wrong citations |
| 4 | **input guard classifier** | §2.1 local classifier + injection rules + PII + language ID | regex only | ~half day | partly — regex catches 30/30 unsafe and 29/30 injection today; PII and language ID are genuinely missing |
| 5 | **semantic cache** | §2.2 cosine ≥ 0.97 against recent queries | lexical content-term key | ~2 hours | marginal — 9.2% hit rate already, and it is 1 ms |
| 6 | **typed boundaries** | §5 Pydantic in/out of every stage | dataclass `Ctx`, dict meta | ~half day | rigor, not behaviour — and it adds a dependency the repo currently does not have |
| 7 | **int8 / ONNX embeddings** | §2.3 2 ms | fp32 sentence-transformers, 7.8 ms | ~half day | more interesting than it was — the reranker took the headroom, so P50 is 48 ms rather than 20, and the same trick applies to the cross-encoder (117M params, fp32) where it would buy more |
| 8 | **WebSocket + loopback benchmark** | §8 | HTTP POST, benchmarked in-process | ~half day | needed *only* for #2, and for honest client RTT |

### Where the spec is wrong about this repo, or about itself

- **§2.6 "rerank — currently missing from the design and must be added."** It was added, and
  the spec was wrong about it twice. **"MiniLM":** an English cross-encoder would reorder
  three quarters of this corpus on nothing, so the model is mMARCO's multilingual MiniLM.
  **"top-20 → top-4":** measured, depth 20 is *worse than not reranking at all*
  (−0.0284 F1 against depth 4, where no reranker at all is −0.0229 — and it spends 139 ms more
  to be worse). Depth 4 is the winner: fusion already puts the right passage in the top 4 for
  68.5% of queries, so reordering four plausible candidates is the whole job, and reaching down
  to rank 20 promotes passages the fusion ranked low for good reason.
  The cheapest depth was the best one, which is not the usual shape of a quality knob and is
  the reason to sweep instead of assume.
- **§2 budgets total ~57 ms and target 64–80 ms P50.** The built path was **20 ms P50** before
  the reranker and is inside the spec's own target with it. The point stands in the direction
  it was written for: the cross-encoder was measured against 20 ms, and the honest cost of
  gap #1 is that it spent most of the headroom an in-window LLM would have needed.
- **§2.7 "extract 26 ms … if it is heuristic, it should not cost 26 ms."** Ours costs
  14.7 ms and is not heuristic: it encodes the candidate sentences and picks by cosine,
  which beat term overlap by +0.016 answer F1 [+0.006, +0.028] over 1200 queries.
- **§10.3 "the fast answer is always produced; no code path returns nothing."** This
  conflicts with §10.5, "abstention is a first-class output." The repo resolves it the way
  §10.5 implies: abstention is an answer, with a gate and a reason, and it is the correct
  output for 150 of the 280 graded queries.
- **§8 "colocated with the LLM provider."** Moot once §0 is taken seriously.

## If a new session picks this up

Do these, in this order, and measure each against the existing harness rather than against
the spec's prose:

1. ~~**Cross-encoder reranker** into the empty `rerank()` slot.~~ **Done.** It cleared the bar
   the same way the last three tuning decisions did: `make tune` over 1200 queries, paired and
   bootstrapped, +0.0229 F1 [+0.0111, +0.0350]. Two things worth carrying forward from doing
   it — a model call with a free fallback needs its wait capped at its *own* cost rather than
   at the whole remaining budget, and it needs a concurrency limit and its own workers, both
   sized from the traces. Neither is in the spec.
2. **Two-phase delivery**: flush the fast answer, then polish, then `novel_fact_check`,
   discard on any unsupported claim. Requires the WebSocket (#8). Publish the percentage of
   answers the polish materially changes — the spec is right that a low number is the
   strongest defense of the budget boundary.
3. **NLI verifier** behind `verify()`. Everything around it — the calibrated floor, D4, the
   confusion matrix — stays as it is. Note that the reranker moved gate 4's inputs, so its
   floors were refitted by `make calibrate` on the same run that adopted it.

Non-negotiables to carry forward, all of them already true here: no LLM inside the window,
no network inside the window, passage-granularity scoring, `perf_counter_ns` only, and every
number in every report regenerable by one make target.
