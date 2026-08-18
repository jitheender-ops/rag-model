# Gate 4 - lexical vs entailment

| field | value |
|---|---|
| machine | arm64 / 10 vCPU |
| region | local |
| provider | Darwin |
| date | 2026-08-18 17:00 IST |
| commit | 4433cea |
| embedder | intfloat/multilingual-e5-small / 384d |
| python | 3.13.7 |
| seed | 42 |
| set | data/guardrails.jsonl (280 rows) |
| index | artifacts/d1/s7 |
| nli model | MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 |
| nli floor | -0.1024 |
| nli budget | 45.0 ms |

One variable: the gate-4 verifier. Correct abstention alone decides nothing -- a verifier that refuses everything scores 100% on it -- so it is reported against the false-abstention rate it costs.

| metric | lexical | nli | verdict |
|---|---|---|---|
| correct abstention (150 should-abstain) | 84.0% | 78.0% | worse (-6.0%) |
| false abstention (130 should-answer) | 3.8% | 9.2% | worse (+5.4%) |
| near-miss caught | 56.7% | 43.3% | worse (-13.3%) |
| hallucination rate | 0.0% | 0.0% | — |
| injection resistance | 100.0% | 100.0% | — |
| answers given | 149 | 151 | — |
| gate 4 firings | 52 | 45 | — |
| end-to-end P50 | 38.2 ms | 74.6 ms | — |
| end-to-end P100 | 112.8 ms | 535.9 ms | — |

**lexical stays**: nli moved correct abstention -6.0% for +5.4% false abstention, which is not an improvement. The default does not change on a result like this.

Latency: P50 38 -> 75 ms, P100 113 -> 536 ms. Gate 4's entailment call is bounded at 45 ms and falls back to the lexical verdict, so this is a quality change, not a budget change.
