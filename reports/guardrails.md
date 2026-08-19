# D4 - Guardrail metrics

| field | value |
|---|---|
| machine | arm64 / 10 vCPU |
| region | local |
| provider | Darwin |
| date | 2026-08-19 22:53 IST |
| commit | 3747067 |
| embedder | intfloat/multilingual-e5-small / 384d |
| python | 3.13.7 |
| seed | 42 |
| set | data/guardrails.jsonl (280 rows) |
| index | artifacts/d1/s7 |
| gate 2 floor | 0.8485 |
| gate 4 coverage floor | 0.2857 |
| floors fitted on corpus | ab0bdc5e0d204eb6 |


## The four numbers

| metric | value | denominator | grading |
|---|---|---|---|
| correct abstention | 71.3% | 150 should-abstain | automatic |
| false abstention | 5.4% | 130 should-answer | automatic, target < 8% |
| hallucination rate | 0.0% | 166 answers given | automatic + human sample |
| injection resistance | 100.0% | 30 injections | canary string match |

## Confusion matrix

|  | abstained | answered |
|---|---|---|
| should abstain (150) | 107 correct | 43 leaked |
| should answer (130) | 7 false refusal | 123 correct |

## Per-gate attribution

| bucket        | n  | abstained | expected gate  | actually fired             | mean ms |
|---------------|----|-----------|----------------|----------------------------|---------|
| off_topic     | 30 |    15/30  | gate 2 (score) | g2_score:9, g4_nli:6       |    58.2 |
| unanswerable  | 30 |    23/30  | gate 2 or 4    | g4_nli:21, g2_score:2      |    71.5 |
| unsafe        | 30 |    30/30  | gate 1 (input) | g1_unsafe:29, g4_nli:1     |     2.1 |
| injection     | 30 |    28/30  | gate 1 + 3     | g1_injection:17, g4_nli:9, g2_score:2 |    30.8 |
| near_miss     | 30 |    11/30  | gate 2 or 4    | g4_nli:9, g2_score:2       |    62.9 |
| control       |100 |     2/100 | --             | g2_score:2                 |    53.3 |
| code_switch   | 30 |     5/30  | --             | g4_nli:5                   |    56.0 |

The `mean ms` column does double duty: off-topic and unsafe queries are rejected in single-digit milliseconds, which is a latency argument and a safety argument in the same row.


## Why the hallucination rate is not the good news it looks like

43 of the 166 answers given came from rows labelled should-abstain. Every one of them is *supported by the passage it cites* -- the generator is extractive, so the answer is a sentence lifted from that passage -- and every one of them is still the wrong answer to the question asked. That is the near-miss failure mode, and it is invisible to a support check by construction. Read the 0.0% beside the confusion matrix, never instead of it: this system's error is citing a real passage that does not answer you, not inventing text.


## On grading hallucination with our own verifier

The automatic pass uses the same entailment check the serving path uses at gate 4, so it is circular by construction and a suspiciously clean number would mean nothing. 50 answers were sampled for human review (`reports/human_sample_50.jsonl`, two reviewers, independent). Fill `data/human_labels.jsonl` with `{"id":..,"unsupported":true|false,"reviewer":..}` and re-run to have the agreement rate printed here.

