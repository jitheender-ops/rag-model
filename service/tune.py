"""Tune the answer path against MS MARCO's own answers, not against taste.

    make tune                 # sweep the variants, print the table
    make tune N=400

Every frozen query carries a human `Answer` string from MS MARCO. Until now this repo scored
retrieval (does the gold PASSAGE come back?) and grounding (does the answer come from the
passage it cites?) but never the thing a user actually hears: is this the right answer? Token
F1 against the human answer is that number, and it is what every constant below was chosen by.

  answer_f1        SQuAD-style token F1 against the human answer, in the query's own script
  cites_gold       the cited chunk came from a passage the humans marked relevant
  ms               the measured window, so a variant cannot buy quality with the budget

WHAT THIS IS NOT
It is not a leaderboard: an extractive system reading a whole sentence is penalised against
a short human answer no matter how right it is, so the ABSOLUTE F1 here is low by
construction and only the DIFFERENCES between variants mean anything. Read the columns
against each other, never on their own.

The sweep runs on D1's frozen held-out queries -- the same set gate 4's floor is fitted on,
and deliberately not D4's grading set.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

from d1.index import tokenize

FROZEN = "data/queries_chunking.jsonl"
DEFAULT_N = int(os.getenv("N", "300"))


def answer_f1(pred: str, gold: str) -> float:
    """Token F1, the SQuAD measure, on the Indic-aware tokenizer.

    Bag-of-tokens on purpose: an extractive answer states the fact in the passage's word
    order, a human answer states it in theirs, and demanding the same order would measure
    phrasing rather than correctness."""
    p, g = tokenize(pred or ""), tokenize(gold or "")
    if not p or not g:
        return 0.0
    from collections import Counter
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not same:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def load_queries(n: int) -> list[dict]:
    with open(FROZEN, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    return [r for r in rows if (r.get("answer") or "").strip()][:n]


# ---------- the variants ----------
# each takes the ranked hits and the chunk texts, and returns (answer_text, cited_chunk_id).
# they share everything upstream, so a difference in the table is a difference in this step.

def _sentences_of(text, P):
    return [s for s, _, _ in P.sentences(P.sanitize(P.display(text)))]


def pick_lexical(ctx, texts, qvec, P, n_ctx=4, cap=96):
    """What shipped before this file existed: the sentence with the most query content
    terms, across n_ctx chunks."""
    q = P.content(ctx.query)
    best, best_score, best_cid = "", -1.0, None
    for cid, _, _ in ctx.hits[:n_ctx]:
        for s in _sentences_of(texts.get(cid, ""), P):
            score = len(q & P.content(s)) / (len(q) or 1)
            if score > best_score:
                best, best_score, best_cid = s, score, cid
    return " ".join(best.split()[:cap]), best_cid


def pick_dense(ctx, texts, qvec, P, n_ctx=4, cap=96):
    """The sentence whose embedding is closest to the query's.

    The query vector is already computed and sitting in the request -- reusing it costs one
    encode of a handful of sentences, and lexical overlap is exactly the signal that fails
    on a paraphrase, which is what a spoken question usually is."""
    from d1.index import cosine, embed_many
    cands = [(cid, s) for cid, _, _ in ctx.hits[:n_ctx]
             for s in _sentences_of(texts.get(cid, ""), P)]
    if not cands:
        return "", None
    vecs = embed_many([s for _, s in cands], "passage")
    i = max(range(len(cands)), key=lambda j: cosine(vecs[j], qvec))
    return " ".join(cands[i][1].split()[:cap]), cands[i][0]


def pick_top_chunk_dense(ctx, texts, qvec, P, n_ctx=1, cap=96):
    """Dense sentence choice, but only inside the top-ranked chunk."""
    return pick_dense(ctx, texts, qvec, P, n_ctx=1, cap=cap)


def pick_whole_top_chunk(ctx, texts, qvec, P, n_ctx=1, cap=96):
    """No sentence selection at all: read the top chunk, capped. The floor to beat."""
    if not ctx.hits:
        return "", None
    cid = ctx.hits[0][0]
    return " ".join(P.sanitize(P.display(texts.get(cid, ""))).split()[:cap]), cid


def pick_llm(ctx, texts, qvec, P, n_ctx=3, cap=96):
    """The harnessed LLM, given the same top chunks the extractive path sees.

    Scored on the same queries and the same metric as everything else, because "an LLM will
    obviously be better" is the kind of claim this harness exists to check rather than
    repeat. A call that fails or times out scores as the empty answer it produced -- that is
    what a user would have got."""
    from service import llm
    cands = [(cid, P.display(texts.get(cid, ""))) for cid, _, _ in ctx.hits[:n_ctx]]
    if not cands:
        return "", None
    try:
        out = llm.complete(ctx.query, [t for _, t in cands])
    except Exception:
        return "", cands[0][0]
    n = out.get("passage")
    cid = cands[n - 1][0] if isinstance(n, int) and 1 <= n <= len(cands) else cands[0][0]
    return " ".join((out.get("answer") or "").split()[:cap]), cid


def pick_cross(depth: int):
    """Cross-encoder over the top `depth` fused hits, then the shipped sentence choice.

    The reranker changes which chunk is #1; generate reads #1 and picks a sentence out of it.
    So this row is the shipped row with one extra step, and the difference between them is the
    reranker and nothing else."""
    def pick(ctx, texts, qvec, P, n_ctx=1, cap=96):
        P.rerank_hits(ctx, depth)
        return pick_top_chunk_dense(ctx, texts, qvec, P, cap=cap)
    return pick


# The baseline row is whatever the serving path currently does, and it moves when the serving
# path moves -- otherwise every later variant is scored against a system that no longer
# exists. It has moved twice: lexical choice -> dense sentence in the top chunk, and dense
# sentence -> cross-encoder in front of it. Every cross-encoder row below ends in that same
# dense top-1 sentence pick, so the only difference between them is the reordering.
SHIPPED = "cross-encoder top 4 (shipped)"
VARIANTS = {
    "lexical sentence, top 4": pick_lexical,
    "dense sentence, top 4": pick_dense,
    "dense sentence, top 1, no rerank": pick_top_chunk_dense,
    "whole top chunk": pick_whole_top_chunk,
    SHIPPED: pick_cross(4),
    "cross-encoder top 8": pick_cross(8),
    "cross-encoder top 20": pick_cross(20),
}
if os.getenv("WITH_LLM"):          # opt-in: each row is a paid API call per query
    VARIANTS["llm (harnessed)"] = pick_llm


def run(variant, queries, ix, texts, parents) -> dict:
    import service.pipeline as P
    from harness.spans import Trace, now_ns, NS_PER_MS
    f1s, gold, ms = [], 0, []
    for q in queries:
        t0 = now_ns()
        trace = Trace(q["qid"], {"query": q["query"]})
        ctx = P.Ctx(q["query"], ix, trace, None)
        qvec = P.embed(q["query"], "query")
        P.retrieve(ctx, qvec)
        # the rerank STAGE is deliberately not called here: it reads RERANK from the
        # environment, which would silently rerank every row including the baseline. The
        # variants that want it call rerank_hits themselves, so the column measures it.
        text, cid = variant(ctx, texts, qvec, P)
        ms.append((now_ns() - t0) / NS_PER_MS)
        f1s.append(answer_f1(text, q["answer"]))
        if cid and ix.get(cid)["pid"] in set(q["qrels"]):
            gold += 1
    return {"answer_f1": statistics.mean(f1s), "cites_gold": gold / len(queries),
            "ms_p50": statistics.median(ms), "n": len(queries), "per_query": f1s}


def paired_delta(a: list[float], b: list[float], rounds: int = 2000) -> tuple[float, float, float]:
    """(mean difference, low, high) at 95%, bootstrapped over the SAME queries.

    Paired, because the alternative is how tuning goes wrong: comparing two independent means
    over 200 queries buries a real 0.01 effect inside a 0.02 standard error, and equally lets
    a lucky 0.01 look like a win. The queries are identical across variants, so the per-query
    difference is the measurement and its spread is the only honest error bar.

    Deterministic: a fixed seed, because a confidence interval that moves between runs is
    another number nobody can reproduce.
    """
    import random
    diffs = [x - y for x, y in zip(a, b)]
    mean = statistics.mean(diffs)
    rng = random.Random(42)
    n = len(diffs)
    means = sorted(statistics.mean(rng.choices(diffs, k=n)) for _ in range(rounds))
    return mean, means[int(0.025 * rounds)], means[int(0.975 * rounds)]


def main():
    from d3.run import winner_dir
    from service.pipeline import load_index
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else DEFAULT_N
    queries = load_queries(n)
    ix, texts, parents = load_index(winner_dir())
    if any(n.startswith("cross-encoder") for n in VARIANTS):
        import service.pipeline as P
        P.cross_encoder()      # 10 s of weight loading, outside the timed loop
    print(f"{len(queries)} held-out queries with human answers, index {winner_dir()}\n")
    print(f"| {'variant':32} | answer F1 | cites gold | ms P50 |")
    print(f"|{'-' * 34}|-----------|------------|--------|")
    rows = {}
    for name, fn in VARIANTS.items():
        r = run(fn, queries, ix, texts, parents)
        rows[name] = r
        print(f"| {name:32} | {r['answer_f1']:9.3f} | {r['cites_gold']:10.1%} | "
              f"{r['ms_p50']:6.1f} |", flush=True)
    ship = SHIPPED
    print(f"\npaired against what ships, 95% bootstrap CI over the same {len(queries)} queries:")
    verdicts = {}
    for name in VARIANTS:
        if name == ship:
            continue
        mean, lo, hi = paired_delta(rows[name]["per_query"], rows[ship]["per_query"])
        real = lo > 0 or hi < 0
        verdicts[name] = (mean, lo, hi, real)
        print(f"  {name:32} {mean:+.4f} F1  [{lo:+.4f}, {hi:+.4f}]  "
              f"{'SIGNIFICANT' if real else 'indistinguishable from noise'}"
              f"   {rows[name]['ms_p50'] - rows[ship]['ms_p50']:+.1f} ms")
    winners = [n for n, (m, _, _, real) in verdicts.items() if real and m > 0]
    print("\n" + ("keep what ships: no variant beat it by more than the error bar."
                  if not winners else
                  f"adopt: {max(winners, key=lambda n: verdicts[n][0])}"))
    print("\nAbsolute F1 is low by construction: an extractive sentence is scored against a "
          "short human answer.\nOnly the differences between rows mean anything.")


def demo():
    assert answer_f1("the bridge opened in 1901", "the bridge opened in 1901") == 1.0
    assert answer_f1("", "anything") == 0.0 and answer_f1("anything", "") == 0.0
    assert answer_f1("completely different words", "the bridge opened") == 0.0
    half = answer_f1("a b c d", "a b")           # p=0.5, r=1.0 -> F1 = 0.667
    assert abs(half - 2 / 3) < 1e-9, half
    # word order must not matter: same facts, different phrasing
    assert answer_f1("opened in 1901", "in 1901 opened") == 1.0
    # and it must work in a script the plain \w tokenizer would shred
    assert answer_f1("কর্পোরেশন একটি সংস্থা", "কর্পোরেশন একটি সংস্থা") == 1.0
    print("tune metric ok")


if __name__ == "__main__":
    demo() if "--selfcheck" in sys.argv else main()
