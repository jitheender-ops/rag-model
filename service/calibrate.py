"""Gate 2's dense-score floor, derived instead of guessed.

    make calibrate      ->  data/score_floor.json   (after `make chunking`)

Cosine scales are not comparable across embedders or corpora: e5 packs unrelated text
around 0.73-0.80, so a floor tuned on one corpus silently passes everything on the next.
The floor is therefore a measured artifact with a provenance stamp, not a constant someone
edited once.

WHICH SET IT IS FITTED ON, AND WHY THAT MATTERS
The sweep runs on D3's frozen latency set (in-domain + spoken + out-of-domain), and D4's
guardrail set is then graded against the result. Fitting the threshold on the set it is
scored against would make D4's off-topic column a report on its own training data.
"""
from __future__ import annotations

import json
import os

from d1 import corpus
from d1.index import BACKEND, MODEL_NAME, embed_many
from service.pipeline import load_index

OUT = "data/score_floor.json"


MAX_FALSE_ABSTENTION = float(os.getenv("MAX_FALSE_ABSTENTION", "0.08"))


def sweep(pos: list[float], neg: list[float],
          max_false_abstention: float = MAX_FALSE_ABSTENTION) -> tuple[float, float]:
    """One-dimensional threshold sweep, constrained by the false-abstention target.

    Not free, and not balanced accuracy. Maximising balanced accuracy on this data picks a
    floor that refuses 17% of legitimate questions to keep the last few out-of-domain ones
    out -- a defensible number on a confusion matrix and a terrible voice assistant. D4
    states the target out loud (false abstention < 8%), so the floor is the most selective
    threshold that still meets it, and gate 4 remains the net for whatever gate 2 lets in.

    Always feasible: a floor of 0 refuses nothing. Ties keep the lower floor.
    """
    best, best_tnr = 0.0, -1.0
    for cand in sorted({round(s, 4) for s in pos + neg}):
        if sum(s < cand for s in pos) / len(pos) > max_false_abstention:
            continue                         # over the refusal budget: not on the table
        tnr = sum(s < cand for s in neg) / len(neg)
        if tnr > best_tnr:
            best, best_tnr = cand, tnr
    return best, best_tnr


GROUNDING_N = int(os.getenv("GROUNDING_N", "300"))


def grounding_samples(ix, texts, parents, queries) -> list[tuple[float, float, bool]]:
    """(coverage, entailment, the citation was a gold passage) for each query reaching gate 4.

    Gate 4's floors are swept on real queries with human qrels, which gives the one label
    that matters: did the passage we are about to answer from actually answer this question?
    Collected with the gate-4 floors dropped to zero so nothing abstains and every citation
    is observable -- gate 2 is left exactly as calibrated, because gate 4 should be fitted
    on the traffic gate 4 really sees, not on traffic gate 2 would have refused.
    """
    import service.pipeline as P
    saved = (P.COVERAGE_FLOOR, P.COVERAGE_FLOOR_CS, P.SUPPORT_FLOOR, P.NLI_FLOOR)
    P.COVERAGE_FLOOR = P.COVERAGE_FLOOR_CS = P.SUPPORT_FLOOR = 0.0
    # ...and the entailment floor too. It is the floor being FITTED here: left at its
    # previous value it abstains, blanks ctx.answer, and every row it refused arrives with
    # nothing to score -- a sweep fitted on the rows the old floor already approved.
    P.NLI_FLOOR = float("-inf")
    out = []
    try:
        for q in queries:
            t = P.answer(q["query"], ix, texts, qid=q["qid"], parents=parents)
            cited = t.meta.get("cited")
            if not cited:                     # gate 1 or gate 2 got there first
                continue
            pid = ix.get(cited)["pid"]
            passage = parents.get(pid) or texts.get(cited, "")
            # the entailment score is collected on the same traffic and the same citations as
            # the coverage score, so the two floors are fitted on one set and are comparable.
            # No deadline here: this is calibration, not serving, and a floor fitted on
            # whichever pairs happened to beat a 45 ms wait would be fitted on the machine.
            nli = P.entails_by(P.display(passage), f"{q['query']} {t.meta.get('answer', '')}",
                               1e9) if P.VERIFY == "nli" and t.meta.get("answer") else None
            out.append((P.covers(P.content(q["query"]), passage),
                        nli, pid in set(q["qrels"])))
    finally:
        P.COVERAGE_FLOOR, P.COVERAGE_FLOOR_CS, P.SUPPORT_FLOOR, P.NLI_FLOOR = saved
    return out


def grounding_queries() -> list[dict]:
    """D1's frozen held-out queries, minus anything D4 grades.

    Both sets are drawn from the same pool of real queries, so the overlap has to be
    subtracted by hand -- otherwise the floor is fitted on rows it is later scored against
    and D4's false-abstention number is a report on its own training data."""
    from d4 import dataset
    graded = {r["query"] for r in dataset.load()} if os.path.exists(dataset.OUT) else set()
    with open("data/queries_chunking.jsonl", encoding="utf-8") as fh:
        pool = [json.loads(l) for l in fh]
    return [q for q in pool if q["query"] not in graded][:GROUNDING_N]


def main():
    from d3.run import freeze_queries, winner_dir
    ix, _texts, _parents = load_index(winner_dir())
    # freeze_queries(), not a bare open(): calibrate runs BEFORE `make latency` in `make
    # submit`, and what it depends on is the frozen query set, not on D3 having run. The
    # set is written once and never regenerated, so both callers see the same 500 queries.
    queries = freeze_queries()
    vecs = embed_many([q["query"] for q in queries], "query")
    best = {"in_domain": [], "spoken": [], "ood": []}
    for q, v in zip(queries, vecs):
        hits = ix.search(v, k=1)
        best[q["kind"]].append(hits[0][1] if hits else 0.0)

    pos = best["in_domain"] + best["spoken"]
    neg = best["ood"]
    floor, tnr = sweep(pos, neg)
    tpr = sum(s >= floor for s in pos) / len(pos)
    cal = {"floor": floor, "ood_rejected": round(tnr, 4),
           "false_abstention": round(1 - tpr, 4),
           "false_abstention_cap": MAX_FALSE_ABSTENTION,
           "balanced_accuracy": round((tpr + tnr) / 2, 4),
           "backend": BACKEND, "model": MODEL_NAME if BACKEND == "st" else "hashed-bow",
           "index": winner_dir(), "corpus_sha": corpus.sha(),
           "n_pos": len(pos), "n_neg": len(neg),
           "in_domain_min": round(min(best["in_domain"]), 4),
           "spoken_min": round(min(best["spoken"]), 4),
           "ood_max": round(max(neg), 4),
           "leaked_ood": sum(s >= floor for s in neg),
           "refused_in_domain": sum(s < floor for s in pos)}
    # gate 4's floor, on a different set again: real queries with qrels, D4's rows removed.
    # The new gate-2 floor is installed first: pipeline reads it at import, so without this
    # the grounding sweep is fitted on the traffic the PREVIOUS floor let through.
    import service.pipeline as P
    P.SCORE_FLOOR = floor
    samples = grounding_samples(ix, _texts, _parents, grounding_queries())
    gold = [c for c, _n, is_gold in samples if is_gold]
    wrong = [c for c, _n, is_gold in samples if not is_gold]
    if gold and wrong:
        cov_floor, cov_tnr = sweep(gold, wrong)
        cal.update({"coverage_floor": cov_floor,
                    "coverage_rejects_wrong_citation": round(cov_tnr, 4),
                    "coverage_false_abstention": round(sum(c < cov_floor for c in gold) / len(gold), 4),
                    "coverage_n_gold": len(gold), "coverage_n_wrong": len(wrong)})
    # gate 4's entailment floor, swept on the same rows and against the same label, so the
    # two verifiers can be compared on one set rather than on two convenient ones.
    n_gold = [n for _c, n, is_gold in samples if is_gold and n is not None]
    n_wrong = [n for _c, n, is_gold in samples if not is_gold and n is not None]
    if n_gold and n_wrong:
        nli_floor, nli_tnr = sweep(n_gold, n_wrong)
        cal.update({"nli_floor": nli_floor,
                    "nli_rejects_wrong_citation": round(nli_tnr, 4),
                    "nli_false_abstention": round(sum(n < nli_floor for n in n_gold) / len(n_gold), 4),
                    "nli_n_gold": len(n_gold), "nli_n_wrong": len(n_wrong),
                    "nli_model": P.NLI_MODEL})
    with open(OUT, "w") as fh:
        json.dump(cal, fh, indent=2)
    print(json.dumps(cal, indent=2), f"-> {OUT}")
    if cal["ood_max"] >= cal["in_domain_min"]:
        print("\nnote: the score ranges OVERLAP -- no threshold separates them cleanly. "
              "The floor above is the best available trade, and the two counts at the "
              "bottom are what it costs in each direction.")


def demo():
    # clean separation: take the highest floor that rejects everything negative
    floor, tnr = sweep([0.9, 0.8, 0.85], [0.2, 0.3, 0.1], max_false_abstention=0.0)
    assert 0.3 < floor <= 0.8 and tnr == 1.0, (floor, tnr)

    # the constraint must bind: refusing 1 of 2 positives (50%) is over an 8% cap, so the
    # sweep gives up the negative rather than the positive.
    floor, tnr = sweep([0.5, 0.9], [0.6], max_false_abstention=0.08)
    assert floor <= 0.5 and tnr == 0.0, (floor, tnr)
    # ...and with the cap lifted it takes the same trade the other way
    assert sweep([0.5, 0.9], [0.6], max_false_abstention=0.5)[1] == 1.0

    assert sweep([1.0], [0.0])[0] <= 1.0     # always feasible: a floor of 0 refuses nothing
    print("calibrate sweep ok")


if __name__ == "__main__":
    import sys

    demo() if "--selfcheck" in sys.argv else main()
