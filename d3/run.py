"""D3 stage 1-3: 500 frozen queries, cold then warm, one JSONL line per query.

    python3 d3/run.py            # cold + warm + a concurrency-4 pass
    python3 d3/run.py --only warm

Never store a pre-computed average -- store samples. All ten span timings, the
degradation verdict, cache hit and token count go on every line; d3/reduce.py does the
arithmetic later, so the same log can answer a question nobody asked at record time.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

from d1 import corpus
from harness.env import pin_randomness
from harness.spans import Recorder
from service.pipeline import answer, load_index

QUERIES = "data/queries_latency.jsonl"
TRACES = "artifacts/d3"
N_IN_DOMAIN, N_SPOKEN, N_OOD = 350, 100, 50
WARMUPS = 50

SPOKEN = ["so um when was the {ent} in {place} finished",
          "hey quick one, {ent} {place} which year",
          "do you know when they completed that {ent} over in {place}",
          "{ent} of {place} - what year was that"]
# Spoken-style disfluency wrapped around a real query, in the query's own language --
# an English filler in front of a Tamil question would make the row a code-switch test
# rather than a spoken-style one, and D4 already owns that bucket.
FILLER = {
    "en": ["so um, {q}", "hey quick one - {q}", "do you know {q}", "{q}, any idea"],
    "hi": ["अच्छा तो {q}", "सुनो, {q}", "एक बात बताओ, {q}", "{q}, पता है क्या"],
    "ta": ["சரி, {q}", "ஒரு விஷயம், {q}", "கொஞ்சம் சொல்லுங்க, {q}", "{q}, தெரியுமா"],
    "bn": ["আচ্ছা, {q}", "একটা কথা, {q}", "শোনো, {q}", "{q}, জানো নাকি"],
}
OOD = ["what is the melting point of tungsten", "who won the world cup in 1998",
       "how do I renew my passport", "what is the capital of Iceland",
       "how many calories in a mango", "what is the exchange rate for the yen",
       "when does the monsoon reach Kerala", "how do solar panels work",
       "who painted the Night Watch", "what is the offside rule"]


def _real_rows(rng) -> list[dict]:
    """In-domain + spoken drawn from the ingested MS MARCO queries. Empty without the
    ingest, and the template generator below takes over."""
    pool = corpus.real_queries()
    if len(pool) < N_IN_DOMAIN:
        return []
    rng.shuffle(pool)
    rows = [{"qid": f"l{i:03d}", "kind": "in_domain", "lang": q["lang"], "query": q["query"]}
            for i, q in enumerate(pool[:N_IN_DOMAIN])]
    # the spoken rows paraphrase queries the in-domain rows did NOT use: the same question
    # twice would let the semantic cache answer the paraphrase and flatter the warm P50.
    tail = pool[N_IN_DOMAIN:] or pool
    for i in range(N_SPOKEN):
        q = tail[i % len(tail)]
        tmpl = FILLER.get(q["lang"], FILLER["en"])
        rows.append({"qid": f"s{i:03d}", "kind": "spoken", "lang": q["lang"],
                     "query": tmpl[i % len(tmpl)].format(q=q["query"].rstrip("? ।"))})
    return rows


def _synth_rows(rng) -> list[dict]:
    facts = [(d, p) for d in corpus.load() for p in d["passages"] if "ent" in p]
    rng.shuffle(facts)
    rows = []
    for i in range(N_IN_DOMAIN):
        d, p = facts[i % len(facts)]
        rows.append({"qid": f"l{i:03d}", "kind": "in_domain", "lang": d["lang"],
                     "query": f"in which year was the {p['ent']} of {p['place']} completed"})
    for i in range(N_SPOKEN):
        d, p = facts[(i * 7) % len(facts)]
        rows.append({"qid": f"s{i:03d}", "kind": "spoken", "lang": d["lang"],
                     "query": SPOKEN[i % len(SPOKEN)].format(ent=p["ent"], place=p["place"])})
    return rows


def freeze_queries(path: str = QUERIES) -> list[dict]:
    """350 in-domain, 100 spoken-style paraphrases, 50 out-of-domain. The OOD ones still
    cost time, so they belong in the distribution."""
    if os.path.exists(path):
        corpus.check_frozen(path)
        with open(path, encoding="utf-8") as fh:
            return [json.loads(l) for l in fh]
    rng = random.Random(43)
    rows = _real_rows(rng) or _synth_rows(rng)
    for i in range(N_OOD):
        rows.append({"qid": f"o{i:03d}", "kind": "ood", "lang": "en",
                     "query": OOD[i % len(OOD)]})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    corpus.bind(path)
    return rows


def winner_dir() -> str:
    try:
        with open("reports/chunking.json") as fh:
            return "artifacts/d1/" + json.load(fh)["winner"].split()[0]
    except Exception:
        return "artifacts/d1/s2"


def one_pass(queries, mode: str, concurrency: int = 1):
    """cold = fresh index, semantic cache off. warm = 50 discarded warmups, cache on."""
    ix, texts, parents = load_index(winner_dir())
    cache = {} if mode != "cold" else None
    if mode != "cold":
        for q in queries[:WARMUPS]:
            answer(q["query"], ix, texts, qid="warmup", cache=cache, parents=parents)
    out = f"{TRACES}/traces_{mode}.jsonl"
    if os.path.exists(out):
        os.remove(out)
    with Recorder(out) as rec:
        def work(q):
            t = answer(q["query"], ix, texts, qid=q["qid"], cache=cache, parents=parents)
            t.meta.update({"kind": q["kind"], "lang": q["lang"], "mode": mode,
                           "concurrency": concurrency})
            return t

        if concurrency > 1:
            with ThreadPoolExecutor(concurrency) as ex:
                for t in ex.map(work, queries):
                    rec.write(t)
        else:
            for q in queries:                       # sequential: no queueing in the numbers
                rec.write(work(q))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["cold", "warm", "conc"], default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    pin_randomness()
    queries = freeze_queries()
    if args.only:
        path = one_pass(queries, args.only, args.concurrency if args.only == "conc" else 1)
        print(f"{args.only:5s} n={len(queries)} -> {path}")
        return
    # each mode gets its own process: "cold" means cold, including the interpreter's own
    # warmed-up code paths, which a same-process second pass would quietly reuse.
    import subprocess
    import sys
    for m in ("cold", "warm", "conc"):
        subprocess.run([sys.executable, __file__, "--only", m,
                        "--concurrency", str(args.concurrency)], check=True,
                       env={**os.environ, "PYTHONPATH": os.getcwd()})


def demo():
    qs = freeze_queries("/tmp/_lq.jsonl")
    assert len(qs) == N_IN_DOMAIN + N_SPOKEN + N_OOD == 500
    kinds = {k: sum(1 for q in qs if q["kind"] == k) for k in ("in_domain", "spoken", "ood")}
    assert kinds == {"in_domain": 350, "spoken": 100, "ood": 50}, kinds
    print("d3 query set ok", kinds)


if __name__ == "__main__":
    import sys

    demo() if "--selfcheck" in sys.argv else main()
