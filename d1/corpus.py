"""Stage 1 of D1: freeze the subset. Sample once with the seed, hash it.

Every strategy reads the identical file -- this is the whole basis of comparability.

Source of docs, in order of preference:
  1. data/raw/*.jsonl written by you   ({"doc_id","lang","script","passages":[...]})
  2. a deterministic synthetic corpus  (seed 42) so the whole pipeline runs offline
     today. It is multilingual on purpose: Hindi/Tamil/Bengali passages end in danda
     (U+0964), which is the exact trap that collapses S2-S4 into S1 if you split on '.'.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random

SEED = 42
CORPUS = "data/corpus.jsonl"
MSMARCO_QUERIES = "data/msmarco_queries.jsonl"   # written by d1/ingest.py

# (lang, script, sentence template, question template, terminator)
LANGS = [
    ("en", "Latin", "The {ent} of {place} was completed in {year} by {who}.",
     "In which year was the {ent} of {place} completed?", "."),
    ("hi", "Devanagari", "{place} का {ent} {year} में {who} द्वारा पूरा किया गया।",
     "{place} का {ent} किस वर्ष में पूरा हुआ?", "।"),
    ("ta", "Tamil", "{place} இன் {ent} {year} ஆம் ஆண்டில் {who} ஆல் நிறைவு செய்யப்பட்டது।",
     "{place} இன் {ent} எந்த ஆண்டில் நிறைவு செய்யப்பட்டது?", "।"),
    ("bn", "Bengali", "{place} এর {ent} {year} সালে {who} দ্বারা সম্পন্ন হয়েছিল।",
     "{place} এর {ent} কোন সালে সম্পন্ন হয়েছিল?", "।"),
]
ENTS = ["bridge", "library", "observatory", "aqueduct", "seawall", "clocktower",
        "granary", "foundry", "planetarium", "arcade"]
PLACES = ["Vasco", "Panaji", "Margao", "Mapusa", "Ponda", "Curchorem", "Bicholim",
          "Canacona", "Sanquelim", "Valpoi"]
WHO = ["the port authority", "a merchant guild", "the state works board",
       "a cooperative trust", "the railway company"]
FILLER = ["Local records describe the surrounding district in some detail",
          "Maintenance schedules were revised several times",
          "The site remains open to visitors on weekdays",
          "Rainfall in the region shapes the annual repair cycle",
          "A second survey confirmed the original measurements"]


def _synth(n_docs: int, rng: random.Random) -> list[dict]:
    docs = []
    for d in range(n_docs):
        lang, script, tmpl, _, term = LANGS[d % len(LANGS)]
        passages = []
        for p in range(rng.randint(4, 7)):
            ent, place = rng.choice(ENTS), rng.choice(PLACES)
            fact = tmpl.format(ent=ent, place=place, year=1850 + rng.randint(0, 160),
                               who=rng.choice(WHO))
            noise = " ".join(f"{rng.choice(FILLER)}{term}" for _ in range(rng.randint(2, 4)))
            passages.append({"pid": f"d{d}:p{p}", "text": f"{fact} {noise}",
                             "ent": ent, "place": place})
        docs.append({"doc_id": f"d{d}", "lang": lang, "script": script, "passages": passages})
    return docs


def _from_raw(paths: list[str]) -> list[dict]:
    docs = []
    for p in sorted(paths):
        with open(p, encoding="utf-8") as fh:
            docs += [json.loads(l) for l in fh if l.strip()]
    return docs


def freeze(n_docs: int = 300, out: str = CORPUS, use_raw: bool = True) -> tuple[str, int]:
    """Write corpus.jsonl and return (sha256, n_docs). Idempotent for a given seed.

    data/raw/*.jsonl (the MSMARCO-XI ingest) wins whenever it exists; use_raw=False forces
    the synthetic corpus, which is what the self-checks run against."""
    rng = random.Random(SEED)
    raw = glob.glob("data/raw/*.jsonl") if use_raw else []
    docs = _from_raw(raw) if raw else _synth(n_docs, rng)
    rng.shuffle(docs)
    docs = docs[:n_docs]        # the ingest writes thousands; the shuffle-then-slice is the
                                # sample, and it is the same sample for every strategy
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for d in docs:
            d["text"] = " ".join(p["text"] for p in d["passages"])
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    return sha(out), len(docs)


def sha(path: str = CORPUS) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def load(path: str = CORPUS) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


FROZEN = "data/frozen.json"


def _frozen() -> dict:
    return json.load(open(FROZEN)) if os.path.exists(FROZEN) else {}


def bind(path: str):
    """Record which corpus a frozen query set was drawn from."""
    reg = _frozen()
    reg[path] = sha()
    os.makedirs(os.path.dirname(FROZEN), exist_ok=True)
    with open(FROZEN, "w") as fh:
        json.dump(reg, fh, indent=2)


def check_frozen(path: str):
    """A frozen set outliving its corpus is silent nonsense: the qrels point at passage ids
    that no longer exist. Fail loudly instead."""
    want = _frozen().get(path)
    if want and want != sha():
        raise SystemExit(
            f"{path} was frozen against corpus {want}, current corpus is {sha()}.\n"
            f"Either restore the corpus or delete {path} (and its row in {FROZEN}) "
            f"to re-freeze -- do not mix them.")


def _row_id(q: dict) -> str:
    """`q:hi:1102432` -> `1102432`. The corpus is parallel, so this id is what identifies
    the same MS MARCO row across hi/ta/bn/en."""
    return q["qid"].split(":")[-1]


def real_queries(docs: list[dict] | None = None) -> list[dict]:
    """Real MS MARCO queries whose gold passages survived into the frozen corpus.

    Empty when the ingest was never run -- callers fall back to their template generators,
    which is what keeps the whole repo runnable with no downloads."""
    if not os.path.exists(MSMARCO_QUERIES):
        return []
    pids = {p["pid"] for d in (load() if docs is None else docs) for p in d["passages"]}
    out = []
    with open(MSMARCO_QUERIES, encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line)
            q["qrels"] = [p for p in q["qrels"] if p in pids]
            if q["qrels"] and q["query"].strip():
                out.append(q)
    return out


def held_out_queries(docs: list[dict] | None = None) -> list[dict]:
    """Real queries whose gold passages are in NO frozen doc, in ANY language.

    The any-language part is the whole point: the corpus is parallel, so dropping only the
    same-language doc leaves the fact answerable from its English twin -- and a row labelled
    unanswerable that is in fact answerable poisons D4's abstention numbers in the flattering
    direction."""
    if not os.path.exists(MSMARCO_QUERIES):
        return []
    kept = {d["doc_id"].split(":", 1)[-1] for d in (load() if docs is None else docs)}
    out = []
    with open(MSMARCO_QUERIES, encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line)
            if _row_id(q) not in kept and q["query"].strip():
                out.append(q)
    return out


def _synth_queries(docs: list[dict]) -> list[dict]:
    """Fallback query set for the synthetic corpus (no MSMARCO ingest present)."""
    cand = []
    for d in docs:
        _, _, _, qt, _ = next(x for x in LANGS if x[0] == d["lang"])
        for p in d["passages"]:
            if "ent" in p:
                cand.append({"qid": f"q:{p['pid']}", "lang": d["lang"],
                             "query": qt.format(ent=p["ent"], place=p["place"]),
                             "qrels": [p["pid"]]})
    return cand


def make_queries(docs: list[dict], n: int, out: str) -> int:
    """Held-out query set + passage-level qrels, frozen to a file and committed.

    Held out before anything is tuned: the sample is drawn from the frozen corpus
    with its own seed and never regenerated at run time.

    With MSMARCO-XI ingested these are real queries with human `is_selected` qrels; the
    template generator is only the fallback for the synthetic corpus.
    """
    if os.path.exists(out):  # frozen means frozen
        check_frozen(out)
        with open(out, encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    rng = random.Random(SEED + 1)
    # Real MS MARCO queries with human is_selected qrels when the ingest has been run;
    # the template generator is the fallback for the synthetic corpus.
    cand = real_queries(docs) or _synth_queries(docs)
    rng.shuffle(cand)
    cand = cand[:n]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for q in cand:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    bind(out)
    return len(cand)


def demo():
    for f in ("/tmp/_c.jsonl", "/tmp/_c2.jsonl", "/tmp/_q.jsonl"):
        if os.path.exists(f):
            os.remove(f)
    s, n = freeze(n_docs=8, out="/tmp/_c.jsonl", use_raw=False)
    assert n == 8 and len(s) == 16
    assert freeze(n_docs=8, out="/tmp/_c2.jsonl", use_raw=False)[0] == s, \
        "corpus freeze is not deterministic"
    docs = load("/tmp/_c.jsonl")
    assert all(p["text"] for d in docs for p in d["passages"])
    global MSMARCO_QUERIES
    keep = MSMARCO_QUERIES

    MSMARCO_QUERIES = "/tmp/_none.jsonl"          # template fallback path
    n_q = make_queries(docs, 10, "/tmp/_q.jsonl")
    assert n_q == 10

    # real-qrels path: a query whose gold passage is not in the frozen corpus is dropped,
    # and a query that keeps at least one gold passage survives with only that one.
    gold = docs[0]["passages"][0]["pid"]
    with open("/tmp/_mq.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"qid": "q1", "lang": "en", "query": "q",
                             "qrels": ["missing:0", gold]}) + "\n")
        fh.write(json.dumps({"qid": "q2", "lang": "en", "query": "q",
                             "qrels": ["missing:1"]}) + "\n")
    MSMARCO_QUERIES = "/tmp/_mq.jsonl"
    os.remove("/tmp/_q.jsonl")
    assert make_queries(docs, 10, "/tmp/_q.jsonl") == 1, "unscoreable query was not dropped"
    kept = json.loads(open("/tmp/_q.jsonl").readline())
    assert kept["qrels"] == [gold], kept
    # held_out_queries must drop a row that survives in ANOTHER language: the corpus is
    # parallel, so an English twin makes an "unanswerable" row answerable.
    parallel_docs = [{"doc_id": "en:7", "lang": "en", "script": "Latin", "passages": []}]
    with open("/tmp/_mq2.jsonl", "w", encoding="utf-8") as fh:
        for lang, rid in (("hi", "7"), ("hi", "8")):
            fh.write(json.dumps({"qid": f"q:{lang}:{rid}", "lang": lang, "query": "q",
                                 "qrels": [f"{lang}:{rid}:0"]}) + "\n")
    MSMARCO_QUERIES = "/tmp/_mq2.jsonl"
    held = [q["qid"] for q in held_out_queries(parallel_docs)]
    assert held == ["q:hi:8"], held
    MSMARCO_QUERIES = keep
    reg = _frozen()
    assert "/tmp/_q.jsonl" in reg
    reg.pop("/tmp/_q.jsonl")                     # keep the real registry clean
    json.dump(reg, open(FROZEN, "w"), indent=2)
    print("corpus ok", s, n, n_q)


if __name__ == "__main__":
    demo()
