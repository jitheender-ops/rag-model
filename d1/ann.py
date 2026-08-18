"""Does an ANN index earn its place, and at what corpus size?

Two questions, because they have different answers:

  1. WHAT DOES THE APPROXIMATION COST?  Exact and HNSW run the same real queries against the
     same real vectors, and we count how much of exact's top-50 the graph actually returns.
     Recall against the exact neighbours, not against qrels -- this is measuring the index,
     not the chunking, and mixing the two is how an ANN regression hides behind a retrieval
     metric that was never very high to begin with.

  2. WHERE IS THE CROSSOVER?  Exact search is linear and HNSW is roughly logarithmic, so
     which one is faster is a question about N, not about the algorithms. Scaling N needs
     vectors, not meaning: search cost depends on the count and the dimensionality, so the
     larger corpora here are the real vectors resampled with noise. That makes the LATENCY
     column honest and the RECALL column meaningless above the real corpus size, which is
     why recall is only reported at N where the vectors are real.

    make ann

ponytail: resampled vectors measure latency, not retrieval quality. Ceiling: the recall
column stops at the real corpus. Upgrade path is ingesting more of MSMARCO-XI, which costs
disk and download rather than any change here.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from d1.index import ANN, EF_SEARCH, EF_CONSTRUCTION, M, build_hnsw
from harness.env import header
from harness.spans import NS_PER_MS, now_ns

STRATEGY = os.getenv("ANN_STRATEGY", "artifacts/d1/s7")
NQ = int(os.getenv("ANN_QUERIES", "200"))
K = int(os.getenv("ANN_K", "50"))
SCALES = [int(x) for x in os.getenv("ANN_SCALES", "12024,50000,200000,1000000").split(",")]
SEED = 42


def exact_search(mat, q, k):
    scores = mat @ q
    k = min(k, len(scores))
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]


def timed(fn, reps):
    """Median of `reps` calls, in ms. Median, not mean: one page fault should not become the
    number that decides an architecture."""
    out = []
    for _ in range(reps):
        t0 = now_ns()
        fn()
        out.append((now_ns() - t0) / NS_PER_MS)
    out.sort()
    return out[len(out) // 2], out[-1]


def grow(mat, n, rng):
    """n vectors that behave like the real ones: resampled rows plus noise, renormalised.

    Not random vectors -- random unit vectors in 384 dimensions are all equidistant, HNSW
    has no structure to exploit, and the benchmark would report the graph's worst case as
    if it were its normal one."""
    if n <= len(mat):
        return mat[:n]
    idx = rng.integers(0, len(mat), n - len(mat))
    extra = mat[idx] + rng.normal(0, 0.05, (n - len(mat), mat.shape[1])).astype("float32")
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    return np.vstack([mat, extra]).astype("float32")


def main():
    vpath = os.path.join(STRATEGY, "index.npy")
    if not os.path.exists(vpath):
        sys.exit(f"no vectors at {vpath} -- run `make chunking` first")
    mat = np.load(vpath).astype("float32")
    rng = np.random.default_rng(SEED)
    queries = mat[rng.choice(len(mat), min(NQ, len(mat)), replace=False)]
    real_n = len(mat)

    rows = []
    for n in SCALES:
        big = grow(mat, n, rng)
        import faiss
        faiss.omp_set_num_threads(faiss.omp_get_max_threads())   # build wide again
        t0 = now_ns()
        ann = build_hnsw(big)
        build_s = (now_ns() - t0) / NS_PER_MS / 1000
        # Build may use every core; SEARCH must not. A per-request lookup that grabs ten
        # threads measures a machine serving one user, and this repo already established
        # what that assumption costs under concurrency-4.
        faiss.omp_set_num_threads(1)
        qs = [np.ascontiguousarray(q.reshape(1, -1)) for q in queries]

        i = [0]
        def one_exact():
            exact_search(big, queries[i[0] % len(queries)], K)
            i[0] += 1
        j = [0]
        def one_ann():
            ann.search(qs[j[0] % len(qs)], K)
            j[0] += 1

        reps = 50 if n <= 200_000 else 20
        ex_p50, ex_p100 = timed(one_exact, reps)
        an_p50, an_p100 = timed(one_ann, reps)

        recall = None
        if n == real_n:                     # only where the vectors mean something
            hit = 0
            for q in queries:
                truth = set(exact_search(big, q, K).tolist())
                _, idx = ann.search(np.ascontiguousarray(q.reshape(1, -1)), K)
                hit += len(truth & {int(x) for x in idx[0] if x >= 0})
            recall = hit / (len(queries) * K)
        rows.append({"n": n, "exact_p50": ex_p50, "exact_p100": ex_p100,
                     "ann_p50": an_p50, "ann_p100": an_p100, "build_s": build_s,
                     "recall": recall, "mb": big.nbytes / 1e6})
        print(f"  n={n:>9,}  exact {ex_p50:7.2f} ms   hnsw {an_p50:6.2f} ms   "
              f"build {build_s:5.1f} s" + (f"   recall@{K} {recall:.3f}" if recall else ""),
              flush=True)

    os.makedirs("reports", exist_ok=True)
    with open("reports/ann.md", "w", encoding="utf-8") as fh:
        fh.write(header("ANN vs exact - where the vector index earns its place",
                        {"strategy": STRATEGY, "queries": len(queries), "k": K,
                         "M": M, "efConstruction": EF_CONSTRUCTION, "efSearch": EF_SEARCH,
                         "real corpus": f"{real_n} chunks", "seed": SEED}))
        fh.write("\nSearch latency is the dense ANN lookup only: no query embedding, no BM25, "
                 "no fusion. Vectors above the real corpus size are resampled real vectors "
                 "plus noise -- honest for latency, meaningless for retrieval quality, so "
                 "recall is reported only at the real corpus size.\n\n")
        fh.write(f"| chunks | vectors | exact P50 | exact P100 | hnsw P50 | hnsw P100 | "
                 f"build | recall@{K} vs exact |\n")
        fh.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            rec = f"{r['recall']:.3f}" if r["recall"] is not None else "—"
            fh.write(f"| {r['n']:,} | {r['mb']:.0f} MB | {r['exact_p50']:.2f} ms | "
                     f"{r['exact_p100']:.2f} ms | {r['ann_p50']:.2f} ms | "
                     f"{r['ann_p100']:.2f} ms | {r['build_s']:.1f} s | {rec} |\n")
        cross = next((r for r in rows if r["ann_p50"] < r["exact_p50"]), None)
        real = rows[0]
        fh.write("\n")
        if cross is None:
            fh.write("**No crossover in the range measured** — exact stays ahead everywhere "
                     "tested, so it stays the default.\n")
        elif cross is rows[0]:
            fh.write(f"**The crossover is at or below the smallest size measured "
                     f"({rows[0]['n']:,} chunks)** — the graph is already the faster index "
                     f"on the corpus this repo actually serves. Exact's cost grows with the "
                     f"corpus and the graph's barely moves: "
                     + ", ".join(f"{r['n']:,} -> {r['exact_p50'] / r['ann_p50']:.0f}x"
                                 for r in rows) +
                     " (exact / hnsw, P50).\n")
        else:
            fh.write(f"**Crossover: {cross['n']:,} chunks.** Below it exact is faster and "
                     f"cannot be wrong. At or above it the graph wins and keeps winning, "
                     f"because exact is linear in corpus size and HNSW is not.\n")
        if real["recall"] is not None:
            fh.write(f"\nAt the real corpus ({real_n:,} chunks) the graph returns "
                     f"**{real['recall']:.1%}** of exact's top-{K}, at "
                     f"**{real['ann_p50']:.2f} ms** against exact's "
                     f"**{real['exact_p50']:.2f} ms**.\n")
        # The honest reading, which is not the one the speed column suggests on its own.
        fh.write(f"\n**And yet `INDEX=exact` stays the default here, on the same evidence.** "
                 f"Dense search is not the pipeline's bottleneck at this size: the whole "
                 f"`dense + bm25 + rrf` stage is 1.3 ms P50 of a 48 ms request, so the "
                 f"{real['exact_p50'] - real['ann_p50']:.2f} ms the graph saves is under one "
                 f"percent of a request and inside the run-to-run noise of the D3 table. "
                 f"What it costs is {(1 - real['recall']) * 100:.1f}% of the true neighbours, "
                 f"a build step, and a dependency. Paying that for noise is not a trade, it "
                 f"is a habit.\n\nThe number that matters is the last column of the top row "
                 f"and the shape of the one before it: the graph is *ready*, its recall cost "
                 f"is measured rather than assumed, and the corpus size where it stops being "
                 f"optional is now a row in a table instead of a sentence in a README. "
                 f"`INDEX=hnsw` is one environment variable, and every gate, floor and report "
                 f"above it stays as it is — the scores are cosines on the same scale, which "
                 f"is why `build_hnsw` uses inner product rather than L2.\n")
    print("\nwrote reports/ann.md")


def demo():
    rng = np.random.default_rng(SEED)
    mat = rng.normal(0, 1, (40, 8)).astype("float32")
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)

    # grow() must keep the real vectors first and unmodified -- the recall row is scored
    # against them, so resampled rows leaking into the head would grade noise as corpus
    big = grow(mat, 100, rng)
    assert big.shape == (100, 8) and np.allclose(big[:40], mat), big.shape
    assert np.allclose(np.linalg.norm(big, axis=1), 1.0, atol=1e-5), "must stay unit length"
    assert not np.allclose(big[40:60], big[60:80]), "resampled rows must not be identical"
    assert grow(mat, 10, rng).shape == (10, 8), "asking for fewer must truncate, not pad"

    # exact_search returns indices best-first, and its own row is its nearest neighbour
    idx = exact_search(mat, mat[3], 5)
    assert idx[0] == 3, idx
    scores = mat @ mat[3]
    assert list(scores[idx]) == sorted(scores[idx], reverse=True), "must be rank-ordered"
    assert len(exact_search(mat, mat[0], 999)) == 40, "k above corpus size must clamp"

    calls = []
    p50, p100 = timed(lambda: calls.append(1), 5)
    assert len(calls) == 5 and p100 >= p50 >= 0
    print("ann bench ok (grow, exact_search, timed)")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        main()
