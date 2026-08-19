"""Stage 3 of D1: one embedder, one set of index params, for all eight strategies.

If you tune per strategy you are measuring your tuning, not the chunking.

Two backends behind one signature, chosen by EMBEDDER (default: st when it imports):

  st    intfloat/multilingual-e5-small -- 384 dims, 512-token window, asymmetric
        "query: " / "passage: " prefixes, L2-normalised. Multilingual because half the
        corpus is Devanagari/Tamil/Bengali; 512 tokens because a 128-token model would
        silently truncate the s1/s2/s7/s8 chunks and penalise them for the model's
        window rather than for their chunking.
  hash  the zero-dependency fallback: hashed bag of words, sparse cosine. Deterministic,
        needs no download, keeps `make check` fast. EMBEDDER=hash forces it.

Two dense search backends behind one signature, chosen by INDEX (default: exact):

  exact  a float32 matrix and a full dot product. Exact neighbours, linear in corpus size.
         The default because every number in this repo was measured on it, and because at
         12k chunks it costs 1.3 ms -- less than building a graph, and it cannot be wrong.
  hnsw   faiss IndexHNSWFlat, inner product over L2-normalised vectors (= cosine). Sublinear,
         approximate, and the one thing that makes the latency table survive a corpus this
         repo does not fit on disk. `make ann` measures both: where the crossover is, and
         what the approximation costs in recall.

M / EF_CONSTRUCTION / EF_SEARCH below were recorded in the manifest long before there was a
graph to apply them to. They now build one.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import zlib
from collections import defaultdict

M = int(os.getenv("HNSW_M", "32"))                    # graph degree
EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
# efSearch is the recall/latency dial and the only one worth tuning per deployment. 128 is
# where `make ann` measures recall@50 agreement with exact at 0.99+ on this corpus; lower it
# for speed once you have re-measured, do not lower it because a blog post said 64.
EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "128"))
# exact by default: every table in this repo was measured on it, and a default that silently
# changes what "recall@50" means would make the D1 column a comparison of two different
# things. INDEX=hnsw opts in, and `make ann` is what justifies opting in.
ANN = os.getenv("INDEX", "exact")
K1, B = 1.2, 0.75   # bm25
# Skip query terms appearing in more than this fraction of chunks. OFF by default, and the
# default is the measurement rather than the temptation: at 300k chunks BM25 is 99% of
# retrieval (13.8 ms against dense search's 0.12 ms behind an HNSW graph), and a 0.05 ceiling
# makes it 1.8 ms -- 7.5x -- but changes 27% of its own top-10. Fusion dilutes that at weight
# 0.1, so the end-to-end cost is probably small; "probably" is not a number, and nothing in
# this repo ships a quality change on one. Measure it with `make tune` before turning it on.
#
# The ceiling is on document frequency rather than a stopword list, so it adapts to the
# corpus instead of asserting which words are dull -- and note this corpus is nine languages,
# so English stopwords sit near 11% document frequency, which is why anything above 0.10 is
# a no-op here.
# ...and it is on now, because the deploy box answered the question the measurement could
# not. At 300k chunks on Modal's cores BM25 ran 82.7 ms of a 200 ms budget, generate was
# skipped for want of room, and every query came back refused with no citation. The choice
# stopped being "7.5x for 27% of BM25's own top-10" and became "that, or a system that
# refuses everything". Fusion weights BM25 at 0.1, so what moves in the final ranking is a
# fraction of that 27%; the guardrail numbers below it are re-graded with this on.
DF_SKIP = float(os.getenv("BM25_DF_SKIP", "0.05"))

HASH_DIM = 4096     # hashed-backend vector dimensionality
MODEL_NAME = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
BATCH = int(os.getenv("EMBED_BATCH", "128"))


def _pick_backend() -> str:
    want = os.getenv("EMBEDDER", "auto")
    if want in ("hash", "st"):
        return want
    try:
        import sentence_transformers  # noqa: F401
        return "st"
    except ImportError:
        return "hash"


BACKEND = _pick_backend()
_model = None


def model():
    """Loaded once, lazily -- importing torch costs seconds and `make check` never needs it."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def DIM() -> int:
    return model().get_embedding_dimension() if BACKEND == "st" else HASH_DIM

# \w alone drops Indic combining vowel signs (category Mn), which shreds every Tamil,
# Bengali and Devanagari word into fragments -- and silently wrecks retrieval for exactly
# the languages this system exists to serve. The Indic blocks are added back explicitly,
# minus the danda U+0964 / double danda U+0965, which are punctuation.
_WORD = re.compile("[\\w\\u0300-\\u036f\\u0900-\\u0963\\u0966-\\u0dff\\u200c\\u200d]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def script_of(tok: str) -> str:
    """Coarse script bucket of a token's first character."""
    c = ord(tok[0])
    if c < 0x0900:
        return "latin"
    for lo, hi, name in ((0x0900, 0x097F, "deva"), (0x0980, 0x09FF, "beng"),
                         (0x0B80, 0x0BFF, "taml")):
        if lo <= c <= hi:
            return name
    return "other"


def _hash_embed(text: str) -> dict[int, float]:
    """Sparse L2-normalised hashed bag of words. Deterministic across processes
    (zlib.crc32, not hash(), which is salted per run)."""
    tf: dict[int, float] = defaultdict(float)
    for t in tokenize(text):
        tf[zlib.crc32(t.encode()) % HASH_DIM] += 1.0
    if not tf:
        return {}
    for k in tf:
        tf[k] = 1.0 + math.log(tf[k])
    norm = math.sqrt(sum(v * v for v in tf.values()))
    return {k: v / norm for k, v in tf.items()}


def embed(text: str, kind: str = "query"):
    """One text -> one vector. kind is the e5 asymmetric prefix; the hashed backend
    ignores it. Returns a dict (hash) or a normalised float32 array (st)."""
    return embed_many([text], kind)[0]


def embed_many(texts: list[str], kind: str = "passage"):
    """Batched: the only way an ST backend is affordable over 10k chunks x 8 strategies."""
    if BACKEND == "hash":
        return [_hash_embed(t) for t in texts]
    prefixed = [f"{kind}: {t}" for t in texts]
    return list(model().encode(prefixed, batch_size=BATCH, normalize_embeddings=True,
                               show_progress_bar=False, convert_to_numpy=True))


def cosine(a, b) -> float:
    if isinstance(a, dict):
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(k, 0.0) for k, v in a.items())
    return float(a @ b)          # both already L2-normalised


def build_hnsw(mat):
    """A faiss HNSW graph over L2-normalised vectors.

    METRIC_INNER_PRODUCT, not L2: the vectors are already unit length, so inner product is
    cosine and the scores come back on the same scale everything downstream is calibrated
    against. Building an L2 graph instead would return the same neighbours and a score gate 2
    would have to be re-fitted for, which is a silent way to break a threshold.
    """
    import faiss
    # Two OpenMP runtimes in one process -- torch ships libomp, faiss ships its own -- and on
    # macOS the second one to touch a parallel region segfaults the interpreter. Not a
    # hypothetical: `import torch` before build_hnsw() is an immediate SIGSEGV at 12k vectors,
    # and it exits 139 with no traceback, which reads like a corrupt index rather than a
    # linker problem. One thread is also what the serving path wants: this repo already
    # measured that four concurrent forward passes on ten cores is thrashing, not
    # parallelism, and a search that grabs every core does the same thing to its neighbours.
    faiss.omp_set_num_threads(1 if "torch" in sys.modules else faiss.omp_get_max_threads())
    ix = faiss.IndexHNSWFlat(mat.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    ix.hnsw.efConstruction = EF_CONSTRUCTION
    ix.hnsw.efSearch = EF_SEARCH
    ix.add(mat)
    return ix


def load_or_build_hnsw(mat, path: str | None):
    """Read the graph from disk, or build it and leave it there for next time.

    A stale graph is worse than no graph -- it would return neighbours for vectors that are
    no longer in the index -- so the row count is checked before it is trusted, and a
    mismatch rebuilds rather than serving a graph that describes a different corpus.
    """
    import faiss
    if path and os.path.exists(path):
        try:
            ix = faiss.read_index(path)
            if ix.ntotal == len(mat):
                ix.hnsw.efSearch = EF_SEARCH
                return ix
            print(f"  hnsw: {path} has {ix.ntotal} vectors, index has {len(mat)} -- rebuilding")
        except Exception as e:
            print(f"  hnsw: could not read {path} ({e!r}) -- rebuilding")
    ix = build_hnsw(mat)
    if path:
        try:
            faiss.write_index(ix, path)
            print(f"  hnsw: wrote {path}")
        except Exception as e:
            print(f"  hnsw: could not write {path} ({e!r})")
    return ix


class Index:
    """Exact search + BM25 over the same corpus. One instance per strategy.

    Dense backend: a float32 matrix and a full dot product. Sparse backend: an inverted
    list over the hashed dims. Both are exact, so recall differences between strategies
    are the chunking, never the ANN's recall curve.
    """

    def __init__(self):
        self.ids: list[str] = []
        self.payload: list[dict] = []
        self.vecs: list = []
        self._mat = None            # dense backend only, built at freeze()
        self._ann = None            # faiss HNSW graph, built at freeze() when INDEX=hnsw
        self._post: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self._tok: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._len: list[int] = []
        self._avglen = 0.0
        self._byid: dict[str, int] | None = None

    def add(self, cid: str, text: str, payload: dict, vec=None):
        i = len(self.ids)
        self.ids.append(cid)
        self.payload.append(payload)
        v = embed(text, "passage") if vec is None else vec
        self.vecs.append(v)
        if isinstance(v, dict):
            for k, w in v.items():
                self._post[k].append((i, w))
        toks = tokenize(text)
        tf: dict[str, int] = defaultdict(int)
        for t in toks:
            tf[t] += 1
        for t, c in tf.items():
            self._tok[t].append((i, c))
        self._len.append(len(toks))

    def add_many(self, rows: list[tuple[str, str, dict]]):
        """(cid, text, payload) triples, embedded in one batched pass."""
        vecs = embed_many([t for _, t, _ in rows], "passage")
        for (cid, text, payload), v in zip(rows, vecs):
            self.add(cid, text, payload, vec=v)

    def freeze(self, ann_path: str | None = None):
        """ann_path: where the HNSW graph lives on disk.

        Building the graph is a minutes-long job at 300k vectors, and a server that builds
        it at startup pays that on every cold start -- which for a scale-to-zero box is a
        cost the first visitor wears, repeatedly. It is written once next to the vectors it
        indexes and read back in seconds after that.
        """
        self._avglen = (sum(self._len) / len(self._len)) if self._len else 0.0
        if self.vecs and not isinstance(self.vecs[0], dict):
            import numpy as np
            self._mat = np.vstack(self.vecs).astype("float32")
            if ANN == "hnsw":
                self._ann = load_or_build_hnsw(self._mat, ann_path)
        return self

    def search(self, qvec, k: int = 50) -> list[tuple[str, float]]:
        if self._ann is not None:
            import numpy as np
            q = np.ascontiguousarray(np.asarray(qvec, dtype="float32").reshape(1, -1))
            scores, idx = self._ann.search(q, min(k, len(self.ids)))
            # faiss pads a short result with -1; a -1 would index the LAST id and quietly
            # return a real chunk that was never a neighbour
            return [(self.ids[int(i)], float(sc))
                    for sc, i in zip(scores[0], idx[0]) if i >= 0]
        if self._mat is not None:
            import numpy as np
            scores = self._mat @ qvec
            k = min(k, len(scores))
            part = np.argpartition(-scores, k - 1)[:k]
            top = part[np.argsort(-scores[part], kind="stable")]
            return [(self.ids[int(i)], float(scores[i])) for i in top]
        scores: dict[int, float] = defaultdict(float)
        for dim, qw in qvec.items():
            for i, w in self._post.get(dim, ()):
                scores[i] += qw * w
        top_sparse = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.ids[i], s) for i, s in top_sparse]

    def bm25(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        """BM25 over the inverted index, skipping terms that appear nearly everywhere.

        At 12k chunks this loop cost nothing. At 300k it became 99% of retrieval -- dense
        search is 0.12 ms behind an HNSW graph and this was 18 ms -- because a term like
        "is" or "the" has a posting list the length of the corpus and gets walked in full.
        Those are exactly the terms BM25 already decides not to care about: IDF at 50%
        document frequency is 0.4 and falling, so the cost is spent to move nothing.

        DF_SKIP is a ceiling on document frequency, not a stopword list: it adapts to the
        corpus instead of asserting which words are dull, and it is off by default at small
        n, where walking everything is cheaper than deciding not to.
        """
        n = len(self.ids)
        cap = int(n * DF_SKIP)
        scores: dict[int, float] = defaultdict(float)
        for t in set(tokenize(query)):
            post = self._tok.get(t)
            if not post:
                continue
            if cap and len(post) > cap:      # near-universal term: high cost, no signal
                continue
            idf = math.log(1 + (n - len(post) + 0.5) / (len(post) + 0.5))
            for i, tf in post:
                dl = self._len[i] or 1
                scores[i] += idf * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / (self._avglen or 1)))
        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.ids[i], s) for i, s in top]

    def get(self, cid: str) -> dict:
        if self._byid is None:
            self._byid = {c: i for i, c in enumerate(self.ids)}
        return self.payload[self._byid[cid]]

    def save(self, path: str) -> int:
        """Persist vectors + payload store; return size on disk in bytes.

        Dense vectors go to a .npy beside the payload jsonl, so the reported index size is
        vectors + payload in both backends and the column stays comparable."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        size = 0
        if self._mat is not None:
            import numpy as np
            vpath = path.replace(".jsonl", ".npy")
            np.save(vpath, self._mat.astype("float32"))
            size += os.path.getsize(vpath)
            with open(path, "w", encoding="utf-8") as fh:
                for cid, p in zip(self.ids, self.payload):
                    fh.write(json.dumps({"id": cid, "p": p}, ensure_ascii=False) + "\n")
        else:
            with open(path, "w", encoding="utf-8") as fh:
                for cid, v, p in zip(self.ids, self.vecs, self.payload):
                    fh.write(json.dumps({"id": cid,
                                         "v": {str(k): round(w, 5) for k, w in v.items()},
                                         "p": p}, ensure_ascii=False) + "\n")
        return size + os.path.getsize(path)


def demo():
    ix = Index()
    ix.add_many([("a", "the bridge of Vasco was completed in 1901", {"pid": "p1"}),
                 ("b", "rainfall in the region shapes the repair cycle", {"pid": "p2"})])
    ix.freeze()
    q = embed("in which year was the bridge of Vasco completed", "query")
    hits = ix.search(q, k=2)
    assert hits[0][0] == "a", hits
    assert ix.get("b")["pid"] == "p2"
    assert ix.bm25("bridge Vasco")[0][0] == "a"
    v = embed("bridge", "passage")
    assert abs(cosine(v, v) - 1.0) < 1e-5
    assert ix.save("/tmp/_ix.jsonl") > 0

    # the ANN backend must agree with exact on the same vectors, and must not pad its
    # result with faiss's -1 sentinel -- a -1 indexes the LAST id and returns a real chunk
    # that was never a neighbour, which is the one ANN bug that looks like a good answer
    if BACKEND == "st":
        try:
            import faiss  # noqa: F401
        except ImportError:
            faiss = None
        if faiss is not None:
            import numpy as np
            ann = build_hnsw(np.vstack(ix.vecs).astype("float32"))
            sc, idx = ann.search(np.ascontiguousarray(
                np.asarray(q, dtype="float32").reshape(1, -1)), 2)
            assert ix.ids[int(idx[0][0])] == "a", idx
            assert abs(float(sc[0][0]) - hits[0][1]) < 1e-4, (sc[0][0], hits[0][1])
            # k larger than the corpus: every padded slot must be dropped, not indexed
            over = Index()
            over.add_many([("only", "a single chunk", {"pid": "p"})])
            over.freeze()
            over._ann = build_hnsw(np.vstack(over.vecs).astype("float32"))
            got = over.search(embed("single", "query"), k=50)
            assert [c for c, _ in got] == ["only"], got

    print(f"index ok [{BACKEND}/{ANN}] dim={DIM()}", [(c, round(s, 3)) for c, s in hits])


if __name__ == "__main__":
    demo()
