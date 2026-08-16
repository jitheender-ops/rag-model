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

ponytail: the index is still exact (dense brute force / sparse postings), not HNSW, so
"index size" is our own serialisation rather than a graph on disk. At this corpus size
exact search is faster than building a graph would be; upgrade path is faiss/qdrant
behind Index.search() -- the six columns and every caller stay as they are.
"""
from __future__ import annotations

import json
import math
import os
import re
import zlib
from collections import defaultdict

M = 32              # recorded in the manifest so the params are visible even though
EF_CONSTRUCTION = 200  # the index is exact -- same values for all strategies
K1, B = 1.2, 0.75   # bm25

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

    def freeze(self):
        self._avglen = (sum(self._len) / len(self._len)) if self._len else 0.0
        if self.vecs and not isinstance(self.vecs[0], dict):
            import numpy as np
            self._mat = np.vstack(self.vecs)
        return self

    def search(self, qvec, k: int = 50) -> list[tuple[str, float]]:
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
        n = len(self.ids)
        scores: dict[int, float] = defaultdict(float)
        for t in set(tokenize(query)):
            post = self._tok.get(t)
            if not post:
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
    print(f"index ok [{BACKEND}] dim={DIM()}", [(c, round(s, 3)) for c, s in hits])


if __name__ == "__main__":
    demo()
