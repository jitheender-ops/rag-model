"""Stage 2 of D1: eight implementations, one interface, no shared state.

Every chunk carries doc_id = the passage it came from -- that is the eval join key,
and the reason the table is valid (score at passage granularity, not chunk granularity).
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from typing import Iterable, Protocol

from d1.index import BACKEND, cosine, embed_many, tokenize

# Sentence terminators including the Devanagari danda U+0964 and double danda U+0965.
# A regex on '.' alone silently produces one giant chunk per Hindi doc and S2-S4
# collapse into S1 -- that is the trap this line exists to defuse.
_SENT = re.compile(r"[^.!?।॥\n]+[.!?।॥]?", re.UNICODE)


@dataclass(frozen=True)
class Chunk:
    id: str            # f"{strategy}:{doc_id}:{ordinal}"
    doc_id: str        # the passage it came from -- the eval join key
    parent_id: str | None   # for S3/S6 window expansion at serve time
    text: str
    span: tuple[int, int]   # char offsets into the source, for the verifier
    lang: str
    script: str
    ordinal: int
    n_tokens: int


@dataclass
class Doc:
    doc_id: str
    lang: str
    script: str
    passages: list[dict]
    text: str = ""


class Chunker(Protocol):
    key: str
    params: dict

    def chunk(self, doc: Doc) -> Iterable[Chunk]: ...


def sentences(text: str) -> list[tuple[str, int, int]]:
    out = []
    for m in _SENT.finditer(text):
        s = m.group().strip()
        if s:
            out.append((s, m.start(), m.end()))
    return out


def _mk(key, doc, p, ordinal, text, span, parent=None) -> Chunk:
    return Chunk(id=f"{key}:{p['pid']}:{ordinal}", doc_id=p["pid"], parent_id=parent,
                 text=text, span=span, lang=doc.lang, script=doc.script,
                 ordinal=ordinal, n_tokens=len(tokenize(text)))


class _Base:
    key = "s0"
    params: dict = {}

    def chunk(self, doc: Doc) -> Iterable[Chunk]:
        raise NotImplementedError


class S1Fixed(_Base):
    """Fixed 256/64 over tokens."""
    key, params = "s1 fixed 256/64", {"size": 256, "overlap": 64}

    def chunk(self, doc):
        size, ov = self.params["size"], self.params["overlap"]
        for p in doc.passages:
            words = p["text"].split()
            i = o = 0
            while i < len(words):
                w = words[i:i + size]
                text = " ".join(w)
                start = p["text"].find(w[0])
                yield _mk(self.key, doc, p, o, text, (start, start + len(text)))
                o += 1
                if i + size >= len(words):
                    break
                i += size - ov


class S2Recursive(_Base):
    """Recursive 320/80: split on paragraph, then sentence, then hard-cut."""
    key, params = "s2 recursive 320/80", {"size": 320, "overlap": 80}

    def chunk(self, doc):
        size, ov = self.params["size"], self.params["overlap"]
        for p in doc.passages:
            buf, buf_start, o = [], 0, 0
            for s, st, en in sentences(p["text"]):
                if buf and sum(len(x.split()) for x in buf) + len(s.split()) > size:
                    text = " ".join(buf)
                    yield _mk(self.key, doc, p, o, text, (buf_start, buf_start + len(text)))
                    o += 1
                    keep, n = [], 0
                    for x in reversed(buf):          # sentence-aligned overlap
                        n += len(x.split())
                        keep.insert(0, x)
                        if n >= ov:
                            break
                    buf, buf_start = keep, st
                else:
                    if not buf:
                        buf_start = st
                    buf.append(s)
            if buf:
                text = " ".join(buf)
                yield _mk(self.key, doc, p, o, text, (buf_start, buf_start + len(text)))


class S3SentenceWindow(_Base):
    """One sentence per chunk; parent_id points at the passage for window expansion."""
    key, params = "s3 sentence-window", {"window": 1}

    def chunk(self, doc):
        for p in doc.passages:
            for o, (s, st, en) in enumerate(sentences(p["text"])):
                yield _mk(self.key, doc, p, o, s, (st, en), parent=p["pid"])


class S4SemanticDrift(_Base):
    """Cut where consecutive-sentence similarity drops below the threshold."""
    # the drift threshold is a property of the embedder's cosine scale, not of the corpus:
    # hashed bag-of-words spreads 0..1, e5 packs unrelated text around 0.7-0.8.
    key = "s4 semantic drift"
    params = {"threshold": 0.45 if BACKEND == "hash" else 0.86, "max_tokens": 320}

    def chunk(self, doc):
        th, mx = self.params["threshold"], self.params["max_tokens"]
        for p in doc.passages:
            sents = sentences(p["text"])
            # one batched call per passage: with a transformer backend, embedding sentences
            # one at a time is the difference between minutes and hours.
            vecs = embed_many([s for s, _, _ in sents], "passage")
            buf, start, o, prev = [], 0, 0, None
            for (s, st, en), v in zip(sents, vecs):
                drift = prev is not None and cosine(prev, v) < th
                too_big = sum(len(x.split()) for x in buf) > mx
                if buf and (drift or too_big):
                    text = " ".join(buf)
                    yield _mk(self.key, doc, p, o, text, (start, start + len(text)))
                    o, buf, start = o + 1, [], st
                if not buf:
                    start = st
                buf.append(s)
                prev = v
            if buf:
                text = " ".join(buf)
                yield _mk(self.key, doc, p, o, text, (start, start + len(text)))


class S5Proposition(_Base):
    """Proposition decomposition. Real version is millions of LLM calls, so it runs on a
    10% slice and the row is marked "sampled" in the report -- said out loud, not hidden.

    ponytail: the decomposer here is clause-splitting on connectives, not an LLM. Ceiling:
    propositions are coarser than an LLM's. Upgrade path: replace decompose() with a
    batched LLM call at temperature 0 and keep the 10% slice.
    """
    key = "s5 proposition"
    params = {"sample_frac": 0.10, "decomposer": "rule-based"}
    sampled = True
    _SPLIT = re.compile(r"\s*(?:,| and | but | तथा | और | மற்றும் | এবং )\s*", re.UNICODE)

    def chunk(self, doc):
        # deterministic 10% slice. crc32, not int(doc_id[1:]): real corpora have ids like
        # "en:137728", and hash() is salted per process so the slice would move between runs.
        if zlib.crc32(doc.doc_id.encode()) % 10 != 0:
            return
        for p in doc.passages:
            o = 0
            for s, st, en in sentences(p["text"]):
                for part in self._SPLIT.split(s):
                    part = part.strip()
                    if len(part.split()) >= 3:
                        yield _mk(self.key, doc, p, o, part, (st, en))
                        o += 1


class S6ParentDocument(_Base):
    """Small children for retrieval, parent_id = the whole passage for serve-time expansion."""
    key, params = "s6 parent-document", {"child_size": 96, "parent": "passage"}

    def chunk(self, doc):
        size = self.params["child_size"]
        for p in doc.passages:
            words = p["text"].split()
            for o in range(0, max(1, len(words)), size):
                text = " ".join(words[o:o + size])
                if text:
                    yield _mk(self.key, doc, p, o // size, text, (0, len(p["text"])),
                              parent=p["pid"])


class S7MetadataFiltered(_Base):
    """S2 chunks with lang/script/doc metadata prefixed into the embedded text, so the
    filter is part of the vector rather than a post-filter that shrinks recall."""
    key, params = "s7 metadata-filtered", {"base": "s2", "fields": ["lang", "script", "doc_id"]}

    def __init__(self):
        self._base = S2Recursive()

    def chunk(self, doc):
        for c in self._base.chunk(doc):
            head = f"[{doc.lang}|{doc.script}|{doc.doc_id}] "
            yield Chunk(id=c.id.replace(S2Recursive.key, self.key), doc_id=c.doc_id,
                        parent_id=c.parent_id, text=head + c.text, span=c.span,
                        lang=c.lang, script=c.script, ordinal=c.ordinal,
                        n_tokens=c.n_tokens)


class S8MultiGranularity(_Base):
    """Union of sentence + fixed + whole-passage. Biggest index, best recall ceiling."""
    key, params = "s8 multi-granularity", {"levels": ["sentence", "256", "passage"]}

    def __init__(self):
        self._s1, self._s3 = S1Fixed(), S3SentenceWindow()

    def chunk(self, doc):
        o = 0
        for sub in (self._s3, self._s1):
            for c in sub.chunk(doc):
                yield Chunk(id=f"{self.key}:{c.doc_id}:{o}", doc_id=c.doc_id,
                            parent_id=c.doc_id, text=c.text, span=c.span, lang=c.lang,
                            script=c.script, ordinal=o, n_tokens=c.n_tokens)
                o += 1
        for p in doc.passages:
            yield _mk(self.key, doc, p, o, p["text"], (0, len(p["text"])), parent=p["pid"])
            o += 1


ALL: list[type[_Base]] = [S1Fixed, S2Recursive, S3SentenceWindow, S4SemanticDrift,
                          S5Proposition, S6ParentDocument, S7MetadataFiltered,
                          S8MultiGranularity]


def demo():
    hi = Doc("d0", "hi", "Devanagari", [{"pid": "d0:p0",
             "text": "पुल १९०१ में पूरा हुआ। दूसरा वाक्य यहाँ है। तीसरा वाक्य भी है।"}])
    en = Doc("d10", "en", "Latin", [{"pid": "d10:p0",
             "text": "The bridge was completed in 1901. Rainfall shapes repairs. Visitors come on weekdays."}])
    for cls in ALL:
        for doc in (hi, en):
            cs = list(cls().chunk(doc))
            if getattr(cls, "sampled", False) and not cs:
                continue                     # out of the 10% slice, which is the point of it
            assert cs, f"{cls.key} produced nothing for {doc.doc_id}"
            assert all(c.doc_id.startswith(doc.doc_id + ":") for c in cs), cls.key
    # the danda trap: Hindi must not collapse to one chunk per doc
    n_hi = len(list(S3SentenceWindow().chunk(hi)))
    assert n_hi == 3, f"Indic sentence splitting broken: {n_hi} chunks"
    assert len(list(S1Fixed().chunk(en))) >= 1
    assert list(S6ParentDocument().chunk(en))[0].parent_id == "d10:p0"
    # the 10% slice must hold for real corpus ids too ("en:137728"), not just "d7"
    ids = [f"en:{i}" for i in range(1000)]
    inside = [i for i in ids if zlib.crc32(i.encode()) % 10 == 0]
    assert 0.05 < len(inside) / len(ids) < 0.20, f"slice is {len(inside)/10:.0f}%, not ~10%"

    def _doc(i):
        return Doc(i, "en", "Latin", [{"pid": f"{i}:p0", "text": en.passages[0]["text"]}])

    assert list(S5Proposition().chunk(_doc(inside[0]))), "in-slice doc produced nothing"
    outside = next(i for i in ids if i not in set(inside))
    assert not list(S5Proposition().chunk(_doc(outside))), "10% slice is not being applied"
    print("chunkers ok", {c.key: len(list(c().chunk(en))) for c in ALL})


if __name__ == "__main__":
    demo()
