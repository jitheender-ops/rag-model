"""The serving path over HTTP, for the browser demo.

    make serve                 # http://localhost:8000  -- page and API on one origin
    make serve PORT=9000

    GET  /             the demo page (same origin as /ask, so no CORS to get wrong)
    GET  /health       readiness: is the index loaded, which one, which floors
    POST /ask          json {"text": "..."}  or  multipart with an `audio` part
    OPTIONS /ask       preflight, for when the page is opened from somewhere else

The response shape is the contract printed inside the demo page itself, not one invented
here -- transcript, answer, abstained, abstain_gate, citations, timings_ms, total_ms,
stt_ms, degraded.

TWO THINGS THIS FILE EXISTS TO GET RIGHT
The index and the encoder are loaded once, at startup, before the socket is opened. A
request that pays a 10 s model load is not a request this system's 200 ms budget describes,
and a server that lazily loads on the first call publishes that lie to whoever tries it first.

`stt_ms` is reported beside `total_ms` and never folded into it. t0 is the instant we hold a
final transcript; the transcription that produced it is a separate, larger number, and one
combined figure would be the most misleading thing this API could return.

ponytail: stdlib http.server, threading, no framework. One endpoint, JSON in and out --
FastAPI would save about forty lines and cost the "runs on stdlib alone" claim that the rest
of the repo keeps. Upgrade path: put uvicorn in front of the same `handle()` if you need
real concurrency, keep-alive or TLS.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from service.pipeline import SCORE_FLOOR, COVERAGE_FLOOR, answer, load_index

PAGE = os.getenv("PAGE", "web/index.html")
MAX_BODY = int(os.getenv("MAX_BODY", str(25 * 1024 * 1024)))     # 25 MB of audio is plenty
N_CITATIONS = 4

STATE: dict = {}          # index, texts, parents, strategy -- filled by boot()


# ---------- the contract ----------

# our span names -> the page's timing keys. `cache` and `assemble` have no span of their
# own: the cache is a dict lookup inside embed_query and context assembly is inside
# generate. They are reported as 0.0 rather than dropped, because the page draws a fixed
# set of bars -- and the untranslated spans go out as `spans_ms` beside them so nothing
# this server measured is only visible after a rename.
TIMING_KEYS = {"input_guards": "guard", "embed_query": "embed", "retrieve": "retrieve",
               "rerank": "rerank", "generate": "generate", "verify": "verify"}


def citations(trace, ctx_hits, texts) -> list[dict]:
    """The cited chunk first, then the runners-up it was chosen from."""
    cited = trace.meta.get("cited")
    out, seen = [], set()
    for cid, score in ([(cited, 1.0)] if cited else []) + list(ctx_hits):
        if not cid or cid in seen:
            continue
        seen.add(cid)
        from service.pipeline import display
        out.append({"chunk_id": cid, "text": display(texts.get(cid, ""))[:400],
                    "score": round(float(score), 4)})
        if len(out) >= N_CITATIONS:
            break
    return out


def spoken(trace) -> dict:
    """Speak the result, or say why not. Never fails the request over it.

    TTS is after t1 and outside the budget, so its cost is reported next to total_ms and
    never inside it -- and a vendor hiccup here must not turn an answer that was produced
    in 8 ms into an HTTP 500."""
    from tts import sarvam as tts
    try:
        wav, ms, provider = tts.speak(trace.meta, script_lang(tts.line_for(trace.meta)))
        import base64
        return {"audio_b64": base64.b64encode(wav).decode(), "audio_mime": "audio/wav",
                "spoken_text": tts.line_for(trace.meta), "tts_ms": round(ms, 1),
                "tts_provider": provider, "tts_error": None}
    except Exception as e:
        return {"audio_b64": None, "audio_mime": None,
                "spoken_text": tts.line_for(trace.meta), "tts_ms": None,
                "tts_provider": None, "tts_error": str(e)[:200]}


def to_payload(trace, hits, texts, stt_ms: float | None, stt_provider: str = "") -> dict:
    m = trace.meta
    timings = {v: round(trace.spans.get(k, 0.0), 3) for k, v in TIMING_KEYS.items()}
    timings["cache"] = 0.0
    timings["assemble"] = 0.0
    return {
        "transcript": m["query"],
        "answer": m["answer"],
        "abstained": bool(m["abstain"]),
        "abstain_gate": m["gate"],
        "abstain_reason": m["reason"],
        "citations": citations(trace, hits, texts),
        "timings_ms": timings,
        "spans_ms": {k: round(v, 3) for k, v in trace.spans.items()},
        "total_ms": round(trace.total_ms, 3),
        # measured for a real round trip, null for typed text, and never added to total_ms
        "stt_ms": None if stt_ms is None else round(stt_ms, 1),
        "stt_provider": stt_provider or None,
        "degraded": (m.get("budget") or {}).get("degraded", []),
        "extractive": m.get("extractive", False),
        "cache_hit": m.get("cache_hit", False),
        "lexical_only": m.get("lexical_only", False),
        "budget_ms": (m.get("budget") or {}).get("total_ms"),
        # excluded legs, both of them, on either side of the measured window:
        #   stt_ms   before t0        tts_ms   after t1
        "excluded_ms": {"stt": stt_ms, "tts": None},
        "within_budget": trace.total_ms <= ((m.get("budget") or {}).get("total_ms") or 200.0),
    }


def script_lang(text: str) -> str:
    """Which voice to use: the script of the text actually being spoken.

    Not the question's script, which was the first thing tried and is wrong for exactly the
    queries this system exists to serve: "consensus definition কী" is majority-Latin, and
    its answer is a Bengali passage. Read the string you are about to say, and a refusal --
    whose canned line is English -- correctly gets an English voice."""
    from d1.index import script_of, tokenize
    seen = {}
    for tok in tokenize(text):
        seen[script_of(tok)] = seen.get(script_of(tok), 0) + 1
    best = max(seen, key=lambda k: seen[k]) if seen else "latin"
    return {"deva": "hi", "beng": "bn", "taml": "ta"}.get(best, "en")


# ---------- request parsing ----------

def parse_multipart(body: bytes, content_type: str) -> dict[str, bytes]:
    """Named parts out of a multipart/form-data body.

    email.parser rather than a hand-rolled boundary split: cgi is gone in 3.13, and getting
    multipart subtly wrong is how you truncate someone's audio at the first CRLF."""
    from email.parser import BytesParser
    msg = BytesParser().parsebytes(b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    parts = {}
    for part in msg.walk() if msg.is_multipart() else []:
        name = part.get_param("name", header="content-disposition")
        if name:
            parts[name] = part.get_payload(decode=True) or b""
    return parts


def question_from(body: bytes, content_type: str) -> tuple[str, float | None, str, str]:
    """(query, stt_ms, stt_provider, filename). Raises ValueError with a usable message."""
    if content_type.startswith("multipart/form-data"):
        parts = parse_multipart(body, content_type)
        audio = parts.get("audio")
        if not audio:
            raise ValueError("multipart body has no `audio` part")
        from stt import sarvam
        mime = sarvam.content_type_of(audio, "clip.webm")
        name = f"clip.{sarvam.ext_for(mime)}"
        t = sarvam.provider().transcribe(audio, filename=name, content_type=mime)
        if not t["text"].strip():
            raise ValueError("the clip transcribed to nothing -- no question to ask")
        return t["text"], t["stt_ms"], t["provider"], name
    payload = json.loads(body or b"{}")
    text = (payload.get("text") or payload.get("query") or "").strip()
    if not text:
        raise ValueError('send {"text": "..."} or a multipart body with an `audio` part')
    return text, None, "", ""


def wants_speech(body: bytes, content_type: str, path: str) -> bool:
    """?speak=1, or {"speak": true}. Off by default: every spoken answer is a vendor call
    and a second of latency, and the API should not spend either without being asked."""
    if "speak=1" in path or "speak=true" in path:
        return True
    if content_type.startswith("application/json"):
        try:
            return bool(json.loads(body or b"{}").get("speak"))
        except ValueError:
            return False
    # the whole body, not the first 4 KB: FormData puts the audio part first and a clip is
    # megabytes, so a windowed search would never see the flag that follows it
    return b'name="speak"' in body


# ---------- server ----------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "mic-rag/1.0"

    def log_message(self, fmt, *args):        # one tidy line per request, not apache noise
        sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, code: int, payload: dict | bytes, ctype="application/json"):
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        # the page may be opened from a file:// or another port; the API is read-only and
        # holds nothing private, so the permissive header is a considered choice, not a shrug
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            if not os.path.exists(PAGE):
                return self._send(404, {"error": f"{PAGE} not found next to the server"})
            with open(PAGE, "rb") as fh:
                return self._send(200, fh.read(), "text/html; charset=utf-8")
        if path == "/health":
            return self._send(200, {"status": "ok" if STATE else "loading",
                                    "index": STATE.get("strategy"),
                                    "chunks": len(STATE.get("texts") or ()),
                                    "score_floor": SCORE_FLOOR,
                                    "coverage_floor": COVERAGE_FLOOR})
        self._send(404, {"error": f"no route {path}; try GET / or POST /ask"})

    def do_POST(self):
        if self.path.split("?")[0] != "/ask":
            return self._send(404, {"error": f"no route {self.path}; POST /ask"})
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._send(413, {"error": f"body over {MAX_BODY} bytes"})
        body = self.rfile.read(length) if length else b""
        try:
            query, stt_ms, provider, _ = question_from(
                body, self.headers.get("Content-Type", "application/json"))
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:                          # STT down, bad audio, bad json
            return self._send(502, {"error": f"could not get a question: {e}"})
        try:
            trace = answer(query, STATE["index"], STATE["texts"], qid="http",
                           parents=STATE["parents"], keep_hits=N_CITATIONS)
            payload = to_payload(trace, trace.meta.get("hits", []), STATE["texts"],
                                 stt_ms, provider)
            if wants_speech(body, self.headers.get("Content-Type", ""), self.path):
                voice = spoken(trace)
                payload.update(voice)
                payload["excluded_ms"]["tts"] = voice["tts_ms"]
        except Exception:
            traceback.print_exc()
            return self._send(500, {"error": "the serving path raised; see server log"})
        self.log_message("ask %-40.40s %6.1f ms %s", query,
                         trace.total_ms, "ABSTAIN " + str(trace.meta["gate"])
                         if trace.meta["abstain"] else "answered")
        self._send(200, payload)


def boot(strategy: str | None = None) -> dict:
    """Load the index and warm the encoder BEFORE the socket opens."""
    from d3.run import winner_dir
    strategy = strategy or winner_dir()
    if not os.path.exists(f"{strategy}/chunks.jsonl"):
        raise SystemExit(f"no index at {strategy} -- run `make chunking` first.")
    print(f"loading {strategy} ...", flush=True)
    ix, texts, parents = load_index(strategy)
    STATE.update({"index": ix, "texts": texts, "parents": parents,
                  "strategy": strategy})
    print(f"ready: {len(texts)} chunks, gate 2 floor {SCORE_FLOOR:.4f}, "
          f"gate 4 coverage floor {COVERAGE_FLOOR:.4f}", flush=True)
    return STATE


def main():
    port = int(os.getenv("PORT", "8000"))
    boot()
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"\n  demo page  http://localhost:{port}/\n"
          f"  endpoint   http://localhost:{port}/ask\n"
          f"  health     http://localhost:{port}/health\n\nctrl-c to stop", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


def demo():
    """Contract checks: the shape the page expects, and the two parsers."""
    from harness.spans import Trace
    t = Trace("x", {"query": "q", "answer": "a", "abstain": False, "gate": None, "reason": "",
                    "cited": "c1", "budget": {"degraded": [], "total_ms": 200.0},
                    "extractive": True, "cache_hit": False, "lexical_only": False})
    t.spans.update({"input_guards": 0.01, "embed_query": 7.0, "retrieve": 1.0})
    t.close()
    p = to_payload(t, [("c1", 0.9)], {"c1": "[en|Latin|en:1] the text"}, None)
    for k in ("transcript", "answer", "abstained", "abstain_gate", "citations",
              "timings_ms", "total_ms", "stt_ms", "degraded"):
        assert k in p, f"the page's contract needs {k}"
    for k in ("guard", "cache", "embed", "retrieve", "rerank", "assemble", "generate", "verify"):
        assert k in p["timings_ms"], f"the page draws a {k} bar"
    assert p["citations"][0]["chunk_id"] == "c1"
    assert p["citations"][0]["text"] == "the text", "s7's prefix must not reach the browser"
    assert p["stt_ms"] is None, "typed text has no STT leg"

    # stt_ms is reported beside total_ms and never inside it
    p2 = to_payload(t, [], {}, 412.0, "sarvam:batch")
    assert p2["stt_ms"] == 412.0 and p2["total_ms"] < 100, (p2["stt_ms"], p2["total_ms"])

    # the voice answers in the script the question was asked in
    assert script_lang("what is a corporation") == "en"
    assert script_lang("কর্পোরেশন কী") == "bn"
    assert script_lang("சிறந்த நார்ச்சத்து") == "ta"
    # the spoken string decides, not the question: a Bengali answer to a mostly-Latin
    # code-switched question is still read by a Bengali voice
    assert script_lang("কর্পোরেশন হল একটি সংস্থা") == "bn"
    assert script_lang("I can't help with that request.") == "en", "refusals are English lines"

    # speaking is opt-in: a vendor call and a second of latency are not a default
    assert wants_speech(b'{"text":"x"}', "application/json", "/ask") is False
    assert wants_speech(b'{"text":"x","speak":true}', "application/json", "/ask") is True
    assert wants_speech(b'{}', "application/json", "/ask?speak=1") is True
    assert wants_speech(b'not json', "application/json", "/ask") is False
    big = b"--B\r\nContent-Disposition: form-data; name=\"audio\"\r\n\r\n" + b"\x00" * 9000 + \
          b"\r\n--B\r\nContent-Disposition: form-data; name=\"speak\"\r\n\r\n1\r\n--B--\r\n"
    assert wants_speech(big, "multipart/form-data; boundary=B", "/ask") is True, \
        "the flag follows the audio part and must still be seen"

    q, ms, prov, _ = question_from(b'{"text": "  what is a corporation "}', "application/json")
    assert (q, ms, prov) == ("what is a corporation", None, ""), (q, ms, prov)
    for bad, why in ((b'{}', "empty json"), (b'{"text": "   "}', "whitespace only")):
        try:
            question_from(bad, "application/json")
            raise AssertionError(f"{why} must be rejected")
        except ValueError:
            pass

    body = (b"--B\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"c.webm\"\r\n"
            b"Content-Type: audio/webm\r\n\r\nAUDIOBYTES\r\n--B--\r\n")
    parts = parse_multipart(body, "multipart/form-data; boundary=B")
    assert parts["audio"] == b"AUDIOBYTES", parts
    print("server contract ok")


if __name__ == "__main__":
    demo() if "--selfcheck" in sys.argv else main()
