"""Ask the thing a question and watch the clock.

    make demo                                   # a real query from the frozen set
    python service/ask.py "what is a corporation"
    python service/ask.py --audio clip.wav      # mic path: STT, then t0
    python service/ask.py --json "..."          # the whole trace, for piping

What it prints is the answer OR the abstention with the gate that fired and the numbers
that fired it -- a refusal is a result here, not an error. Under the answer is the per-stage
breakdown and the total against the 200 ms budget.

The STT round trip is printed on its own line, outside the total, because that is where the
budget is drawn: t0 is the instant the server holds a final transcript. Printing one number
that adds the two together would be the single most misleading thing this file could do.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from harness.budget import TOTAL_MS
from service.pipeline import SCORE_FLOOR, answer, load_index

FROZEN = "data/queries_chunking.jsonl"
STAGES = ["input_guards", "embed_query", "retrieve", "rerank", "generate", "verify"]
SHORT = {"input_guards": "guards", "embed_query": "embed", "retrieve": "retrieve",
         "rerank": "rerank", "generate": "generate", "verify": "verify"}


def sample_query(n: int = 0) -> str:
    """A question the corpus can actually answer, so `make demo` shows the answer path.

    Taken from the frozen held-out set rather than invented: a hand-written demo query that
    happens to work is a demo of the query, not of the system."""
    if not os.path.exists(FROZEN):
        return "what is a corporation"
    with open(FROZEN, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    return rows[n % len(rows)]["query"] if rows else "what is a corporation"


def transcribe(path: str) -> tuple[str, float, str]:
    """The excluded leg. Returns (text, stt_ms, provider)."""
    from stt import sarvam
    with open(path, "rb") as fh:
        audio = fh.read()
    mime = sarvam.content_type_of(audio, path)
    t = sarvam.provider().transcribe(
        audio, filename=f"{os.path.basename(path)}.{sarvam.ext_for(mime)}", content_type=mime)
    return t["text"], t["stt_ms"], t["provider"]


def say_it(trace, path: str = "/tmp/mic_rag_answer.wav") -> dict | None:
    """Speak the result and play it. After t1, so it is never inside the measured window.

    Playback failing is not the request failing: the answer already happened, and a missing
    audio player is a fact about this laptop, not about the system under test."""
    from tts import sarvam as tts
    from service.server import script_lang
    text = tts.line_for(trace.meta)
    try:
        wav, ms, provider = tts.speak(trace.meta, script_lang(text))
    except Exception as e:
        print(f"spoken   : (tts failed: {str(e)[:90]})")
        return None
    with open(path, "wb") as fh:
        fh.write(wav)
    for player in (["afplay", path], ["aplay", "-q", path]):
        try:
            import subprocess
            subprocess.run(player, check=True, capture_output=True, timeout=60)
            break
        except Exception:
            continue
    return {"text": text, "tts_ms": ms, "provider": provider,
            "seconds": tts.duration_s(wav), "path": path}


def render(trace, stt_ms: float | None, stt_provider: str = "") -> str:
    m = trace.meta
    out = [f"question : {m['query']}"]
    if stt_ms is not None:
        how = (f"transcribed in {stt_ms:.0f} ms -- excluded leg, before t0"
               if stt_provider != "mock" else
               "transcribed by the MOCK -- its 0 ms is not a measurement")
        out.append(f"           [{how}]")
    if m["abstain"]:
        out.append(f"ABSTAINED: {m['gate']} -- {m['reason'] or 'no reason recorded'}")
        out.append("           (a refusal is a result: the gate that fired is named above)")
    else:
        out.append(f"answer   : {m['answer']}")
        if m.get("cited"):
            out.append(f"cited    : {m['cited']}")
    spans = "  ".join(f"{SHORT[s]} {trace.spans[s]:.2f}" for s in STAGES if s in trace.spans)
    out.append("")
    out.append(f"spans ms : {spans}")
    verdict = "within budget" if trace.total_ms <= TOTAL_MS else "OVER BUDGET"
    out.append(f"total    : {trace.total_ms:.2f} ms of {TOTAL_MS:.0f} ms -- {verdict}")
    if m.get("cache_hit"):
        out.append("           (served from the semantic cache: no stage ran)")
    if m.get("lexical_only"):
        out.append("           (encoder missed its deadline: retrieved lexically, gate 2 off)")
    if m.get("degraded"):
        out.append(f"           (degraded: {', '.join(m['budget']['degraded'])})")
    if stt_ms is not None and stt_provider != "mock":
        # only ever printed for a real round trip: adding a mock's 0 ms to the measured
        # window produces a mic-to-answer figure that is pure fiction, and it is precisely
        # the number someone would quote.
        out.append(f"end-to-end: {stt_ms + trace.total_ms:.0f} ms including the STT leg, "
                   f"which the 200 ms budget deliberately excludes")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="ask one question through the serving path")
    ap.add_argument("query", nargs="*", help="the question; omitted = one from the frozen set")
    ap.add_argument("--audio", help="transcribe this clip first (needs SARVAM_API_KEY, "
                                    "or STT_PROVIDER=mock to exercise the path offline)")
    ap.add_argument("--nth", type=int, default=0, help="which frozen query, when none is given")
    ap.add_argument("--json", action="store_true", help="print the whole trace as JSON")
    ap.add_argument("--speak", action="store_true",
                    help="say the answer out loud (Sarvam TTS; after t1, outside the budget)")
    args = ap.parse_args()

    from d3.run import winner_dir
    strategy = winner_dir()
    if not os.path.exists(f"{strategy}/chunks.jsonl"):
        raise SystemExit(f"no index at {strategy} -- run `make chunking` first "
                         "(it builds what this serves).")

    stt_ms, stt_provider = None, ""
    if args.audio:
        query, stt_ms, stt_provider = transcribe(args.audio)
        if not query:
            raise SystemExit(f"{args.audio} transcribed to nothing -- no question to ask.")
    else:
        query = " ".join(args.query) or sample_query(args.nth)

    ix, texts, parents = load_index(strategy)          # model load happens here, before t0
    trace = answer(query, ix, texts, qid="ask", parents=parents)
    voice = say_it(trace) if args.speak else None
    if args.json:
        row = trace.row()
        if voice:
            row["tts"] = voice
        print(json.dumps(row, ensure_ascii=False, indent=2))
    else:
        print(render(trace, stt_ms, stt_provider))
        if voice:
            print(f"spoken   : {voice['text'][:70]}")
            print(f"           {voice['tts_ms']:.0f} ms, {voice['seconds']:.1f}s of audio "
                  f"-> {voice['path']}  [after t1, outside the budget]")
        print(f"\nindex {strategy}, gate 2 floor {SCORE_FLOOR:.4f}")


def demo():
    """Rendering only -- the serving path has its own checks in service/pipeline.py."""
    from harness.spans import Trace
    t = Trace("x", {"query": "q", "abstain": False, "answer": "an answer", "cited": "c1",
                    "gate": None, "reason": "", "cache_hit": False, "lexical_only": False,
                    "degraded": False})
    t.spans["embed_query"] = 7.0
    t.close()
    out = render(t, None)
    assert "answer   : an answer" in out and "within budget" in out, out
    assert "end-to-end" not in out, "no STT leg means no combined number"

    t2 = Trace("y", {"query": "q", "abstain": True, "gate": "gate4_nli", "reason": "support=0.1",
                     "answer": "", "cache_hit": False, "lexical_only": False, "degraded": False})
    t2.close()
    out2 = render(t2, 412.0, "sarvam:batch")
    assert "ABSTAINED: gate4_nli -- support=0.1" in out2, out2
    # the two legs must stay visibly separate, and the combined number must be labelled
    assert "excluded leg, before t0" in out2 and "deliberately excludes" in out2, out2
    mock = render(t2, 0.0, "mock")
    assert "end-to-end" not in mock, "a mock leg must never produce a mic-to-answer number"
    assert "not a measurement" in mock, mock
    print("ask ok")


if __name__ == "__main__":
    demo() if "--selfcheck" in sys.argv else main()
